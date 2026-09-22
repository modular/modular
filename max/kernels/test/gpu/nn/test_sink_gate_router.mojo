# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
#
# `sink_gate_router` against an independent host reference. The kernel selects
# the top `n_experts_per_tok` routed experts by `sigmoid(logit) + bias`, then
# softmax-normalizes the log-sigmoid of those experts' raw logits together with
# the always-on sink logits. A wrong index routes a token to the wrong expert,
# which is a large discrete error rather than a rounding one, so indices are
# compared exactly and only the weights carry a tolerance.

from std.math import exp
from std.random import seed

from internal_utils import assert_almost_equal, assert_equal
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from max.gpu.host import DeviceContext
from nn.moe import sink_gate_router


def sigmoid_ref(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))


def test_sink_gate_router[
    n_routed: Int,
    topk: Int,
    n_shared: Int,
    dtype: DType = .float32,
    pad: Int = 0,
](num_tokens: Int, ctx: DeviceContext) raises:
    comptime n_total = n_routed + n_shared
    # Padding for alignment gives the router wider rows; the extra columns
    # hold no expert and must not be read.
    comptime n_stride = n_total + pad
    comptime route_scale: Float32 = 8.0
    comptime global_scale_val: Float32 = 1.3

    var logits_host = ctx.enqueue_create_host_buffer[dtype](
        num_tokens * n_stride
    )
    var bias_host = ctx.enqueue_create_host_buffer[.float32](n_routed)
    var gscale_host = ctx.enqueue_create_host_buffer[.float32](1)

    # Continuous random values keep the selection order unambiguous: ties would
    # be broken by lower index in the kernel but are measure-zero here.
    random(
        TileTensor(logits_host, row_major(Coord(num_tokens, Idx[n_stride]))),
        min=-4.0,
        max=4.0,
    )
    # Poison the padding: this would take every top-k slot if scored.
    for t in range(num_tokens):
        for c in range(n_total, n_stride):
            logits_host[t * n_stride + c] = 1.0e4
    random(TileTensor(bias_host, row_major(Idx[n_routed])), min=-0.1, max=0.1)
    gscale_host[0] = global_scale_val

    var logits_dev = ctx.enqueue_create_buffer[dtype](num_tokens * n_stride)
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_routed)
    var gscale_dev = ctx.enqueue_create_buffer[.float32](1)
    var idx_dev = ctx.enqueue_create_buffer[.int32](num_tokens * topk)
    var w_dev = ctx.enqueue_create_buffer[dtype](num_tokens * topk)
    var sink_dev = ctx.enqueue_create_buffer[dtype](num_tokens * n_shared)
    ctx.enqueue_copy(logits_dev, logits_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    ctx.enqueue_copy(gscale_dev, gscale_host)

    sink_gate_router[n_routed, topk, n_shared, "gpu"](
        TileTensor(idx_dev, row_major(Coord(num_tokens, Idx[topk]))),
        TileTensor(w_dev, row_major(Coord(num_tokens, Idx[topk]))),
        TileTensor(sink_dev, row_major(Coord(num_tokens, Idx[n_shared]))),
        TileTensor(
            logits_dev, row_major(Coord(num_tokens, Idx[n_stride]))
        ).as_imm(),
        TileTensor(bias_dev, row_major(Idx[n_routed])).as_imm(),
        TileTensor(gscale_dev, row_major(Idx[1])).as_imm(),
        route_scale,
        ctx,
    )

    var idx_out = ctx.enqueue_create_host_buffer[.int32](num_tokens * topk)
    var w_out = ctx.enqueue_create_host_buffer[dtype](num_tokens * topk)
    var sink_out = ctx.enqueue_create_host_buffer[dtype](num_tokens * n_shared)
    ctx.enqueue_copy(idx_out, idx_dev)
    ctx.enqueue_copy(w_out, w_dev)
    ctx.enqueue_copy(sink_out, sink_dev)
    ctx.synchronize()

    var idx_ref = ctx.enqueue_create_host_buffer[.int32](num_tokens * topk)
    var w_ref = ctx.enqueue_create_host_buffer[dtype](num_tokens * topk)
    var sink_ref = ctx.enqueue_create_host_buffer[dtype](num_tokens * n_shared)

    for t in range(num_tokens):
        # Score the logits as stored, so the oracle sees the same rounding
        # the kernel does.
        var sigmoids = List[Float32]()
        for c in range(n_total):
            sigmoids.append(sigmoid_ref(Float32(logits_host[t * n_stride + c])))

        var winners = List[Int]()
        for _ in range(topk):
            var best = -1
            var best_score = Float32(0)
            for e in range(n_routed):
                if e in winners:
                    continue
                var s = sigmoids[e] + bias_host[e]
                if best == -1 or s > best_score:
                    best = e
                    best_score = s
            winners.append(best)

        for j in range(topk):
            idx_ref[t * topk + j] = Int32(winners[j])

        # The kernel's log-space softmax equals a plain sigmoid share.
        # Asserting the identity keeps this oracle independent of the
        # kernel's own expression.
        var total = Float32(0)
        for j in range(topk):
            total += sigmoids[winners[j]]
        for s in range(n_shared):
            total += sigmoids[n_routed + s]
        var factor = route_scale * global_scale_val / total

        for j in range(topk):
            w_ref[t * topk + j] = (sigmoids[winners[j]] * factor).cast[dtype]()
        for s in range(n_shared):
            var sink = sigmoids[n_routed + s] * factor
            sink_ref[t * n_shared + s] = sink.cast[dtype]()

    assert_equal(
        idx_out.as_span(),
        idx_ref.as_span(),
        "expert index mismatch",
        shape=[num_tokens, topk],
    )
    # A bf16 store rounds the f32 result, so allow it about two ulp.
    comptime rtol = 1e-5 if dtype == .float32 else 1e-2
    comptime atol = 1e-6 if dtype == .float32 else 1e-4
    assert_almost_equal(
        w_out.as_span(),
        w_ref.as_span(),
        "expert weight mismatch",
        shape=[num_tokens, topk],
        rtol=rtol,
        atol=atol,
    )
    assert_almost_equal(
        sink_out.as_span(),
        sink_ref.as_span(),
        "sink weight mismatch",
        shape=[num_tokens, n_shared],
        rtol=rtol,
        atol=atol,
    )


def test_every_shape[dtype: DType](ctx: DeviceContext) raises:
    # Inkling-Small geometry: 256 routed experts, 6 selected, 2 sinks.
    test_sink_gate_router[256, 6, 2, dtype](1, ctx)
    test_sink_gate_router[256, 6, 2, dtype](17, ctx)
    test_sink_gate_router[256, 6, 2, dtype](64, ctx)
    # k_total = 4, and a routed count of exactly one AMD wavefront.
    test_sink_gate_router[64, 2, 2, dtype](5, ctx)


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        test_every_shape[.float32](ctx)
        # The kernel scores in f32 either way, so bf16 meets the same oracle
        # up to the store rounding.
        test_every_shape[.bfloat16](ctx)
        # Inkling pads its 258 gate rows to 264 for an aligned router GEMM.
        test_sink_gate_router[256, 6, 2, pad=6](17, ctx)
