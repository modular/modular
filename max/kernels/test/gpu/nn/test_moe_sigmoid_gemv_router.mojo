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

from std.math import exp
from std.random import rand, seed

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from nn.moe import sigmoid_gemv_single_group_router
from std.testing import assert_almost_equal, assert_equal, assert_true


def test_sigmoid_gemv_router[
    num_tokens: Int,
    n_experts: Int,
    topk: Int,
    hidden_size: Int,
](ctx: DeviceContext) raises:
    """Checks the fused router against a float32 host reference."""
    comptime routed_scaling_factor = Float32(2.5)

    var hidden_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_tokens * hidden_size
    )
    var weight_host = ctx.enqueue_create_host_buffer[.float32](
        n_experts * hidden_size
    )
    var bias_host = ctx.enqueue_create_host_buffer[.float32](n_experts)
    var indices_host = ctx.enqueue_create_host_buffer[.int32](num_tokens * topk)
    var weights_host = ctx.enqueue_create_host_buffer[.float32](
        num_tokens * topk
    )
    ctx.synchronize()

    seed(7)
    rand(hidden_host.unsafe_ptr(), num_tokens * hidden_size, min=-1, max=1)
    rand(weight_host.unsafe_ptr(), n_experts * hidden_size, min=-0.05, max=0.05)
    # The bias only steers selection, so make it big enough to change winners.
    rand(bias_host.unsafe_ptr(), n_experts, min=-0.2, max=0.2)

    var hidden_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_tokens * hidden_size
    )
    var weight_dev = ctx.enqueue_create_buffer[.float32](
        n_experts * hidden_size
    )
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[.int32](num_tokens * topk)
    var weights_dev = ctx.enqueue_create_buffer[.float32](num_tokens * topk)
    ctx.enqueue_copy(hidden_dev, hidden_host)
    ctx.enqueue_copy(weight_dev, weight_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    sigmoid_gemv_single_group_router[n_experts, topk, True, "gpu"](
        TileTensor(indices_dev, row_major[num_tokens, topk]()),
        TileTensor(weights_dev, row_major[num_tokens, topk]()),
        TileTensor(hidden_dev, row_major[num_tokens, hidden_size]()).as_imm(),
        TileTensor(weight_dev, row_major[n_experts, hidden_size]()).as_imm(),
        TileTensor(bias_dev, row_major[n_experts]()).as_imm(),
        routed_scaling_factor,
        ctx,
    )
    ctx.enqueue_copy(indices_host, indices_dev)
    ctx.enqueue_copy(weights_host, weights_dev)
    ctx.synchronize()

    var scores = List[Float32](length=n_experts, fill=0)
    var taken = List[Bool](length=n_experts, fill=False)
    for t in range(num_tokens):
        for e in range(n_experts):
            var acc = Float32(0)
            for k in range(hidden_size):
                acc += (
                    hidden_host[t * hidden_size + k].cast[.float32]()
                    * weight_host[e * hidden_size + k]
                )
            scores[e] = 1 / (1 + exp(-acc))
            taken[e] = False

        # Reference top-k on the biased scores, highest first.
        var winners = List[Int]()
        var total = Float32(0)
        for _ in range(topk):
            var best = -1
            for e in range(n_experts):
                if not taken[e] and (
                    best < 0
                    or scores[e] + bias_host[e] > scores[best] + bias_host[best]
                ):
                    best = e
            taken[best] = True
            winners.append(best)
            total += scores[best]

        for r in range(topk):
            var got = Int(indices_host[t * topk + r])
            assert_equal(
                got,
                winners[r],
                msg=String("token ", t, " rank ", r),
            )
            assert_almost_equal(
                weights_host[t * topk + r],
                scores[winners[r]] / total * routed_scaling_factor,
                rtol=1e-4,
                msg=String("token ", t, " rank ", r),
            )


def test_sigmoid_gemv_router_no_tokens(ctx: DeviceContext) raises:
    """A zero-token batch, as in warmup, must not launch or write anything."""
    comptime n_experts = 128
    comptime topk = 6
    comptime hidden_size = 2688
    # One row of guard values the router must leave untouched.
    var indices_host = ctx.enqueue_create_host_buffer[.int32](topk)
    var weights_host = ctx.enqueue_create_host_buffer[.float32](topk)
    ctx.synchronize()
    for i in range(topk):
        indices_host[i] = -1
        weights_host[i] = Float32.MAX
    var hidden_dev = ctx.enqueue_create_buffer[.bfloat16](hidden_size)
    var weight_dev = ctx.enqueue_create_buffer[.float32](
        n_experts * hidden_size
    )
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[.int32](topk)
    var weights_dev = ctx.enqueue_create_buffer[.float32](topk)
    ctx.enqueue_copy(indices_dev, indices_host)
    ctx.enqueue_copy(weights_dev, weights_host)

    sigmoid_gemv_single_group_router[n_experts, topk, True, "gpu"](
        TileTensor(indices_dev, row_major((0, Idx[topk]))),
        TileTensor(weights_dev, row_major((0, Idx[topk]))),
        TileTensor(hidden_dev, row_major((0, Idx[hidden_size]))).as_imm(),
        TileTensor(weight_dev, row_major[n_experts, hidden_size]()).as_imm(),
        TileTensor(bias_dev, row_major[n_experts]()).as_imm(),
        Float32(2.5),
        ctx,
    )
    ctx.enqueue_copy(indices_host, indices_dev)
    ctx.enqueue_copy(weights_host, weights_dev)
    ctx.synchronize()
    for i in range(topk):
        assert_equal(indices_host[i], -1)
        assert_equal(weights_host[i], Float32.MAX)


def test_sigmoid_gemv_router_bf16_weight[
    num_tokens: Int, n_experts: Int, topk: Int, hidden_size: Int
](ctx: DeviceContext) raises:
    """Runs a bf16 gate weight whose shared-memory rows are 8- not 16-byte
    aligned, and checks the routing is well formed."""
    var hidden_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_tokens * hidden_size
    )
    var weight_dev = ctx.enqueue_create_buffer[.bfloat16](
        n_experts * hidden_size
    )
    var bias_dev = ctx.enqueue_create_buffer[.bfloat16](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[.int32](num_tokens * topk)
    var weights_dev = ctx.enqueue_create_buffer[.bfloat16](num_tokens * topk)
    with hidden_dev.map_to_host() as h:
        rand(h.unsafe_ptr(), num_tokens * hidden_size, min=-1, max=1)
    with weight_dev.map_to_host() as w:
        rand(w.unsafe_ptr(), n_experts * hidden_size, min=-0.05, max=0.05)
    bias_dev.enqueue_fill(0)

    sigmoid_gemv_single_group_router[n_experts, topk, True, "gpu"](
        TileTensor(indices_dev, row_major[num_tokens, topk]()),
        TileTensor(weights_dev, row_major[num_tokens, topk]()),
        TileTensor(hidden_dev, row_major[num_tokens, hidden_size]()).as_imm(),
        TileTensor(weight_dev, row_major[n_experts, hidden_size]()).as_imm(),
        TileTensor(bias_dev, row_major[n_experts]()).as_imm(),
        Float32(1),
        ctx,
    )
    with indices_dev.map_to_host() as idx, weights_dev.map_to_host() as wts:
        for t in range(num_tokens):
            var total = Float32(0)
            for r in range(topk):
                var e = Int(idx[t * topk + r])
                assert_true(
                    e >= 0 and e < n_experts,
                    msg=String("token ", t, " rank ", r),
                )
                for q in range(r):
                    assert_true(
                        Int(idx[t * topk + q]) != e,
                        msg=String("token ", t, " rank ", r),
                    )
                total += wts[t * topk + r].cast[.float32]()
            # Normalized weights sum to the scaling factor.
            assert_almost_equal(total, 1, rtol=2e-2, msg=String("token ", t))


def main() raises:
    with DeviceContext() as ctx:
        # Nemotron-3.5-Lightning: 128 experts, top-6, hidden 2688.
        test_sigmoid_gemv_router[1, 128, 6, 2688](ctx)
        test_sigmoid_gemv_router[7, 128, 6, 2688](ctx)
        test_sigmoid_gemv_router[64, 128, 6, 2688](ctx)
        test_sigmoid_gemv_router[300, 128, 6, 2688](ctx)
        # A wider expert count and a power-of-two top-k.
        test_sigmoid_gemv_router[16, 256, 8, 1024](ctx)
        test_sigmoid_gemv_router_no_tokens(ctx)
        # bf16 gate weight with hidden 128: 24-byte shared-memory rows.
        test_sigmoid_gemv_router_bf16_weight[9, 128, 6, 128](ctx)
