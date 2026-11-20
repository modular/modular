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
from nn.moe import (
    sigmoid_gemv_group_limited_router,
    sigmoid_gemv_single_group_router,
)
from std.testing import assert_almost_equal, assert_equal, assert_true


def test_sigmoid_gemv_router[
    num_tokens: Int,
    n_experts: Int,
    topk: Int,
    hidden_size: Int,
    weight_type: DType = .float32,
    ties: Bool = False,
](ctx: DeviceContext) raises:
    """Checks the fused router, accumulating in float32, against a float32
    host reference.

    With `ties`, experts 3, 11, 150 and 287 share one gate row and a bias of
    1.0, so their scores are equal: every token must pick them with equal
    weights, in ascending id order.
    """
    comptime routed_scaling_factor = Float32(2.5)
    comptime assert not ties or n_experts > 287, "the tied experts must exist"
    var tied: List[Int] = [3, 11, 150, 287]

    var hidden_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_tokens * hidden_size
    )
    var weight_host = ctx.enqueue_create_host_buffer[weight_type](
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
    comptime if ties:
        for i in range(len(tied)):
            for k in range(hidden_size):
                weight_host[tied[i] * hidden_size + k] = weight_host[
                    tied[0] * hidden_size + k
                ]
            bias_host[tied[i]] = 1.0

    var hidden_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_tokens * hidden_size
    )
    var weight_dev = ctx.enqueue_create_buffer[weight_type](
        n_experts * hidden_size
    )
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[.int32](num_tokens * topk)
    var weights_dev = ctx.enqueue_create_buffer[.float32](num_tokens * topk)
    ctx.enqueue_copy(hidden_dev, hidden_host)
    ctx.enqueue_copy(weight_dev, weight_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    sigmoid_gemv_single_group_router[n_experts, topk, True](
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
                    * weight_host[e * hidden_size + k].cast[.float32]()
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

        comptime if ties:
            # The tied experts take consecutive ranks in id order, and their
            # equal scores give bit-equal weights.
            var first = -1
            for r in range(topk):
                if Int(indices_host[t * topk + r]) == tied[0]:
                    first = r
            assert_true(
                first >= 0 and first + len(tied) <= topk,
                msg=String("token ", t, " skips a tied expert"),
            )
            for i in range(len(tied)):
                var r = t * topk + first + i
                assert_equal(
                    Int(indices_host[r]), tied[i], msg=String("token ", t)
                )
                assert_equal(
                    weights_host[r],
                    weights_host[t * topk + first],
                    msg=String("token ", t),
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

    sigmoid_gemv_single_group_router[n_experts, topk, True](
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

    sigmoid_gemv_single_group_router[n_experts, topk, True](
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


def test_sigmoid_gemv_group_limited_router[
    num_tokens: Int,
    n_experts: Int,
    topk: Int,
    hidden_size: Int,
    n_groups: Int,
    topk_group: Int,
](ctx: DeviceContext) raises:
    """Checks the fused group-limited router and its packed output against a
    float32 host reference."""
    comptime routed_scaling_factor = Float32(2.5)
    comptime group_size = n_experts // n_groups

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
    var packed_host = ctx.enqueue_create_host_buffer[.float32](
        num_tokens * 2 * topk
    )
    ctx.synchronize()

    seed(11)
    rand(hidden_host.unsafe_ptr(), num_tokens * hidden_size, min=-1, max=1)
    rand(weight_host.unsafe_ptr(), n_experts * hidden_size, min=-0.05, max=0.05)
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
    var packed_dev = ctx.enqueue_create_buffer[.float32](num_tokens * 2 * topk)
    ctx.enqueue_copy(hidden_dev, hidden_host)
    ctx.enqueue_copy(weight_dev, weight_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    var packed_t = TileTensor(packed_dev, row_major[num_tokens, 2 * topk]())
    sigmoid_gemv_group_limited_router[
        n_experts, topk, n_groups, topk_group, True, packed=True
    ](
        TileTensor(indices_dev, row_major[num_tokens, topk]()),
        TileTensor(weights_dev, row_major[num_tokens, topk]()),
        TileTensor(hidden_dev, row_major[num_tokens, hidden_size]()).as_imm(),
        TileTensor(weight_dev, row_major[n_experts, hidden_size]()).as_imm(),
        TileTensor(bias_dev, row_major[n_experts]()).as_imm(),
        routed_scaling_factor,
        ctx,
        packed_t.ptr.unsafe_origin_cast[MutUntrackedOrigin](),
    )
    ctx.enqueue_copy(indices_host, indices_dev)
    ctx.enqueue_copy(weights_host, weights_dev)
    ctx.enqueue_copy(packed_host, packed_dev)
    ctx.synchronize()

    var scores = List[Float32](length=n_experts, fill=0)
    var taken = List[Bool](length=n_experts, fill=False)
    var group_kept = List[Bool](length=n_groups, fill=False)
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

        # A group scores the sum of its two best biased experts.
        var group_score = List[Float32](length=n_groups, fill=0)
        for g in range(n_groups):
            var first = Float32(-1e30)
            var second = Float32(-1e30)
            for e in range(g * group_size, (g + 1) * group_size):
                var b = scores[e] + bias_host[e]
                if b > first:
                    second = first
                    first = b
                elif b > second:
                    second = b
            group_score[g] = first + second
            group_kept[g] = False
        for _ in range(topk_group):
            var best = -1
            for g in range(n_groups):
                if not group_kept[g] and (
                    best < 0 or group_score[g] > group_score[best]
                ):
                    best = g
            group_kept[best] = True

        var winners = List[Int]()
        var total = Float32(0)
        for _ in range(topk):
            var best = -1
            for e in range(n_experts):
                if (
                    group_kept[e // group_size]
                    and not taken[e]
                    and (
                        best < 0
                        or scores[e] + bias_host[e]
                        > scores[best] + bias_host[best]
                    )
                ):
                    best = e
            taken[best] = True
            winners.append(best)
            total += scores[best]

        for r in range(topk):
            var msg = String("token ", t, " rank ", r)
            var got = Int(indices_host[t * topk + r])
            assert_equal(got, winners[r], msg=msg)
            assert_almost_equal(
                weights_host[t * topk + r],
                scores[winners[r]] / total * routed_scaling_factor,
                rtol=1e-4,
                msg=msg,
            )
            assert_equal(packed_host[t * 2 * topk + r], Float32(got), msg=msg)
            assert_equal(
                packed_host[t * 2 * topk + topk + r],
                weights_host[t * topk + r],
                msg=msg,
            )


def main() raises:
    with DeviceContext() as ctx:
        # Nemotron-3.5-Lightning: 128 experts, top-6, hidden 2688.
        test_sigmoid_gemv_router[1, 128, 6, 2688](ctx)
        test_sigmoid_gemv_router[7, 128, 6, 2688](ctx)
        test_sigmoid_gemv_router[64, 128, 6, 2688](ctx)
        test_sigmoid_gemv_router[300, 128, 6, 2688](ctx)
        # A wider expert count and a power-of-two top-k.
        test_sigmoid_gemv_router[16, 256, 8, 1024](ctx)
        # GLM-5.3-Flash's bf16 gate, accumulated in float32: 288 experts,
        # top-8, hidden 4096. On NVIDIA GPUs the logits come from tensor
        # cores up to 128 rows (one row, a ragged second tile, a device's
        # rows at MTP 5 with 32 requests, the most rows the kernel takes) and
        # from `matmul` above.
        test_sigmoid_gemv_router[1, 288, 8, 4096, .bfloat16](ctx)
        test_sigmoid_gemv_router[29, 288, 8, 4096, .bfloat16](ctx)
        test_sigmoid_gemv_router[48, 288, 8, 4096, .bfloat16](ctx)
        test_sigmoid_gemv_router[128, 288, 8, 4096, .bfloat16](ctx)
        test_sigmoid_gemv_router[130, 288, 8, 4096, .bfloat16](ctx)
        test_sigmoid_gemv_router[5, 288, 8, 4096, .bfloat16, ties=True](ctx)
        # A hidden size the tensor-core slices do not divide keeps the
        # CUDA-core split-K GEMV, accumulating the bf16 gate in float32: one
        # warp a split up to 16 rows, shared-memory tiles above.
        test_sigmoid_gemv_router[7, 128, 6, 2688, .bfloat16](ctx)
        test_sigmoid_gemv_router[33, 128, 6, 2688, .bfloat16](ctx)
        test_sigmoid_gemv_router_no_tokens(ctx)
        # bf16 gate weight with hidden 128: 24-byte shared-memory rows.
        test_sigmoid_gemv_router_bf16_weight[9, 128, 6, 128](ctx)
        # Beam-1.0: 256 experts in 8 groups, 4 kept, top-8, hidden 5120.
        test_sigmoid_gemv_group_limited_router[1, 256, 8, 5120, 8, 4](ctx)
        test_sigmoid_gemv_group_limited_router[16, 256, 8, 5120, 8, 4](ctx)
        test_sigmoid_gemv_group_limited_router[37, 256, 8, 5120, 8, 4](ctx)
