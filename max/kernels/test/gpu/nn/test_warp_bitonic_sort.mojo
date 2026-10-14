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

"""Test warp-level bitonic sort correctness in MoE router."""

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from nn.moe import (
    router_group_limited,
    single_group_router,
    single_group_router_eplb,
)
from std.testing import assert_equal


def test_warp_bitonic_sort_interleaved[
    n_experts: Int,
    n_groups: Int,
    topk_experts: Int,
    topk_group: Int,
](ctx: DeviceContext) raises:
    """Verify bitonic sort correctly orders experts interleaved across groups.

    Catches a bug where _warp_bitonic_sort used lane_id() instead of
    lane_id() % num_lanes, causing lanes 8-31 to compute wrong merge directions.
    """
    comptime experts_per_group = n_experts // n_groups

    var scores_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[.int32](topk_experts)
    var weights_dev = ctx.enqueue_create_buffer[.float32](topk_experts)
    var scores_host = ctx.enqueue_create_host_buffer[.float32](n_experts)

    ctx.synchronize()

    with bias_dev.map_to_host() as bias_map:
        for i in range(n_experts):
            bias_map[i] = 0.0

    var indices_tensor = TileTensor(
        indices_dev,
        row_major[1, topk_experts](),
    )
    var weights_tensor = TileTensor(
        weights_dev,
        row_major[1, topk_experts](),
    )
    var scores_tensor = TileTensor(
        scores_dev,
        row_major[1, n_experts](),
    )
    var bias_tensor = TileTensor(
        bias_dev,
        row_major[n_experts](),
    )

    # Test each starting_group to exercise all lane groups (0-7, 8-15, 16-23, 24-31).
    # For each iteration, 4 consecutive groups (g0-g3) are selected as top groups.
    # Within selected groups: position 0 gets scores 170-167, position 1 gets 160-157.
    # Example for starting_group=0 with 64 experts (8 groups of 8):
    #   g0=group0, g1=group1, g2=group2, g3=group3
    #   expert 0 (g0, pos0) -> score 167    expert 8  (g1, pos0) -> score 168
    #   expert 1 (g0, pos1) -> score 157    expert 9  (g1, pos1) -> score 158
    #   expert 16 (g2, pos0) -> score 169   expert 24 (g3, pos0) -> score 170
    #   expert 17 (g2, pos1) -> score 159   expert 25 (g3, pos1) -> score 160
    # Expected top-8 sorted by score: [24, 16, 8, 0, 25, 17, 9, 1]
    # This interleaving requires correct cross-group sorting to pass.
    for starting_group in range(n_groups):
        var g0 = starting_group
        var g1 = (starting_group + 1) % n_groups
        var g2 = (starting_group + 2) % n_groups
        var g3 = (starting_group + 3) % n_groups

        with scores_dev.map_to_host() as scores_map:
            for i in range(n_experts):
                var group, pos_in_group = divmod(i, experts_per_group)
                var score: Float32

                var rank = -1
                if group == g3:
                    rank = 0
                elif group == g2:
                    rank = 1
                elif group == g1:
                    rank = 2
                elif group == g0:
                    rank = 3

                if rank >= 0:
                    if pos_in_group == 0:
                        score = 170.0 - Float32(rank)
                    elif pos_in_group == 1:
                        score = 160.0 - Float32(rank)
                    else:
                        score = Float32(pos_in_group)
                else:
                    score = Float32(pos_in_group)

                scores_map[i] = score
                scores_host[i] = score

        var expected = List[Int]()
        expected.append(g3 * experts_per_group + 0)
        expected.append(g2 * experts_per_group + 0)
        expected.append(g1 * experts_per_group + 0)
        expected.append(g0 * experts_per_group + 0)
        expected.append(g3 * experts_per_group + 1)
        expected.append(g2 * experts_per_group + 1)
        expected.append(g1 * experts_per_group + 1)
        expected.append(g0 * experts_per_group + 1)

        router_group_limited[
            n_experts, topk_experts, n_groups, topk_group, True, target="gpu"
        ](
            indices_tensor,
            weights_tensor,
            scores_tensor.as_imm(),
            bias_tensor.as_imm(),
            Float32(1.0),
            ctx,
        )

        with indices_dev.map_to_host() as indices_map:
            for i in range(topk_experts):
                var actual_idx = Int(indices_map[i])
                var expected_idx = expected[i]
                assert_equal(
                    actual_idx,
                    expected_idx,
                    msg=String(
                        "starting_group=",
                        starting_group,
                        " index[",
                        i,
                        "]: got ",
                        actual_idx,
                        " (score=",
                        scores_host[actual_idx],
                        "), expected ",
                        expected_idx,
                        " (score=",
                        scores_host[expected_idx],
                        ")",
                    ),
                )


def test_router_group_limited_fused_scores[
    n_experts: Int,
    n_groups: Int,
    topk_experts: Int,
    topk_group: Int,
](ctx: DeviceContext) raises:
    """Verifies the fused `scores_input_fn` branch of `router_group_limited`.

    `expert_scores` holds a decoy ranking and the real scores exist only
    inside the lambda. The unfused branch therefore cannot produce the
    expected order, so this fails loudly if `scores_input_fn` ever stops
    resolving — the `comptime if` would otherwise take the else branch and
    read the tensor whose producer the fusion elided, silently.

    The lambda computes its scores rather than reading a captured tensor: a
    captured `TileTensor` does not survive the trip to the device through
    this kernel's `OptionalReg` closure parameter, which faults on AMD and
    reads a wrong pointer on Apple. That is a pre-existing defect in
    `router_group_limited`, unreachable from the graph-compiler caller, whose
    lambda closes over a kernel argument rather than a local.
    """
    var scores_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[.int32](topk_experts)
    var weights_dev = ctx.enqueue_create_buffer[.float32](topk_experts)

    ctx.synchronize()

    with bias_dev.map_to_host() as bias_map:
        for i in range(n_experts):
            bias_map[i] = 0.0

    # The decoy ascends with the expert index, so reading `expert_scores`
    # directly ranks the last group first — the inverse of what the lambda's
    # scores produce.
    with scores_dev.map_to_host() as decoy_map:
        for i in range(n_experts):
            decoy_map[i] = Float32(i)

    var indices_tensor = TileTensor(indices_dev, row_major[1, topk_experts]())
    var weights_tensor = TileTensor(weights_dev, row_major[1, topk_experts]())
    var scores_tensor = TileTensor(scores_dev, row_major[1, n_experts]())
    var bias_tensor = TileTensor(bias_dev, row_major[n_experts]())

    # Descends strictly with the expert index, so group 0 wins the group stage
    # and its experts come back in index order.
    @inline(.always)
    def scores_input_fn[
        width: Int
    ](coords: Coord) capturing -> SIMD[.float32, width]:
        comptime assert (
            coords.rank == 2
        ), "the expert-scores lambda contract is rank 2"
        return SIMD[.float32, width](
            Float32(n_experts) - Float32(Int(coords[1].value()))
        )

    router_group_limited[
        n_experts,
        topk_experts,
        n_groups,
        topk_group,
        True,
        target="gpu",
        scores_input_fn=scores_input_fn,
    ](
        indices_tensor,
        weights_tensor,
        scores_tensor.layout,
        bias_tensor.as_imm(),
        Float32(1.0),
        ctx,
    )

    # The lambda ranks expert 0 highest and descends, so the first
    # `topk_experts` come back in index order. Under the decoy the winners
    # would come from the last group instead.
    with indices_dev.map_to_host() as indices_map:
        for i in range(topk_experts):
            assert_equal(
                Int(indices_map[i]),
                i,
                msg=String(
                    "fused scores index[",
                    i,
                    "]: got ",
                    Int(indices_map[i]),
                    ", expected ",
                    i,
                    " (a decoy-ranked result means the unfused branch ran)",
                ),
            )


def test_single_group_router_fused_scores[
    n_experts: Int,
    topk_experts: Int,
](ctx: DeviceContext) raises:
    """Verifies the fused `scores_input_fn` branch of `single_group_router`.

    `expert_scores` holds the inverse of the lambda's ranking, per token, so
    the unfused branch returns each token's winners in the opposite order.
    Either token's assertion therefore fails if `scores_input_fn` stops
    resolving.

    The two tokens rank experts oppositely, which pins the token coordinate
    as well: a lambda that dropped `coords[0]` would hand both tokens the
    same ranking and fail on the second.

    The lambda computes its scores rather than reading a captured tensor,
    for the reason given on `test_router_group_limited_fused_scores`.
    """
    comptime num_tokens = 2
    var scores_dev = ctx.enqueue_create_buffer[.float32](num_tokens * n_experts)
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[.int32](
        num_tokens * topk_experts
    )
    var weights_dev = ctx.enqueue_create_buffer[.float32](
        num_tokens * topk_experts
    )

    ctx.synchronize()

    with bias_dev.map_to_host() as bias_map:
        for i in range(n_experts):
            bias_map[i] = 0.0

    # Exactly the inverse of the lambda, per token.
    with scores_dev.map_to_host() as decoy_map:
        for e in range(n_experts):
            decoy_map[e] = Float32(e)
            decoy_map[n_experts + e] = Float32(n_experts - e)

    var indices_tensor = TileTensor(
        indices_dev, row_major[num_tokens, topk_experts]()
    )
    var weights_tensor = TileTensor(
        weights_dev, row_major[num_tokens, topk_experts]()
    )
    var scores_tensor = TileTensor(
        scores_dev, row_major[num_tokens, n_experts]()
    )
    var bias_tensor = TileTensor(bias_dev, row_major[n_experts]())

    @inline(.always)
    def scores_input_fn[
        width: Int
    ](coords: Coord) capturing -> SIMD[.float32, width]:
        comptime assert (
            coords.rank == 2
        ), "the expert-scores lambda contract is rank 2"
        var token = Int(coords[0].value())
        var expert = Int(coords[1].value())
        var score = Float32(expert)
        if token == 0:
            score = Float32(n_experts - expert)
        return SIMD[.float32, width](score)

    single_group_router[
        n_experts,
        topk_experts,
        True,
        "gpu",
        scores_input_fn=scores_input_fn,
    ](
        indices_tensor,
        weights_tensor,
        scores_tensor.layout,
        bias_tensor.as_imm(),
        Float32(1.0),
        ctx,
    )
    ctx.synchronize()

    with indices_dev.map_to_host() as indices_map:
        for i in range(topk_experts):
            # Token 0 descends from expert 0; token 1 descends from the last.
            assert_equal(
                Int(indices_map[i]),
                i,
                msg=String(
                    "fused single-group token 0 index[",
                    i,
                    "]: got ",
                    Int(indices_map[i]),
                    ", expected ",
                    i,
                    " (an inverted result means the unfused branch ran)",
                ),
            )
            var expected_1 = n_experts - 1 - i
            assert_equal(
                Int(indices_map[topk_experts + i]),
                expected_1,
                msg=String(
                    "fused single-group token 1 index[",
                    i,
                    "]: got ",
                    Int(indices_map[topk_experts + i]),
                    ", expected ",
                    expected_1,
                    (
                        " (matching token 0 means the token coordinate was"
                        " dropped)"
                    ),
                ),
            )


def test_single_group_router_eplb_fused_scores[
    n_experts: Int,
    topk_experts: Int,
](ctx: DeviceContext) raises:
    """Verifies the fused `scores_input_fn` branch of the EPLB router.

    `log2phy` is the identity with one replica per logical expert, so the
    remap is a no-op and the physical indices equal the logical ones. That
    keeps the assertion on the coordinate the lambda receives rather than on
    the remap, which has its own coverage in the benchmark.
    """
    comptime num_tokens = 2
    comptime num_layers = 2
    comptime num_log = n_experts
    comptime max_replicas = 1

    var scores_dev = ctx.enqueue_create_buffer[.float32](num_tokens * n_experts)
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var phy_dev = ctx.enqueue_create_buffer[.int32](num_tokens * topk_experts)
    var log_dev = ctx.enqueue_create_buffer[.int32](num_tokens * topk_experts)
    var weights_dev = ctx.enqueue_create_buffer[.float32](
        num_tokens * topk_experts
    )
    var logcnt_dev = ctx.enqueue_create_buffer[.int32](num_layers * num_log)
    var log2phy_dev = ctx.enqueue_create_buffer[.int32](
        num_layers * num_log * max_replicas
    )
    var layer_idx_dev = ctx.enqueue_create_buffer[.int32](1)

    ctx.synchronize()

    with bias_dev.map_to_host() as bias_map:
        for i in range(n_experts):
            bias_map[i] = 0.0

    # Inverse of the lambda, per token, exactly as in the non-EPLB test.
    with scores_dev.map_to_host() as decoy_map:
        for e in range(n_experts):
            decoy_map[e] = Float32(e)
            decoy_map[n_experts + e] = Float32(n_experts - e)

    # One replica each, mapped to itself: physical == logical.
    with logcnt_dev.map_to_host() as logcnt_map:
        for i in range(num_layers * num_log):
            logcnt_map[i] = Int32(1)
    with log2phy_dev.map_to_host() as log2phy_map:
        for L in range(num_layers):
            for log in range(num_log):
                log2phy_map[L * num_log + log] = Int32(log)
    with layer_idx_dev.map_to_host() as layer_idx_map:
        layer_idx_map[0] = Int32(0)

    var expert_indices = TileTensor(
        phy_dev, row_major[num_tokens, topk_experts]()
    )
    var expert_indices_log = TileTensor(
        log_dev, row_major[num_tokens, topk_experts]()
    )
    var expert_weights = TileTensor(
        weights_dev, row_major[num_tokens, topk_experts]()
    )
    var expert_scores = TileTensor(
        scores_dev, row_major[num_tokens, n_experts]()
    )
    var expert_bias = TileTensor(bias_dev, row_major[n_experts]())
    var logcnt = TileTensor(logcnt_dev, row_major[num_layers, num_log]())
    var log2phy = TileTensor(
        log2phy_dev, row_major[num_layers, num_log, max_replicas]()
    )
    var layer_idx = TileTensor(layer_idx_dev, row_major[1]())

    @inline(.always)
    def scores_input_fn[
        width: Int
    ](coords: Coord) capturing -> SIMD[.float32, width]:
        comptime assert (
            coords.rank == 2
        ), "the expert-scores lambda contract is rank 2"
        var token = Int(coords[0].value())
        var expert = Int(coords[1].value())
        var score = Float32(expert)
        if token == 0:
            score = Float32(n_experts - expert)
        return SIMD[.float32, width](score)

    single_group_router_eplb[
        n_routed_experts=n_experts,
        n_experts_per_tok=topk_experts,
        norm_weights=True,
        num_log=num_log,
        max_replicas=max_replicas,
        hash_decorrelate=False,
        target="gpu",
        scores_input_fn=scores_input_fn,
    ](
        expert_indices,
        expert_indices_log,
        expert_weights,
        expert_scores.layout,
        expert_bias.as_imm(),
        logcnt.as_imm(),
        log2phy.as_imm(),
        layer_idx.as_imm(),
        Float32(1.0),
        ctx,
    )
    ctx.synchronize()

    with log_dev.map_to_host() as log_map:
        for i in range(topk_experts):
            assert_equal(
                Int(log_map[i]),
                i,
                msg=String(
                    "fused eplb token 0 logical[",
                    i,
                    "]: got ",
                    Int(log_map[i]),
                    ", expected ",
                    i,
                    " (an inverted result means the unfused branch ran)",
                ),
            )
            var expected_1 = n_experts - 1 - i
            assert_equal(
                Int(log_map[topk_experts + i]),
                expected_1,
                msg=String(
                    "fused eplb token 1 logical[",
                    i,
                    "]: got ",
                    Int(log_map[topk_experts + i]),
                    ", expected ",
                    expected_1,
                    (
                        " (matching token 0 means the token coordinate was"
                        " dropped)"
                    ),
                ),
            )


def test_single_group_router[
    n_experts: Int,
    topk_experts: Int,
    bias_val: Float32,
    routed_scaling_factor: Float32,
](ctx: DeviceContext) raises:
    """Verify single group router correctly routes experts."""
    var num_warps = n_experts // 32
    var scores_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[.int32](topk_experts)
    var weights_dev = ctx.enqueue_create_buffer[.float32](topk_experts)
    var scores_host = ctx.enqueue_create_host_buffer[.float32](n_experts)
    ctx.synchronize()

    with bias_dev.map_to_host() as bias_map:
        for i in range(n_experts):
            bias_map[i] = bias_val

    var indices_tensor = TileTensor(indices_dev, row_major[1, topk_experts]())
    var weights_tensor = TileTensor(weights_dev, row_major[1, topk_experts]())
    var scores_tensor = TileTensor(scores_dev, row_major[1, n_experts]())
    var bias_tensor = TileTensor(bias_dev, row_major[n_experts]())

    # Rotate which warps hold the top-k winners so every smem slot gets tested.
    # For starting_warp=W, winners are the first expert in warps W, W+1, ..., W+k-1
    # (wrapping around). Winner in rank r gets score (1000 - r*10), all others get 1.
    for starting_warp in range(num_warps):
        with scores_dev.map_to_host() as scores_map:
            for i in range(n_experts):
                scores_map[i] = 1.0
                scores_host[i] = 1.0

            for r in range(topk_experts):
                var warp = (starting_warp + r) % num_warps
                var expert_idx = warp * 32  # first expert in that warp
                var score = 1000.0 - Float32(r) * 10.0
                scores_map[expert_idx] = score
                scores_host[expert_idx] = score

        # Build expected: sorted descending by score, i.e. rank 0 first.
        var expected = List[Int]()
        for r in range(topk_experts):
            var warp = (starting_warp + r) % num_warps
            expected.append(warp * 32)

        single_group_router[n_experts, topk_experts, True, "gpu"](
            indices_tensor,
            weights_tensor,
            scores_tensor.as_imm(),
            bias_tensor.as_imm(),
            routed_scaling_factor,
            ctx,
        )
        ctx.synchronize()

        with indices_dev.map_to_host() as indices_map:
            for i in range(topk_experts):
                var actual_idx = Int(indices_map[i])
                var expected_idx = expected[i]
                assert_equal(
                    actual_idx,
                    expected_idx,
                    msg=String(
                        "starting_warp=",
                        starting_warp,
                        " index[",
                        i,
                        "]: got ",
                        actual_idx,
                        " (score=",
                        scores_host[actual_idx],
                        "), expected ",
                        expected_idx,
                        " (score=",
                        scores_host[expected_idx],
                        ")",
                    ),
                )


def test_single_group_router_raw_score_used_for_weights[
    n_experts: Int,
    topk_experts: Int,
](ctx: DeviceContext) raises:
    """Weights are computed from raw scores, not biased scores.

    This is the root cause of the decoder emitting None tokens.
    Sets bias[i] = large value so biased score >> raw score.
    If the kernel uses biased scores for weights, the normalized
    weights will be wrong and downstream hidden states will shift
    into special-token logit regions.

    Uses integer-valued raw scores so expected weights are exact.
    """
    var scores_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[.int32](topk_experts)
    var weights_dev = ctx.enqueue_create_buffer[.float32](topk_experts)
    ctx.synchronize()

    # Raw scores: experts 0..k-1 get scores k, k-1, ..., 1
    # Bias: all experts get 1000.0 so biased >> raw
    # Without bias, top-k = experts 0..k-1 in descending score order
    # Weights must use raw scores, not biased scores
    with scores_dev.map_to_host() as s, bias_dev.map_to_host() as b:
        for i in range(n_experts):
            b[i] = 1000.0
            s[i] = Float32(topk_experts - i) if i < topk_experts else 0.0

    var indices_tensor = TileTensor(indices_dev, row_major[1, topk_experts]())
    var weights_tensor = TileTensor(weights_dev, row_major[1, topk_experts]())
    var scores_tensor = TileTensor(scores_dev, row_major[1, n_experts]())
    var bias_tensor = TileTensor(bias_dev, row_major[n_experts]())

    single_group_router[n_experts, topk_experts, True, "gpu"](
        indices_tensor,
        weights_tensor,
        scores_tensor.as_imm(),
        bias_tensor.as_imm(),
        Float32(1.0),
        ctx,
    )
    ctx.synchronize()

    # raw_sum = k + (k-1) + ... + 1 = k*(k+1)/2
    var raw_sum = Float32(topk_experts * (topk_experts + 1)) / 2.0

    with indices_dev.map_to_host() as idx_map, weights_dev.map_to_host() as wgt_map:
        for i in range(topk_experts):
            assert_equal(
                Int(idx_map[i]),
                i,
                msg=String(
                    "index[", i, "]: got ", Int(idx_map[i]), " expected ", i
                ),
            )
            # expected weight = raw_score / raw_sum = (k-i) / raw_sum
            # all values exactly representable in float32
            var expected_weight = Float32(topk_experts - i) / raw_sum
            assert_equal(
                wgt_map[i],
                expected_weight,
                msg=String(
                    "weight[",
                    i,
                    "]: got ",
                    wgt_map[i],
                    " expected ",
                    expected_weight,
                    ". Non-equal weights mean biased score was used.",
                ),
            )


def test_single_group_router_scores_from_layout_and_lambda[
    n_experts: Int,
    topk_experts: Int,
](ctx: DeviceContext) raises:
    """Scores come only from the lambda; the router sees just their layout.

    Token `t` scores expert `e` as `n_experts - (e - t) mod n_experts`, so its
    top-k are experts `t, t+1, ..., t+k-1` with raw scores `n, n-1, ...`.
    """
    var num_tokens = 3
    var bias_dev = ctx.enqueue_create_buffer[.float32](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[.int32](
        num_tokens * topk_experts
    )
    var weights_dev = ctx.enqueue_create_buffer[.float32](
        num_tokens * topk_experts
    )
    bias_dev.enqueue_fill(0.0)

    var out_layout = row_major(Coord(num_tokens, Idx[topk_experts]))
    var indices_tensor = TileTensor(indices_dev, out_layout)
    var weights_tensor = TileTensor(weights_dev, out_layout)
    var bias_tensor = TileTensor(bias_dev, row_major[n_experts]())

    def scores_fn[width: Int](coords: Coord) capturing -> SIMD[.float32, width]:
        comptime assert (
            coords.rank == 2
        ), "the expert-scores lambda contract is rank 2"
        var rotated = (
            Int(coords[1].value()) - Int(coords[0].value()) + n_experts
        ) % n_experts
        return SIMD[.float32, width](Float32(n_experts - rotated))

    single_group_router[
        n_experts, topk_experts, True, "gpu", scores_input_fn=scores_fn
    ](
        indices_tensor,
        weights_tensor,
        row_major(Coord(num_tokens, Idx[n_experts])),
        bias_tensor.as_imm(),
        Float32(1.0),
        ctx,
    )
    ctx.synchronize()

    var raw_sum: Float32 = 0
    for r in range(topk_experts):
        raw_sum += Float32(n_experts - r)

    with indices_dev.map_to_host() as idx_map, weights_dev.map_to_host() as wgt_map:
        for t in range(num_tokens):
            for r in range(topk_experts):
                var i = t * topk_experts + r
                assert_equal(Int(idx_map[i]), t + r)
                assert_equal(wgt_map[i], Float32(n_experts - r) / raw_sum)


def main() raises:
    with DeviceContext() as ctx:
        # academic-ds-9b: 64 experts, 8 groups, 8 experts/group
        test_warp_bitonic_sort_interleaved[64, 8, 8, 4](ctx)
        # deepseek-r1: 256 experts, 8 groups, 32 experts/group
        test_warp_bitonic_sort_interleaved[256, 8, 8, 4](ctx)

        # single_group router (Kimi K2.5)
        test_single_group_router[384, 8, 1.0, 1.0](ctx)
        test_single_group_router_raw_score_used_for_weights[384, 8](ctx)
        # Nemotron-H: a top-k that is not a power of two.
        test_single_group_router_raw_score_used_for_weights[128, 6](ctx)
        test_single_group_router_scores_from_layout_and_lambda[384, 8](ctx)
        # GLM-5.3-Flash: 288 experts is not a whole number of 64-lane
        # wavefronts, so the launch pads to the next one.
        test_single_group_router[288, 8, 1.0, 2.5](ctx)
        test_single_group_router_raw_score_used_for_weights[288, 8](ctx)

        # Fused `scores_input_fn` branch of each router that offers one.
        test_router_group_limited_fused_scores[64, 8, 8, 4](ctx)
        test_single_group_router_fused_scores[384, 8](ctx)
        test_single_group_router_eplb_fused_scores[384, 8](ctx)
