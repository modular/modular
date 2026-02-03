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

from gpu.host import DeviceContext
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.moe import router_group_limited
from testing import assert_equal


fn test_warp_bitonic_sort_interleaved[
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

    var scores_dev = ctx.enqueue_create_buffer[DType.float32](n_experts)
    var bias_dev = ctx.enqueue_create_buffer[DType.float32](n_experts)
    var indices_dev = ctx.enqueue_create_buffer[DType.int32](topk_experts)
    var weights_dev = ctx.enqueue_create_buffer[DType.float32](topk_experts)
    var scores_host = ctx.enqueue_create_host_buffer[DType.float32](n_experts)

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
                var group = i // experts_per_group
                var pos_in_group = i % experts_per_group
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
            scores_tensor.as_immut(),
            bias_tensor.as_immut(),
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


def main():
    with DeviceContext() as ctx:
        # academic-ds-9b: 64 experts, 8 groups, 8 experts/group
        test_warp_bitonic_sort_interleaved[64, 8, 8, 4](ctx)
        # deepseek-r1: 256 experts, 8 groups, 32 experts/group
        test_warp_bitonic_sort_interleaved[256, 8, 8, 4](ctx)
