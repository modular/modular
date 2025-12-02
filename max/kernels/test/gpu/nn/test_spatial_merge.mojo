# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from gpu.host import DeviceContext
from testing import assert_equal
from nn.spatial_merge import spatial_merge
from layout import RuntimeLayout, LayoutTensor, Layout, UNKNOWN_VALUE
from utils import IndexList


def test_spatial_merge(ctx: DeviceContext):
    comptime dtype = DType.float32
    comptime merge_size = 2
    comptime hidden_size = 4
    comptime batch_size = 2

    # Batch item 0: t=1, h=4, w=4
    # Batch item 1: t=2, h=2, w=2
    var grid_thw_list: List[Int64] = [1, 4, 4, 2, 2, 2]

    var total_input_patches = 20
    var total_output_patches = 6
    var C_out = hidden_size * merge_size * merge_size

    # Define layouts
    comptime input_static_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    var input_shape = IndexList[2](total_input_patches, hidden_size)
    var input_runtime_layout = RuntimeLayout[input_static_layout].row_major(
        input_shape
    )

    comptime output_static_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE
    )
    var output_shape = IndexList[2](total_output_patches, C_out)
    var output_runtime_layout = RuntimeLayout[output_static_layout].row_major(
        output_shape
    )

    comptime grid_thw_static_layout = Layout.row_major(batch_size, 3)
    var grid_thw_shape = IndexList[2](batch_size, 3)
    var grid_thw_runtime_layout = RuntimeLayout[
        grid_thw_static_layout
    ].row_major(grid_thw_shape)

    # Create device buffers
    var input_device = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var output_device = ctx.enqueue_create_buffer[dtype](
        output_shape.flattened_length()
    )
    var grid_thw_device = ctx.enqueue_create_buffer[DType.int64](
        grid_thw_shape.flattened_length()
    )

    # Initialize input data on host
    with input_device.map_to_host() as input_host:
        var input_host_tensor = LayoutTensor[dtype, input_static_layout](
            input_host, input_runtime_layout
        )
        for patch_idx in range(total_input_patches):
            for feat_idx in range(hidden_size):
                input_host_tensor[patch_idx, feat_idx] = Float32(
                    patch_idx * 100 + feat_idx
                )

    # Initialize grid_thw data on host
    with grid_thw_device.map_to_host() as grid_thw_host:
        var grid_thw_host_tensor = LayoutTensor[
            DType.int64, grid_thw_static_layout
        ](grid_thw_host, grid_thw_runtime_layout)
        for i in range(batch_size):
            for j in range(3):
                grid_thw_host_tensor[i, j] = grid_thw_list[i * 3 + j]

    # Create layout tensors for GPU operations
    var input_tensor = LayoutTensor[dtype, input_static_layout](
        input_device, input_runtime_layout
    )
    var output_tensor = LayoutTensor[dtype, output_static_layout](
        output_device, output_runtime_layout
    )
    var grid_thw_tensor = LayoutTensor[DType.int64, grid_thw_static_layout](
        grid_thw_device, grid_thw_runtime_layout
    )

    spatial_merge[dtype](
        output_tensor,
        input_tensor,
        grid_thw_tensor,
        hidden_size,
        merge_size,
        ctx,
    )

    ctx.synchronize()

    # Verify batch 0: t=1, h=4, w=4, merge_size=2
    # Output has 1 × 2 × 2 = 4 merged patches.
    # Each merged patch contains 2×2 spatial positions.

    # Expected mapping for batch 0:
    # Output patch 0 (ho=0, wo=0) contains input patches: [0,1,4,5]
    # Output patch 1 (ho=0, wo=1) contains input patches: [2,3,6,7]
    # Output patch 2 (ho=1, wo=0) contains input patches: [8,9,12,13]
    # Output patch 3 (ho=1, wo=1) contains input patches: [10,11,14,15]

    var batch0_expected_list: List[Int] = [
        0,
        1,
        4,
        5,
        2,
        3,
        6,
        7,
        8,
        9,
        12,
        13,
        10,
        11,
        14,
        15,
    ]
    var batch0_expected = batch0_expected_list.unsafe_ptr()

    # Verify results
    with output_device.map_to_host() as output_host:
        # Verify batch 0 (4 output patches × 4 spatial positions × 4 features).
        for out_patch in range(4):
            for spatial_pos in range(4):
                var expected_input_patch = batch0_expected[
                    out_patch * 4 + spatial_pos
                ]
                for feat in range(hidden_size):
                    var output_idx = (
                        out_patch * C_out + spatial_pos * hidden_size + feat
                    )
                    var actual_val = output_host[output_idx]
                    var expected_val = Float32(
                        expected_input_patch * 100 + feat
                    )
                    assert_equal(
                        actual_val,
                        expected_val,
                        "Batch 0: patch="
                        + String(out_patch)
                        + " pos="
                        + String(spatial_pos)
                        + " feat="
                        + String(feat),
                    )

        # Verify batch 1: t=2, h=2, w=2, merge_size=2
        # Output has 2 × 1 × 1 = 2 merged patches (one per frame).
        # Each merged patch contains 2×2 spatial positions (entire spatial grid).
        # Input patches are 16, 17, 18, 19.

        var batch1_expected_list: List[Int] = [
            16,
            17,
            18,
            19,
            16,
            17,
            18,
            19,  # Frame 0 and Frame 1 (repeated)
        ]
        var batch1_expected = batch1_expected_list.unsafe_ptr()

        var batch1_start = 4 * C_out  # Batch 0 had 4 output patches

        # Verify batch 1 (2 output patches × 4 spatial positions × 4 features)
        for out_patch in range(2):
            for spatial_pos in range(4):
                var expected_input_patch = batch1_expected[
                    out_patch * 4 + spatial_pos
                ]
                for feat in range(hidden_size):
                    var output_idx = (
                        batch1_start
                        + out_patch * C_out
                        + spatial_pos * hidden_size
                        + feat
                    )
                    var actual_val = output_host[output_idx]
                    var expected_val = Float32(
                        expected_input_patch * 100 + feat
                    )
                    assert_equal(
                        actual_val,
                        expected_val,
                        "Batch 1: patch="
                        + String(out_patch)
                        + " pos="
                        + String(spatial_pos)
                        + " feat="
                        + String(feat),
                    )


def main():
    with DeviceContext() as ctx:
        test_spatial_merge(ctx)
