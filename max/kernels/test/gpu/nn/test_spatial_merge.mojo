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
from internal_utils import DeviceNDBuffer, HostNDBuffer
from layout import RuntimeLayout, LayoutTensor, Layout
from utils.index import Index
from buffer.dimlist import DimList


def test_spatial_merge(ctx: DeviceContext):
    alias dtype = DType.float32
    alias merge_size = 2
    alias hidden_size = 4
    alias batch_size = 2

    # Batch item 0: t=1, h=4, w=4
    # Batch item 1: t=2, h=2, w=2
    var grid_thw_list: List[Int64] = [1, 4, 4, 2, 2, 2]

    var total_input_patches = 20
    var total_output_patches = 6
    var C_out = hidden_size * merge_size * merge_size

    # Create host buffers
    var input_host = HostNDBuffer[dtype, 2](
        DimList(total_input_patches, hidden_size)
    )
    for patch_idx in range(total_input_patches):
        for feat_idx in range(hidden_size):
            input_host.tensor[patch_idx, feat_idx] = Float32(
                patch_idx * 100 + feat_idx
            )

    var output_host = HostNDBuffer[dtype, 2](
        DimList(total_output_patches, C_out)
    )

    var grid_thw_host = HostNDBuffer[DType.int64, 2](DimList(batch_size, 3))
    for i in range(batch_size):
        for j in range(3):
            grid_thw_host.tensor[i, j] = grid_thw_list[i * 3 + j]

    var input_device = DeviceNDBuffer[dtype, 2](
        DimList(total_input_patches, hidden_size), ctx=ctx
    )
    var output_device = DeviceNDBuffer[dtype, 2](
        DimList(total_output_patches, C_out), ctx=ctx
    )
    var grid_thw_device = DeviceNDBuffer[DType.int64, 2](
        DimList(batch_size, 3), ctx=ctx
    )

    ctx.enqueue_copy(input_device.buffer, input_host.tensor.data)
    ctx.enqueue_copy(grid_thw_device.buffer, grid_thw_host.tensor.data)

    spatial_merge[dtype](
        output_device.to_layout_tensor(),
        input_device.to_layout_tensor(),
        grid_thw_device.to_layout_tensor(),
        batch_size,
        hidden_size,
        merge_size,
        total_output_patches,
        ctx,
    )

    ctx.enqueue_copy(output_host.tensor.data, output_device.buffer)

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
                var actual_val = output_host.tensor.data[output_idx]
                var expected_val = Float32(expected_input_patch * 100 + feat)
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
                var actual_val = output_host.tensor.data[output_idx]
                var expected_val = Float32(expected_input_patch * 100 + feat)
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

    _ = input_device
    _ = output_device
    _ = grid_thw_device


def main():
    with DeviceContext() as ctx:
        test_spatial_merge(ctx)
