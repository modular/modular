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
"""Causal mask for the attention block.

For each per-lane element of `att_block` (col_l rt_32x32 FP32), set
to `-inf` whenever the absolute K position would be strictly to the
right of the lane's Q position. Subsequent `exp2` then sends those
slots to zero.

Per-element row-within-tile mapping comes from the
`v_mfma_f32_32x32x16_bf16` accumulator fragment geometry; see
`MhaMmaOp.ACC_ROW_OFFSETS_32x32`.
"""

from layout import TensorLayout

from .hk_mha_mma_op import ACC_ROW_OFFSETS_32x32
from structured_kernels.amd_tile_io import RegTile


@always_inline
def mask_kv_tile[
    layout: TensorLayout,
    //,
    Q_BLOCK_SIZE: Int,
    KV_BLOCK_SIZE: Int,
](
    mut dst: RegTile[DType.float32, layout, MutExternalOrigin],
    q_abs: Int32,
    k_abs: Int32,
    lane: Int32,
):
    """Sets `dst` slots whose K position exceeds Q position to `-inf`.

    Position arithmetic uses `Int32` to keep AMDGPU VALU at 32-bit ops.

    Parameters:
        layout: `dst` layout, `row_major[height, width, 16]` for col_l
            rt_32x32 (16 = fragment elements per lane).
        Q_BLOCK_SIZE: Q rows per tile.
        KV_BLOCK_SIZE: K rows per tile.

    Args:
        dst: Attention block tile (mutated in place).
        q_abs: Absolute Q tile index.
        k_abs: Absolute K tile index.
        lane: `lane_id()` cast to `Int32`.
    """
    comptime dst_height = layout.static_shape[0]
    comptime dst_width = layout.static_shape[1]
    comptime base_tile_elts = layout.static_shape[2]

    comptime assert (
        base_tile_elts == 16
    ), "mask_kv_tile: requires base_tile_elts == 16 (col_l rt_32x32 fragment)"

    comptime _NEG_INF_VEC = SIMD[DType.float32, 16](Float32(-3.4028235e38))
    comptime _ROW_OFFSETS = ACC_ROW_OFFSETS_32x32

    var col_in_tile = lane & Int32(31)
    var row_extra = (lane >> Int32(5)) << Int32(2)
    var q_pos = q_abs * Int32(Q_BLOCK_SIZE) + col_in_tile
    var k_base = k_abs * Int32(KV_BLOCK_SIZE)

    var dst_vec = dst.vectorize[1, 1, 16]()
    comptime assert dst_vec.flat_rank == 3

    comptime for i in range(dst_height):
        # `rel < _ROW_OFFSETS[p]` is the per-element form of `k_pos > q_pos`
        # once the per-element row offset is folded in.
        var rel = q_pos - (k_base + Int32(i * 32) + row_extra)
        var mask = SIMD[DType.int32, 16](rel).lt(_ROW_OFFSETS)
        comptime for j in range(dst_width):
            dst_vec[i, j, 0] = mask.select(_NEG_INF_VEC, dst_vec[i, j, 0])
