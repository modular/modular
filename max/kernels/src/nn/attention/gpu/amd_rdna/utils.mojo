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
"""Shared helpers for RDNA Wave32 attention kernels."""

from std.sys import simd_width_of
from std.sys.info import _is_amd_rdna4
from max.gpu import warp_id as get_warp_id
from std.math.uutils import udivmod


@inline(.always)
def wmma_acc_row(elem: Int, lane_half: Int) -> Int:
    """Row of WMMA C/D element `elem` held by a lane in half `lane_half`.

    RDNA3 (gfx11) interleaves the two lane halves: row = elem*2 + lane_half.
    RDNA4 (gfx12) gives lanes 0-15 rows 0-7 and lanes 16-31 rows 8-15:
    row = elem + 8*lane_half.

    Args:
        elem: Index of the element in the lane's C/D fragment.
        lane_half: `lane // 16`.
    """
    comptime if _is_amd_rdna4():
        return elem + 8 * lane_half
    else:
        return elem * 2 + lane_half


@inline(.always)
def pad[dtype: DType, depth: Int, size: Int]() -> Int:
    """V-SMEM row padding to avoid bank conflicts. Skipped for depth=64.

    Parameters:
        dtype: Element type of the stored values; determines the SIMD width
            used to compute the padding stride.
        depth: Matrix depth in elements; padding is skipped when equal to 64.
        size: Base row size in elements before padding is applied.
    """
    comptime simd_width = simd_width_of[dtype]()
    comptime padding = 0 if depth == 64 else size // simd_width
    return size + padding


@inline(.always)
def get_warp_coords[BN: Int, WN: Int]() -> Tuple[Int, Int]:
    """Return `(warp_row, warp_col)` for the current warp given a BN×WN grid.

    Parameters:
        BN: Total block size along the N dimension of the BN×WN grid.
        WN: Warp tile size along the N dimension; the number of warps along N
            is `BN // WN`.
    """
    comptime num_warps_n = BN // WN
    return udivmod(get_warp_id(), num_warps_n)
