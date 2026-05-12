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
"""Per-lane DRAM->VGPR loaders for the preshuffled MXFP4 MoE matmul.

Both loaders consume buffers produced by `mxfp4_preshuffle_layouts` and emit
one `buffer_load_*` per call — no LDS round-trip. Each lane reads exactly the
fragment / scale word the MFMA needs at its `(lane_nlane, lane_klane)` slot.

`PreshuffledBLoader[N, K_BYTES]`:
    Loads one 16-byte FP4 B fragment per lane via `buffer_load_dwordx4`,
    indexed by logical `(n, k_byte)` through `b_5d_layout`.

`PreshuffledScaleLoader[MN_padded, K_SCALES]`:
    Loads one packed Int32 scale word per lane (4 E8M0 bytes covering
    `MNXdlPack=2 x KXdlPack=2` sub-MMAs) via `buffer_load_dword`, indexed
    by logical `(mn, k_scale)` through `scale_4d_layout`.
"""

from std.gpu.intrinsics import AMDBufferResource
from std.memory.unsafe import bitcast

from layout import Coord, Idx, TileTensor
from layout._utils import make_amd_buffer_resource

from .mxfp4_preshuffle_layouts import Shuffler


struct PreshuffledBLoader[N: Int, K_BYTES: Int](TrivialRegisterPassable):
    """Per-lane B fragment loader from preshuffled GMEM (DRAM -> VGPR direct).

    The 5D layout places each lane's 16-byte fragment at a contiguous DRAM
    offset, so a single `buffer_load_dwordx4` per lane delivers the MFMA's
    B operand with no LDS staging. OOB lanes are clamped to zero by the
    buffer-resource bounds.

    Parameters:
        N: Per-expert N dimension (rows of the logical [N, K_BYTES] tile).
        K_BYTES: Per-expert FP4-packed K dimension (= K // 2).
    """

    var bc: AMDBufferResource

    @always_inline
    def __init__(out self, b_gmem_tile: TileTensor[DType.uint8, ...]):
        """Builds the V# from a preshuffled per-expert B byte buffer."""
        self.bc = make_amd_buffer_resource(b_gmem_tile)

    @always_inline
    def load_fragment(self, n: Int, k_byte: Int) -> SIMD[DType.uint8, 16]:
        """Loads the 16-byte B fragment at logical `(n, k_byte)`.

        For one MFMA dispatch a lane calls this with
        `(n = warp_n_off + n_mma * 16 + lane % 16,
          k_byte = k_tile * 64 + (lane // 16) * 16)`.
        """
        var byte_off = Int32(
            Shuffler[1].b_5d_grouped_layout[N=Self.N, K_BYTES=Self.K_BYTES](
                Coord(Idx(0), Idx(n), Idx(k_byte))
            )
        )
        return self.bc.load[DType.uint8, 16](byte_off)


struct PreshuffledScaleLoader[MN_padded: Int, K_SCALES: Int](
    TrivialRegisterPassable
):
    """Per-lane packed-Int32 scale loader from preshuffled GMEM.

    Each i32 cell holds 4 E8M0 bytes packed in `(k_pack, mn_pack)` order;
    the MFMA's `opsel` byte index selects the right byte per sub-MMA.
    OOB lanes (past `MN_padded * K_SCALES`) read as zero.

    Parameters:
        MN_padded: MN dimension rounded up to 32 (the scale-block stride).
        K_SCALES: K // 32 — one E8M0 byte per 32 FP4 elements.
    """

    var bc: AMDBufferResource

    @always_inline
    def __init__(out self, scale_gmem_tile: TileTensor[DType.uint8, ...]):
        """Builds the V# from a preshuffled per-expert scale byte buffer."""
        self.bc = make_amd_buffer_resource(scale_gmem_tile)

    @always_inline
    def load_packed(self, mn: Int, k_scale: Int) -> Int32:
        """Loads the packed Int32 scale word containing logical `(mn, k_scale)`.

        Pass `(mn, k_scale)` at `(mn_pack=0, k_pack=0)` — the cell base —
        and all 4 bytes of the cell come back in the returned i32. The
        MFMA's `opsel` then selects the byte for each sub-MMA.

        Per-lane usage:
            mn       = warp_mn_off + lane % 16            # mn_lane within block
            k_scale  = scale_pack_idx * 8 + (lane // 16)  # k_lane within block
        """
        var byte_off = Int32(
            Shuffler[1].scale_4d_grouped_layout[
                MN_padded=Self.MN_padded, K_SCALES=Self.K_SCALES
            ](Coord(Idx(0), Idx(mn), Idx(k_scale)))
        )
        var v = self.bc.load[DType.uint8, 4](byte_off)
        return bitcast[DType.int32, 1](v)[0]
