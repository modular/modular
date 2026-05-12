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
"""Q, P, and Output register buffers for gfx950 attention kernels.

TileTensor-only: no LayoutTensor imports, aliases, or return types.
"""

from std.math import ceildiv, recip
from std.math.uutils import umod, ufloordiv
from std.sys import simd_width_of, size_of

from std.gpu import lane_id, WARP_SIZE
from std.gpu.intrinsics import cvt_pk_fp8_f32_raw
from layout import TensorLayout, TileTensor
from layout.coord import Coord, ComptimeInt, Idx
from layout.swizzle import Swizzle
from layout.tile_layout import (
    Layout as TileLayout,
    col_major,
    row_major,
)
from layout.tensor_core import num_matrix_reg
from layout.tile_tensor import stack_allocation
from std.utils import IndexList

from structured_kernels.amd_tile_io import RegTileLoader

from .mma import TiledMmaOp
from .utils import get_warp_coords
import std.itertools


@always_inline
def _cast_f32_to_fp8_raw[
    src_dtype: DType,
    size: Int,
    //,
    dtype: DType,
](src: SIMD[src_dtype, size]) -> SIMD[dtype, size]:
    """Cast N f32 → N fp8 without the compiler's clamp + NaN-scrub wrapper.

    Chunks into groups of 4 and calls `cvt_pk_fp8_f32_raw` per chunk.
    Only safe when inputs are provably bounded and finite — used by the
    P→PV cast where softmax output is in (0, 1].
    """
    comptime assert (
        src_dtype == DType.float32
    ), "_cast_f32_to_fp8_raw source dtype must be float32."
    comptime assert (
        size % 4 == 0
    ), "_cast_f32_to_fp8_raw requires size divisible by 4."
    comptime assert (
        dtype == DType.float8_e4m3fn or dtype == DType.float8_e5m2
    ), "_cast_f32_to_fp8_raw requires E4M3FN or E5M2 destination dtype."

    var f32_src = rebind[SIMD[DType.float32, size]](src)
    var result = SIMD[dtype, size]()
    comptime for i in range(size // 4):
        var chunk = cvt_pk_fp8_f32_raw[dtype](f32_src.slice[4, offset=i * 4]())
        comptime for j in range(4):
            result[i * 4 + j] = chunk[j]
    return result


struct QRegisterBuffer[
    dtype: DType,
    mma_shape: IndexList[3],
    WM: Int,
    WN: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    thread_rows: Int,
    thread_cols: Int,
]:
    comptime reg_dtype = Self.dtype
    comptime mma_dtype = Self.dtype
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_K = Self.mma_shape[2]
    # A-operand fragment size per lane: num_matrix_reg[MMA_M, MMA_K].
    # For bf16 [32,32,16]: (32*16)/64 = 8.
    # For fp8  [32,32,64]: (32*64)/64 = 32.
    comptime input_frag_size = num_matrix_reg[Self.MMA_M, Self.MMA_K]()
    comptime num_mmas = ceildiv(Self.WM, Self.MMA_M)
    comptime num_k_tiles = ceildiv(Self.BK, Self.MMA_K)

    comptime num_tiles = Self.depth // Self.BK
    comptime _total_rows = Self.num_mmas * Self.num_k_tiles * Self.num_tiles
    comptime _rows_per_tile = Self.num_mmas * Self.num_k_tiles

    # TileTensor storage in registers.
    comptime reg_layout = row_major[Self._total_rows, Self.input_frag_size]()
    comptime RegType = TileTensor[
        Self.dtype,
        type_of(Self.reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var reg_tile: Self.RegType

    # Thread layout for warp-scoped RegTileLoader (col-major to match
    # get_warp_layout[mma_shape] used by the original copy_dram_to_local).
    comptime _q_thread_rows = Self.thread_rows
    comptime _q_thread_cols = Self.thread_cols

    @always_inline
    def __init__[
        q_tile_layout: TensorLayout
    ](out self, q_tile: TileTensor[Self.dtype, q_tile_layout, ImmutAnyOrigin],):
        """Load Q tile from DRAM into registers via buffer_load intrinsics.

        Each warp loads its [WM, depth] sub-tile using col-major thread
        distribution (matching get_warp_layout[mma_shape]), then tiles
        it into BK-wide strips stored in register memory.

        Args:
            q_tile: The full Q tile as a DRAM TileTensor.
        """
        self.reg_tile = stack_allocation[Self.dtype, AddressSpace.LOCAL](
            Self.reg_layout
        )

        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        # Warp's portion of Q: [WM, depth] sub-tile at (warp_row, 0).
        var warp_tile = q_tile.tile[Self.WM, Self.depth](warp_row, 0)

        # `RegTileLoader` stores the per-thread (M, N) fragment row-major in
        # the dst register tile, so dst row i is m_mma=i's contiguous frag —
        # matching the `mma_tile[i, k_idx]` consumer convention for any
        # `num_mmas`. (For M=1 cases — BM=32 FP8, BF16 multi-k — col_major
        # would coincide with row_major, but BM=64 with M=2 needs row_major.)
        comptime load_width = simd_width_of[Self.dtype]()
        var reg_loader = RegTileLoader[
            Self.dtype,
            col_major[Self._q_thread_rows, Self._q_thread_cols](),
            warp_scope=True,
        ](warp_tile)
        comptime for i in range(Self.num_tiles):
            var src = warp_tile.tile[Self.WM, Self.BK](0, i)
            var dst = self.reg_tile.tile[
                Self._rows_per_tile, Self.input_frag_size
            ](i, 0)
            reg_loader.load(
                dst,
                src.vectorize[1, load_width](),
            )

    @always_inline
    def mma_tile[
        tile_idx: Int, k_idx: Int
    ](self) -> TileTensor[
        Self.dtype,
        type_of(row_major[Self.num_mmas, Self.input_frag_size]()),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        """Return MMA-sized sub-tile for the given tile and k indices."""
        return rebind[
            TileTensor[
                Self.dtype,
                type_of(row_major[Self.num_mmas, Self.input_frag_size]()),
                MutExternalOrigin,
                address_space=AddressSpace.LOCAL,
            ]
        ](
            self.reg_tile.tile[Self._rows_per_tile, Self.input_frag_size](
                tile_idx, 0
            ).tile[Self.num_mmas, Self.input_frag_size](k_idx, 0)
        )

    @always_inline
    def scale[accum_type: DType](self, scale_factor: Scalar[accum_type]):
        """Scale all Q register elements in-place.

        Casts bf16 -> f32, multiplies by scale_factor, casts back to bf16.
        Used for pre-scaling Q by (1/sqrt(d) * log2e) so that QK matmul
        produces already-scaled scores, eliminating scale from the hot loop.
        """
        comptime for tile in range(Self.num_tiles):
            comptime for k in range(Self.num_k_tiles):
                var sub = self.reg_tile.tile[
                    Self._rows_per_tile, Self.input_frag_size
                ](tile, 0).tile[Self.num_mmas, Self.input_frag_size](k, 0)
                var vec = sub.vectorize[1, Self.input_frag_size]()
                comptime for row in range(Self.num_mmas):
                    var q_f32 = vec[row, 0].cast[accum_type]()
                    q_f32 *= scale_factor
                    vec[row, 0] = q_f32.cast[Self.dtype]()

    @always_inline
    def zero(self):
        _ = self.reg_tile.fill(0)


struct OutputRegisterBuffer[
    dtype: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    output_frag_size: Int,
]:
    comptime reg_dtype = Self.dtype

    comptime _total_rows = Self.num_n_mmas * Self.num_m_mmas

    # TileTensor storage in registers.
    comptime reg_layout = row_major[Self._total_rows, Self.output_frag_size]()
    comptime RegType = TileTensor[
        Self.dtype,
        type_of(Self.reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var reg_tile: Self.RegType

    @always_inline
    def __init__(out self):
        self.reg_tile = stack_allocation[Self.dtype, AddressSpace.LOCAL](
            Self.reg_layout
        )

    @always_inline
    def apply_softmax_denominator[
        layout_type: TensorLayout, //
    ](self, rowsum: TileTensor[Self.dtype, layout_type, ...]):
        comptime assert rowsum.flat_rank == 2
        var reg_vec = self.reg_tile.vectorize[1, Self.output_frag_size]()
        comptime for m_mma in range(Self.num_m_mmas):
            var rowsum_inv = recip(rowsum[m_mma, 0][0])

            comptime for n_mma in range(Self.num_n_mmas):
                reg_vec[n_mma * Self.num_m_mmas + m_mma, 0] *= rowsum_inv

    @always_inline
    def zero(self):
        _ = self.reg_tile.fill(0)


struct PRegisterBuffer[
    accum_type_: DType,
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    output_frag_size: Int,
    shared_memory_backed: Bool,
    mma_shape: IndexList[3],
    tr_load_enabled: Bool = False,
    num_stages: Int = 1,
    p_swizzle: Optional[Swizzle] = None,
    # When True, use raw `v_cvt_pk_fp8_f32` without the compiler's
    # clamp + NaN-scrub wrapper. Safe only when inputs are provably
    # bounded and finite (e.g. softmax output in (0, 1]). Saves ~200
    # VALU ops per iteration in the FP8 MLA prefill hot path.
    raw_fp8_cast: Bool = False,
]:
    comptime reg_dtype = Self.accum_type_
    comptime mma_dtype = Self.dtype

    # TileTensor storage (with staging dimension).
    comptime _staged_rows = (
        Self.num_stages * Self.num_n_mmas * Self.num_m_mmas
    )
    comptime _tiles_per_stage = Self.num_n_mmas * Self.num_m_mmas
    comptime reg_layout = row_major[Self._staged_rows, Self.output_frag_size]()
    comptime RegType = TileTensor[
        Self.accum_type_,
        type_of(Self.reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var reg_tile: Self.RegType

    # TileTensor type for a single pipeline stage sub-tile.
    comptime stage_layout = row_major[
        Self._tiles_per_stage, Self.output_frag_size
    ]()
    comptime StageTileType = TileTensor[
        Self.accum_type_,
        type_of(Self.stage_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    # TiledMmaOp for SMEM→register A-matrix loads.
    comptime _TiledMma = TiledMmaOp[
        out_type=Self.accum_type_,
        in_type=Self.dtype,
        shape=Self.mma_shape,
        transpose_b=False,
    ]

    # P SMEM is a BM×BN region (when shared_memory_backed) carved into
    # `_num_blocks` blocked BM×BK sub-tiles stacked vertically. Plain
    # row-major `[_num_blocks * BM, BK]` matches the actual in-memory
    # layout (each block is a contiguous BM×BK row-major region), so
    # `.tile[BM, BK](block_idx, 0)` yields the natural
    # `block_idx * BM * BK` offset with no stride override.
    comptime _num_blocks = (
        Self.BN // Self.BK
    ) if Self.shared_memory_backed else 0
    comptime _smem_layout = row_major[Self._num_blocks * Self.BM, Self.BK]()
    comptime SmemTileType = TileTensor[
        Self.dtype,
        type_of(Self._smem_layout),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    var smem_tile: Self.SmemTileType

    # TileTensor type for a single BM×BK blocked SMEM slice. Parent is
    # plain row-major, so `.tile[BM, BK]` inherits the `(BK, 1)` strides
    # naturally — no stride override needed.
    comptime _block_smem_layout = row_major[Self.BM, Self.BK]()
    comptime BlockSmemType = TileTensor[
        Self.dtype,
        type_of(Self._block_smem_layout),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    @always_inline
    def __init__(out self, smem_tile: Self.SmemTileType):
        self.reg_tile = stack_allocation[Self.accum_type_, AddressSpace.LOCAL](
            Self.reg_layout
        )
        self.smem_tile = smem_tile

    @always_inline
    def _block_smem(self, block_idx: Int) -> Self.BlockSmemType:
        """TileTensor view of the `block_idx`-th blocked SMEM slice."""
        return self.smem_tile.tile[Self.BM, Self.BK](block_idx, 0)

    @always_inline
    def get_mma_tile_shared[
        tile_idx: Int, k_idx: Int
    ](self) -> Self.MmaTileType:
        var result = stack_allocation[Self.mma_dtype, AddressSpace.LOCAL](
            Self._mma_layout
        )
        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        var smem_block = self._block_smem(tile_idx)
        var warp_tile = smem_block.tile[Self.WM, Self.BK](warp_row, 0)

        comptime if (Self.mma_shape[0] == 32 and Self.input_frag_size == 32):
            # MFMA_F32_32x32x64_FP8 B-operand register layout = 2-MMA-tile
            # C-output join (same as prefill's register path). With
            # warps_per_block=2, each P block is filled by 2 warps — warp
            # with n_mma_in_block=0 owns keys 0..31, warp with
            # n_mma_in_block=1 owns keys 32..63.  copy_to_shared writes
            # each warp's 32x32 MMA tile lane-contiguously (64 lanes x
            # 16B).  Reader lane l's slots 0..15 come from warp0 lane l,
            # slots 16..31 from warp1 lane l — same lane_id across both
            # warps because C-output M=l%32 matches the B-operand K-slot.
            # So two ds_read_b128 + SIMD.join reconstruct the 32-fp8 fragment.
            #
            # For num_m_mmas > 1 (BM=64 MLA decode), each m_mma is laid
            # out as its own 1024B-per-warp lane-contiguous stripe at
            # offset `m_mma * MMA_M * BK` within the BM×BK SMEM block.
            comptime warps_per_block = Self.BK // Self.WN
            comptime num_gather = (
                Self.input_frag_size // Self.output_frag_size
            )
            comptime warp_stride = (
                Self.mma_shape[0] * Self.BK
            ) // warps_per_block
            comptime m_mma_stride = Self.mma_shape[0] * Self.BK
            comptime assert num_gather == warps_per_block, (
                "FP8 MLA P SMEM round-trip expects num_gather =="
                " warps_per_block."
            )
            comptime assert (
                num_gather == 2
            ), "FP8 MLA P SMEM round-trip only handles num_gather==2 for now."
            comptime lane_bytes = size_of[Scalar[Self.dtype]]() * (
                Self.output_frag_size
            )
            comptime assert (
                lane_bytes == 16
            ), "FP8 MLA lane MMA-tile size must be 16B for ds_read_b128."
            var lid = lane_id()
            var block_base = smem_block.ptr
            var result_vec = result.vectorize[1, Self.input_frag_size]()
            comptime for m_mma in range(Self.num_m_mmas):
                var m_off = m_mma * m_mma_stride
                var lo = (
                    block_base + m_off + Int(lid) * Self.output_frag_size
                ).load[width=Self.output_frag_size]()
                var hi = (
                    block_base
                    + m_off
                    + warp_stride
                    + Int(lid) * Self.output_frag_size
                ).load[width=Self.output_frag_size]()
                var joined = lo.join(hi)
                result_vec[m_mma, 0] = rebind[type_of(result_vec[m_mma, 0])](
                    joined
                )
        elif (
            Self.mma_shape[0] == 16
            and Self.tr_load_enabled
            and Self.input_frag_size <= 2 * Self.output_frag_size
        ):
            # 16x16 MMA: match register mma_tile interleaving.
            # The [WM, BK] block has two [16, 16] MMA tiles side-by-side.
            # Each thread reads 4 bf16 from left half (lo) and 4 from
            # right half (hi), then joins -> bf16[8], matching the
            # hardware B-operand register layout.
            comptime mma_M = Self.mma_shape[0]
            comptime mma_N = Self.mma_shape[1]
            comptime frag_w = Self.output_frag_size
            comptime warp_m = mma_M
            comptime warp_n = WARP_SIZE // warp_m

            var result_vec = result.vectorize[1, Self.input_frag_size]()
            var smem_base = smem_block.ptr

            comptime for m in range(Self.num_m_mmas):
                var lo_tile = warp_tile.tile[mma_M, mma_N](m, 2 * k_idx)
                var lo_dist = lo_tile.vectorize[1, frag_w]().distribute[
                    col_major[warp_m, warp_n]()
                ](lane_id())

                var hi_tile = warp_tile.tile[mma_M, mma_N](m, 2 * k_idx + 1)
                var hi_dist = hi_tile.vectorize[1, frag_w]().distribute[
                    col_major[warp_m, warp_n]()
                ](lane_id())

                # Interleaved P read: single 16-byte load (ds_read_b128).
                # Each 8-element group = [lo_frag(4), hi_frag(4)].
                comptime if Self.p_swizzle:
                    comptime simd_w = simd_width_of[Self.dtype]()
                    var r = umod(lane_id(), warp_m)
                    var c = ufloordiv(lane_id(), warp_m)
                    var group_idx = r * (Self.BK // simd_w) + c
                    var swizzled_group = Self.p_swizzle.value()(group_idx)
                    var data = (smem_base + swizzled_group * simd_w).load[
                        width=simd_w
                    ]()
                    result_vec[m, 0] = rebind[type_of(result_vec[m, 0])](data)
                else:
                    var joined = lo_dist[0, 0].join(hi_dist[0, 0])
                    result_vec[m, 0] = rebind[type_of(result_vec[m, 0])](
                        joined.slice[Self.input_frag_size, offset=0]()
                    )
        else:
            Self._TiledMma.load_a[swizzle=Self.p_swizzle](
                warp_tile, result, k_idx
            )

        return result

    @always_inline
    def stage_tile[stage: Int = 0](self) -> Self.StageTileType:
        """Return the TileTensor sub-tile for the given pipeline stage."""
        return rebind[Self.StageTileType](
            self.reg_tile.tile[Self._tiles_per_stage, Self.output_frag_size](
                stage, 0
            )
        )

    # TileTensor type for MMA operand (cast from accum_type to mma_dtype).
    # Fragment width = A-operand regs per thread.
    comptime input_frag_size = num_matrix_reg[
        Self.mma_shape[0], Self.mma_shape[2]
    ]()
    comptime _mma_layout = row_major[Self.num_m_mmas, Self.input_frag_size]()
    comptime MmaTileType = TileTensor[
        Self.mma_dtype,
        type_of(Self._mma_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    @always_inline
    def mma_tile[
        tile_idx: Int, k_idx: Int, stage: Int = 0
    ](self) -> Self.MmaTileType:
        """TileTensor MMA operand with cast+interleave via SIMD whole-vector ops.

        Converts f32 accumulator rows to bf16 MMA fragments using SIMD cast,
        interleave, and slice — no per-element [j] indexing needed.
        """
        comptime if Self.shared_memory_backed:
            return self.get_mma_tile_shared[tile_idx, k_idx]()

        var result = stack_allocation[Self.mma_dtype, AddressSpace.LOCAL](
            Self._mma_layout
        )
        var result_vec = result.vectorize[1, Self.input_frag_size]()

        # Read source tile for this stage.
        var src = self.stage_tile[stage]()
        var src_vec = src.vectorize[1, Self.output_frag_size]()

        comptime if Self.tr_load_enabled:
            comptime if Self.mma_shape[0] == 32:
                # 32x32 MMA: cast full row, then slice to k_idx chunk.
                # src is SIMD[f32, output_frag_size], cast → SIMD[mma_dtype, frag].
                # Result takes input_frag_size elements starting at k_idx offset.
                comptime assert (
                    Self.output_frag_size == 16
                ), "output_frag_size must be 16 for 32x32 mma shape"

                # When input_frag_size > output_frag_size (e.g. fp8 [32,32,64]),
                # gather from multiple consecutive stage tile rows.
                comptime num_gather = Self.input_frag_size // Self.output_frag_size
                comptime if num_gather <= 1:
                    # input_frag_size <= output_frag_size: cast one row, slice.
                    # bf16 [32,32,16]: input_frag_size=8, frag=16 → slice 8 of 16.
                    var casted = src_vec[tile_idx, 0].cast[Self.mma_dtype]()
                    result_vec[0, 0] = rebind[type_of(result_vec[0, 0])](
                        casted.slice[
                            Self.input_frag_size,
                            offset=k_idx * Self.input_frag_size,
                        ]()
                    )
                elif num_gather == 2:
                    # Gather 2 rows, cast each to mma_dtype, join to full frag.
                    var lo: SIMD[Self.mma_dtype, Self.output_frag_size]
                    var hi: SIMD[Self.mma_dtype, Self.output_frag_size]
                    comptime if (
                        Self.raw_fp8_cast and Self.mma_dtype.is_float8()
                    ):
                        # Raw cvt path: softmax output is in (0, 1], so the
                        # compiler's clamp + NaN-scrub wrapper is a no-op
                        # that just adds ~6 VALU ops per f32→fp8 pair.
                        lo = _cast_f32_to_fp8_raw[Self.mma_dtype](
                            src_vec[tile_idx * 2, 0]
                        )
                        hi = _cast_f32_to_fp8_raw[Self.mma_dtype](
                            src_vec[tile_idx * 2 + 1, 0]
                        )
                    else:
                        lo = src_vec[tile_idx * 2, 0].cast[Self.mma_dtype]()
                        hi = src_vec[tile_idx * 2 + 1, 0].cast[Self.mma_dtype]()
                    var joined = lo.join(hi)
                    result_vec[0, 0] = rebind[type_of(result_vec[0, 0])](
                        joined.slice[
                            Self.input_frag_size,
                            offset=k_idx * Self.input_frag_size,
                        ]()
                    )
                else:
                    comptime assert False, String(
                        "Unsupported num_gather: ", num_gather
                    )

            elif Self.mma_shape[0] == 16:
                # 16x16 MMA: cast two halves and join.
                # reg_tile_split = stage[tile_idx * 2*num_m_mmas : ...], shape
                # [2*num_m_mmas, output_frag_size].  Row m → first half,
                # row m+num_m_mmas → second half.
                comptime assert (
                    Self.output_frag_size == 4
                ), "output_frag_size must be 4 for 16x16 mma shape"

                # The stage tile is [num_n_mmas * num_m_mmas, frag].
                # For 16x16, split into num_n_mmas//2 groups.
                comptime group_rows = 2 * Self.num_m_mmas
                var group = src.tile[group_rows, Self.output_frag_size](
                    tile_idx, 0
                ).vectorize[1, Self.output_frag_size]()

                comptime for m in range(Self.num_m_mmas):
                    var lo = group[m, 0].cast[Self.mma_dtype]()
                    var hi = group[m + Self.num_m_mmas, 0].cast[
                        Self.mma_dtype
                    ]()
                    var joined = lo.join(hi)
                    result_vec[m, 0] = rebind[type_of(result_vec[m, 0])](
                        joined.slice[
                            Self.input_frag_size,
                            offset=k_idx * Self.input_frag_size,
                        ]()
                    )
            else:
                comptime assert False, String(
                    "Unsupported mma shape: ", Self.mma_shape[0]
                )
        else:
            # Non-tr_load: cast + interleave pairs of 4-element sub-groups.
            # Input [0..3, 4..7, 8..11, 12..15] →
            # Output [0,4, 1,5, 2,6, 3,7, 8,12, 9,13, 10,14, 11,15]
            var row = src_vec[tile_idx, 0].cast[Self.mma_dtype]()
            var lo4 = row.slice[4, offset=0]()
            var hi4 = row.slice[4, offset=4]()
            var lo_half = lo4.interleave(hi4)  # 8 elems

            var lo4b = row.slice[4, offset=8]()
            var hi4b = row.slice[4, offset=12]()
            var hi_half = lo4b.interleave(hi4b)  # 8 elems

            var packed = lo_half.join(hi_half)  # 16 elems
            result_vec[0, 0] = rebind[type_of(result_vec[0, 0])](
                packed.slice[
                    Self.input_frag_size, offset=k_idx * Self.input_frag_size
                ]()
            )

        return result

    @always_inline
    def zero[stage: Int](self):
        _ = self.stage_tile[stage]().fill(0)

    @always_inline
    def _store_mma_tile[
        reg_idx: Int
    ](
        self,
        smem_base: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
        byte_offset: Int,
        m_mma: Int,
        n_mma: Int,
    ):
        """Store one MMA tile from register index `reg_idx` into the
        `(m_mma, n_mma)` position of a `[WM, BK]` SMEM warp tile at
        `smem_base + byte_offset`.

        Extracted from `copy_to_shared` — used by the non-swizzle paths
        in both `WN < BK` and `WN >= BK` branches. Not a free function
        because it depends on half a dozen `Self.*` comptime params.
        """
        comptime frag_w = Self.output_frag_size
        comptime warp_m = Self.mma_shape[0]
        comptime warp_n = WARP_SIZE // warp_m
        comptime tl_3d = col_major[warp_m, warp_n, 1]()

        var smem_warp_tile = TileTensor[
            Self.dtype,
            type_of(row_major[Self.WM, Self.BK]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ](smem_base + byte_offset, row_major[Self.WM, Self.BK]())

        var mma_tile_res = smem_warp_tile.tile_with_offset[
            Self.mma_shape[0], Self.mma_shape[1]
        ](Coord(Idx(m_mma), Idx(n_mma)))
        var mma_tile = mma_tile_res[0]
        # Element offset of mma_tile from smem_warp_tile.ptr
        # (== block_base), used below to build the swizzle argument
        # without resorting to `Int(ptr_a) - Int(ptr_b)` math.
        var mma_tile_off = mma_tile_res[2]

        comptime mma_M = Self.mma_shape[0]
        comptime mma_N = Self.mma_shape[1]
        comptime S0 = type_of(mma_tile).LayoutType._stride_types[0].static_value
        comptime layout_3d = type_of(
            TileLayout(
                Coord(
                    ComptimeInt[mma_M](),
                    ComptimeInt[mma_N // frag_w](),
                    ComptimeInt[frag_w](),
                ),
                Coord(
                    ComptimeInt[S0](),
                    ComptimeInt[frag_w](),
                    ComptimeInt[1](),
                ),
            )
        )
        var mma_3d = TileTensor[
            Self.dtype,
            layout_3d,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            element_size=1,
        ](mma_tile.ptr, layout_3d())

        var dist_res = mma_3d.distribute_with_offset[tl_3d](lane_id())
        var dst_frag = dist_res[0]
        var lane_off_in_mma = dist_res[2]

        var p_reg_vec = self.stage_tile[0]().vectorize[
            1, Self.output_frag_size
        ]()
        var p_reg_tile = p_reg_vec.tile[1, 1](reg_idx, 0)
        comptime frag_size = p_reg_tile.element_size
        var reg_val = p_reg_tile.ptr.load[width=frag_size]().cast[Self.dtype]()

        # Apply swizzle to spread rows across different LDS banks.
        # Use block base (smem_base + byte_offset) so the swizzle
        # offset matches the read path in get_mma_tile_shared.
        comptime if Self.p_swizzle:
            var block_base = smem_base + byte_offset
            var elem_off = mma_tile_off + lane_off_in_mma
            var swizzled = Self.p_swizzle.value()(elem_off // frag_w) * frag_w
            (block_base + swizzled).store[width=frag_w](reg_val)
        else:
            dst_frag.raw_store[width=frag_w](0, reg_val)

    @always_inline
    def copy_to_shared(self):
        comptime frag_w = Self.output_frag_size
        var warp_row, warp_col = get_warp_coords[Self.BN, Self.WN]()

        var p_reg_vec = self.stage_tile[0]().vectorize[
            1, Self.output_frag_size
        ]()

        # MFMA 32x32x64 FP8: lane-contiguous SMEM layout.  Each lane
        # packs its 16 fp32 C-output (cast to fp8 = 16 bytes) into ONE
        # ds_write_b128.  Within each [BM, BK] P block, the two warps
        # contributing (warps_per_block=2) write into disjoint 1024B
        # regions.  Reader in get_mma_tile_shared pairs halves lane-by-
        # lane — the MFMA C-output (M=l%32, N blocked) matches the MFMA
        # B-operand pattern the PV MMA needs, so no per-element
        # reordering is required.
        #
        # INTENTIONALLY NOT wrapped in a structured `PSmemWriter` or
        # `TiledMmaOp`-style abstraction. This is a single specialized
        # path gated by a narrow comptime condition
        # (`mma_shape[0] == 32 && !p_swizzle && num_m_mmas == 1 &&
        # lane_bytes == 16`). The per-lane `ds_write_b128` is driven by
        # `lid * output_frag_size` — a lane-contiguous scalar packing
        # that no existing distribute / TileWriter primitive expresses.
        # Wrapping would add a struct for ONE write pattern with no
        # other consumers. Keep inline; if a second FP8 MLA write ever
        # needs the same lane-contiguous packing, extract a helper at
        # that point.
        comptime if (Self.mma_shape[0] == 32 and not Self.p_swizzle):
            comptime warps_per_block = Self.BK // Self.WN
            comptime warp_stride = (
                Self.mma_shape[0] * Self.BK
            ) // warps_per_block
            comptime m_mma_stride = Self.mma_shape[0] * Self.BK
            comptime lane_bytes = size_of[Scalar[Self.dtype]]() * (
                Self.output_frag_size
            )
            comptime assert (
                lane_bytes == 16
            ), "FP8 MLA lane MMA-tile size must be 16B for ds_write_b128."
            comptime assert (
                Self.BM * Self.BK
                == Self.num_m_mmas * warps_per_block * Self.num_n_mmas * 64 * 16
            ), (
                "P SMEM block size must equal"
                " num_m_mmas*warps_per_block*num_n_mmas*64*16B."
            )
            var block_idx = warp_col // Int(warps_per_block)
            var n_mma_in_block = warp_col % Int(warps_per_block)
            var block_base = self.smem_tile.tile[Self.BM, Self.BK](
                block_idx, 0
            ).ptr
            var lid = lane_id()

            comptime for m_mma in range(Self.num_m_mmas):
                comptime for n_mma in range(Self.num_n_mmas):
                    # Reg layout is m_mma INNER (matches `mma`'s c_idx).
                    comptime reg_idx = n_mma * Self.num_m_mmas + m_mma
                    var p_reg_ptr = p_reg_vec.tile[1, 1](reg_idx, 0).ptr
                    var loaded = p_reg_ptr.load[width=Self.output_frag_size]()
                    var reg16: SIMD[Self.dtype, Self.output_frag_size]
                    comptime if Self.raw_fp8_cast and Self.dtype.is_float8():
                        # Raw cvt path: softmax output is bounded in (0, 1],
                        # so the compiler's clamp + NaN-scrub wrapper around
                        # pop.cast is a no-op that just adds ~6 VALU ops per
                        # f32→fp8 pair.
                        reg16 = _cast_f32_to_fp8_raw[Self.dtype](loaded)
                    else:
                        reg16 = loaded.cast[Self.dtype]()
                    var warp_off = m_mma * Int(m_mma_stride) + (
                        n_mma_in_block * Int(Self.num_n_mmas) + n_mma
                    ) * Int(warp_stride)
                    (
                        block_base + warp_off + Int(lid) * Self.output_frag_size
                    ).store[width=Self.output_frag_size](reg16)
            return

        comptime if Self.WN < Self.BK:
            # WN < BK: multiple warps fill each [BM, BK] SMEM block.
            comptime warps_per_block = Self.BK // Self.WN
            var block_idx = warp_col // Int(warps_per_block)
            var n_mma_in_block = warp_col % Int(warps_per_block)
            var smem_offset = block_idx * Self.BM * Self.BK
            var warp_offset = smem_offset + warp_row * Self.WM * Self.BK

            comptime if Self.p_swizzle:
                # Interleaved P layout for 16-byte reads: each 8-element
                # group = [warp0_frag(4), warp1_frag(4)].
                comptime simd_w = simd_width_of[Self.dtype]()
                comptime warp_m = Self.mma_shape[0]
                var r = umod(lane_id(), warp_m)
                var c = ufloordiv(lane_id(), warp_m)
                var block_base = self.smem_tile.tile[Self.BM, Self.BK](
                    block_idx, 0
                ).ptr

                comptime for m_mma in range(Self.num_m_mmas):
                    comptime for n_mma in range(Self.num_n_mmas):
                        comptime reg_idx = n_mma * Self.num_m_mmas + m_mma
                        var p_reg_tile = p_reg_vec.tile[1, 1](reg_idx, 0)
                        var reg_val = p_reg_tile.raw_load[
                            width=p_reg_tile.element_size
                        ](0).cast[Self.dtype]()

                        var group_idx = r * (Self.BK // simd_w) + c
                        var swizzled_group = Self.p_swizzle.value()(group_idx)
                        var elem_off = (
                            swizzled_group * simd_w + n_mma_in_block * frag_w
                        )
                        (block_base + elem_off).store[width=frag_w](reg_val)
            else:
                comptime for m_mma in range(Self.num_m_mmas):
                    comptime for n_mma in range(Self.num_n_mmas):
                        comptime reg_idx = n_mma * Self.num_m_mmas + m_mma
                        self._store_mma_tile[reg_idx](
                            self.smem_tile.ptr,
                            warp_offset,
                            m_mma,
                            n_mma + n_mma_in_block * Self.num_n_mmas,
                        )
        else:
            # WN >= BK: each warp fills one or more [BM, BK] blocks.
            comptime num_n_mmas_per_bk = Self.num_n_mmas // (Self.WN // Self.BK)

            comptime if Self.p_swizzle and num_n_mmas_per_bk == 2:
                # Interleaved 16-byte write: join n_mma=0 (lo) and n_mma=1
                # (hi) frags into one 8-bf16 group, matching the read
                # layout in get_mma_tile_shared.
                comptime group_w = 2 * frag_w
                comptime warp_m_ = Self.mma_shape[0]
                var r = umod(lane_id(), warp_m_)
                var c = ufloordiv(lane_id(), warp_m_)

                comptime for i in range(Self.WN // Self.BK):
                    var block_idx_ = Int(i) + warp_col * (Self.WN // Self.BK)
                    var block_base = self.smem_tile.tile[Self.BM, Self.BK](
                        block_idx_, 0
                    ).ptr

                    comptime for m_mma in range(Self.num_m_mmas):
                        comptime lo_idx = (
                            0 + i * num_n_mmas_per_bk
                        ) * Self.num_m_mmas + m_mma
                        comptime hi_idx = (
                            1 + i * num_n_mmas_per_bk
                        ) * Self.num_m_mmas + m_mma
                        var lo = (
                            p_reg_vec.tile[1, 1](lo_idx, 0)
                            .raw_load[width=frag_w](0)
                            .cast[Self.dtype]()
                        )
                        var hi = (
                            p_reg_vec.tile[1, 1](hi_idx, 0)
                            .raw_load[width=frag_w](0)
                            .cast[Self.dtype]()
                        )
                        var joined = lo.join(hi)

                        var group_idx = r * (Self.BK // group_w) + c
                        var swizzled_group = Self.p_swizzle.value()(group_idx)
                        (block_base + swizzled_group * group_w).store[
                            width=group_w
                        ](joined)
            else:
                comptime for i in range(Self.WN // Self.BK):
                    var block_idx_ = Int(i) + warp_col * (Self.WN // Self.BK)
                    var smem_offset = block_idx_ * Self.BM * Self.BK
                    var warp_offset = smem_offset + warp_row * Self.WM * Self.BK

                    comptime for m_mma, n_mma in std.itertools.product(
                        range(Self.num_m_mmas), range(num_n_mmas_per_bk)
                    ):
                        comptime reg_idx = (
                            n_mma + i * num_n_mmas_per_bk
                        ) * Self.num_m_mmas + m_mma
                        self._store_mma_tile[reg_idx](
                            self.smem_tile.ptr,
                            warp_offset,
                            m_mma,
                            n_mma,
                        )
