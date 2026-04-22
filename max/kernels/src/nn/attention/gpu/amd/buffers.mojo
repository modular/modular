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

from std.math import ceildiv, recip
from std.math.uutils import umod, ufloordiv
from std.sys import simd_width_of, size_of

from std.gpu import lane_id, WARP_SIZE
from layout import Layout, LayoutTensor, TensorLayout, TileTensor
from layout.coord import Coord, ComptimeInt
from layout.tile_layout import Layout as TileLayout, col_major as tt_col_major
from layout.swizzle import Swizzle
from layout.tensor_core import num_matrix_reg
from layout.tile_layout import row_major as tt_row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from std.utils import IndexList

from structured_kernels.amd_tile_io import RegTileLoader

from .mma import TiledMmaOp
from .utils import (
    LocalLayoutTensor,
    get_warp_coords,
)
import std.itertools


# Legacy traits — only used by the RDNA attention path
# (amd_rdna/buffers_rdna.mojo).  CDNA attention uses TiledMmaOp with
# TileTensor directly, bypassing these abstractions.
trait KVBuffer:
    comptime _dtype: DType
    comptime mma_tile_layout: Layout
    comptime _num_stages: Int

    def load_from_dram(mut self):
        ...

    def get_mma_tile(
        self,
    ) -> LocalLayoutTensor[Self._dtype, Self.mma_tile_layout,]:
        ...

    def copy_to_shared[
        tile_id: Int = 0
    ](self,):
        ...

    def load_from_shared[
        k_mma: Int,
    ](self):
        ...


trait RegisterBuffer:
    comptime reg_dtype: DType
    comptime reg_tile_layout: Layout

    @staticmethod
    def get_dtype() -> DType:
        ...

    def zero(self):
        ...

    def get_reg_tile[
        stage: Int = 0
    ](self,) -> LocalLayoutTensor[Self.reg_dtype, Self.reg_tile_layout,]:
        ...


trait RegisterMMABuffer(RegisterBuffer):
    comptime mma_dtype: DType
    comptime mma_tile_layout: Layout

    def get_mma_tile[
        tile_idx: Int, k_idx: Int
    ](self,) -> LocalLayoutTensor[Self.mma_dtype, Self.mma_tile_layout,]:
        ...


struct QRegisterBuffer[
    dtype: DType,
    mma_shape: IndexList[3],
    k_group_size: Int,
    WM: Int,
    WN: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    thread_layout: Layout,
]:
    comptime reg_dtype = Self.dtype
    comptime mma_dtype = Self.dtype
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_K = Self.mma_shape[2]
    # A-operand fragment size per lane: num_matrix_reg[MMA_M, MMA_K].
    # For bf16 [32,32,16]: (32*16)/64 = 8.
    # For fp8  [32,32,64]: (32*64)/64 = 32.
    comptime input_frag_size = num_matrix_reg[
        Self.MMA_M, Self.MMA_K
    ]() * Self.k_group_size
    comptime num_mmas = ceildiv(Self.WM, Self.MMA_M)
    comptime num_k_tiles = ceildiv(Self.BK, Self.MMA_K * Self.k_group_size)

    comptime num_tiles = Self.depth // Self.BK
    comptime _total_rows = Self.num_mmas * Self.num_k_tiles * Self.num_tiles
    comptime _rows_per_tile = Self.num_mmas * Self.num_k_tiles

    # TileTensor storage in registers.
    comptime reg_layout = tt_row_major[Self._total_rows, Self.input_frag_size]()
    comptime RegType = TileTensor[
        Self.dtype,
        type_of(Self.reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var reg_tile: Self.RegType

    # Thread layout for warp-scoped RegTileLoader (col-major to match
    # get_warp_layout[mma_shape] used by the original copy_dram_to_local).
    comptime _q_thread_rows = Self.thread_layout.shape[0].value()
    comptime _q_thread_cols = Self.thread_layout.shape[1].value()

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
        self.reg_tile = tt_stack_allocation[Self.dtype, AddressSpace.LOCAL](
            Self.reg_layout
        )

        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        # Warp's portion of Q: [WM, depth] sub-tile at (warp_row, 0).
        var warp_tile = q_tile.tile[Self.WM, Self.depth](warp_row, 0)
        var reg_loader = RegTileLoader[
            Self.dtype,
            tt_col_major[Self._q_thread_rows, Self._q_thread_cols](),
            warp_scope=True,
        ](warp_tile)

        # Load each BK-wide strip along the depth axis.
        comptime load_width = simd_width_of[Self.dtype]()
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
        type_of(tt_row_major[Self.num_mmas, Self.input_frag_size]()),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        """Return MMA-sized sub-tile for the given tile and k indices."""
        return rebind[
            TileTensor[
                Self.dtype,
                type_of(tt_row_major[Self.num_mmas, Self.input_frag_size]()),
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

        Casts each element to accum_type, multiplies by scale_factor, then
        casts back to Self.dtype. Used for pre-scaling Q by
        (1/sqrt(d) * log2e) so that QK matmul produces already-scaled
        scores, eliminating scale from the hot loop.
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
    comptime reg_layout = tt_row_major[
        Self._total_rows, Self.output_frag_size
    ]()
    comptime RegType = TileTensor[
        Self.dtype,
        type_of(Self.reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var reg_tile: Self.RegType

    @always_inline
    def __init__(out self):
        self.reg_tile = tt_stack_allocation[Self.dtype, AddressSpace.LOCAL](
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
    k_group_size: Int,
    tr_load_enabled: Bool = False,
    num_stages: Int = 1,
    p_swizzle: Optional[Swizzle] = None,
]:
    comptime reg_dtype = Self.accum_type_
    comptime mma_dtype = Self.dtype
    comptime mma_tile_layout = Layout.row_major(
        Self.num_m_mmas, simd_width_of[Self.dtype]()
    )
    comptime reg_tile_layout = Layout.row_major(
        Self.num_n_mmas * Self.num_m_mmas, Self.output_frag_size
    )

    # TileTensor storage (with staging dimension).
    comptime _staged_rows = (
        Self.num_stages * Self.num_n_mmas * Self.num_m_mmas
    )
    comptime _tiles_per_stage = Self.num_n_mmas * Self.num_m_mmas
    comptime reg_layout = tt_row_major[
        Self._staged_rows, Self.output_frag_size
    ]()
    comptime RegType = TileTensor[
        Self.accum_type_,
        type_of(Self.reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var reg_tile: Self.RegType

    # TileTensor type for a single pipeline stage sub-tile.
    comptime stage_layout = tt_row_major[
        Self._tiles_per_stage, Self.output_frag_size
    ]()
    comptime StageTileType = TileTensor[
        Self.accum_type_,
        type_of(Self.stage_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    # TileTensor type for MMA operand (cast from accum_type to mma_dtype).
    # Fragment width = A-operand regs per thread * k_group_size.
    comptime input_frag_size = num_matrix_reg[
        Self.mma_shape[0], Self.mma_shape[2]
    ]() * Self.k_group_size
    comptime _mma_layout = tt_row_major[Self.num_m_mmas, Self.input_frag_size]()
    comptime MmaTileType = TileTensor[
        Self.mma_dtype,
        type_of(Self._mma_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    # TiledMmaOp for SMEM→register A-matrix loads.
    comptime _TiledMma = TiledMmaOp[
        out_type=Self.accum_type_,
        in_type=Self.dtype,
        shape=Self.mma_shape,
        group_size=Self.k_group_size,
        transpose_b=False,
    ]

    var shared_memory_ptr: UnsafePointer[
        Scalar[Self.dtype],
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    @always_inline
    def __init__(
        out self,
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
    ):
        self.reg_tile = tt_stack_allocation[
            Self.accum_type_, AddressSpace.LOCAL
        ](Self.reg_layout)
        self.shared_memory_ptr = shared_ptr

    @always_inline
    def get_mma_tile_shared[
        tile_idx: Int, k_idx: Int
    ](self) -> Self.MmaTileType:
        var result = tt_stack_allocation[Self.mma_dtype, AddressSpace.LOCAL](
            Self._mma_layout
        )
        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        # Inline blocked offset: each BM×BK block is contiguous.
        var smem_block = TileTensor[
            Self.dtype,
            type_of(tt_row_major[Self.BM, Self.BK]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ](
            self.shared_memory_ptr + tile_idx * Self.BM * Self.BK,
            tt_row_major[Self.BM, Self.BK](),
        )
        var warp_tile = smem_block.tile[Self.WM, Self.BK](warp_row, 0)

        comptime if (
            Self.mma_shape[0] == 32
            and Self.input_frag_size == 32
            and Self.num_m_mmas == 1
        ):
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
            comptime warps_per_block = Self.BK // Self.WN
            comptime num_gather = (
                Self.input_frag_size // Self.output_frag_size
            )
            comptime warp_stride = (Self.BM * Self.BK) // warps_per_block
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
            var block_base = (
                self.shared_memory_ptr + tile_idx * Self.BM * Self.BK
            )
            var result_vec = result.vectorize[1, Self.input_frag_size]()
            var lo = (block_base + Int(lid) * Self.output_frag_size).load[
                width=Self.output_frag_size
            ]()
            var hi = (
                block_base + warp_stride + Int(lid) * Self.output_frag_size
            ).load[width=Self.output_frag_size]()
            var joined = lo.join(hi)
            result_vec[0, 0] = rebind[type_of(result_vec[0, 0])](joined)
        elif (
            Self.mma_shape[0] == 16
            and Self.tr_load_enabled
            # Only use the lo/hi join path when input_frag fits in 2
            # output_frag casts (bf16 16x16x32: 8 = 2*4).  For FP8
            # 16x16x128 (input_frag=32), fall through to load_a.
            and Self.input_frag_size <= 2 * Self.output_frag_size
        ):
            # 16x16 MMA: match the register mma_tile interleaving.
            # The [WM, BK] block has two [16, 16] MMA tiles side-by-side.
            # Each thread reads 4 bf16 from the left half (lo) and 4 from
            # the right half (hi), then joins → bf16[8].  This matches
            # the hardware B-operand register layout that the register
            # mma_tile path produces via cast+join.
            comptime mma_M = Self.mma_shape[0]
            comptime mma_N = Self.mma_shape[1]
            comptime frag_w = Self.output_frag_size  # 4
            comptime warp_m = mma_M  # 16
            comptime warp_n = WARP_SIZE // warp_m  # 4

            var result_vec = result.vectorize[1, Self.input_frag_size]()
            var smem_base = smem_block.ptr

            comptime for m in range(Self.num_m_mmas):
                # Left half [16, 16] tile — n_mma=0 within the BK block.
                var lo_tile = warp_tile.tile[mma_M, mma_N](m, 2 * k_idx)
                var lo_dist = lo_tile.vectorize[1, frag_w]().distribute[
                    tt_col_major[warp_m, warp_n]()
                ](lane_id())

                # Right half [16, 16] tile — n_mma=1 within the BK block.
                var hi_tile = warp_tile.tile[mma_M, mma_N](m, 2 * k_idx + 1)
                var hi_dist = hi_tile.vectorize[1, frag_w]().distribute[
                    tt_col_major[warp_m, warp_n]()
                ](lane_id())

                # Interleaved P read: single 16-byte load (ds_read_b128).
                # Each 8-element group = [lo_frag(4), hi_frag(4)].
                # Swizzle in 8-element units → 4 banks per read → 0
                # bank conflicts (same pattern as K loads).
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
            Self._TiledMma.load_a[swizzle=None](warp_tile, result, k_idx)

        return result

    @always_inline
    def stage_tile[stage: Int = 0](self) -> Self.StageTileType:
        """Return the TileTensor sub-tile for the given pipeline stage."""
        return rebind[Self.StageTileType](
            self.reg_tile.tile[Self._tiles_per_stage, Self.output_frag_size](
                stage, 0
            )
        )

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

        var result = tt_stack_allocation[Self.mma_dtype, AddressSpace.LOCAL](
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
                    var lo = src_vec[tile_idx * 2, 0].cast[Self.mma_dtype]()
                    var hi = src_vec[tile_idx * 2 + 1, 0].cast[Self.mma_dtype]()
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
    def zero(self):
        self.zero[0]()

    @always_inline
    def copy_to_shared(self):
        comptime frag_w = Self.output_frag_size
        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]

        var p_reg_vec = self.stage_tile[0]().vectorize[
            1, Self.output_frag_size
        ]()

        # MFMA 32x32x64 FP8: lane-contiguous SMEM layout.  Each lane
        # packs its 16 fp32 C-output (cast to fp8 = 16 bytes) into ONE
        # ds_write_b128.  Within each [BM, BK] P block, the two warps
        # contributing (warps_per_block=2) write into disjoint 1024B
        # regions: warp with n_mma_in_block=0 at [0..1023], with
        # n_mma_in_block=1 at [1024..2047].
        # Reader pairs up the two halves lane-by-lane (see
        # get_mma_tile_shared): slots 0..15 = warp0 lane l's 16 fp8,
        # slots 16..31 = warp1 lane l's 16 fp8 — the MFMA C-output
        # (M=l%32, N blocked) is exactly the MFMA B-operand pattern the
        # PV MMA needs, so no per-element reordering is required.
        comptime if (
            Self.mma_shape[0] == 32
            and not Self.p_swizzle
            and Self.num_m_mmas == 1
        ):
            comptime warps_per_block = Self.BK // Self.WN
            comptime warp_stride = (Self.BM * Self.BK) // warps_per_block
            comptime lane_bytes = size_of[Scalar[Self.dtype]]() * (
                Self.output_frag_size
            )
            comptime assert (
                lane_bytes == 16
            ), "FP8 MLA lane MMA-tile size must be 16B for ds_write_b128."
            comptime assert (
                Self.BM * Self.BK == warps_per_block * Self.num_n_mmas * 64 * 16
            ), "P SMEM block size must equal warps_per_block*num_n_mmas*64*16B."
            var block_idx = warp_col // Int(warps_per_block)
            var n_mma_in_block = warp_col % Int(warps_per_block)
            var block_base = (
                self.shared_memory_ptr + block_idx * Self.BM * Self.BK
            )
            var lid = lane_id()

            comptime for n_mma in range(Self.num_n_mmas):
                var p_reg_ptr = p_reg_vec.tile[1, 1](n_mma, 0).ptr
                var reg16 = p_reg_ptr.load[width=Self.output_frag_size]().cast[
                    Self.dtype
                ]()
                var warp_off = (
                    n_mma_in_block * Int(Self.num_n_mmas) + n_mma
                ) * Int(warp_stride)
                (
                    block_base + warp_off + Int(lid) * Self.output_frag_size
                ).store[width=Self.output_frag_size](reg16)
            return

        # 3D thread layout: (warp_rows, warp_cols, 1).
        # The trailing 1 leaves the inner frag_w elements undistributed,
        # so each thread gets frag_w contiguous scalars.
        comptime warp_m = Self.mma_shape[0]
        comptime warp_n = WARP_SIZE // warp_m
        comptime tl_3d = tt_col_major[warp_m, warp_n, 1]()

        # Stores one MMA tile from register index reg_idx into the
        # (m_mma, n_mma) position of a [WM, BK] SMEM warp tile.
        # The tile pointer is computed from smem_base + byte_offset.
        @always_inline
        @parameter
        def _store_mma_tile[
            reg_idx: Int
        ](
            smem_base: UnsafePointer[
                Scalar[Self.dtype],
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ],
            byte_offset: Int,
            m_mma: Int,
            n_mma: Int,
        ):
            var smem_warp_tile = TileTensor[
                Self.dtype,
                type_of(tt_row_major[Self.WM, Self.BK]()),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ](smem_base + byte_offset, tt_row_major[Self.WM, Self.BK]())

            var mma_tile = smem_warp_tile.tile[
                Self.mma_shape[0], Self.mma_shape[1]
            ](m_mma, n_mma)

            # Reshape MMA tile to 3D: (mma_M, mma_N/frag_w, frag_w)
            # with stride (S0, frag_w, 1). The inner frag_w dimension
            # embeds the element layout as a regular dimension instead
            # of tracking it via element_size.
            comptime mma_M = Self.mma_shape[0]
            comptime mma_N = Self.mma_shape[1]
            comptime S0 = type_of(mma_tile).LayoutType._stride_types[
                0
            ].static_value
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

            # Distribute: each thread gets shape (1, 1, frag_w) =
            # frag_w contiguous scalars at stride 1.
            var dst_frag = mma_3d.distribute[tl_3d](lane_id())

            # Load register SIMD vector and store to SMEM.
            var p_reg_tile = p_reg_vec.tile[1, 1](reg_idx, 0)
            comptime frag_size = p_reg_tile.element_size
            var reg_val = p_reg_tile.ptr.load[width=frag_size]().cast[
                Self.dtype
            ]()

            # Apply swizzle to spread rows across different LDS banks.
            # Use block base (smem_base + byte_offset) so the swizzle
            # offset matches the read path in get_mma_tile_shared.
            comptime if Self.p_swizzle:
                comptime elem_size = size_of[Scalar[Self.dtype]]()
                var block_base = smem_base + byte_offset
                var elem_off = (
                    Int(dst_frag.ptr) - Int(block_base)
                ) // elem_size
                var swizzled = (
                    Self.p_swizzle.value()(elem_off // frag_w) * frag_w
                )
                (block_base + swizzled).store[width=frag_w](reg_val)
            else:
                dst_frag.raw_store[width=frag_w](0, reg_val)

        comptime if Self.WN < Self.BK:
            # WN < BK: multiple warps fill each [BM, BK] SMEM block.
            # E.g. WN=16, BK=32: warps 0,1 fill block 0; warps 2,3
            # fill block 1.  Each warp writes its MMA tiles into the
            # appropriate MMA_N-sized column slot within the BK block.
            comptime warps_per_block = Self.BK // Self.WN
            var block_idx = warp_col // Int(warps_per_block)
            var n_mma_in_block = warp_col % Int(warps_per_block)
            var smem_offset = block_idx * Self.BM * Self.BK
            var warp_offset = smem_offset + warp_row * Self.WM * Self.BK

            comptime if Self.p_swizzle:
                # Interleaved P layout for 16-byte reads: each 8-element
                # group = [warp0_frag(4), warp1_frag(4)].  Warp N writes
                # to the N-th half of each group.  Swizzle in 8-element
                # units (matching the read granularity) so reads use
                # ds_read_b128 with 0 bank conflicts.
                comptime simd_w = simd_width_of[Self.dtype]()
                comptime warp_m = Self.mma_shape[0]
                comptime warp_n = WARP_SIZE // warp_m
                var r = umod(lane_id(), warp_m)
                var c = ufloordiv(lane_id(), warp_m)
                var block_base = self.shared_memory_ptr + smem_offset

                comptime for m_mma in range(Self.num_m_mmas):
                    comptime for n_mma in range(Self.num_n_mmas):
                        comptime reg_idx = n_mma * Self.num_m_mmas + m_mma
                        var p_reg_tile = p_reg_vec.tile[1, 1](reg_idx, 0)
                        var reg_val = p_reg_tile.raw_load[
                            width=p_reg_tile.element_size
                        ](0).cast[Self.dtype]()

                        # Group index = r * (BK/simd_w) + c
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
                        _store_mma_tile[reg_idx](
                            self.shared_memory_ptr,
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
                # layout in get_mma_tile_shared.  Both halves in the same
                # warp → ds_write_b128 → zero bank conflicts.
                comptime group_w = 2 * frag_w
                comptime warp_m_ = Self.mma_shape[0]
                var r = umod(lane_id(), warp_m_)
                var c = ufloordiv(lane_id(), warp_m_)

                comptime for i in range(Self.WN // Self.BK):
                    var block_idx_ = Int(i) + warp_col * (Self.WN // Self.BK)
                    var block_base = (
                        self.shared_memory_ptr + block_idx_ * Self.BM * Self.BK
                    )

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
                        _store_mma_tile[reg_idx](
                            self.shared_memory_ptr,
                            warp_offset,
                            m_mma,
                            n_mma,
                        )
