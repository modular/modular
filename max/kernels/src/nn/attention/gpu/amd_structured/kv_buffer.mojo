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
"""KV cache buffers for MHA/MLA prefill and decode kernels.

Provides KVCacheIterator (TileTensor-based DRAM tile iteration), the
KVBufferConfig trait + K/V implementors, and three KV buffer structs:

- KVBuffer: double-buffered DMA + LDS + register tile management
  used by MHA/MLA prefill (owns its DRAM iterator).
- DecodeStreamingKVBuffer: single-buffer per-strip DMA used by the
  streaming decode kernel (takes an external DRAM tile per iteration).
- DecodeKVBuffer: double-buffered register staging used by the decode
  mirror path (parametrized by KVBufferConfig for K vs V roles).

TileTensor is used throughout:
  - DRAM tiles: TileTensor with RuntimeInt valid_rows (KVCacheIterator)
  - SMEM sub-tiles: `.tile()` views on a strided parent TileTensor that
    mirrors the blocked (BN × BK) SMEM layout
  - DMA: SubTileLoaderLDS / RegTileLoader (both src and dst are TileTensor)
  - LDS loads: KVMmaOp.load_prefill / load_v_bf16 / load_v_fp8_strip
    (TileTensor SMEM -> reg-tile fragments)
  - MMA register tiles: TileTensor in LOCAL with stack_allocation
"""

from std.math import ceildiv
from std.math.uutils import umod, ufloordiv
from std.sys import simd_width_of, llvm_intrinsic
from std.gpu import WARP_SIZE, lane_id
from std.memory.pointer import AddressSpace as BaseAddressSpace
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    Layout,
    MixedLayout,
    RuntimeInt,
    TensorLayout,
    TileTensor,
)
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from layout.tensor_core import TiledTensorCore
from layout.swizzle import Swizzle
from std.utils.numerics import get_accum_type
from nn.attention.mha_operand import MHAOperand
from .mma import KVMmaOp, TiledMmaOp
from .utils import get_warp_coords
from structured_kernels.amd_tile_io import (
    RegTileLoader,
    RegTileWriterLDS,
    SubTileLoaderLDS,
    load_lds_fragment,
)

from std.utils import IndexList


struct KVCacheIterator[
    cache_t: MHAOperand,
    tile_size: Int,
    kv_num_heads: Int,
    depth: Int,
    cache_depth: Int = depth,
    head_dim_offset: Int = 0,
]:
    """TileTensor-based DRAM tile iterator.

    Returns a TileTensor with RuntimeInt for the row dimension (valid-row
    count) and ComptimeInt for depth and strides. No RuntimeLayout storage.

    When cache_depth != depth, the DRAM stride uses cache_depth (e.g., MLA
    K_rope reads 64 columns from a 576-wide cache row). head_dim_offset
    shifts the column start (e.g., skip to rope portion at column 512).
    """

    comptime GmemTileLayout = MixedLayout[
        Coord[RuntimeInt[DType.int64], ComptimeInt[Self.depth]].element_types,
        Coord[
            ComptimeInt[Self.kv_num_heads * Self.cache_depth], ComptimeInt[1]
        ].element_types,
    ]
    comptime GmemTileType = TileTensor[
        Self.cache_t.dtype,
        Self.GmemTileLayout,
        ImmutAnyOrigin,
    ]

    var cache: Self.cache_t
    var end: Int
    var tile_start_row: Int
    var batch_idx: Int
    var kv_head_idx: Int

    @always_inline
    def __init__(
        out self,
        cache: Self.cache_t,
        batch_idx: Int,
        kv_head_idx: Int,
        end: Int,
    ):
        self.cache = cache
        self.end = end
        self.tile_start_row = 0
        self.batch_idx = batch_idx
        self.kv_head_idx = kv_head_idx

    @always_inline
    def next_tile(mut self) -> Self.GmemTileType:
        """Returns a TileTensor for the next DRAM tile."""
        var valid_rows = max(
            0,
            min(Self.tile_size, self.end - self.tile_start_row),
        )
        var tile = self.cache.block_paged_tile[Self.tile_size](
            UInt32(self.batch_idx),
            UInt32(self.tile_start_row),
            UInt32(self.kv_head_idx),
            Self.GmemTileLayout(
                Coord(
                    RuntimeInt[DType.int64](Int64(valid_rows)),
                    Idx[Self.depth](),
                ),
                Coord(
                    Idx[Self.kv_num_heads * Self.cache_depth](),
                    Idx[1](),
                ),
            ),
            UInt32(Self.head_dim_offset),
        )
        self.tile_start_row += Self.tile_size
        return tile

    @always_inline
    def increment(mut self):
        self.tile_start_row += Self.tile_size


@always_inline
def _get_k_swizzle[mma_m: Int, bk: Int]() -> Optional[Swizzle]:
    """K swizzle for decode.

    XORs upper row bits into lower address bits to spread different rows
    across LDS bank groups in the col_major thread distribution. Returns
    Swizzle(3,0,4) for 32x32 MMA, Swizzle(3,0,3) for 16x16, and None for
    BK > 64 (fp8 16x16x128).
    """
    comptime if bk > 64:
        return None
    comptime if mma_m == 32:
        return Swizzle(3, 0, 4)
    return Swizzle(3, 0, 3)


struct KVBuffer[
    kv_t: MHAOperand,
    //,
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    num_threads: Int,
    depth: Int,
    kv_num_heads: Int,
    transpose: Bool,
    full_kv: Bool = True,
    cache_depth: Int = depth,
    head_dim_offset: Int = 0,
    reg_chunk_depth: Int = depth,
    smem_depth: Int = depth,
]:
    """KV cache buffer managing DMA, LDS staging, and register tiles.

    Handles the full data path: DRAM -> LDS (shared memory) -> registers.

    SMEM is navigated via `.tile()` on a strided parent TileTensor whose
    (BK, BN) strides encode the blocked layout (num_repeats contiguous
    BN×BK blocks per stage, two stages). Stage selection and in-stage
    block selection both happen via the tile column index — no pointer
    arithmetic required. smem_mma_subtile is still used for V-operand
    MMA sub-tiles which have mma_cols != BK.

    When full_kv=True (depth<=256), each SMEM stage holds BN x smem_depth
    elements — the full tile. When full_kv=False (depth=512), each stage
    holds only BN x BK elements, and the caller iterates over BK blocks.

    `smem_depth` defaults to `depth`. It exists for per-warp V buffers
    whose `depth` is smaller than `BK` (e.g. depth_per_warp=16 with
    BK=32): the SMEM layout stays valid with `smem_depth = max(depth,
    BK)` while `depth` keeps driving register-tile sizing and the column
    count read by `load_from_shared`.

    MMA register tiles (mma_tile) are TileTensor in LOCAL address space.
    TiledMmaOp (mma.mojo) handles SMEM→register loads and MMA dispatch.
    """

    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_mmas = ceildiv(
        Self.WN if Self.transpose else Self.depth, Self.MMA_N
    )
    comptime num_k_mmas2 = ceildiv(Self.BK, Self.MMA_K)
    # B-operand fragment size: matches the old prefill KV buffer's
    # input_frag_size = num_matrix_reg[MMA_K, MMA_N].
    # For BF16 [32,32,16]: 8.  For FP8 [32,32,64]: 32.
    comptime input_frag_size = (Self.MMA_K * Self.MMA_N) // WARP_SIZE
    comptime simd_width = simd_width_of[Self.kv_t.dtype]()
    comptime num_k_tiles = ceildiv(
        Self.depth if Self.transpose else Self.WN, Self.BK
    )
    # Register-side strip count. For the K path (transpose=True) the caller
    # can cap depth coverage via reg_chunk_depth so the reg tile stays small
    # even when SMEM holds the full depth; global strip index maps into the
    # reg tile via `bk_tile % _reg_num_k_tiles`.
    comptime _reg_num_k_tiles = (
        ceildiv(
            Self.reg_chunk_depth, Self.BK
        ) if Self.transpose else Self.num_k_tiles
    )

    comptime warp_tile_rows = 32
    comptime num_repeats = Self.smem_depth // Self.BK
    comptime smem_cols = Self.smem_depth if Self.full_kv else Self.BK
    comptime smem_stage_size = Self.BN * Self.smem_cols

    # Strided parent view over the full 2-stage SMEM allocation.
    # Shape (BN, _smem_total_cols) with stride (BK, BN) so that
    # `.tile[tile_rows, BK]((tile_row, tile_col))` produces
    # `tile_row * tile_rows * BK + tile_col * BN * BK`, matching the
    # block-aligned offsets of the blocked (BN × BK) SMEM layout.
    # tile_col indexes linearly over all blocks across both stages, so
    # stage selection happens via coordinate arithmetic rather than
    # pointer arithmetic (col = buffer_idx * blocks_per_stage + block).
    comptime _blocks_per_stage = Self.num_repeats if Self.full_kv else 1
    comptime _smem_total_cols = 2 * Self._blocks_per_stage * Self.BK
    comptime _SmemParentLayout = MixedLayout[
        Coord[
            ComptimeInt[Self.BN], ComptimeInt[Self._smem_total_cols]
        ].element_types,
        Coord[ComptimeInt[Self.BK], ComptimeInt[Self.BN]].element_types,
    ]
    # Strides for sub-tiles of width BK: plain row-major (BK, 1) so
    # element indexing within a block behaves normally.
    comptime _SmemTileStrides = MixedLayout[
        Coord[ComptimeInt[Self.BK], ComptimeInt[1]].element_types,
        Coord[ComptimeInt[1], ComptimeInt[1]].element_types,
    ]

    comptime _num_warps = Self.num_threads // WARP_SIZE
    comptime _dma_col_groups = (
        Self.smem_depth // Self.BK
    ) if Self.full_kv else 1
    comptime _total_tiles = (
        Self.BN // Self.warp_tile_rows
    ) * Self._dma_col_groups
    comptime _tiles_per_warp = ceildiv(Self._total_tiles, Self._num_warps)
    comptime vm_instrs_per_load = UInt32(Self._tiles_per_warp * 2)

    comptime _mma_total_rows = Self.num_mmas * Self.num_k_mmas2 * Self._reg_num_k_tiles
    comptime mma_layout = row_major[
        Self._mma_total_rows, Self.input_frag_size
    ]()
    comptime MMATileType = TileTensor[
        Self.kv_t.dtype,
        type_of(Self.mma_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    comptime KVMmaOpType = KVMmaOp[
        Self.kv_t.dtype,
        Self.mma_shape,
        Self.num_mmas,
        Self.num_k_mmas2,
        Self._reg_num_k_tiles,
        Self.BN,
        Self.BK,
        transpose_b=Self.transpose,
        swizzle=Self.swizzle,
    ]
    var kv_mma_op: Self.KVMmaOpType

    comptime wtile_dim0 = Self.WN
    comptime wtile_dim1 = Self.BK

    comptime SmemParentType = TileTensor[
        Self.kv_t.dtype,
        Self._SmemParentLayout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    var smem_tile: Self.SmemParentType

    var kv_cache_iter: KVCacheIterator[
        Self.kv_t,
        Self.BN,
        Self.kv_num_heads,
        Self.depth,
        Self.cache_depth,
        Self.head_dim_offset,
    ]

    var warp_id: UInt32

    @always_inline
    def _smem_view(self) -> Self.SmemParentType:
        """Full 2-stage SMEM view with strides (BK, BN)."""
        return self.smem_tile

    @always_inline
    def smem_block_tile[
        tile_rows: Int,
    ](self, tile_row: Int, block_col: Int) -> TileTensor[
        Self.kv_t.dtype,
        type_of(row_major[tile_rows, Self.BK]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]:
        """Get a (tile_rows, BK) row-major sub-tile from SMEM.

        tile_row indexes along BN (rows within a BN×BK block), block_col
        indexes linearly across all BN×BK blocks in both stages.
        """
        return self._smem_view().tile[
            tile_rows,
            Self.BK,
            stride_layout=Self._SmemTileStrides,
        ](Coord(Idx(tile_row), Idx(block_col)))

    @always_inline
    def __init__(
        out self,
        k_cache: Self.kv_t,
        batch_idx: Int,
        head_idx: Int,
        smem_tile: Self.SmemParentType,
        end: Int,
        warp_id: UInt32,
    ):
        self.kv_mma_op = Self.KVMmaOpType()
        self.smem_tile = smem_tile

        self.kv_cache_iter = type_of(self.kv_cache_iter)(
            k_cache, batch_idx, head_idx, end
        )

        self.warp_id = warp_id

    @always_inline
    def load_from_dram[buffer_idx: Int](mut self):
        var gmem_tile = self.kv_cache_iter.next_tile()
        var loader = SubTileLoaderLDS[Self.kv_t.dtype, Self.swizzle](gmem_tile)

        # Stage offset for this buffer_idx within the linear block index.
        comptime _stage_block_base = buffer_idx * Self._blocks_per_stage

        comptime if not Self.full_kv:
            comptime num_warps = Self.num_threads // WARP_SIZE
            comptime num_row_groups = Self.BN // Self.warp_tile_rows
            comptime tiles_per_warp = ceildiv(num_row_groups, num_warps)

            comptime for t in range(tiles_per_warp):
                comptime tile_idx = Int(t) * num_warps
                var warp_tile_idx = UInt32(tile_idx) + self.warp_id
                var warp_row = Int(warp_tile_idx)
                var smem_warp = self.smem_block_tile[Self.warp_tile_rows](
                    warp_row, _stage_block_base
                )
                var gmem_warp_tile = gmem_tile.tile[
                    Self.warp_tile_rows, Self.BK
                ](warp_row, 0)
                loader.load(smem_warp, gmem_warp_tile)
        elif (
            Self.smem_depth == 64
            and Self.BN <= Self.warp_tile_rows * 2
            and Self.smem_depth // Self.BK >= 2
        ):
            var warp_r, warp_c = divmod(Int(self.warp_id), 2)
            var smem_warp = self.smem_block_tile[Self.warp_tile_rows](
                warp_r, _stage_block_base + warp_c
            )
            var gmem_warp_tile = gmem_tile.tile[Self.warp_tile_rows, Self.BK](
                warp_r, warp_c
            )
            loader.load(smem_warp, gmem_warp_tile)
        else:
            comptime num_warps = Self.num_threads // WARP_SIZE
            comptime num_row_groups = Self.BN // Self.warp_tile_rows
            comptime num_col_groups = Self.smem_depth // Self.BK
            comptime total_tiles = num_row_groups * num_col_groups
            comptime tiles_per_warp = ceildiv(total_tiles, num_warps)

            comptime for t in range(tiles_per_warp):
                comptime tile_idx = Int(t) * num_warps
                var warp_tile = UInt32(tile_idx) + self.warp_id
                var warp_row, warp_col = divmod(
                    warp_tile, UInt32(num_col_groups)
                )
                var smem_warp = self.smem_block_tile[Self.warp_tile_rows](
                    Int(warp_row), _stage_block_base + Int(warp_col)
                )
                var gmem_warp_tile = gmem_tile.tile[
                    Self.warp_tile_rows, Self.BK
                ](Int(warp_row), Int(warp_col))
                loader.load(smem_warp, gmem_warp_tile)

    # split[N]()[idx] → tile[rows_per_split, cols](idx, 0)
    comptime _rows_per_k_tile = Self.num_mmas * Self.num_k_mmas2
    comptime _rows_per_k_mma = Self.num_mmas

    @always_inline
    def get_mma_tile[
        k_mma_tile_idx: Int,
        bk_tile_idx: Int,
    ](self) -> TileTensor[
        Self.kv_t.dtype,
        type_of(row_major[Self._rows_per_k_mma, Self.input_frag_size]()),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        comptime reg_slot = bk_tile_idx % Self._reg_num_k_tiles
        return self.kv_mma_op.mma_tile_at[reg_slot, k_mma_tile_idx]()

    @always_inline
    def mma_subtile[
        k_mma_tile_idx: Int,
        bk_tile_idx: Int,
    ](self) -> TileTensor[
        Self.kv_t.dtype,
        type_of(row_major[Self._rows_per_k_mma, Self.input_frag_size]()),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        """Alias for get_mma_tile, kept for decode-call-site symmetry."""
        return self.get_mma_tile[k_mma_tile_idx, bk_tile_idx]()

    @always_inline
    def load_from_shared(self, buffer: Int):
        # The no-index form loads every strip into the reg tile.  When the
        # reg tile is chunked (_reg_num_k_tiles < num_k_tiles) slots alias,
        # so the caller must drive the chunk loop explicitly via the
        # indexed overload.
        comptime assert Self._reg_num_k_tiles == Self.num_k_tiles, (
            "load_from_shared(buffer) requires full-depth reg tile; use"
            " load_from_shared[bk_tile](buffer) for chunked buffers"
        )
        comptime if (
            not Self.transpose
            and Self.kv_t.dtype.is_float8()
            and Self.mma_shape[0] == 16
        ):
            # FP8 16x16x128 V load: scalar path.
            # A-operand register layout via col_major(16, 4):
            # thread l at (row=l%16, col_group=l//16) holds 32 FP8
            # values from V^T[depth_row, keys g*32..(g+1)*32].
            # V SMEM is row_major(BN, BK): V[key, d] = base + key*BK + d.
            #
            # INTENTIONALLY NOT wrapped in `TiledMmaOp.load_b` /
            # `SubTileLoaderLDS.load` / an equivalent structured abstraction.
            # The A-operand layout is a per-lane **permuted scalar gather**
            # (each lane pulls 32 FP8 scalars from non-contiguous SMEM
            # rows parameterized by `lane_id()`), which no existing
            # distribute / vectorize primitive encodes. Wrapping it in a
            # new struct would just be a paperweight around the same
            # `(v_base + (key + j) * BK + depth_idx).load[width=1]()`
            # loop below — zero reuse beyond this one site, gated by the
            # narrow `mma_shape[0] == 16 && !transpose && is_float8`
            # comptime guard.
            # If this pattern ever appears elsewhere, the right move is
            # a `FP8VScalarGather` helper alongside `ds_read_tr16_b64_*`
            # in `structured_kernels/amd_tile_io.mojo`.
            comptime MMA_M_ = Self.mma_shape[0]
            comptime num_depth_tiles = Self.depth // MMA_M_

            var lid = lane_id()
            var v_base = self.smem_tile.tile[Self.BN, Self.BK](
                0, buffer * Self._blocks_per_stage
            ).ptr
            var row = umod(lid, MMA_M_)
            var col_group = ufloordiv(lid, MMA_M_)
            var reg_vec = self.kv_mma_op.reg_tile.vectorize[
                1, Self.input_frag_size
            ]()

            comptime for bk_tile in range(Self.num_k_tiles):
                comptime for dt in range(num_depth_tiles):
                    var depth_idx = Int(row) + dt * MMA_M_
                    var key_start = bk_tile * Self.BK + Int(col_group) * 32
                    var vals = SIMD[Self.kv_t.dtype, Self.input_frag_size]()
                    for j in range(Self.input_frag_size):
                        vals[j] = (
                            v_base + (key_start + j) * Self.BK + depth_idx
                        ).load[width=1]()
                    reg_vec[bk_tile * num_depth_tiles + dt, 0] = rebind[
                        type_of(reg_vec[0, 0])
                    ](vals)
        elif not Self.transpose and Self.kv_t.dtype.is_float8():
            # FP8 V vector load using ds_read_tr8_b64 with paired-lane
            # addressing.  Replaces ~128 scalar LDS reads with ~16 vector
            # reads (8x fewer instructions).
            #
            # Paired lanes (even/odd) access the same key at depth offsets
            # differing by 8.  After the hardware 8x8 transpose, each lane
            # holds 8 contiguous depth values.  Even lanes cover depths
            # [d, d+8), odd lanes [d+8, d+16) -> 16 unique depths per
            # 16-lane row.  Two rows within hw0 (depth_base 0 and 16) give
            # 32 depths; hw1 shifts keys by +4 for the complementary MFMA
            # C-output column pattern, covering all 64 BN keys per MFMA
            # tile. The per-strip load lives in
            # `TiledMmaLoader.load_v_fp8_strip` — see that method for
            # the addressing details; here we precompute the lane-only
            # coords once and iterate (bk_tile, dt).
            var v_base = self.smem_tile.tile[Self.BN, Self.BK](
                0, buffer * Self._blocks_per_stage
            ).ptr

            # Per-lane address components (computed once, reused across
            # all bk_tile iterations by `KVMmaOp.load_v_fp8_strip`).
            var lid = lane_id()
            var lane_in_row = umod(lid, 16)
            var pair_idx = ufloordiv(lane_in_row, 2)
            var is_odd = umod(lane_in_row, 2)
            var row_in_warp = ufloordiv(lid, 16)
            var is_hw1 = ufloordiv(lid, 32)
            var rel_key = Int(umod(pair_idx, 4) + ufloordiv(pair_idx, 4) * 8)
            var depth_base = Int(umod(row_in_warp, 2)) * 16 + Int(is_odd) * 8
            var hw_key_shift = Int(is_hw1) * 4

            comptime for bk_tile in range(Self.num_k_tiles):
                self.kv_mma_op.load_v_fp8_strip[bk_tile](
                    v_base, rel_key, hw_key_shift, depth_base
                )
        else:
            comptime for bk_tile in range(Self.num_k_tiles):
                self.load_from_shared[bk_tile](buffer)

    @always_inline
    def load_from_shared[bk_tile: Int](self, buffer: Int):
        var smem_base = self.smem_tile.tile[Self.BN, Self.BK](
            0, buffer * Self._blocks_per_stage
        ).ptr

        comptime if Self.transpose:
            # K (transpose) path: delegate to KVMmaOp.load_prefill.
            # BF16: single load per MMA tile.
            # FP8:  two half-K loads joined (num_packs=2 branch inside
            #       TiledMmaLoader.load_b matches the FP8 MMA K=128 layout).
            comptime num_warps_n = Self.BN // Self.WN
            var warp_col = umod(Int(self.warp_id), num_warps_n)

            var warp_smem = self.smem_block_tile[Self.wtile_dim0](
                Int(warp_col), buffer * Self._blocks_per_stage + bk_tile
            )

            comptime reg_slot = bk_tile % Self._reg_num_k_tiles
            self.kv_mma_op.load_prefill[reg_slot](warp_smem)

        else:
            comptime if Self.kv_t.dtype.is_float8():
                # FP8 V per-strip path — delegate to
                # `KVMmaOp.load_v_fp8_strip`. Single-bk_tile entry point
                # recomputes the lane-only coords each call (the bulk
                # `load_all_from_shared` variant hoists them across a
                # multi-bk loop).
                var lid = lane_id()
                var lane_in_row = umod(lid, 16)
                var pair_idx = ufloordiv(lane_in_row, 2)
                var is_odd = umod(lane_in_row, 2)
                var row_in_warp = ufloordiv(lid, 16)
                var is_hw1 = ufloordiv(lid, 32)
                var rel_key = Int(
                    umod(pair_idx, 4) + ufloordiv(pair_idx, 4) * 8
                )
                var depth_base = (
                    Int(umod(row_in_warp, 2)) * 16 + Int(is_odd) * 8
                )
                var hw_key_shift = Int(is_hw1) * 4

                self.kv_mma_op.load_v_fp8_strip[bk_tile](
                    smem_base, rel_key, hw_key_shift, depth_base
                )
            else:
                # BF16 V path — delegate to `KVMmaOp.load_v_bf16`.
                self.kv_mma_op.load_v_bf16[bk_tile](smem_base)


struct DecodeStreamingKVBuffer[
    kv_t: MHAOperand,
    //,
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    num_threads: Int,
    depth: Int,
    kv_num_heads: Int,
    transpose: Bool,
]:
    """Streaming-decode KV buffer: single-buffer SMEM staging with per-strip DMA.

    Unlike KVBuffer, this takes an external DRAM tile per
    outer-loop iteration and loads BK-wide strips one at a time.

    K (transpose=True): BN x BK SMEM, column strips from BN x depth.
    V (transpose=False): BK x depth SMEM (blocked BK x BK), row strips.
    """

    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]
    comptime simd_width = simd_width_of[Self.kv_t.dtype]()
    comptime input_frag_size = (Self.MMA_K * Self.MMA_N) // WARP_SIZE

    # MMA tiling within one strip.
    # K (transpose=True): WN MMA tiles per warp (warp tiling in key dim).
    # V (transpose=False): warp's depth portion — each warp handles
    # depth // num_warps_n columns, matching the output register tiling.
    comptime _num_warps_n = Self.BN // Self.WN
    comptime num_mmas = ceildiv(
        Self.WN if Self.transpose else Self.depth // Self._num_warps_n,
        Self.MMA_N,
    )
    comptime num_k_mmas2 = ceildiv(Self.BK, Self.MMA_K)

    # Register tile: single strip only.
    comptime _mma_total_rows = Self.num_mmas * Self.num_k_mmas2
    comptime mma_layout = row_major[
        Self._mma_total_rows, Self.input_frag_size
    ]()
    comptime MMATileType = TileTensor[
        Self.kv_t.dtype,
        type_of(Self.mma_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    comptime KVMmaOpType = KVMmaOp[
        Self.kv_t.dtype,
        Self.mma_shape,
        Self.num_mmas,
        Self.num_k_mmas2,
        1,  # num_k_tiles: single strip
        Self.BN,
        Self.BK,
        transpose_b=Self.transpose,
        swizzle=Self.swizzle,
    ]

    # DMA tiling: warp-cooperative load of one strip.
    comptime warp_tile_rows = 32
    comptime _num_warps = Self.num_threads // WARP_SIZE
    comptime _strip_rows = Self.BN if Self.transpose else Self.BK
    comptime _strip_cols = Self.BK if Self.transpose else Self.depth
    comptime _num_row_groups = Self._strip_rows // Self.warp_tile_rows
    comptime _num_col_groups = Self._strip_cols // Self.BK
    comptime _total_dma_tiles = Self._num_row_groups * Self._num_col_groups
    comptime _tiles_per_warp = ceildiv(Self._total_dma_tiles, Self._num_warps)

    # Strided parent view over the single-stage K SMEM (BN × BK).
    # Strides (BK, BN) so `.tile[tile_rows, BK]((tile_row, 0))` yields
    # `tile_row * tile_rows * BK`, matching smem_subtile for 1 block.
    # Only used for the K (transpose=True) path; V uses a plain
    # row_major[BK, depth] view constructed inline.
    comptime _KSmemParentLayout = MixedLayout[
        Coord[ComptimeInt[Self.BN], ComptimeInt[Self.BK]].element_types,
        Coord[ComptimeInt[Self.BK], ComptimeInt[Self.BN]].element_types,
    ]
    comptime _KSmemTileStrides = MixedLayout[
        Coord[ComptimeInt[Self.BK], ComptimeInt[1]].element_types,
        Coord[ComptimeInt[1], ComptimeInt[1]].element_types,
    ]

    var kv_mma_op: Self.KVMmaOpType
    var smem_ptr: UnsafePointer[
        Scalar[Self.kv_t.dtype],
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]
    var warp_id: UInt32

    @always_inline
    def __init__(
        out self,
        cache: Self.kv_t,
        batch_idx: Int,
        head_idx: Int,
        smem_ptr: UnsafePointer[
            Scalar[Self.kv_t.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
        num_keys: Int,
        warp_id: UInt32,
    ):
        self.kv_mma_op = Self.KVMmaOpType()
        self.smem_ptr = smem_ptr
        self.warp_id = warp_id

    @always_inline
    def _k_smem_view(
        self,
    ) -> TileTensor[
        Self.kv_t.dtype,
        Self._KSmemParentLayout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]:
        """Single-stage K SMEM view (BN × BK) with strides (BK, BN)."""
        return TileTensor[
            Self.kv_t.dtype,
            Self._KSmemParentLayout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ](self.smem_ptr, Self._KSmemParentLayout())

    @always_inline
    def k_smem_block_tile[
        tile_rows: Int,
    ](self, tile_row: Int) -> TileTensor[
        Self.kv_t.dtype,
        type_of(row_major[tile_rows, Self.BK]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]:
        """Get a (tile_rows, BK) row-major sub-tile from the K SMEM.

        Single-stage K SMEM has one BN×BK block, so only the row index
        along BN varies.
        """
        return self._k_smem_view().tile[
            tile_rows,
            Self.BK,
            stride_layout=Self._KSmemTileStrides,
        ](Coord(Idx(tile_row), Idx(0)))

    @always_inline
    def load_from_dram[
        strip_idx: Int
    ](self, gmem_tile: TileTensor[Self.kv_t.dtype, ...],):
        """Load one BK-wide strip from an external DRAM tile to SMEM.

        K (transpose=True): columns [strip*BK, (strip+1)*BK] from BN x depth.
        V (transpose=False): rows [strip*BK, (strip+1)*BK] from BN x depth.
        """
        comptime if Self.transpose:
            comptime if Self.kv_t.dtype.is_float8():
                # FP8 K decode: BK=32 is narrower than the SubTileLoaderLDS
                # DMA sub-tile (simd_width_of[fp8]()*4 = 64), so use a
                # DRAM→regs→SMEM path (same pattern as V below).
                comptime _k_sw = simd_width_of[Self.kv_t.dtype]()
                comptime _k_btile0 = Self.BN
                comptime _k_btile1 = Self.BK
                comptime _k_thr_rows = (
                    min(
                        Self.num_threads,
                        (_k_btile0 * _k_btile1) // _k_sw,
                    )
                    * _k_sw
                    // _k_btile1
                )
                comptime _k_thr_cols = _k_btile1 // _k_sw
                comptime _k_rows_per_stage = (
                    _k_btile0 * _k_btile1
                ) // Self.num_threads // _k_sw

                var reg_loader = RegTileLoader[
                    Self.kv_t.dtype,
                    row_major[_k_thr_rows, _k_thr_cols](),
                    Self.num_threads,
                ](gmem_tile)

                var load_buf = stack_allocation[
                    Self.kv_t.dtype, AddressSpace.LOCAL
                ](row_major[_k_rows_per_stage, _k_sw]())

                var dram_strip = gmem_tile.tile[Self.BN, Self.BK](0, strip_idx)
                reg_loader.load(
                    load_buf,
                    dram_strip.vectorize[1, _k_sw](),
                )

                comptime _k_smem_layout = row_major[Self.BN, Self.BK]()
                var k_smem = TileTensor[
                    Self.kv_t.dtype,
                    type_of(_k_smem_layout),
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                ](self.smem_ptr, _k_smem_layout)

                RegTileWriterLDS[
                    row_major[_k_thr_rows, _k_thr_cols](),
                    Self.swizzle,
                    Self.num_threads,
                ].copy(
                    k_smem.vectorize[1, _k_sw](),
                    load_buf.vectorize[1, _k_sw](),
                )
            else:
                # BF16 K: column strip → BN x BK to SMEM via LDS DMA.
                var loader = SubTileLoaderLDS[Self.kv_t.dtype, Self.swizzle](
                    gmem_tile
                )
                comptime for t in range(Self._tiles_per_warp):
                    comptime tile_base = Int(t) * Self._num_warps
                    var warp_tile = UInt32(tile_base) + self.warp_id
                    if warp_tile < UInt32(Self._total_dma_tiles):
                        var warp_row = Int(warp_tile)
                        var smem_warp = self.k_smem_block_tile[
                            Self.warp_tile_rows
                        ](warp_row)
                        var gmem_warp = gmem_tile.tile[Self.BN, Self.BK](
                            0, strip_idx
                        ).tile[Self.warp_tile_rows, Self.BK](warp_row, 0)
                        loader.load(smem_warp, gmem_warp)
        else:
            # V: row strip → BK x depth to flat SMEM [BK, depth].
            # DRAM→regs→SMEM matching the decode DecodeKVBuffer token_gen V.
            # Flat layout (stride=depth) is required so that load_b
            # sees the correct strides for warp sub-tile distribution.
            comptime _v_sw = simd_width_of[Self.kv_t.dtype]()
            comptime _v_btile0 = Self.BK
            comptime _v_btile1 = Self.depth
            comptime _v_thr_rows = (
                min(
                    Self.num_threads,
                    (_v_btile0 * _v_btile1) // _v_sw,
                )
                * _v_sw
                // _v_btile1
            )
            comptime _v_thr_cols = _v_btile1 // _v_sw
            comptime _v_warp_depth = Self.depth // Self._num_warps_n
            comptime _v_num_mmas_v = ceildiv(_v_warp_depth, Self.MMA_N)
            comptime _v_rows_per_stage = (_v_num_mmas_v * Self.num_k_mmas2)

            var reg_loader = RegTileLoader[
                Self.kv_t.dtype,
                row_major[_v_thr_rows, _v_thr_cols](),
                Self.num_threads,
            ](gmem_tile)

            var load_buf = stack_allocation[
                Self.kv_t.dtype, AddressSpace.LOCAL
            ](row_major[_v_rows_per_stage, _v_sw]())

            var dram_strip = gmem_tile.tile[Self.BK, Self.depth](strip_idx, 0)
            reg_loader.load(
                load_buf,
                dram_strip.vectorize[1, _v_sw](),
            )

            # Flat SMEM tile [BK, depth] — row_major stride = depth.
            comptime _v_smem_layout = row_major[Self.BK, Self.depth]()
            var v_smem = TileTensor[
                Self.kv_t.dtype,
                type_of(_v_smem_layout),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ](self.smem_ptr, _v_smem_layout)

            RegTileWriterLDS[
                row_major[_v_thr_rows, _v_thr_cols](),
                None,
                Self.num_threads,
            ].copy(
                v_smem.vectorize[1, _v_sw](),
                load_buf.vectorize[1, _v_sw](),
            )

    @always_inline
    def load_from_shared(self):
        """Load from SMEM to MMA registers."""
        comptime if Self.transpose:
            comptime if Self.kv_t.dtype.is_float8():
                # FP8 K (decode): MMA_M=16, MMA_K=32 → frag_width=8, which
                # load_lds_fragment handles with a single 8-element FP8
                # load per lane (use_fp8_split only fires at MMA_K=128).
                var warp_col = umod(Int(self.warp_id), Self._num_warps_n)
                var warp_smem = self.k_smem_block_tile[Self.WN](Int(warp_col))
                load_lds_fragment[Self.MMA_K, Self.swizzle](
                    warp_smem, self.kv_mma_op.reg_tile
                )
            else:
                # BF16 K (transpose): TiledMmaOp.load_b on a [WN, BK] SMEM
                # sub-view. Distribute+swizzle matches TensorCore.load_b
                # semantics (vector-granularity) and handles the plain
                # row-major [BN, BK] SMEM that MHA's DRAM→LDS writes.
                var warp_col = umod(Int(self.warp_id), Self._num_warps_n)
                var warp_tile = self.k_smem_block_tile[Self.WN](Int(warp_col))

                comptime for kg in range(Self.num_k_mmas2):
                    TiledMmaOp[
                        get_accum_type[Self.kv_t.dtype](),
                        Self.kv_t.dtype,
                        Self.mma_shape,
                        transpose_b=True,
                    ].load_b[swizzle=Self.swizzle](
                        warp_tile, self.get_mma_tile[Int(kg)](), Int(kg)
                    )
        else:
            # V (non-transpose): MFMA B non-transpose register layout.
            #
            # warp_tile is a [BK, warp_depth] sub-view of [BK, depth] SMEM,
            # so its row stride is `depth` (NOT `warp_depth`) — the tile is
            # strided, not dense. TileTensor's `vectorize[simd, 1]` tracks
            # only a scalar `element_size` and loses the element-layout
            # stride, so a subsequent `distribute` emits a contiguous
            # `load[width=simd]` that reads the wrong bytes on a strided
            # tile. LayoutTensor's `vectorize` preserves the element
            # layout via `zipped_divide` and iterates element-by-element
            # in `copy_from`, which is why the LayoutTensor path works.
            #
            # The MFMA B non-transpose layout is BLOCK-distributed:
            # lane (tr, tc) with tr = lane // MMA_N, tc = lane % MMA_N
            # holds `input_frag_size` CONSECUTIVE K-rows at column tc.
            # (`distribute_with_offset[row_major[WARP_SIZE/MMA_N, MMA_N]]`
            # would give CYCLIC distribution — rows [tr, tr+4, tr+8, …] —
            # which is the wrong register permutation for MFMA.) Emit
            # the block load explicitly with scalar strided reads.
            comptime _v_warp_depth = Self.depth // Self._num_warps_n
            comptime _v_num_mmas_v = ceildiv(_v_warp_depth, Self.MMA_N)
            comptime _v_k_rows = Self.MMA_K
            comptime _v_smem_layout = row_major[Self.BK, Self.depth]()

            var warp_col = umod(Int(self.warp_id), Self._num_warps_n)
            var smem_tile = TileTensor[
                Self.kv_t.dtype,
                type_of(_v_smem_layout),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ](self.smem_ptr, _v_smem_layout)
            var warp_tile = smem_tile.tile[Self.BK, _v_warp_depth](
                0, Int(warp_col)
            )

            var reg_vec = self.kv_mma_op.reg_tile.vectorize[
                1, Self.input_frag_size
            ]()

            var tr, tc = divmod(Int(lane_id()), Self.MMA_N)

            comptime for kg in range(Self.num_k_mmas2):
                comptime for i in range(_v_num_mmas_v):
                    # mma_view: [_v_k_rows, MMA_N] at (kg, i) within
                    # warp_tile; row stride = depth (strided tile).
                    var mma_view = warp_tile.tile[_v_k_rows, Self.MMA_N](kg, i)
                    var frag = SIMD[Self.kv_t.dtype, Self.input_frag_size]()
                    comptime for r in range(Self.input_frag_size):
                        frag[r] = mma_view[tr * Self.input_frag_size + r, tc][0]
                    reg_vec[kg * _v_num_mmas_v + i, 0] = frag

    @always_inline
    def get_mma_tile[
        k_mma_idx: Int,
    ](self) -> TileTensor[
        Self.kv_t.dtype,
        type_of(row_major[Self.num_mmas, Self.input_frag_size]()),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        """Get register tile for one k_mma group within the single strip."""
        return self.kv_mma_op.mma_tile_at[0, k_mma_idx]()


# ===----------------------------------------------------------------------=== #
# DecodeKVBuffer — register-staged buffer for the decode mirror path
# ===----------------------------------------------------------------------=== #


trait KVBufferConfig:
    comptime wsize: Int
    comptime wtile_dim0: Int
    comptime wtile_dim1: Int

    comptime btile_dim0: Int
    comptime btile_dim1: Int

    comptime iterator_axis: Int

    @staticmethod
    @always_inline
    def get_wtile_coord() -> IndexList[2]:
        ...


@fieldwise_init
struct KBufferConfig[BN: Int, BK: Int, WN: Int](KVBufferConfig):
    comptime wsize = Self.wtile_dim0
    comptime wtile_dim0 = Self.WN
    comptime wtile_dim1 = Self.BK

    comptime btile_dim0 = Self.BN
    comptime btile_dim1 = Self.BK

    comptime iterator_axis = 1

    @staticmethod
    @always_inline
    def get_wtile_coord() -> IndexList[2]:
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]
        return IndexList[2](warp_col, 0)


@fieldwise_init
struct VBufferConfig[BN: Int, BK: Int, WN: Int, depth: Int](KVBufferConfig):
    comptime wsize = Self.wtile_dim1
    comptime wtile_dim0 = Self.BK
    comptime wtile_dim1 = Self.depth // (Self.BN // Self.WN)

    comptime btile_dim0 = Self.BK
    comptime btile_dim1 = Self.depth

    comptime iterator_axis = 0

    @staticmethod
    @always_inline
    def get_wtile_coord() -> IndexList[2]:
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]
        return IndexList[2](0, warp_col)


struct DecodeKVBuffer[
    dtype: DType,
    kv_tile_layout: TensorLayout,
    //,
    config: KVBufferConfig,
    tensor_core_mma: TiledTensorCore,
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
    token_gen: Bool = False,
]:
    comptime _dtype = Self.dtype
    comptime _num_stages = Self.num_stages
    comptime MMA_N = Self.tensor_core_mma.shape[1]
    comptime MMA_K = Self.tensor_core_mma.shape[2]
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_mmas = ceildiv(Self.config.wsize, Self.MMA_N)

    comptime num_k_tiles = ceildiv(
        Self.BK, Self.MMA_K * Self.tensor_core_mma.group_size
    )
    comptime simd_width = simd_width_of[Self.dtype]()

    # Thread layout for DRAM→register and register→SMEM distribution.
    # token_gen uses a layout matching the vectorized smem tile shape;
    # non-token_gen uses the standard (num_threads//4, 4) grid.
    comptime _btile_dim0 = Self.config.btile_dim0
    comptime _btile_dim1 = Self.config.btile_dim1
    comptime _thread_rows = (
        min(
            Self.num_threads,
            (Self._btile_dim0 * Self._btile_dim1) // Self.simd_width,
        )
        * Self.simd_width
        // Self._btile_dim1
    ) if Self.token_gen else Self.num_threads // 4
    comptime _thread_cols = (
        Self._btile_dim1 // Self.simd_width
    ) if Self.token_gen else 4

    # TileTensor register storage for DMA staging.
    comptime _rows_per_stage = Self.num_mmas * Self.num_k_tiles
    comptime _load_rows = Self.num_stages * Self._rows_per_stage
    comptime _load_layout = row_major[Self._load_rows, Self.simd_width]()
    comptime LoadTile = TileTensor[
        Self.dtype,
        type_of(Self._load_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var load_tile: Self.LoadTile

    # TileTensor register storage for MMA operand.
    comptime _mma_rows = Self.num_mmas
    comptime _mma_cols = Self.simd_width
    comptime _mma_layout = row_major[Self._mma_rows, Self._mma_cols]()
    comptime MmaTile = TileTensor[
        Self.dtype,
        type_of(Self._mma_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var mma_tile: Self.MmaTile

    comptime wtile_dim0 = Self.config.wtile_dim0
    comptime wtile_dim1 = Self.config.wtile_dim1

    # TiledMmaOp for SMEM→register loads.
    comptime _TiledMma = TiledMmaOp[
        out_type=Self.tensor_core_mma.out_type,
        in_type=Self.dtype,
        shape=Self.tensor_core_mma.shape,
        transpose_b=Self.tensor_core_mma.transpose_b,
    ]

    # SMEM TileTensor (stored once, used by copy_to_shared and load_from_shared).
    comptime _smem_layout = row_major[Self._btile_dim0, Self._btile_dim1]()
    comptime SmemTile = TileTensor[
        Self.dtype,
        type_of(Self._smem_layout),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]
    var smem_tile: Self.SmemTile

    # DRAM tile and loader.
    comptime GmemTileType = TileTensor[
        Self.dtype, Self.kv_tile_layout, ImmutAnyOrigin
    ]
    # num_threads overridden: block has more threads than the load layout.
    comptime RegLoaderType = RegTileLoader[
        Self.dtype,
        row_major[Self._thread_rows, Self._thread_cols](),
        Self.num_threads,
    ]
    var gmem_tile: Self.GmemTileType
    var reg_loader: Self.RegLoaderType
    var tile_idx: Int
    var load_tile_id: Int

    @always_inline
    def __init__(
        out self,
        gmem_tile: Self.GmemTileType,
        smem_tile: Self.SmemTile,
    ):
        self.load_tile = stack_allocation[Self.dtype, AddressSpace.LOCAL](
            Self._load_layout
        )
        self.mma_tile = stack_allocation[Self.dtype, AddressSpace.LOCAL](
            Self._mma_layout
        )
        self.smem_tile = smem_tile
        self.gmem_tile = gmem_tile
        self.reg_loader = Self.RegLoaderType(gmem_tile)
        self.tile_idx = 0
        self.load_tile_id = 0

    @always_inline
    def load_from_dram(
        mut self,
    ):
        # Build per-iteration DRAM sub-tile on-the-fly.
        var row_idx = 0 if Self.config.iterator_axis == 1 else self.tile_idx
        var col_idx = self.tile_idx if Self.config.iterator_axis == 1 else 0
        var src = self.gmem_tile.tile[Self._btile_dim0, Self._btile_dim1](
            row_idx, col_idx
        )
        var dst = self.load_tile.tile[Self._rows_per_stage, Self.simd_width](
            self.load_tile_id, 0
        )
        self.reg_loader.load(
            dst,
            src.vectorize[1, Self.simd_width](),
        )
        self.tile_idx += 1
        self.load_tile_id = (self.load_tile_id + 1) % Self.num_stages

    @always_inline
    def get_mma_tile(
        self,
    ) -> TileTensor[
        Self.dtype,
        type_of(row_major[Self._mma_rows, Self._mma_cols]()),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        return rebind[
            TileTensor[
                Self.dtype,
                type_of(row_major[Self._mma_rows, Self._mma_cols]()),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ]
        ](self.mma_tile)

    @always_inline
    def copy_to_shared[
        tile_id: Int = 0
    ](self,):
        RegTileWriterLDS[
            row_major[Self._thread_rows, Self._thread_cols](),
            Self.swizzle,
            Self.num_threads,
        ].copy(
            self.smem_tile.vectorize[1, Self.simd_width](),
            self.load_tile.tile[Self._rows_per_stage, Self.simd_width](
                tile_id, 0
            ).vectorize[1, Self.simd_width](),
        )

    @always_inline
    def load_from_shared[
        k_mma: Int,
    ](self):
        var wtile_coord0 = Self.config.get_wtile_coord()[0]
        var wtile_coord1 = Self.config.get_wtile_coord()[1]
        var warp_tile = self.smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
            wtile_coord0, wtile_coord1
        )

        comptime if Self.tensor_core_mma.transpose_b:
            # K (transpose_b=True): warp_tile is a contiguous [WN, BK]
            # sub-view, TiledMmaOp.load_b handles swizzle + distribute.
            Self._TiledMma.load_b[swizzle=Self.swizzle](
                warp_tile, self.mma_tile, k_mma
            )
        else:
            # V (transpose_b=False): warp_tile is a strided [BK, warp_depth]
            # sub-view of [BK, depth] (row stride = depth). TileTensor
            # vectorize drops the element-layout stride, so emit the MFMA
            # B non-transpose BLOCK distribution explicitly: lane
            # (tr, tc) with tr = lane // MMA_N, tc = lane % MMA_N holds
            # `simd_width` consecutive K-rows at column tc.
            comptime k_rows = (Self.MMA_K * Self.tensor_core_mma.group_size)
            var tr, tc = divmod(Int(lane_id()), Self.MMA_N)
            var reg_vec = self.mma_tile.vectorize[1, Self.simd_width]()

            comptime for i in range(Self.num_mmas):
                var mma_view = warp_tile.tile[k_rows, Self.MMA_N](k_mma, i)
                var frag = SIMD[Self.dtype, Self.simd_width]()
                comptime for r in range(Self.simd_width):
                    frag[r] = mma_view[tr * Self.simd_width + r, tc][0]
                reg_vec[i, 0] = frag


comptime KBuffer[
    dtype: DType,
    kv_tile_layout: TensorLayout,
    tensor_core_mma: TiledTensorCore,
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
    token_gen: Bool = False,
] = DecodeKVBuffer[
    dtype=dtype,
    kv_tile_layout=kv_tile_layout,
    config=KBufferConfig[BN, BK, WN],
    tensor_core_mma=tensor_core_mma,
    swizzle=swizzle,
    BN=BN,
    WN=WN,
    BK=BK,
    depth=depth,
    num_threads=num_threads,
    num_stages=num_stages,
    token_gen=token_gen,
]

comptime VBuffer[
    dtype: DType,
    kv_tile_layout: TensorLayout,
    tensor_core_mma: TiledTensorCore,
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
    token_gen: Bool = False,
] = DecodeKVBuffer[
    dtype=dtype,
    kv_tile_layout=kv_tile_layout,
    config=VBufferConfig[BN, BK, WN, depth],
    tensor_core_mma=tensor_core_mma,
    swizzle=swizzle,
    BN=BN,
    WN=WN,
    BK=BK,
    depth=depth,
    num_threads=num_threads,
    num_stages=num_stages,
    token_gen=token_gen,
]
