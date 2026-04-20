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

from std.math import ceildiv
from std.math.uutils import umod, ufloordiv
from std.sys import simd_width_of, llvm_intrinsic, get_defined_bool
from std.sys.intrinsics import readfirstlane, _type_is_eq
from std.gpu import WARP_SIZE, block_idx, lane_id
from std.gpu import warp_id as get_warp_id
from std.gpu.sync import schedule_barrier, s_waitcnt
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    TileTensor,
)
from layout.tile_layout import row_major as tt_row_major
from layout.layout import blocked_product
from layout.swizzle import Swizzle
from layout.tensor_core import num_matrix_reg
from .mma import TiledMmaOp
from std.memory import bitcast
from std.gpu.intrinsics import ds_read_tr8_b64
from nn.attention.mha_mask import TileMaskStatus, CausalMask
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_utils import get_start_and_end_for_partitions

from std.utils import IndexList
from std.utils.numerics import get_accum_type
from nn.attention.mha_utils import MHAConfig
from .attention import Attention, AttentionConfig
from .utils import (
    load_b,
    load_b_tr,
    copy_dram_to_sram_lds,
)


@fieldwise_init
struct MHAAttentionConfig[token_gen: Bool, config: MHAConfig, group: Int](
    AttentionConfig
):
    # share shared memory for k and v
    # depth=512 with BN=128: separate K+V = 256KB > 160KB LDS limit.
    # Share K/V SMEM (max(K,V) = 128KB + 4KB P = 132KB, fits 1 WG).
    comptime shared_kv = Self.token_gen and Self.config.depth > 256
    # shared memory for the full tile vs BK blocks
    comptime full_kv = True
    # pad the depth for v smem
    comptime depth_padded = False
    # double shared memory for k and v (prefill only; decode uses single buffer)
    comptime double_buffer = not Self.token_gen
    # double buffer K only for decode with small BN (V stays single-buffered).
    # BN=128 uses single-buffer K to fit 2 WGs in MI355's 160KB LDS.
    comptime double_buffer_k_only = (
        Self.token_gen and Self.config.block_n() <= 64
    )

    @staticmethod
    @always_inline
    def q_head_idx() -> Int:
        comptime if Self.token_gen:
            comptime mma_shape = Self.get_mma_shape()
            var group_idx = umod(lane_id(), mma_shape[0])
            return block_idx.y * Self.group + group_idx
        else:
            return block_idx.x

    @staticmethod
    @always_inline
    def q_tile_idx() -> Int:
        return block_idx.y if not Self.token_gen else 0

    @staticmethod
    @always_inline
    def kv_head_idx() -> Int:
        # decode and prefill have different launch configs
        return block_idx.y if Self.token_gen else ufloordiv(
            Self.q_head_idx(), Self.group
        )

    @staticmethod
    @always_inline
    def get_mma_shape() -> IndexList[3]:
        # FP8: decode uses 16x16x128 when depth%128==0, else 32x32x64.
        # Prefill always uses 32x32x64.
        comptime if Self.config.dtype.is_float8():
            comptime if Self.token_gen:
                comptime if Self.config.depth % 128 == 0:
                    return IndexList[3](16, 16, 128)
                return IndexList[3](32, 32, 64)
            return IndexList[3](32, 32, 64)
        # BF16: prefill uses 32x32x16, decode uses 16x16x32.
        comptime if Self.token_gen:
            return IndexList[3](16, 16, 32)
        return IndexList[3](32, 32, 16)

    @staticmethod
    @always_inline
    def get_q_offset[q_depth: Int]() -> UInt32:
        return UInt32(
            q_depth
            * (
                (
                    Self.kv_head_idx()
                    * Self.group if Self.token_gen else Self.q_head_idx()
                )
                + Self.config.num_heads
                * Self.q_tile_idx()
                * Self.config.block_m()
            )
        )

    @staticmethod
    @always_inline
    def get_output_offset[output_depth: Int]() -> UInt32:
        return Self.get_q_offset[output_depth]()


@always_inline
def barrier[
    *, schedule_barrier_before: Bool = True, schedule_barrier_after: Bool = True
]():
    comptime if schedule_barrier_before:
        schedule_barrier()

    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

    comptime if schedule_barrier_after:
        schedule_barrier()


@always_inline
def block_sync_lds[
    *,
    lgkmcnt: UInt32 = 0,
]():
    """
    Synchronize LDS (local data share) with waitcnt barrier.
    """

    s_waitcnt[lgkmcnt=lgkmcnt]()


@always_inline
def block_sync_lds_direct_load[
    *,
    vmcnt: UInt32 = 0,
]():
    """
    Synchronize LDS for direct load with waitcnt barrier.
    """
    s_waitcnt[vmcnt=vmcnt]()


struct KVCacheIterator[
    cache_t: MHAOperand,
    tile_size: Int,
    kv_num_heads: Int,
    depth: Int,
    cache_depth: Int = depth,
    head_dim_offset: Int = 0,
]:
    comptime kv_gmem_layout = Layout(
        IntTuple(Self.tile_size, Self.depth),
        IntTuple(Self.kv_num_heads * Self.cache_depth, 1),
    )
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
    def next_unsafe(
        mut self,
        out result: LayoutTensor[
            Self.cache_t.dtype,
            Self.kv_gmem_layout,
            ImmutAnyOrigin,
            masked=True,
        ],
    ):
        var kv_tile_num_rows = min(
            Self.tile_size,
            self.end - self.tile_start_row,
        )
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = type_of(result.runtime_layout)(
            type_of(result.runtime_layout.shape)(kv_tile_num_rows, Self.depth),
            type_of(result.runtime_layout.stride)(
                Self.kv_num_heads * Self.cache_depth, 1
            ),
        )
        var out = type_of(result)(
            self.cache.block_paged_ptr[Self.tile_size](
                UInt32(self.batch_idx),
                UInt32(self.tile_start_row),
                UInt32(self.kv_head_idx),
                UInt32(Self.head_dim_offset),
            ),
            kv_runtime_layout,
        )
        self.tile_start_row += Self.tile_size
        return out

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
    k_group_size: Int,
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    num_threads: Int,
    depth: Int,
    kv_num_heads: Int,
    transpose: Bool,
    cache_depth: Int = depth,
    head_dim_offset: Int = 0,
    # SMEM block width for the blocked-product layout. When depth < BK
    # (e.g. depth_per_warp=16 with BK=32), num_repeats = depth//BK = 0
    # which makes the layout degenerate. Set smem_depth = BK so
    # num_repeats >= 1. 'depth' still controls register tile sizing and
    # the column count in load_from_shared.
    smem_depth: Int = depth,
]:
    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_mmas = ceildiv(
        Self.WN if Self.transpose else Self.depth, Self.MMA_N
    )
    comptime num_k_mmas2 = ceildiv(Self.BK, Self.MMA_K * Self.k_group_size)
    # B-operand fragment size per lane: reg[MMA_K, MMA_N] * k_group_size.
    comptime input_frag_size = num_matrix_reg[
        Self.MMA_K, Self.MMA_N
    ]() * Self.k_group_size
    comptime num_k_tiles = ceildiv(
        Self.depth if Self.transpose else Self.WN, Self.BK
    )

    comptime warp_tile_rows = 32
    # Use smem_depth (not depth) for SMEM layout — depth may be smaller
    # than BK when multiple warps share a single BK-wide block.
    comptime num_repeats = Self.smem_depth // Self.BK
    comptime tiler_layout = Layout.row_major(1, Self.num_repeats)
    comptime base_layout = Layout.row_major(Self.BN, Self.BK)

    # Number of VM (vector memory) instructions each warp issues per
    # load_from_dram call.  Used by block_sync_lds_direct_load to
    # wait for DMA completion.  Assumes a 16×4 thread distribution
    # (2 sub-iters per 32×BK warp tile); callers that use larger BK
    # with an 8×8 thread layout should pass vmcnt=0 and wait for all
    # VM ops instead of relying on this count.
    comptime _num_warps = Self.num_threads // WARP_SIZE
    comptime _total_tiles = (Self.BN // Self.warp_tile_rows) * (
        Self.smem_depth // Self.BK
    )
    comptime _tiles_per_warp = ceildiv(Self._total_tiles, Self._num_warps)
    comptime vm_instrs_per_load = UInt32(Self._tiles_per_warp * 2)
    comptime smem_layout = blocked_product(Self.base_layout, Self.tiler_layout)

    comptime MMATileType = LayoutTensor[
        Self.kv_t.dtype,
        Layout.row_major(
            Self.num_mmas * Self.num_k_mmas2 * Self.num_k_tiles,
            Self.input_frag_size,
        ),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var mma_tile: Self.MMATileType

    comptime wtile_dim0 = Self.WN
    comptime wtile_dim1 = Self.BK

    comptime SharedTileType = LayoutTensor[
        Self.kv_t.dtype,
        Self.smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]
    comptime SharedWarpTileType = Self.SharedTileType.TileType[
        Self.wtile_dim0, Self.wtile_dim1
    ]

    var smem_ptr: UnsafePointer[
        Scalar[Self.kv_t.dtype],
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    comptime smem_stage_size = Self.smem_layout.size()

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
    def __init__(
        out self,
        k_cache: Self.kv_t,
        batch_idx: Int,
        head_idx: Int,
        shared_ptr: UnsafePointer[
            Scalar[Self.kv_t.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
        end: Int,
        warp_id: UInt32,
    ):
        self.mma_tile = type_of(self.mma_tile).stack_allocation()
        self.smem_ptr = shared_ptr

        self.kv_cache_iter = type_of(self.kv_cache_iter)(
            k_cache, batch_idx, head_idx, end
        )

        self.warp_id = warp_id

    @always_inline
    def load_from_dram[buffer_idx: Int](mut self):
        var global_tile = self.kv_cache_iter.next_unsafe()

        var smem_tile = Self.SharedTileType(
            self.smem_ptr + buffer_idx * Self.smem_stage_size
        )

        comptime if Self.smem_depth == 64 and Self.BN <= Self.warp_tile_rows * 2 and Self.smem_depth // Self.BK >= 2:
            var smem_warp_tile = smem_tile.tile[Self.warp_tile_rows, Self.BK](
                Int(self.warp_id) // 2, Int(self.warp_id) % 2
            )
            var gmem_warp_tile = global_tile.tile[Self.warp_tile_rows, Self.BK](
                Int(self.warp_id) // 2, Int(self.warp_id) % 2
            )
            var lds_base = UInt32(readfirstlane(Int32(Int(smem_warp_tile.ptr))))
            # load from dram to sram directly
            copy_dram_to_sram_lds[swizzle=Self.swizzle,](
                smem_warp_tile,
                gmem_warp_tile,
                lds_base,
            )
        else:
            comptime num_warps = Self.num_threads // WARP_SIZE
            # Number of row groups and depth groups (warp_tile_rows×BK tiles)
            comptime num_row_groups = Self.BN // Self.warp_tile_rows
            comptime num_col_groups = Self.smem_depth // Self.BK
            # Total tiles to cover: num_row_groups × num_col_groups
            comptime total_tiles = num_row_groups * num_col_groups
            # Each warp handles ceil(total_tiles / num_warps) tiles
            comptime tiles_per_warp = ceildiv(total_tiles, num_warps)

            comptime for t in range(tiles_per_warp):
                comptime tile_idx = Int(t) * num_warps
                # Each warp's tile index
                var warp_tile = UInt32(tile_idx) + self.warp_id
                var warp_row, warp_col = divmod(
                    warp_tile, UInt32(num_col_groups)
                )
                var smem_warp_tile = smem_tile.tile[
                    Self.warp_tile_rows, Self.BK
                ](Int(warp_row), Int(warp_col))
                var gmem_warp_tile = global_tile.tile[
                    Self.warp_tile_rows, Self.BK
                ](Int(warp_row), Int(warp_col))
                # Compute LDS base pointer for this specific tile
                var lds_base = UInt32(
                    readfirstlane(Int32(Int(smem_warp_tile.ptr)))
                )
                # load from dram to sram directly
                copy_dram_to_sram_lds[swizzle=Self.swizzle,](
                    smem_warp_tile,
                    gmem_warp_tile,
                    lds_base,
                )

    @always_inline
    def get_mma_tile[
        k_mma_tile_idx: Int,
        bk_tile_idx: Int,
    ](self) -> Self.MMATileType.SplitElementType[
        Self.num_k_tiles
    ].SplitElementType[Self.num_k_mmas2]:
        return self.mma_tile.split[Self.num_k_tiles]()[bk_tile_idx].split[
            self.num_k_mmas2
        ]()[k_mma_tile_idx]

    comptime _total_rows = Self.num_mmas * Self.num_k_mmas2 * Self.num_k_tiles
    comptime _rows_per_k_tile = Self.num_mmas * Self.num_k_mmas2
    comptime _layout = tt_row_major[Self._total_rows, Self.input_frag_size]()

    @always_inline
    def mma_subtile[
        k_mma_tile_idx: Int,
        bk_tile_idx: Int,
    ](self) -> TileTensor[
        Self.kv_t.dtype,
        type_of(tt_row_major[Self.num_mmas, Self.input_frag_size]()),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        """Return MMA-sized sub-tile for the given k_mma and bk indices."""
        comptime full_layout = Self._layout
        var full = TileTensor[
            Self.kv_t.dtype,
            type_of(full_layout),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ](self.mma_tile.ptr, full_layout)
        return rebind[
            TileTensor[
                Self.kv_t.dtype,
                type_of(tt_row_major[Self.num_mmas, Self.input_frag_size]()),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ]
        ](
            full.tile[Self._rows_per_k_tile, Self.input_frag_size](
                bk_tile_idx, 0
            ).tile[Self.num_mmas, Self.input_frag_size](k_mma_tile_idx, 0)
        )

    @always_inline
    def load_from_shared(self, buffer: Int):
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
            comptime MMA_M_ = Self.mma_shape[0]
            comptime num_depth_tiles = Self.depth // MMA_M_

            var lid = lane_id()
            var v_base = self.smem_ptr + buffer * Self.smem_stage_size
            var row = umod(lid, MMA_M_)
            var col_group = ufloordiv(lid, MMA_M_)
            var reg_vec = self.mma_tile.vectorize[1, Self.input_frag_size]()

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
            # FP8 32x32x64 V vector load using ds_read_b64_tr_b8 with
            # paired-lane addressing.  Replaces ~128 scalar LDS reads
            # with ~16 vector reads (8× fewer instructions).
            #
            # Paired lanes (even/odd) access the same key at depth
            # offsets differing by 8.  After the hardware 8×8
            # transpose, each lane holds 8 contiguous depth values.
            # Even lanes cover depths [d, d+8), odd lanes [d+8, d+16)
            # → 16 unique depths per 16-lane row.  Two rows within
            # hw0 (depth_base 0 and 16) give 32 depths; hw1 shifts
            # keys by +4 for the complementary MFMA C-output column
            # pattern, covering all 64 BN keys per MFMA tile.
            #
            # 4 ds_reads per A-fragment (key groups 0,16,32,48) ×
            # 4 depth tiles = 16 vector reads for depth=128.
            comptime kv_type = Self.kv_t.dtype
            comptime BN_ = Self.BN
            comptime BK_ = Self.BK
            comptime MMA_M_ = Self.mma_shape[0]
            comptime num_depth_tiles = Self.depth // MMA_M_

            var lid = lane_id()
            var v_base = self.smem_ptr + buffer * Self.smem_stage_size

            # Per-lane address components (computed once, reused
            # across all bk_tile × depth_tile iterations).
            var lane_in_row = umod(lid, 16)
            var pair_idx = ufloordiv(lane_in_row, 2)
            var is_odd = umod(lane_in_row, 2)
            var row_in_warp = ufloordiv(lid, 16)
            var is_hw1 = ufloordiv(lid, 32)

            # Key mapping within a 16-lane row: 8 unique keys
            # matching the MFMA C-output column pattern.
            var rel_key = Int(umod(pair_idx, 4) + ufloordiv(pair_idx, 4) * 8)

            # Depth sub-range per lane: rows 0,2 → +0; rows 1,3 →
            # +16.  Even lanes → +0; odd lanes → +8.
            var depth_base = Int(umod(row_in_warp, 2)) * 16 + Int(is_odd) * 8

            # hw1 key shift: complementary C-output column keys.
            var hw_key_shift = Int(is_hw1) * 4

            var reg_vec = self.mma_tile.vectorize[1, Self.input_frag_size]()

            comptime for bk_tile in range(Self.num_k_tiles):
                comptime row_offset = bk_tile * 64

                comptime for dt in range(num_depth_tiles):
                    comptime depth_offset = dt * MMA_M_
                    comptime blk = depth_offset // BK_
                    comptime d_in_blk = depth_offset % BK_

                    var block_base = v_base + blk * BN_ * BK_

                    @always_inline
                    @parameter
                    def _load_keys[key_base: Int]() -> SIMD[kv_type, 8]:
                        var key = row_offset + key_base + rel_key + hw_key_shift
                        return ds_read_tr8_b64(
                            block_base + key * BK_ + d_in_blk + depth_base
                        )

                    var r0 = _load_keys[0]()
                    var r1 = _load_keys[16]()
                    var r2 = _load_keys[32]()
                    var r3 = _load_keys[48]()
                    var joined = r0.join(r1).join(r2.join(r3))

                    reg_vec[bk_tile * num_depth_tiles + dt, 0] = rebind[
                        type_of(reg_vec[0, 0])
                    ](joined)
        else:
            comptime for bk_tile in range(Self.num_k_tiles):
                self.load_from_shared[bk_tile](buffer)

    @always_inline
    def load_from_shared[bk_tile: Int](self, buffer: Int):
        comptime if Self.transpose:
            comptime num_warps_n = Self.BN // Self.WN
            var warp_col = umod(get_warp_id(), num_warps_n)
            var smem_base = Self.SharedTileType(
                self.smem_ptr + buffer * Self.smem_stage_size
            )
            var smem_tile = smem_base.tile[Self.BN, Self.BK](0, bk_tile)

            var wtile_coord0 = warp_col
            var wtile_coord1 = 0
            var warp_tile = smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
                wtile_coord0, wtile_coord1
            )
            var load_b_tile = load_b[Self.mma_shape, swizzle=Self.swizzle](
                warp_tile
            )

            self.mma_tile.split[Self.num_k_tiles]()[bk_tile].vectorize[
                1, Self.input_frag_size
            ]().copy_from(load_b_tile.vectorize[1, Self.input_frag_size]())

        else:
            comptime MMA_M = Self.mma_shape[0]
            comptime MMA_K = Self.mma_shape[2]

            comptime for k in range(Self.BK // MMA_K):
                var smem_tile = (
                    Self.SharedTileType(
                        self.smem_ptr + buffer * Self.smem_stage_size
                    )
                    .tile[Self.BK, Self.depth](bk_tile, 0)
                    .tile[MMA_K, Self.depth](k, 0)
                )
                var frags = (
                    type_of(
                        self.mma_tile.split[Self.num_k_tiles]()[bk_tile].split[
                            Self.num_k_mmas2
                        ]()[k]
                    )
                    .stack_allocation()
                    .vectorize[1, Self.input_frag_size]()
                )

                comptime for i in range(Self.depth // MMA_M):
                    comptime tile_layout = type_of(
                        smem_tile.tile[MMA_K, MMA_M](0, i)
                    ).layout
                    # TODO: KERN-2173, the offset calculation is a workaround
                    # a bug in tile, remove this once the bug is fixed
                    comptime tiles_per_bk = Self.BK // MMA_M
                    comptime stride = self.base_layout.size()
                    comptime offset = (
                        MMA_M * (i % tiles_per_bk)
                        + (i // tiles_per_bk) * stride
                    )
                    var tile = LayoutTensor[
                        smem_tile.dtype,
                        tile_layout,
                        MutAnyOrigin,
                        address_space=smem_tile.address_space,
                    ](smem_tile.ptr + offset)
                    frags[i, 0] = rebind[frags.element_type](
                        load_b_tr[Self.mma_shape](tile)
                    )
                var mma_tile = self.get_mma_tile[k, bk_tile]()
                mma_tile.vectorize[1, Self.input_frag_size]().copy_from(frags)


__extension Attention:
    @always_inline
    def get_num_rows(self) -> UInt32:
        var end = min(
            self.kv_start_row + UInt32(Self.BN), UInt32(self.num_keys)
        )
        var num_rows = max(
            min(Int32(end - self.kv_start_row), Int32(UInt32(Self.BN))), 0
        )
        return UInt32(num_rows)

    @always_inline
    def apply_mask[
        stage: Int, scale: Bool = True
    ](mut self, not_last_iter: Bool = False):
        comptime if scale:
            self.scale_p_reg[stage]()
        var num_rows = self.get_num_rows()
        self.mask_apply[stage](self.kv_start_row, num_rows, not_last_iter)
        self.kv_start_row += UInt32(Self.BN)

    @always_inline
    def online_softmax_step_0[stage: Int, mask: Bool = True](mut self):
        comptime if mask:
            self.apply_mask[stage]()
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Self.num_warps_n, Self.WM
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.calculate_qk_max(score_tile, warp_scratch)
        self.softmax.exp[start=0, stride=2](score_tile)

    @always_inline
    def online_softmax_step_0_fma[stage: Int, mask: Bool = True](mut self):
        """Step 0 with deferred scaling: mask (no scale), max, exp_scaled.

        Avoids pre-scaling all scores by deferring the scale multiply into the
        exp computation. Uses exp_scaled which subtracts the unscaled max first
        (exact for the maximum element), then scales inside exp2.
        score_frag_rowmax remains unscaled after this call — scale_rowmax is
        deferred to step_1_fma (before calculate_correction needs it).
        """
        comptime if mask:
            self.apply_mask[stage, scale=False]()

        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Self.num_warps_n, Self.WM
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.calculate_qk_max(score_tile, warp_scratch)
        self.softmax.exp_scaled[start=0, stride=2](score_tile, self.scale)

    @always_inline
    def online_softmax_step_1[stage: Int](mut self):
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Self.num_warps_n, Self.WM
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.exp[start=1, stride=2](score_tile)
        self.softmax.calculate_qk_sum(score_tile, warp_scratch)
        self.softmax.calculate_correction()
        self.softmax.update_max()
        self.softmax.update_sum()

    @always_inline
    def online_softmax_step_1_fma[stage: Int](mut self):
        """Step 1 with deferred scaling for odd-indexed tiles.

        Processes remaining score tiles with exp_scaled, then scales the
        rowmax before calculate_correction (which compares against the
        previous iteration's scaled rowmax_tensor).
        """
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Self.num_warps_n, Self.WM
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.exp_scaled[start=1, stride=2](score_tile, self.scale)
        self.softmax.scale_rowmax(self.scale)
        self.softmax.calculate_qk_sum(score_tile, warp_scratch)
        self.softmax.calculate_correction()
        self.softmax.update_max()
        self.softmax.update_sum()

    @always_inline
    def online_softmax_step_0_prescaled[
        stage: Int, mask: Bool = True
    ](mut self):
        """Softmax step 0 for pre-scaled Q: no scale needed on scores.

        When Q is pre-multiplied by (scale * log2e), the QK matmul already
        produces scaled scores. We just need mask + max + exp2(score - max).
        """
        comptime if mask:
            self.apply_mask[stage, scale=False]()
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Self.num_warps_n, Self.WM
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.calculate_qk_max(score_tile, warp_scratch)
        self.softmax.exp[start=0, stride=2](score_tile)

    @always_inline
    def online_softmax_step_1_prescaled[stage: Int](mut self):
        """Softmax step 1 for pre-scaled Q: no scale needed on scores."""
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Self.num_warps_n, Self.WM
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.exp[start=1, stride=2](score_tile)
        self.softmax.calculate_qk_sum(score_tile, warp_scratch)
        self.softmax.calculate_correction()
        self.softmax.update_max()
        self.softmax.update_sum()

    @always_inline
    def online_softmax_update_output(mut self):
        self.softmax.update_output(self.out_reg_buffer.reg_tile)

    @always_inline
    def mha_prefill(mut self):
        """Double-buffered gfx950 MHA with V-load-later optimization.

        Uses both LDS slots for K and V: compute from one slot while
        prefetching the next tile into the other slot. V registers are
        loaded just before PV matmul (not during K load) to reduce peak
        VGPR usage by ~21, enabling double-buffer without spills.
        """
        comptime assert (
            Self.BK == 32 or Self.BK == 64 or Self.BK == 128
        ), "BK must be 32, 64, or 128"
        comptime assert (
            Self.depth == 64
            or Self.depth == 128
            or Self.depth == 256
            or Self.depth == 512
        ), "depth must be 64, 128, 256, or 512"
        comptime assert not Self.q_type.is_float8() or (
            Self.depth == 128 or Self.depth == 256
        ), "fp8 only supports depth=128 or 256"
        comptime assert (
            Self.depth % Self.BK == 0
        ), "depth must be a multiple of BK"
        comptime assert Self.BN % Self.BK == 0, "BN must be a multiple of BK"
        # Pre-scale Q by default for depth<=128 to eliminate per-element
        # scale multiply from the softmax hot loop. Disabled for depth>128
        # to work around LLVM Machine Instruction Scheduler crash (isReg
        # assertion in RewriteMFMAFormStage).
        # Disable Q pre-scaling for fp8: quantization back to fp8 loses
        # too much precision.
        comptime prescale_q = get_defined_bool[
            "PRESCALE_Q", True
        ]() and Self.depth <= 128 and not Self.q_type.is_float8()

        comptime k_swizzle = (
            Swizzle(3, 0, 4) if Self.mma_shape[0]
            == 32 else Optional[Swizzle](None)
        )

        var warp_id = UInt32(
            readfirstlane(bitcast[DType.int32](UInt32(get_warp_id())))
        )
        var k_buffer = KVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=k_swizzle,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=True,
        ](
            self.k,
            self.batch_idx,
            self.kv_head_idx(),
            self.smem_manager.get_k_ptr[type_of(self.k).dtype](),
            self.num_keys,
            warp_id,
        )

        var v_buffer = KVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=None,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
        ](
            self.v,
            self.batch_idx,
            self.kv_head_idx(),
            self.smem_manager.get_v_ptr[type_of(self.v).dtype](),
            self.num_keys,
            warp_id,
        )

        comptime accum_type = get_accum_type[type_of(self.k).dtype]()

        @always_inline
        @parameter
        def mma_qk():
            comptime MmaOp = TiledMmaOp[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]
            self.zero_p_buffer[0]()

            comptime for i in range(Self.depth // Self.BK):
                comptime for k_mma in range(Self.num_k_mmas2):
                    MmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[i, k_mma](),
                        k_buffer.mma_subtile[k_mma, i](),
                        self.p_reg_buffer.stage_tile[0](),
                    )

        @always_inline
        @parameter
        def mma_pv():
            comptime PVMmaOp = TiledMmaOp[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]

            comptime for i in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    PVMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.p_reg_buffer.mma_tile[i, k_mma, 0](),
                        v_buffer.mma_subtile[k_mma, i](),
                        self.out_reg_buffer.reg_tile,
                    )

        # Calculate iteration bounds using mask helpers.
        # start_column returns the first non-fully-masked column,
        # aligned down to BN.  last_masked_set_end returns the total
        # number of BN-wide tiles to process (a count, not an index).
        var score_row = UInt32(self.mask_block_row + UInt32(self.start_pos))
        var start_col = self.mask.start_column[Self.BM, Self.BN, 1](score_row)
        var num_tiles = Int(
            self.mask.last_masked_set_end[Self.BM, Self.BN, 1](
                score_row, UInt32(self.num_keys)
            )
        )

        # Advance KV iterators and mask tracking to start_col.
        k_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        v_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        self.kv_start_row = start_col
        self.mask_warp_col += start_col

        var num_pairs = num_tiles // 2

        # Pre-scale Q registers so QK matmul produces already-scaled scores.
        # This eliminates the per-element scale multiply from the hot loop
        # at the cost of bf16 quantization of (Q * scale * log2e).
        comptime if prescale_q:
            self.scale_q_buffer()

        # Pre-load first tile K+V into LDS slot 0
        _ = k_buffer.load_from_dram[0]()
        _ = v_buffer.load_from_dram[0]()

        # For CausalMask, start_column/last_masked_set_end already
        # guarantee no FULL_MASK tiles in the iteration range. For other
        # masks (e.g. ChunkedCausalMask), FULL_MASK tiles can appear in
        # the middle, so we need per-tile status checks.
        comptime has_interior_full_mask = not _type_is_eq[
            Self.mask_t, CausalMask
        ]()

        @always_inline
        @parameter
        def process_tile[slot: Int, has_next: Bool]():
            """Process a single KV tile from the given LDS slot.

            Waits for DRAM→LDS to complete, runs QK→softmax→PV. If has_next,
            prefetches the next tile into the other slot before softmax.
            For non-causal masks, checks tile status and skips FULL_MASK.
            """
            comptime next_slot = 1 - slot

            # Wait for K+V loads to this slot.
            # When has_next, the previous iteration's prefetch for the
            # *other* slot's V is still in-flight.
            comptime if has_next:
                block_sync_lds_direct_load[vmcnt=v_buffer.vm_instrs_per_load]()
            else:
                block_sync_lds_direct_load[vmcnt=0]()
            barrier[
                schedule_barrier_before=False, schedule_barrier_after=False
            ]()

            # For masks that can have FULL_MASK in the middle (e.g.
            # ChunkedCausalMask), check status and skip if fully masked.
            comptime if has_interior_full_mask:
                var tile_status = self.mask_status(self.kv_start_row)
                if tile_status == TileMaskStatus.FULL_MASK:
                    self.kv_start_row += UInt32(Self.BN)
                    self.mask_advance()
                    comptime if has_next:
                        _ = k_buffer.load_from_dram[next_slot]()
                        _ = v_buffer.load_from_dram[next_slot]()
                        barrier[
                            schedule_barrier_before=False,
                            schedule_barrier_after=False,
                        ]()
                    return

            k_buffer.load_from_shared(slot)
            mma_qk()

            # Prefetch next tile before softmax to hide page table
            # s_load latency during softmax VALU.
            comptime if has_next:
                _ = k_buffer.load_from_dram[next_slot]()
                _ = v_buffer.load_from_dram[next_slot]()

            comptime if prescale_q:
                self.online_softmax_step_0_prescaled[0]()
                self.online_softmax_step_1_prescaled[0]()
            elif Self.depth > 128 or Self.mask_t.apply_log2e_after_mask:
                self.online_softmax_step_0[0]()
                self.online_softmax_step_1[0]()
            else:
                self.online_softmax_step_0_fma[0]()
                self.online_softmax_step_1_fma[0]()

            # Wait for V loads from current tile before reading.
            # When has_next, the next slot's K+V prefetch (2 loads) is
            # still in-flight.
            comptime if has_next:
                block_sync_lds_direct_load[
                    vmcnt=k_buffer.vm_instrs_per_load
                    + v_buffer.vm_instrs_per_load
                ]()
            # Barrier ensures all waves' V DMA is visible before any
            # wave reads V from LDS (cross-wave coherence).
            barrier[
                schedule_barrier_before=False,
                schedule_barrier_after=False,
            ]()

            v_buffer.load_from_shared(slot)

            self.online_softmax_update_output()

            mma_pv()

        # Main loop: process tiles in pairs (double-buffered).
        # Even tile: compute from slot 0, prefetch to slot 1.
        # Odd tile: compute from slot 1, prefetch to slot 0.
        for _ in range(num_pairs):
            process_tile[0, True]()
            process_tile[1, True]()

        # Remainder: last odd tile from slot 0 (already prefetched).
        if num_tiles % 2 != 0:
            process_tile[0, False]()

        # Apply final softmax denominator and store
        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_output()

    @always_inline
    def mha_decoding(
        mut self,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        num_partitions: Int,
    ):
        """GFX950 MHA decode using KVBuffer with direct LDS loads.

        BN=128 WN=32: 4 warps (256 threads). Single-buffer K+V. Uses
        16x16x32 MMA (bf16), 16x16x128 MMA (FP8 depth%128==0), or
        32x32x64 MMA (FP8 depth%128!=0) and split-K partitioning.
        """
        comptime assert (
            Self.BK == 32 or Self.BK == 64 or Self.BK == 128
        ), "BK must be 32, 64, or 128"
        comptime assert (
            Self.depth % Self.BK == 0
        ), "depth must be a multiple of BK"
        comptime assert Self.BN % Self.BK == 0, "BN must be a multiple of BK"

        comptime k_swizzle = _get_k_swizzle[Self.mma_shape[0], Self.BK]()

        var warp_id = UInt32(
            readfirstlane(bitcast[DType.int32](UInt32(get_warp_id())))
        )

        # Split-K: compute this partition's key range.
        start, end = get_start_and_end_for_partitions[Self.BN](
            self.num_keys, num_partitions, block_idx.x
        )

        var k_buffer = KVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=k_swizzle,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=True,
        ](
            self.k,
            self.batch_idx,
            self.kv_head_idx(),
            self.smem_manager.get_k_ptr[type_of(self.k).dtype](),
            end,
            warp_id,
        )

        # V uses two buffers to avoid loading full depth into registers:
        # - v_dma_buffer: V depth (= output_depth), for cooperative DRAM→LDS
        # - v_buffer: depth_per_warp, for LDS→REG and mma_subtile
        # The per-warp buffer's SMEM pointer is offset into the V SMEM
        # region. When depth_per_warp < BK, multiple warps share one
        # BK-wide block; the offset accounts for the within-block position.
        # Note: for MHA, output_depth == depth; for MLA, output_depth < depth
        # (V's effective depth, which equals the output depth).
        comptime depth_per_warp = Self.output_depth // Self.num_warps_n
        # smem_depth_per_warp is the SMEM block width seen by each warp.
        # Must be >= BK so the blocked-product layout has at least 1 repeat.
        comptime smem_depth_per_warp = max(depth_per_warp, Self.BK)
        var v_warp_col = umod(Int(warp_id), Self.num_warps_n)
        var v_smem_ptr = self.smem_manager.get_v_ptr[type_of(self.v).dtype]()
        var v_dma_buffer = KVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=None,
            BN=Self.BN,
            WN=Self.BN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.output_depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
            cache_depth=Self.cache_depth,
        ](
            self.v,
            self.batch_idx,
            self.kv_head_idx(),
            v_smem_ptr,
            end,
            warp_id,
        )
        # Compute per-warp SMEM offset into V's blocked-product layout.
        # Each warp's depth slice starts at column warp_col * depth_per_warp.
        # In blocked_product(row_major(BN, BK), ...), the offset for
        # (row=0, col) is: (col // BK) * BN * BK + col % BK.
        var v_col_start = v_warp_col * depth_per_warp
        var v_warp_smem_offset = ufloordiv(
            v_col_start, Self.BK
        ) * Self.BN * Self.BK + umod(v_col_start, Self.BK)
        var v_buffer = KVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=None,
            BN=Self.BN,
            WN=Self.BN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=depth_per_warp,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
            cache_depth=Self.cache_depth,
            smem_depth=smem_depth_per_warp,
        ](
            self.v,
            self.batch_idx,
            self.kv_head_idx(),
            v_smem_ptr + v_warp_smem_offset,
            end,
            warp_id,
        )

        # Advance iterators and mask tracking to partition start.
        k_buffer.kv_cache_iter.tile_start_row = start
        v_dma_buffer.kv_cache_iter.tile_start_row = start
        self.kv_start_row = UInt32(start)

        # Pre-scale Q by (scale * log2e) so QK scores are already in the
        # correct scaled-log2 space for softmax.  Default on for bf16 to
        # avoid the Qwen decode NaN in the `_fma` path.  Disabled for fp8
        # because scaling Q and quantizing back to fp8 loses too much
        # precision — fp8 stays on `_fma`.  Sink still requires pre-scaling
        # because `rowmax_tensor` is initialized in scaled-log2 space.
        comptime prescale_q = (
            get_defined_bool["PRESCALE_Q", True]()
            and not Self.q_type.is_float8()
        ) or Self.sink
        comptime if prescale_q:
            self.scale_q_buffer()

        comptime accum_type = get_accum_type[type_of(self.k).dtype]()

        @always_inline
        @parameter
        def mma_qk():
            comptime MmaOp = TiledMmaOp[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]
            self.zero_p_buffer[0]()

            comptime for i in range(Self.depth // Self.BK):
                comptime for k_mma in range(Self.num_k_mmas2):
                    MmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[i, k_mma](),
                        k_buffer.mma_subtile[k_mma, i](),
                        self.p_reg_buffer.stage_tile[0](),
                    )

        @always_inline
        @parameter
        def mma_pv():
            comptime PVMmaOp = TiledMmaOp[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]

            # Each warp's v_buffer holds only depth_per_warp tiles
            # (loaded from the warp's LDS depth offset), so
            # mma_subtile directly gives the warp's depth range.
            comptime for i in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    PVMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.p_reg_buffer.mma_tile[i, k_mma, 0](),
                        v_buffer.mma_subtile[k_mma, i](),
                        self.out_reg_buffer.reg_tile,
                    )

        comptime shared_kv = Self.attention_config_t.shared_kv

        # Masks with interior FULL_MASK tiles (e.g. ChunkedCausalMask) need
        # a per-tile status check: when a tile is fully masked, skip QK,
        # softmax and PV entirely but keep the DMA prefetch pipeline moving.
        # CausalMask during decode never produces FULL_MASK (all keys up to
        # num_keys-1 are valid), so this flag is False for it.
        comptime has_interior_full_mask = (
            Self.mask_t.check_mask_during_decoding
        )

        @always_inline
        @parameter
        def prefetch_next[slot: Int]():
            """Prefetch the next tile's K (and V if not shared_kv) after
            the current tile's LDS reads have drained."""
            s_waitcnt[lgkmcnt=0]()
            barrier[
                schedule_barrier_before=False,
                schedule_barrier_after=False,
            ]()
            # Double-buffer K: prefetch to OTHER slot.
            # Single-buffer K: overwrite slot 0.
            comptime if Self.attention_config_t.double_buffer_k_only:
                _ = k_buffer.load_from_dram[1 - slot]()
            else:
                _ = k_buffer.load_from_dram[0]()
            comptime if not shared_kv:
                _ = v_dma_buffer.load_from_dram[0]()

        @always_inline
        @parameter
        def process_tile[slot: Int, has_next: Bool]():
            """Process one KV tile.

            K[slot] is already in LDS when this is called.
            If not shared_kv, V is also in LDS (parallel DMA).
            If shared_kv, V is loaded mid-tile after K is consumed.
            """
            # Wait for K DMA (and V DMA if not shared_kv).
            s_waitcnt[vmcnt=0, lgkmcnt=0]()
            barrier[
                schedule_barrier_before=False,
                schedule_barrier_after=False,
            ]()

            # For masks that can have FULL_MASK tiles mid-sequence (e.g.
            # ChunkedCausalMask), skip fully-masked tiles. Without this the
            # mask_apply path treats FULL_MASK as `masked=False` and
            # leaves raw QK scores in P, which softmax then consumes.
            comptime if has_interior_full_mask:
                var tile_status = self.mask_status(self.kv_start_row)
                if tile_status == TileMaskStatus.FULL_MASK:
                    self.kv_start_row += UInt32(Self.BN)
                    self.mask_advance()
                    comptime if has_next:
                        prefetch_next[slot]()
                    return

            k_buffer.load_from_shared(slot)
            mma_qk()

            # Online softmax on P (cross-warp reduction via SMEM).
            # Default is prescaled (matches scale_q_buffer above) — the
            # prescaled path is numerically safer and is required for
            # Qwen-class models, where the deferred-scale (`_fma`) path
            # accumulates enough numerical error to produce NaN logits
            # after a few decode steps on real weights.  The `_fma`
            # variants fold the score scale into exp2 and save VALU in
            # the hot loop; only opt in via `-D PRESCALE_Q=False` if you
            # know your workload tolerates the extra error.
            comptime if prescale_q:
                self.online_softmax_step_0_prescaled[0]()
                self.online_softmax_step_1_prescaled[0]()
            else:
                self.online_softmax_step_0_fma[0]()
                self.online_softmax_step_1_fma[0]()

            # Write P to shared memory so all warps can read it for PV.
            self.p_reg_buffer.copy_to_shared()
            s_waitcnt[lgkmcnt=0]()
            barrier[
                schedule_barrier_before=False,
                schedule_barrier_after=False,
            ]()

            comptime if shared_kv:
                # K consumed + P synced. Load V into shared SMEM
                # (overwrites K). All warps finished K LDS reads above.
                _ = v_dma_buffer.load_from_dram[0]()
                s_waitcnt[vmcnt=0, lgkmcnt=0]()
                barrier[
                    schedule_barrier_before=False,
                    schedule_barrier_after=False,
                ]()

            v_buffer.load_from_shared(0)

            # Prefetch next tile after V→REG.
            # Barrier ensures all warps finished V LDS reads before DMA.
            comptime if has_next:
                prefetch_next[slot]()

            self.online_softmax_update_output()
            mma_pv()

            # Iter-end drain + barrier. The entry barrier at the next
            # iteration is insufficient on its own: without this
            # explicit full drain here, mma_pv's pipelined P SMEM
            # ds_reads (and prefetch_next's in-flight DMA) can overlap
            # with iter N+1's copy_to_shared writes and K LDS reads.
            # Waitcnt-only and barrier-only were both empirically
            # insufficient — both are required.
            s_waitcnt[vmcnt=0, lgkmcnt=0]()
            barrier[
                schedule_barrier_before=False,
                schedule_barrier_after=False,
            ]()

        var num_tiles = ceildiv(end - start, Self.BN)

        # Prologue: load first tile's K (slot 0).
        # V loaded in parallel when separate SMEM, or mid-tile when shared.
        _ = k_buffer.load_from_dram[0]()
        comptime if not shared_kv:
            _ = v_dma_buffer.load_from_dram[0]()

        comptime if Self.attention_config_t.double_buffer_k_only:
            # Double-buffer K: process tiles in pairs (slot 0, slot 1).
            var num_pairs = num_tiles // 2
            for _ in range(num_pairs):
                process_tile[0, True]()
                process_tile[1, True]()

            # Remainder: last odd tile from slot 0.
            if num_tiles % 2 != 0:
                process_tile[0, False]()
        else:
            # Single-buffer K: simple linear loop.
            for _ in range(num_tiles - 1):
                process_tile[0, True]()

            if num_tiles > 0:
                process_tile[0, False]()

        # Apply final softmax denominator.
        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
        self.store_output()
