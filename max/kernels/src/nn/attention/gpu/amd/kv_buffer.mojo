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
"""KV cache buffer for structured MHA kernels (TileTensor hot path).

Provides KVCacheIterator (TileTensor-based DRAM tile iteration) and
KVBuffer (DMA + LDS + register tile management).

TileTensor is used throughout:
  - DRAM tiles: TileTensor with RuntimeInt valid_rows (KVCacheIterator)
  - SMEM sub-tiles: flat TileTensor views via smem_subtile/smem_mma_subtile
  - DMA: tt_copy_dram_to_sram_lds (both src and dst are TileTensor)
  - LDS loads: tt_load_b / tt_load_b_tr (TileTensor SMEM -> SIMD)
  - MMA register tiles: TileTensor in LOCAL with stack_allocation
"""

from std.math import ceildiv
from std.math.uutils import umod
from std.sys import simd_width_of, llvm_intrinsic
from std.sys.intrinsics import readfirstlane
from std.gpu import WARP_SIZE
from std.gpu import warp_id as get_warp_id
from std.memory.pointer import AddressSpace as BaseAddressSpace
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    MixedLayout,
    RuntimeInt,
    TileTensor,
)
from layout.tile_layout import row_major as tt_row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from layout.swizzle import Swizzle
from nn.attention.mha_operand import MHAOperand
from structured_kernels.amd_tile_io import (
    LdsTileLoader,
    smem_subtile,
    smem_mma_subtile,
    tt_copy_dram_to_sram_lds,
    tt_load_b,
    tt_load_b_tr,
)

from std.utils import IndexList


struct KVCacheIterator[
    cache_t: MHAOperand, tile_size: Int, kv_num_heads: Int, depth: Int
]:
    """TileTensor-based DRAM tile iterator.

    Returns a TileTensor with RuntimeInt for the row dimension (valid-row
    count) and ComptimeInt for depth and strides. No RuntimeLayout storage.
    """

    comptime GmemTileLayout = MixedLayout[
        TypeListOf[
            type=CoordLike,
            RuntimeInt[DType.int64],
            ComptimeInt[Self.depth],
        ](),
        TypeListOf[
            type=CoordLike,
            ComptimeInt[Self.kv_num_heads * Self.depth],
            ComptimeInt[1],
        ](),
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
        var valid_rows = min(
            Self.tile_size,
            self.end - self.tile_start_row,
        )
        var ptr = self.cache.block_paged_ptr[Self.tile_size](
            UInt32(self.batch_idx),
            UInt32(self.tile_start_row),
            UInt32(self.kv_head_idx),
            0,
        )
        self.tile_start_row += Self.tile_size
        var tile_layout = Self.GmemTileLayout(
            Coord(
                RuntimeInt[DType.int64](Int64(valid_rows)), Idx[Self.depth]()
            ),
            Coord(
                Idx[Self.kv_num_heads * Self.depth](),
                Idx[1](),
            ),
        )
        return Self.GmemTileType(ptr=ptr, layout=tile_layout)

    @always_inline
    def increment(mut self):
        self.tile_start_row += Self.tile_size


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
    full_kv: Bool = True,
]:
    """KV cache buffer managing DMA, LDS staging, and register tiles.

    Handles the full data path: DRAM -> LDS (shared memory) -> registers.

    SMEM is addressed via smem_subtile / smem_mma_subtile which compute
    block-aligned offsets from the blocked_product layout structure
    (num_repeats contiguous BN×BK blocks) and return flat TileTensor views.

    When full_kv=True (depth<=256), each SMEM stage holds BN x depth
    elements — the full tile. When full_kv=False (depth=512), each stage
    holds only BN x BK elements, and the caller iterates over BK blocks.

    MMA register tiles (mma_tile) are TileTensor in LOCAL address space.
    TiledMmaOp (mma.mojo) handles SMEM→register loads and MMA dispatch.
    """

    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_mmas = ceildiv(
        Self.WN if Self.transpose else Self.depth, Self.MMA_N
    )
    comptime num_k_mmas2 = ceildiv(Self.BK, Self.MMA_K * Self.k_group_size)
    comptime simd_width = simd_width_of[Self.kv_t.dtype]()
    comptime num_k_tiles = ceildiv(
        Self.depth if Self.transpose else Self.WN, Self.BK
    )

    comptime warp_tile_rows = 32
    comptime num_repeats = Self.depth // Self.BK
    comptime smem_cols = Self.depth if Self.full_kv else Self.BK
    comptime smem_stage_size = Self.BN * Self.smem_cols

    comptime _num_warps = Self.num_threads // WARP_SIZE
    comptime _dma_col_groups = (Self.depth // Self.BK) if Self.full_kv else 1
    comptime _total_tiles = (
        Self.BN // Self.warp_tile_rows
    ) * Self._dma_col_groups
    comptime _tiles_per_warp = ceildiv(Self._total_tiles, Self._num_warps)
    comptime vm_instrs_per_load = UInt32(Self._tiles_per_warp * 2)

    comptime _mma_total_rows = Self.num_mmas * Self.num_k_mmas2 * Self.num_k_tiles
    comptime mma_layout = tt_row_major[Self._mma_total_rows, Self.simd_width]()
    comptime MMATileType = TileTensor[
        Self.kv_t.dtype,
        type_of(Self.mma_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var mma_tile: Self.MMATileType

    comptime wtile_dim0 = Self.WN
    comptime wtile_dim1 = Self.BK

    var smem_ptr: UnsafePointer[
        Scalar[Self.kv_t.dtype],
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    var kv_cache_iter: KVCacheIterator[
        Self.kv_t, Self.BN, Self.kv_num_heads, Self.depth
    ]

    var lds_base_ptrs: InlineArray[UInt32, 2]

    var warp_id: UInt32

    @always_inline
    def __init__(
        out self,
        k_cache: Self.kv_t,
        batch_idx: UInt,
        head_idx: UInt,
        shared_ptr: UnsafePointer[
            Scalar[Self.kv_t.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
        end: UInt,
        warp_id: UInt32,
    ):
        self.mma_tile = tt_stack_allocation[
            Self.kv_t.dtype, AddressSpace.LOCAL
        ](Self.mma_layout)
        self.smem_ptr = shared_ptr

        self.kv_cache_iter = type_of(self.kv_cache_iter)(
            k_cache, Int(batch_idx), Int(head_idx), Int(end)
        )

        self.warp_id = warp_id

        var warp_row, warp_col = divmod(self.warp_id, UInt32(4))

        self.lds_base_ptrs = type_of(self.lds_base_ptrs)(uninitialized=True)

        comptime for i in range(2):
            var warp_smem = smem_subtile[
                Self.warp_tile_rows, Self.BK, Self.BN, Self.BK, Self.kv_t.dtype
            ](
                self.smem_ptr + i * Self.smem_stage_size,
                Int(warp_row),
                Int(warp_col),
            )
            self.lds_base_ptrs[i] = UInt32(
                readfirstlane(Int32(Int(warp_smem.ptr)))
            )

    @always_inline
    def load_from_dram[buffer_idx: Int](mut self):
        var gmem_tile = self.kv_cache_iter.next_tile()
        var loader = LdsTileLoader[Self.kv_t.dtype, Self.swizzle](gmem_tile)

        var smem_base = self.smem_ptr + buffer_idx * Self.smem_stage_size

        comptime if not Self.full_kv:
            comptime num_warps = Self.num_threads // WARP_SIZE
            comptime num_row_groups = Self.BN // Self.warp_tile_rows
            comptime tiles_per_warp = ceildiv(num_row_groups, num_warps)

            comptime for t in range(tiles_per_warp):
                comptime tile_idx = Int(t) * num_warps
                var warp_tile_idx = UInt32(tile_idx) + self.warp_id
                var warp_row = Int(warp_tile_idx)
                var smem_warp = smem_subtile[
                    Self.warp_tile_rows,
                    Self.BK,
                    Self.BN,
                    Self.BK,
                    Self.kv_t.dtype,
                ](smem_base, warp_row, 0)
                var gmem_warp_tile = gmem_tile.tile[
                    Self.warp_tile_rows, Self.BK
                ](warp_row, 0)
                var lds_base = UInt32(readfirstlane(Int32(Int(smem_warp.ptr))))
                loader.load(smem_warp, gmem_warp_tile, lds_base)
        elif Self.depth == 64:
            var warp_r, warp_c = divmod(Int(self.warp_id), 2)
            var smem_warp = smem_subtile[
                Self.warp_tile_rows,
                Self.BK,
                Self.BN,
                Self.BK,
                Self.kv_t.dtype,
            ](
                smem_base,
                warp_r,
                warp_c,
            )
            var gmem_warp_tile = gmem_tile.tile[Self.warp_tile_rows, Self.BK](
                warp_r, warp_c
            )
            var lds_base = UInt32(readfirstlane(Int32(Int(smem_warp.ptr))))
            loader.load(smem_warp, gmem_warp_tile, lds_base)
        else:
            comptime num_warps = Self.num_threads // WARP_SIZE
            comptime num_row_groups = Self.BN // Self.warp_tile_rows
            comptime num_col_groups = Self.depth // Self.BK
            comptime total_tiles = num_row_groups * num_col_groups
            comptime tiles_per_warp = ceildiv(total_tiles, num_warps)

            comptime for t in range(tiles_per_warp):
                comptime tile_idx = Int(t) * num_warps
                var warp_tile = UInt32(tile_idx) + self.warp_id
                var warp_row, warp_col = divmod(
                    warp_tile, UInt32(num_col_groups)
                )
                var smem_warp = smem_subtile[
                    Self.warp_tile_rows,
                    Self.BK,
                    Self.BN,
                    Self.BK,
                    Self.kv_t.dtype,
                ](smem_base, Int(warp_row), Int(warp_col))
                var gmem_warp_tile = gmem_tile.tile[
                    Self.warp_tile_rows, Self.BK
                ](Int(warp_row), Int(warp_col))
                var lds_base = UInt32(readfirstlane(Int32(Int(smem_warp.ptr))))
                loader.load(smem_warp, gmem_warp_tile, lds_base)

    # split[N]()[idx] → tile[rows_per_split, cols](idx, 0)
    comptime _rows_per_k_tile = Self.num_mmas * Self.num_k_mmas2
    comptime _rows_per_k_mma = Self.num_mmas

    @always_inline
    def get_mma_tile[
        k_mma_tile_idx: Int,
        bk_tile_idx: Int,
    ](self) -> TileTensor[
        Self.kv_t.dtype,
        type_of(tt_row_major[Self._rows_per_k_mma, Self.simd_width]()),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        return self.mma_tile.tile[Self._rows_per_k_tile, Self.simd_width](
            bk_tile_idx, 0
        ).tile[Self._rows_per_k_mma, Self.simd_width](k_mma_tile_idx, 0)

    @always_inline
    def copy_to_shared(
        self,
    ):
        ...

    @always_inline
    def load_from_shared(self, buffer: UInt):
        comptime for bk_tile in range(Self.num_k_tiles):
            self.load_from_shared[bk_tile](buffer)

    @always_inline
    def load_from_shared[bk_tile: Int](self, buffer: UInt):
        var smem_base = self.smem_ptr + Int(buffer) * Self.smem_stage_size

        comptime if Self.transpose:
            comptime num_warps_n = Self.BN // Self.WN
            var warp_col = umod(get_warp_id(), num_warps_n)

            var warp_smem = smem_subtile[
                Self.wtile_dim0,
                Self.wtile_dim1,
                Self.BN,
                Self.BK,
                Self.kv_t.dtype,
            ](smem_base + bk_tile * Self.BN * Self.BK, warp_col, 0)

            comptime total_frags = Self.num_mmas * Self.num_k_mmas2
            var frags = tt_load_b[
                Self.mma_shape, Self.swizzle, total_frags, Self.simd_width
            ](warp_smem)

            var mma_dst = self.mma_tile.tile[
                Self._rows_per_k_tile, Self.simd_width
            ](bk_tile, 0).vectorize[1, Self.simd_width]()

            comptime for i in range(total_frags):
                mma_dst[Int(i), 0] = rebind[type_of(mma_dst[Int(i), 0])](
                    frags[Int(i)]
                )

        else:
            comptime MMA_M = Self.mma_shape[0]
            comptime MMA_K = Self.mma_shape[2]

            comptime for k in range(Self.BK // MMA_K):
                comptime for i in range(Self.depth // MMA_M):
                    var mma_smem = smem_mma_subtile[
                        MMA_K, MMA_M, Self.BN, Self.BK, Self.kv_t.dtype
                    ](smem_base, bk_tile, Int(k), Int(i))
                    var frag = tt_load_b_tr[Self.mma_shape](mma_smem)

                    var mma_dst = self.get_mma_tile[
                        Int(k), bk_tile
                    ]().vectorize[1, Self.simd_width]()
                    mma_dst[Int(i), 0] = rebind[type_of(mma_dst[Int(i), 0])](
                        frag
                    )
