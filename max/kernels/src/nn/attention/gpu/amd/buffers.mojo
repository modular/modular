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
from std.math.uutils import umod, ufloordiv, udivmod
from std.sys import simd_width_of
from std.sys.intrinsics import readfirstlane

from std.gpu import lane_id, warp_id as get_warp_id, WARP_SIZE
from std.gpu.sync import s_waitcnt
from layout import Layout, LayoutTensor, TensorLayout, TileTensor
from layout._utils import idx2crd
from layout.coord import Coord, ComptimeInt
from layout.tile_layout import Layout as TileLayout, col_major as tt_col_major
from layout.swizzle import Swizzle
from layout.tensor_core import TiledTensorCore
from layout.tile_layout import row_major as tt_row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from std.utils import IndexList

from structured_kernels.amd_tile_io import (
    LdsTileLoader,
    RegTileLoader,
    tt_copy_local_to_shared,
)

from .mma import TiledMmaOp
from .utils import (
    LocalLayoutTensor,
    get_warp_coords,
    pad,
)
import std.itertools


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


# Legacy traits — only used by RDNA mma_rdna.mojo.
# CDNA attention paths use TiledMmaOp with TileTensor directly.
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


struct KVBufferImpl[
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
](KVBuffer):
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
    comptime _load_layout = tt_row_major[Self._load_rows, Self.simd_width]()
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
    comptime mma_tile_layout = Layout.row_major(Self._mma_rows, Self._mma_cols)
    comptime _mma_layout = tt_row_major[Self._mma_rows, Self._mma_cols]()
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
        group_size=Self.tensor_core_mma.group_size,
        transpose_b=Self.tensor_core_mma.transpose_b,
    ]

    # SMEM TileTensor (stored once, used by copy_to_shared and load_from_shared).
    comptime _smem_layout = tt_row_major[Self._btile_dim0, Self._btile_dim1]()
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
    comptime RegLoaderType = RegTileLoader[
        Self.dtype, Self._thread_rows, Self._thread_cols, Self.num_threads
    ]
    var gmem_tile: Self.GmemTileType
    var reg_loader: Self.RegLoaderType
    var tile_idx: Int
    var load_tile_id: Int

    @always_inline
    def __init__(
        out self,
        gmem_tile: Self.GmemTileType,
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
    ):
        self.load_tile = tt_stack_allocation[Self.dtype, AddressSpace.LOCAL](
            Self._load_layout
        )
        self.mma_tile = tt_stack_allocation[Self.dtype, AddressSpace.LOCAL](
            Self._mma_layout
        )
        self.smem_tile = Self.SmemTile(shared_ptr, Self._smem_layout)
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
    ) -> LocalLayoutTensor[Self.dtype, Self.mma_tile_layout]:
        return rebind[LocalLayoutTensor[Self.dtype, Self.mma_tile_layout]](
            self.mma_tile.to_layout_tensor()
        )

    @always_inline
    def copy_to_shared[
        tile_id: Int = 0
    ](self,):
        tt_copy_local_to_shared[
            Self._thread_rows,
            Self._thread_cols,
            Self.swizzle,
            Self.num_threads,
        ](
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

        comptime if not Self.token_gen:
            var warp_tile = self.smem_tile.tile[
                Self.wtile_dim0, Self.wtile_dim1
            ](wtile_coord0, wtile_coord1)
            Self._TiledMma.load_b[swizzle=Self.swizzle](
                warp_tile, self.mma_tile, UInt(k_mma)
            )
        else:
            # Token-gen: use LayoutTensor path (TileTensor distribute
            # produces different offsets for single-row token-gen tiles).
            comptime _smem_row_major = Layout.row_major(
                Self._btile_dim0, Self._btile_dim1
            )
            var smem_warp = LayoutTensor[
                Self.dtype,
                _smem_row_major,
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ](self.smem_tile.ptr)
            var warp_tile = smem_warp.tile[Self.wtile_dim0, Self.wtile_dim1](
                wtile_coord0, wtile_coord1
            )
            comptime MMATileType = LocalLayoutTensor[
                Self.dtype,
                Layout.row_major(Self._mma_rows, Self._mma_cols),
            ]
            var mma_frag = rebind[MMATileType](self.mma_tile.to_layout_tensor())
            Self.tensor_core_mma.mma_op.load_b[swizzle=Self.swizzle](
                warp_tile,
                mma_frag.vectorize[1, Self.simd_width](),
                UInt(k_mma),
            )


comptime KBuffer[
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
] = KVBufferImpl[
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
] = KVBufferImpl[
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


# ===----------------------------------------------------------------------=== #
# KVBufferLDS: Direct DRAM→LDS DMA (Option B: serialized, single SMEM)
# ===----------------------------------------------------------------------=== #


struct KVBufferLDS[
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
    token_gen: Bool = False,
](KVBuffer):
    """KV buffer with direct DRAM→LDS DMA (no register staging).

    Serialized single-buffer variant for comparison with KVBufferImpl's
    register double-buffered approach. Each iteration:
      1. load_from_dram(): issues DRAM→LDS DMA via LdsTileLoader
      2. copy_to_shared(): s_waitcnt vmcnt(0) to wait for DMA
      3. barrier(): make SMEM visible
      4. load_from_shared(): SMEM→registers via TiledMmaOp.load_b
      5. MMA
      6. barrier()

    No register staging buffer, no prefetch overlap between iterations.
    Requires warp_id for DMA work distribution.
    """

    comptime _dtype = Self.dtype
    comptime _num_stages = 1
    comptime MMA_N = Self.tensor_core_mma.shape[1]
    comptime MMA_K = Self.tensor_core_mma.shape[2]
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_mmas = ceildiv(Self.config.wsize, Self.MMA_N)

    comptime num_k_tiles = ceildiv(
        Self.BK, Self.MMA_K * Self.tensor_core_mma.group_size
    )
    comptime simd_width = simd_width_of[Self.dtype]()

    comptime _btile_dim0 = Self.config.btile_dim0
    comptime _btile_dim1 = Self.config.btile_dim1

    # DMA distribution across warps.
    comptime warp_tile_rows = 32
    comptime _num_warps = Self.num_threads // WARP_SIZE
    comptime _num_row_groups = Self._btile_dim0 // Self.warp_tile_rows
    comptime _num_col_groups = max(Self._btile_dim1 // 32, 1)
    comptime _total_tiles = Self._num_row_groups * Self._num_col_groups
    comptime _tiles_per_warp = ceildiv(Self._total_tiles, Self._num_warps)
    # Each 32×32 warp-tile produces 2 VM instructions inside
    # tt_copy_dram_to_sram_lds (loop: 32/16 = 2 sub-tiles per tile).
    comptime vm_instrs_per_load = UInt32(Self._tiles_per_warp * 2)

    # TileTensor register storage for MMA operand.
    comptime _mma_rows = Self.num_mmas
    comptime _mma_cols = Self.simd_width
    comptime mma_tile_layout = Layout.row_major(Self._mma_rows, Self._mma_cols)
    comptime _mma_layout = tt_row_major[Self._mma_rows, Self._mma_cols]()
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
        group_size=Self.tensor_core_mma.group_size,
        transpose_b=Self.tensor_core_mma.transpose_b,
    ]

    # SMEM TileTensor (stored once, used by load_from_dram and load_from_shared).
    comptime _smem_layout = tt_row_major[Self._btile_dim0, Self._btile_dim1]()
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
    comptime LdsLoaderType = LdsTileLoader[Self.dtype, Self.swizzle]
    var gmem_tile: Self.GmemTileType
    var lds_loader: Self.LdsLoaderType
    var tile_idx: Int
    var warp_id: UInt32

    @always_inline
    def __init__(
        out self,
        gmem_tile: Self.GmemTileType,
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
        warp_id: UInt32,
    ):
        self.mma_tile = tt_stack_allocation[Self.dtype, AddressSpace.LOCAL](
            Self._mma_layout
        )
        self.smem_tile = Self.SmemTile(shared_ptr, Self._smem_layout)
        self.gmem_tile = gmem_tile
        self.lds_loader = Self.LdsLoaderType(gmem_tile)
        self.tile_idx = 0
        self.warp_id = warp_id

    @always_inline
    def load_from_dram(mut self):
        """Issue direct DRAM→LDS DMA distributed across warps."""
        # Build per-iteration DRAM sub-tile on-the-fly.
        var row_idx = 0 if Self.config.iterator_axis == 1 else self.tile_idx
        var col_idx = self.tile_idx if Self.config.iterator_axis == 1 else 0
        var dram_tile = self.gmem_tile.tile[Self._btile_dim0, Self._btile_dim1](
            row_idx, col_idx
        )

        # Distribute work: each warp handles tiles_per_warp sub-tiles.
        comptime for t in range(Self._tiles_per_warp):
            comptime tile_offset = Int(t) * Self._num_warps
            var warp_tile_idx = UInt32(tile_offset) + self.warp_id

            comptime if Self._num_col_groups > 1:
                var warp_row, warp_col = divmod(
                    warp_tile_idx, UInt32(Self._num_col_groups)
                )
                var smem_warp = self.smem_tile.tile[Self.warp_tile_rows, 32](
                    Int(warp_row), Int(warp_col)
                )
                var gmem_warp = dram_tile.tile[Self.warp_tile_rows, 32](
                    Int(warp_row), Int(warp_col)
                )
                var lds_base = UInt32(readfirstlane(Int32(Int(smem_warp.ptr))))
                self.lds_loader.load(smem_warp, gmem_warp, lds_base)
            else:
                var smem_warp = self.smem_tile.tile[
                    Self.warp_tile_rows, Self._btile_dim1
                ](Int(warp_tile_idx), 0)
                var gmem_warp = dram_tile.tile[
                    Self.warp_tile_rows, Self._btile_dim1
                ](Int(warp_tile_idx), 0)
                var lds_base = UInt32(readfirstlane(Int32(Int(smem_warp.ptr))))
                self.lds_loader.load(smem_warp, gmem_warp, lds_base)

        self.tile_idx += 1

    @always_inline
    def get_mma_tile(
        self,
    ) -> LocalLayoutTensor[Self.dtype, Self.mma_tile_layout]:
        return rebind[LocalLayoutTensor[Self.dtype, Self.mma_tile_layout]](
            self.mma_tile.to_layout_tensor()
        )

    @always_inline
    def copy_to_shared[
        tile_id: Int = 0
    ](self,):
        """Wait for outstanding DRAM→LDS DMA to complete."""
        s_waitcnt[vmcnt=0]()

    @always_inline
    def load_from_shared[
        k_mma: Int,
    ](self):
        """Load MMA operands from SMEM to registers."""
        var wtile_coord0 = Self.config.get_wtile_coord()[0]
        var wtile_coord1 = Self.config.get_wtile_coord()[1]
        var warp_tile = self.smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
            wtile_coord0, wtile_coord1
        )
        Self._TiledMma.load_b[swizzle=Self.swizzle](
            warp_tile, self.mma_tile, UInt(k_mma)
        )


comptime KBufferLDS[
    kv_tile_layout: TensorLayout,
    tensor_core_mma: TiledTensorCore,
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    token_gen: Bool = False,
] = KVBufferLDS[
    kv_tile_layout=kv_tile_layout,
    config=KBufferConfig[BN, BK, WN],
    tensor_core_mma=tensor_core_mma,
    swizzle=swizzle,
    BN=BN,
    WN=WN,
    BK=BK,
    depth=depth,
    num_threads=num_threads,
    token_gen=token_gen,
]


struct VBufferTransposeLoads[
    dtype: DType,
    kv_tile_layout: TensorLayout,
    //,
    tensor_core_mma: TiledTensorCore,
    BN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
](KVBuffer):
    comptime _dtype = Self.dtype
    comptime _num_stages = Self.num_stages
    comptime simd_width = simd_width_of[Self.dtype]()
    comptime num_repeats = Self.BK // Self.simd_width

    # V Buffer shared memory layout:
    # num_repeats contiguous row_major(pad_depth, simd_width) blocks.
    # Block b starts at offset b * pad_depth * simd_width.
    # Padding = depth//simd_width (0 for depth=64) avoids bank conflicts.

    comptime MMA_M = Self.tensor_core_mma.shape[0]
    comptime MMA_K = Self.tensor_core_mma.shape[2]
    comptime num_k_tiles = ceildiv(
        Self.BK, Self.MMA_K * Self.tensor_core_mma.group_size
    )
    comptime num_depth_tiles = Self.depth // Self.MMA_M

    comptime depth_tile_size = min(Self.depth, 128)

    # for depth = 64, we use 8B loads instead of 16B loads
    comptime load_width = 4 if Self.depth == 64 else Self.simd_width
    comptime loads_per_thread_per_depth_tile = (
        Self.depth_tile_size * Self.BK
    ) // (Self.load_width * Self.num_threads)

    # TileTensor register storage for DMA staging.
    comptime _load_rows_per_stage = (
        Self.loads_per_thread_per_depth_tile
        * (Self.depth // Self.depth_tile_size)
    )
    comptime _load_rows = Self._load_rows_per_stage * Self.num_stages
    comptime _load_layout = tt_row_major[Self._load_rows, Self.load_width]()
    comptime LoadTile = TileTensor[
        Self.dtype,
        type_of(Self._load_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var load_tile: Self.LoadTile

    # TileTensor register storage for MMA operand.
    comptime _mma_rows = Self.depth // Self.MMA_M
    comptime _mma_cols = Self.simd_width
    comptime mma_tile_layout = Layout.row_major(Self._mma_rows, Self._mma_cols)
    comptime _mma_layout = tt_row_major[Self._mma_rows, Self._mma_cols]()
    comptime MmaTile = TileTensor[
        Self.dtype,
        type_of(Self._mma_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var mma_tile: Self.MmaTile

    # Ephemeral LayoutTensor type for copy_to_shared and load_from_shared
    # (complex blocked_product SMEM layout + interleaved transpose).
    # Constructed inline from smem_ptr at each call site.
    var smem_ptr: UnsafePointer[
        Scalar[Self.dtype],
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    # DRAM tile and loader.
    comptime GmemTileType = TileTensor[
        Self.dtype, Self.kv_tile_layout, ImmutAnyOrigin
    ]
    # Warp-scoped RegTileLoader: thread_layout = row_major(4, 16), warp_scope=True
    comptime _dram_thread_rows = 4
    comptime _dram_thread_cols = Self.depth_tile_size // Self.load_width
    comptime RegLoaderType = RegTileLoader[
        Self.dtype,
        Self._dram_thread_rows,
        Self._dram_thread_cols,
        warp_scope=True,
    ]
    var gmem_tile: Self.GmemTileType
    var reg_loader: Self.RegLoaderType
    var tile_idx: Int
    var current_stage: Int

    @always_inline
    def __init__(
        out self,
        gmem_tile: Self.GmemTileType,
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
    ):
        comptime assert Self.depth in (
            64,
            128,
            256,
            512,
        ), "depth must be 64, 128, 256, or 512"
        comptime assert (
            Self.tensor_core_mma.shape[2] * Self.tensor_core_mma.group_size
            == 16
        ), "tensor_core_mma.shape[2] * tensor_core_mma.group_size must be 16"

        self.gmem_tile = gmem_tile
        self.reg_loader = Self.RegLoaderType(gmem_tile)

        self.load_tile = tt_stack_allocation[Self.dtype, AddressSpace.LOCAL](
            Self._load_layout
        )
        self.mma_tile = tt_stack_allocation[Self.dtype, AddressSpace.LOCAL](
            Self._mma_layout
        )
        self.smem_ptr = shared_ptr
        self.tile_idx = 0
        self.current_stage = 0

    @always_inline
    @staticmethod
    def pad[dim: Int]() -> Int:
        return pad[Self.dtype, Self.depth, dim]()

    @always_inline
    def load_from_dram(
        mut self,
    ):
        # Build per-iteration DRAM sub-tile: BK × depth (axis=0 iteration).
        var dram_tile = self.gmem_tile.tile[Self.BK, Self.depth](
            self.tile_idx, 0
        )
        var warp_id = get_warp_id()

        comptime assert (
            Self.loads_per_thread_per_depth_tile == 2
        ), "loads_per_thread_per_depth_tile must be 2"

        comptime for depth_idx in range(Self.depth // Self.depth_tile_size):
            comptime for i in range(Self.loads_per_thread_per_depth_tile):
                # Same warp-level tiling as before:
                # 16-row sub-tile → 8-row sub-tile → 4×depth_tile_size sub-tile
                var warp_tile = (
                    dram_tile.tile[16, Self.depth](
                        warp_id // 2,
                        0,
                    )
                    .tile[8, Self.depth](i, 0)
                    .tile[4, Self.depth_tile_size](warp_id % 2, depth_idx)
                )
                var dst_stage = self.load_tile.tile[
                    Self._load_rows_per_stage, Self.load_width
                ](self.current_stage, 0)
                self.reg_loader.load(
                    dst_stage.tile[1, Self.load_width](
                        i + depth_idx * Self.loads_per_thread_per_depth_tile,
                        0,
                    ),
                    warp_tile.vectorize[1, Self.load_width](),
                )

        self.current_stage = (self.current_stage + 1) % Self.num_stages
        self.tile_idx += 1

    @always_inline
    def get_mma_tile(
        self,
    ) -> LocalLayoutTensor[Self.dtype, Self.mma_tile_layout]:
        return rebind[LocalLayoutTensor[Self.dtype, Self.mma_tile_layout]](
            self.mma_tile.to_layout_tensor()
        )

    @always_inline
    def copy_to_shared[
        tile_id: Int = 0
    ](self,):
        # Each warp writes to its block in the blocked SMEM layout.
        # SMEM = num_repeats contiguous row_major(pad_depth, simd_width) blocks.
        var warp_id = get_warp_id()
        var lane_coords = idx2crd[Layout.col_major(16, 4)](lane_id())
        var lane_row = lane_coords[0]
        var lane_col = lane_coords[1]

        comptime pad_depth = Self.pad[Self.depth]()
        comptime pad_depth_tile = Self.pad[Self.depth_tile_size]()
        comptime pad_lw = Self.pad[Self.load_width]()
        comptime block_size = pad_depth * Self.simd_width

        # Warp's block base pointer.
        var block_ptr = self.smem_ptr + warp_id * block_size

        # Source register tile for this pipeline stage.
        var load_stage = self.load_tile.tile[
            Self._load_rows_per_stage, Self.load_width
        ](tile_id, 0)

        comptime for depth_idx in range(Self.depth // Self.depth_tile_size):
            # Depth chunk within the block (row_major stride = simd_width).
            var chunk_ptr = (
                block_ptr + depth_idx * pad_depth_tile * Self.simd_width
            )

            # Per-lane sub-tile: (pad_load_width, 2) at (lane_row, lane_col).
            # Offset = lane_row * pad_lw * simd_width + lane_col * 2
            # Then write at row stride = simd_width within the sub-tile.
            var lane_base = (
                chunk_ptr + lane_row * pad_lw * Self.simd_width + lane_col * 2
            )

            comptime for j in range(Self.load_width):
                var reg_tile_0 = load_stage[0 + depth_idx * 2, j][0]
                var reg_tile_1 = load_stage[1 + depth_idx * 2, j][0]
                var reg_pair = SIMD[Self.dtype, 2](reg_tile_0, reg_tile_1)
                (lane_base + j * Self.simd_width).store(reg_pair)

    @always_inline
    def load_from_shared[
        k_mma: Int,
    ](self):
        # Load MMA operands from padded blocked SMEM to registers.
        # SMEM = contiguous row_major(pad_depth, simd_width) blocks.
        # Each thread reads one SIMD[simd_width] row per depth_idx.
        var col_idx, lane_ = udivmod(lane_id(), 32)
        var lane = Int(lane_)

        comptime pad_depth = Self.pad[Self.depth]()
        comptime pad_mma_m = Self.pad[Self.MMA_M]()
        comptime pad_sw = Self.pad[Self.simd_width]()
        comptime block_size = pad_depth * Self.simd_width

        # Select the block for this k_mma group.
        var block_ptr = self.smem_ptr + (col_idx + k_mma * 2) * block_size

        # MMA tile register storage — vectorize for SIMD writes.
        var mma_vec = self.mma_tile.vectorize[1, Self.simd_width]()

        comptime for depth_idx in range(Self.num_depth_tiles):
            # Row within block:
            #   depth_idx * pad_MMA_M + (lane // sw) * pad_sw + (lane % sw)
            var row = (
                depth_idx * pad_mma_m
                + (lane // Self.simd_width) * pad_sw
                + (lane % Self.simd_width)
            )
            var elem_ptr = block_ptr + row * Self.simd_width
            mma_vec[depth_idx, 0] = elem_ptr.load[width=Self.simd_width]()


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
    comptime simd_width = simd_width_of[Self.dtype]()
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_mmas = ceildiv(Self.WM, Self.MMA_M)
    comptime num_k_tiles = ceildiv(Self.BK, Self.MMA_K * Self.k_group_size)

    comptime num_tiles = Self.depth // Self.BK
    comptime _total_rows = Self.num_mmas * Self.num_k_tiles * Self.num_tiles
    comptime _rows_per_tile = Self.num_mmas * Self.num_k_tiles

    # TileTensor storage in registers.
    comptime reg_layout = tt_row_major[Self._total_rows, Self.simd_width]()
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
            Self._q_thread_rows,
            Self._q_thread_cols,
            warp_scope=True,
            col_major_threads=True,
        ](warp_tile)

        # Load each BK-wide strip along the depth axis.
        comptime for i in range(Self.num_tiles):
            var src = warp_tile.tile[Self.WM, Self.BK](0, i)
            var dst = self.reg_tile.tile[Self._rows_per_tile, Self.simd_width](
                i, 0
            )
            reg_loader.load(
                dst,
                src.vectorize[1, Self.simd_width](),
            )

    @always_inline
    def mma_tile[
        tile_idx: Int, k_idx: Int
    ](self) -> TileTensor[
        Self.dtype,
        type_of(tt_row_major[Self.num_mmas, Self.simd_width]()),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        """Return MMA-sized sub-tile for the given tile and k indices."""
        return rebind[
            TileTensor[
                Self.dtype,
                type_of(tt_row_major[Self.num_mmas, Self.simd_width]()),
                MutExternalOrigin,
                address_space=AddressSpace.LOCAL,
            ]
        ](
            self.reg_tile.tile[Self._rows_per_tile, Self.simd_width](
                tile_idx, 0
            ).tile[Self.num_mmas, Self.simd_width](k_idx, 0)
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
                    Self._rows_per_tile, Self.simd_width
                ](tile, 0).tile[Self.num_mmas, Self.simd_width](k, 0)
                var vec = sub.vectorize[1, Self.simd_width]()
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
        Self._TiledMma.load_a[swizzle=None](warp_tile, result, UInt(k_idx))
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
    comptime _mma_simd = simd_width_of[Self.mma_dtype]()
    comptime _mma_layout = tt_row_major[Self.num_m_mmas, Self._mma_simd]()
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

        var result = tt_stack_allocation[Self.mma_dtype, AddressSpace.LOCAL](
            Self._mma_layout
        )
        var result_vec = result.vectorize[1, Self._mma_simd]()

        # Read source tile for this stage.
        var src = self.stage_tile[stage]()
        var src_vec = src.vectorize[1, Self.output_frag_size]()

        comptime if Self.tr_load_enabled:
            comptime if Self.mma_shape[0] == 32:
                # 32x32 MMA: cast full row, then slice to k_idx chunk.
                # src is SIMD[f32, output_frag_size], cast → SIMD[bf16, frag].
                # Result takes _mma_simd elements starting at k_idx offset.
                comptime assert (
                    Self.output_frag_size == 16
                ), "output_frag_size must be 16 for 32x32 mma shape"

                var casted = src_vec[tile_idx, 0].cast[Self.mma_dtype]()
                result_vec[0, 0] = rebind[type_of(result_vec[0, 0])](
                    casted.slice[
                        Self._mma_simd, offset=k_idx * Self._mma_simd
                    ]()
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
                            Self._mma_simd,
                            offset=k_idx * Self._mma_simd,
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
                packed.slice[Self._mma_simd, offset=k_idx * Self._mma_simd]()
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
        comptime num_n_mmas_per_bk = Self.num_n_mmas // (Self.WN // Self.BK)

        comptime assert (
            Self.WN == Self.BK or Self.WN == Self.BN
        ), "WN must be equal to BN or BK"

        var p_reg_vec = self.stage_tile[0]().vectorize[
            1, Self.output_frag_size
        ]()

        # 3D thread layout: (warp_rows, warp_cols, 1).
        # The trailing 1 leaves the inner frag_w elements undistributed,
        # so each thread gets frag_w contiguous scalars.
        comptime warp_m = Self.mma_shape[0]
        comptime warp_n = WARP_SIZE // warp_m
        comptime tl_3d = tt_col_major[warp_m, warp_n, 1]()

        comptime for i in range(Self.WN // Self.BK):
            var block_idx = Int(i) + warp_col * (Self.WN // Self.BK)
            # Inline blocked offset: each BM×BK block is contiguous.
            var smem_block = TileTensor[
                Self.dtype,
                type_of(tt_row_major[Self.BM, Self.BK]()),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ](
                self.shared_memory_ptr + block_idx * Self.BM * Self.BK,
                tt_row_major[Self.BM, Self.BK](),
            )
            var p_smem_warp_tile = smem_block.tile[Self.WM, Self.BK](
                warp_row, 0
            )

            comptime for m_mma, n_mma in std.itertools.product(
                range(Self.num_m_mmas), range(num_n_mmas_per_bk)
            ):
                var mma_tile = p_smem_warp_tile.tile[
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
                var p_reg_tile = p_reg_vec.tile[1, 1](
                    (n_mma + i * num_n_mmas_per_bk) * Self.num_m_mmas + m_mma,
                    0,
                )
                comptime frag_size = p_reg_tile.element_size
                var reg_val = p_reg_tile.ptr.load[width=frag_size]().cast[
                    Self.dtype
                ]()
                dst_frag.ptr.store[width=frag_w](reg_val)
