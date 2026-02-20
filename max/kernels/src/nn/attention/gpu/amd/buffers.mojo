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

from math import ceildiv, recip
from sys import simd_width_of
from sys.intrinsics import readfirstlane

from gpu import barrier, block_idx, lane_id
from gpu import warp_id as get_warp_id
from layout import Layout, LayoutTensor
from layout._utils import idx2crd
from layout.layout import blocked_product
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    copy_dram_to_local,
    copy_local_to_shared,
)
from layout.swizzle import Swizzle
from layout.tensor_core import TiledTensorCore
from memory.pointer import AddressSpace as BaseAddressSpace
from nn.mha_utils import _kernel_mask

from utils import IndexList

from .utils import (
    LocalLayoutTensor,
    SharedLayoutTensor,
    get_fragment_layout,
    get_warp_coords,
    get_warp_layout,
    pad,
)
import itertools


trait KVBuffer:
    comptime _dtype: DType
    comptime mma_tile_layout: Layout
    comptime _num_stages: Int

    @staticmethod
    fn get_dtype() -> DType:
        ...

    fn load_from_dram(mut self):
        ...

    fn get_mma_tile(
        self,
    ) -> LocalLayoutTensor[Self._dtype, Self.mma_tile_layout,]:
        ...

    fn copy_to_shared[
        tile_id: Int = 0
    ](self,):
        ...

    fn load_from_shared[
        k_mma: Int,
    ](self):
        ...


trait RegisterBuffer:
    comptime reg_dtype: DType
    comptime reg_tile_layout: Layout

    @staticmethod
    fn get_dtype() -> DType:
        ...

    fn zero(self):
        ...

    fn get_reg_tile[
        stage: Int = 0
    ](self,) -> LocalLayoutTensor[Self.reg_dtype, Self.reg_tile_layout,]:
        ...


trait RegisterMMABuffer(RegisterBuffer):
    comptime mma_dtype: DType
    comptime mma_tile_layout: Layout

    fn get_mma_tile[
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
    fn get_wtile_coord() -> IndexList[2]:
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
    fn get_wtile_coord() -> IndexList[2]:
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
    fn get_wtile_coord() -> IndexList[2]:
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]
        return IndexList[2](0, warp_col)


struct KVBufferImpl[
    dtype: DType,
    layout: Layout,
    address_space: BaseAddressSpace,
    alignment: Int,
    origin: Origin,
    masked: Bool,
    layout_int_type: DType,
    linear_idx_type: DType,
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

    comptime num_repeats = Self.config.btile_dim1 // Self.simd_width

    # Shared memory layout
    # Layout construction for standard memory access:
    # - base_layout: Layout.row_major(BN, simd_width) -> BNxsimd_width tiles
    # - tiler_layout: Layout.row_major(1, num_repeats) -> repeat tiles num_repeats times horizontally
    # - smem_layout: blocked_product(base_layout, tiler_layout) -> tiled blocked layout
    #
    # Resulting shape: BNx(simd_width x num_repeats) = BNxBK tensor
    # Where BK = simd_width x num_repeats, typically simd_width=8, num_repeats=BK/8
    #
    # This creates num_repeats blocks of BNxsimd_width arranged horizontally:
    # Within each simd_width-column block, elements are consecutive (stride 1)
    # Between blocks: stride = BN x simd_width
    #
    # ASCII diagram for BN=128, simd_width=8, BK=32 (showing first 2 of 4 blocks):
    # ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    # │        Block 0 (128x8)                     │        Block 1 (128x8)                     │     ... 2 more blocks           │
    # ├────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
    # │   0    1    2    3    4    5    6    7     │ 1024 1025 1026 1027 1028 1029 1030 1031    │ (Block 2: 2048-3071)            │
    # │   8    9   10   11   12   13   14   15     │ 1032 1033 1034 1035 1036 1037 1038 1039    │ (Block 3: 3072-4095)            │
    # │  16   17   18   19   20   21   22   23     │ 1040 1041 1042 1043 1044 1045 1046 1047    │                                 │
    # │  24   25   26   27   28   29   30   31     │ 1048 1049 1050 1051 1052 1053 1054 1055    │                                 │
    # │ ...                                        │  ...                                       │                                 │
    # │1016 1017 1018 1019 1020 1021 1022 1023     │ 2040 2041 2042 2043 2044 2045 2046 2047    │                                 │
    # └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    # stride between blocks = BN x simd_width = 128 x 8 = 1024

    comptime base_layout = Layout.row_major(
        Self.config.btile_dim0, Self.simd_width
    )
    comptime tiler_layout = Layout.row_major(1, Self.num_repeats)
    comptime smem_layout = blocked_product(
        Self.base_layout,
        Self.tiler_layout,
        coalesce_output=True,
    ) if not Self.token_gen else Layout.row_major(
        Self.config.btile_dim0, Self.config.btile_dim1
    )

    comptime thread_layout = Layout.row_major(
        min(
            Self.num_threads,
            (Self.config.btile_dim0 * Self.config.btile_dim1)
            // Self.simd_width,
        )
        * Self.simd_width
        // Self.smem_layout.stride[0].value(),
        Self.smem_layout.stride[0].value() // Self.simd_width,
    ) if Self.token_gen else Layout.row_major(Self.num_threads // 4, 4)

    comptime LoadTileType = LocalLayoutTensor[
        Self.dtype,
        Layout.row_major(
            Self.num_stages * Self.num_mmas * Self.num_k_tiles,
            Self.simd_width,
        ),
    ]
    var load_tile: Self.LoadTileType

    comptime mma_tile_layout = Layout.row_major(Self.num_mmas, Self.simd_width)

    comptime MMATileType = LocalLayoutTensor[
        Self.dtype,
        Self.mma_tile_layout,
    ]
    var mma_tile: Self.MMATileType

    comptime wtile_dim0 = Self.config.wtile_dim0
    comptime wtile_dim1 = Self.config.wtile_dim1

    comptime SharedIterType = LayoutTensorIter[
        Self.dtype,
        Self.smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]

    var smem_iter: Self.SharedIterType

    comptime SharedTileType = Self.SharedIterType.LayoutTensorType
    comptime SharedWarpTileType = Self.SharedTileType.TileType[
        Self.wtile_dim0, Self.wtile_dim1
    ]

    var bounds: Int
    var load_tile_id: Int

    comptime GlobalTensorType = LayoutTensor[
        Self.dtype,
        Self.layout,
        Self.origin,
        address_space = Self.address_space,
        alignment = Self.alignment,
        masked = Self.masked,
        layout_int_type = Self.layout_int_type,
        linear_idx_type = Self.linear_idx_type,
    ]

    comptime GlobalTiledIteratorType = Self.GlobalTensorType.TiledIteratorType[
        Self.config.btile_dim0,
        Self.config.btile_dim1,
        axis = Self.config.iterator_axis,
    ]

    var global_iterator: Self.GlobalTiledIteratorType

    @always_inline
    fn __init__(
        out self,
        global_tile: Self.GlobalTensorType,
        num_b_rows: Optional[Int],
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
    ):
        # comptime assert
        self.load_tile = type_of(self.load_tile).stack_allocation()
        self.mma_tile = type_of(self.mma_tile).stack_allocation()
        self.smem_iter = type_of(self.smem_iter)(shared_ptr, 0)
        comptime stride = Self.GlobalTiledIteratorType.layout.stride[0].value()
        self.bounds = num_b_rows.value() * stride if num_b_rows else Int.MAX
        self.global_iterator = global_tile.tiled_iterator[
            Self.config.btile_dim0,
            Self.config.btile_dim1,
            axis = Self.config.iterator_axis,
        ](0, 0)
        self.load_tile_id = 0

    @always_inline
    @staticmethod
    fn get_dtype() -> DType:
        return Self._dtype

    @always_inline
    fn load_from_dram(
        mut self,
    ):
        copy_dram_to_local[src_thread_layout = Self.thread_layout,](
            self.load_tile.split[Self.num_stages]()[
                self.load_tile_id
            ].vectorize[1, Self.simd_width](),
            self.global_iterator,
            UInt32(self.bounds),
        )
        self.global_iterator._incr()
        self.load_tile_id = (self.load_tile_id + 1) % Self.num_stages

    @always_inline
    fn get_mma_tile(self) -> Self.MMATileType:
        return self.mma_tile

    @always_inline
    fn copy_to_shared[
        tile_id: Int = 0
    ](self,):
        var smem_tile = self.smem_iter.next_unsafe(0)[]
        copy_local_to_shared[
            thread_layout = Self.thread_layout,
            swizzle = Self.swizzle,
            row_major=True,
        ](
            smem_tile.vectorize[1, Self.simd_width](),
            self.load_tile.split[Self.num_stages]()[tile_id].vectorize[
                1, Self.simd_width
            ](),
        )

    @always_inline
    fn load_from_shared[
        k_mma: Int,
    ](self):
        comptime num_warps_n = Self.BN // Self.WN
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]
        var smem_tile = self.smem_iter.next_unsafe(0)[]

        var wtile_coord0 = Self.config.get_wtile_coord()[0]
        var wtile_coord1 = Self.config.get_wtile_coord()[1]
        var warp_tile = smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
            wtile_coord0, wtile_coord1
        )

        Self.tensor_core_mma.mma_op.load_b[swizzle = Self.swizzle](
            warp_tile,
            self.get_mma_tile().vectorize[1, Self.simd_width](),
            UInt(k_mma),
        )


comptime KBuffer[
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
    config = KBufferConfig[BN, BK, WN],
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
    config = VBufferConfig[BN, BK, WN, depth],
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


struct VBufferTransposeLoads[
    dtype: DType,
    layout: Layout,
    address_space: BaseAddressSpace,
    alignment: Int,
    origin: Origin,
    masked: Bool,
    layout_int_type: DType,
    linear_idx_type: DType,
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

    # V Buffer shared memory layout
    # - base_layout: Layout.row_major(depth + padding, simd_width) -> (depth+padding)xsimd_width tiles
    # - tiler_layout: Layout.row_major(1, num_repeats) -> repeat tiles num_repeats times horizontally
    # - smem_layout: blocked_product(base_layout, tiler_layout) -> tiled blocked layout with padding
    #
    # Resulting shape: (depth + padding)x(simd_width x num_repeats) = (depth + depth//8)xBK tensor
    # Where padding = depth//8 helps avoid bank conflicts, BK = simd_width x num_repeats
    #
    # This creates num_repeats blocks of (depth+padding)xsimd_width arranged horizontally:
    # Within each simd_width-column block, elements are consecutive (stride 1)
    # Between blocks: stride = (depth + padding) x simd_width
    #
    # ASCII diagram for depth=128, padding=16, simd_width=8, BK=32 (showing first 2 of 4 blocks):
    # ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    # │        Block 0 (144x8)                     │        Block 1 (144x8)                     │     ... 2 more blocks           │
    # ├────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
    # │   0    1    2    3    4    5    6    7     │ 1152 1153 1154 1155 1156 1157 1158 1159    │ (Block 2: 2304-3455)            │
    # │   8    9   10   11   12   13   14   15     │ 1160 1161 1162 1163 1164 1165 1166 1167    │ (Block 3: 3456-4607)            │
    # │  16   17   18   19   20   21   22   23     │ 1168 1169 1170 1171 1172 1173 1174 1175    │                                 │
    # │  24   25   26   27   28   29   30   31     │ 1176 1177 1178 1179 1180 1181 1182 1183    │                                 │
    # │ ...                                        │  ...                                       │                                 │
    # │1144 1145 1146 1147 1148 1149 1150 1151     │ 2296 2297 2298 2299 2300 2301 2302 2303    │                                 │
    # └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    # stride between blocks = (depth + padding) x simd_width = 144 x 8 = 1152

    comptime base_layout = Layout.row_major(
        Self.pad[Self.depth](),
        Self.simd_width,
    )
    comptime tiler_layout = Layout.row_major(1, Self.num_repeats)
    comptime smem_layout = blocked_product(
        Self.base_layout,
        Self.tiler_layout,
        coalesce_output=True,
    )

    comptime MMA_M = Self.tensor_core_mma.shape[0]
    comptime MMA_K = Self.tensor_core_mma.shape[2]
    comptime num_k_tiles = ceildiv(
        Self.BK, Self.MMA_K * Self.tensor_core_mma.group_size
    )
    comptime num_depth_tiles = Self.depth // Self.MMA_M

    comptime depth_tile_size = min(Self.depth, 128)

    # for depth = 64, we use 8B loads instead of 16B loads
    # this keeps the layout of the memory access the same but may not be optimal
    # can come back to this if perf becomes an issue
    comptime load_width = 4 if Self.depth == 64 else Self.simd_width
    comptime loads_per_thread_per_depth_tile = (
        Self.depth_tile_size * Self.BK
    ) // (Self.load_width * Self.num_threads)

    comptime LoadTileType = LocalLayoutTensor[
        Self.dtype,
        Layout.row_major(
            (
                Self.loads_per_thread_per_depth_tile
                * (Self.depth // Self.depth_tile_size)
            )
            * Self.num_stages,
            Self.load_width,
        ),
    ]

    var load_tile: Self.LoadTileType

    comptime mma_tile_layout = Layout.row_major(
        Self.depth // Self.MMA_M, Self.simd_width
    )

    comptime MMATileType = LocalLayoutTensor[
        Self.dtype,
        Self.mma_tile_layout,
    ]

    var mma_tile: Self.MMATileType

    comptime SharedIterType = LayoutTensorIter[
        Self.dtype,
        Self.smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]

    var smem_iter: Self.SharedIterType

    comptime SharedTileType = Self.SharedIterType.LayoutTensorType

    comptime GlobalTensorType = LayoutTensor[
        Self.dtype,
        Self.layout,
        Self.origin,
        address_space = Self.address_space,
        alignment = Self.alignment,
        masked = Self.masked,
        layout_int_type = Self.layout_int_type,
        linear_idx_type = Self.linear_idx_type,
    ]

    comptime GlobalTiledIteratorType = Self.GlobalTensorType.TiledIteratorType[
        Self.BK,
        Self.depth,
        axis=0,
    ]

    var global_iterator: Self.GlobalTiledIteratorType
    var global_base_tile: Self.GlobalTensorType
    var current_stage: Int

    @always_inline
    fn __init__(
        out self,
        global_tile: Self.GlobalTensorType,
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
    ):
        comptime assert Self.depth in (
            64,
            128,
            256,
        ), "depth must be 64, 128, or 256"
        comptime assert (
            Self.tensor_core_mma.shape[2] * Self.tensor_core_mma.group_size
            == 16
        ), "tensor_core_mma.shape[2] * tensor_core_mma.group_size must be 16"

        self.global_base_tile = global_tile
        self.global_iterator = global_tile.tiled_iterator[
            Self.BK, Self.depth, axis=0
        ](0, 0)

        self.load_tile = type_of(self.load_tile).stack_allocation()
        self.mma_tile = type_of(self.mma_tile).stack_allocation()
        self.smem_iter = type_of(self.smem_iter)(shared_ptr, 0)
        self.current_stage = 0

    @always_inline
    @staticmethod
    fn get_dtype() -> DType:
        return Self._dtype

    @always_inline
    @staticmethod
    fn pad[dim: Int]() -> Int:
        return pad[Self.dtype, Self.depth, dim]()

    @always_inline
    fn load_from_dram(
        mut self,
    ):
        var global_tile = self.global_iterator[]
        var warp_id = get_warp_id()

        comptime assert (
            Self.loads_per_thread_per_depth_tile == 2
        ), "loads_per_thread_per_depth_tile must be 2"
        var load_tile = self.load_tile.split[Self.num_stages]()[
            self.current_stage
        ]

        comptime for depth_idx in range(Self.depth // Self.depth_tile_size):
            # every lane loads 2 elements (=8B for depth=64 and 16B for depth=128)
            # we transpose the global tile when writing to shared memory
            # the load pattern here is such that it enables us to use 16B loads
            # from shared memory and use p from registers instead of going through the shared memory.
            # warp 0 lane 0 will load first element of row 0 and row 8
            # warp 0 lane 16 will load first element of row 1 and row 9
            # warp 0 lane 32 will load first element of row 2 and row 10
            # warp 0 lane 48 will load first element of row 3 and row 11
            # warp 1 lane 0 will load first element of row 4 and row 12
            # warp 1 lane 16 will load first element of row 5 and row 13
            # warp 1 lane 32 will load first element of row 6 and row 14
            # warp 1 lane 48 will load first element of row 7 and row 15
            # warp 2 lane 0 will load first element of row 16 and row 24
            # warp 2 lane 16 will load first element of row 17 and row 25
            # warp 2 lane 32 will load first element of row 18 and row 26
            # warp 2 lane 48 will load first element of row 19 and row 27
            # warp 3 lane 0 will load first element of row 20 and row 28
            # warp 3 lane 16 will load first element of row 21 and row 29
            # warp 3 lane 32 will load first element of row 22 and row 30
            # warp 3 lane 48 will load first element of row 23 and row 31

            # so when we transpose and write to shared memory, the shared memory tile (of size depthxBK)
            # will effectively have its columns permuted as:
            # 0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15,16,24,17,25,18,26,19,27,20,28,21,29,22,30,23,31

            # we will have to interleave the elements of p in register to match this pattern for second mma to be correct.
            # which means that the output of softmax(which will be of size 16), we will have to be divided into into 2x8 and first 8 will be
            # interleaved and second 8 will be interleaved independently and use for two different mma operations.
            # This explanation will likely be clearer with a diagram, I will come back to this later.

            comptime for i in range(Self.loads_per_thread_per_depth_tile):
                var warp_tile = (
                    global_tile.tile[16, Self.depth](
                        Int(warp_id) // 2,
                        0,
                    )
                    .tile[8, Self.depth](i, 0)
                    .tile[4, Self.depth_tile_size](Int(warp_id) % 2, depth_idx)
                )
                copy_dram_to_local[
                    src_thread_layout = Layout.row_major(4, 16),
                    thread_scope = ThreadScope.WARP,
                ](
                    load_tile.tile[1, Self.load_width](
                        i + depth_idx * Self.loads_per_thread_per_depth_tile,
                        0,
                    ).vectorize[1, Self.load_width](),
                    warp_tile.vectorize[1, Self.load_width](),
                    self.global_base_tile,
                )

        self.current_stage = (self.current_stage + 1) % Self.num_stages
        self.global_iterator._incr()

    @always_inline
    fn get_mma_tile(self) -> Self.MMATileType:
        return self.mma_tile

    @always_inline
    fn copy_to_shared[
        tile_id: Int = 0
    ](self,):
        # we multiply v^T x p^T instead of p x v
        # here all threads work to load 16xdepth tile at a time
        # with each warp loading 4xdepth tile
        # each thread loads v_reg_tile is therefore BK//MMA_N 16B elements

        # transpose v_global_tile to v_smem
        # each thread writes 8x2 elements to smem using 4x4B writes
        # shared memory layout is row_major(depth, BK // num_warps) repeated num_warps times
        # and each warp writes to a different tile in smem

        var warp_id = get_warp_id()
        var lane_coords = idx2crd[Layout.col_major(16, 4)](Int(lane_id()))
        var lane_row = lane_coords[0]
        var lane_col = lane_coords[1]

        var smem_iter_tensor = self.smem_iter.next_unsafe(0)[]
        var load_tile = self.load_tile.split[Self.num_stages]()[tile_id]

        comptime for depth_idx in range(Self.depth // Self.depth_tile_size):
            var smem_warp_tile = smem_iter_tensor.tile[
                Self.pad[Self.depth](),
                Self.simd_width,
            ](0, Int(warp_id)).tile[
                Self.pad[Self.depth_tile_size](),
                Self.simd_width,
            ](
                depth_idx, 0
            )

            var lane_tile = (
                smem_warp_tile.tile[Self.pad[Self.load_width](), 2](
                    lane_row, lane_col
                )
                .slice[: Self.load_width, :]()
                .vectorize[1, 2]()
            )

            comptime for j in range(Self.load_width):
                # each thread loads 2x8 elements from gmem
                # they are interleaved and written to smem
                var reg_tile_0 = load_tile[0 + depth_idx * 2, j][0]
                var reg_tile_1 = load_tile[1 + depth_idx * 2, j][0]
                var reg_pair = SIMD[Self.dtype, 2](reg_tile_0, reg_tile_1)
                lane_tile[j, 0] = rebind[lane_tile.element_type](reg_pair)

    @always_inline
    fn load_from_shared[
        k_mma: Int,
    ](self):
        # MMA
        # threads in 16x4 layout
        # each column loads depth x 8 elements from smem
        var col_idx, lane = divmod(lane_id(), 32)
        var smem_iter_tensor = self.smem_iter.next_unsafe(0)[]

        comptime for depth_idx in range(Self.num_depth_tiles):
            # TODO: document and parameterize this magic
            var smem_fragment = (
                smem_iter_tensor.tile[Self.pad[Self.depth](), 8](
                    0, Int(col_idx + UInt(k_mma * 2))
                )
                .vectorize[1, Self.simd_width]()
                .tile[Self.pad[Self.MMA_M](), 1](depth_idx, 0)
                .tile[Self.pad[Self.simd_width](), 1](
                    Int(lane // UInt(Self.simd_width)), 0
                )
                .slice[: Self.simd_width, :]()
                .tile[1, 1](Int(lane % UInt(Self.simd_width)), 0)
            )
            self.mma_tile.vectorize[1, Self.simd_width]().tile[1, 1](
                depth_idx, 0
            ).copy_from(smem_fragment)


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
](RegisterMMABuffer):
    comptime reg_dtype = Self.dtype
    comptime mma_dtype = Self.dtype
    comptime simd_width = simd_width_of[Self.dtype]()
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_mmas = ceildiv(Self.WM, Self.MMA_M)
    comptime num_k_tiles = ceildiv(Self.BK, Self.MMA_K * Self.k_group_size)

    comptime MMATileType = Self.RegisterTileType.SplitElementType[
        Self.num_tiles
    ].SplitElementType[Self.num_k_tiles]
    comptime mma_tile_layout = Self.MMATileType.layout

    comptime num_tiles = Self.depth // Self.BK
    comptime reg_tile_layout = Layout.row_major(
        Self.num_mmas * Self.num_k_tiles * Self.num_tiles, Self.simd_width
    )
    comptime RegisterTileType = LocalLayoutTensor[
        Self.dtype,
        Self.reg_tile_layout,
    ]

    var reg_tile: Self.RegisterTileType

    comptime TiledIteratorType = Self.RegisterTileType.TiledIteratorType[
        Self.num_mmas * Self.num_k_tiles, Self.simd_width, axis=0
    ]

    # TODO: This is expensive, dereferencing q_gmem_warp_iter[] is expensive and
    # using its dim() is also expensive. Need to find a better way to do this.

    @staticmethod
    @always_inline
    fn get_dtype() -> DType:
        return Self.reg_dtype

    @always_inline
    fn __init__(out self, tensor: LayoutTensor[Self.dtype, ...]):
        self.reg_tile = type_of(self.reg_tile).stack_allocation()

        comptime num_warps_n = Self.BN // Self.WN
        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        var bounds = max(
            min(Int32(Self.WM), Int32(tensor.dim[0]() - Self.WM * warp_row))
            * Int32(tensor.stride[0]()),
            0,
        )
        var gmem_warp_iter = tensor.tiled_iterator[Self.WM, Self.BK, axis=1](
            warp_row, 0
        )
        var mma_tiles = self.reg_tile.split[Self.num_tiles]()

        comptime for i in range(Self.num_tiles):
            var reg_tile = mma_tiles[i]
            copy_dram_to_local[
                src_thread_layout = Self.thread_layout,
                thread_scope = ThreadScope.WARP,
            ](
                reg_tile.vectorize[1, Self.simd_width](),
                gmem_warp_iter,
                UInt32(Int(readfirstlane(bounds))),
            )
            gmem_warp_iter._incr()

    @always_inline
    fn get_iter(self) -> Self.TiledIteratorType:
        return self.reg_tile.tiled_iterator[
            Self.num_mmas * Self.num_k_tiles, Self.simd_width, axis=0
        ]()

    @always_inline
    fn get_mma_tile[tile_idx: Int, k_idx: Int](self) -> Self.MMATileType:
        return self.reg_tile.split[Self.num_tiles]()[tile_idx].split[
            Self.num_k_tiles
        ]()[k_idx]

    @always_inline
    fn get_reg_tile[stage: Int = 0](self) -> Self.RegisterTileType:
        return self.reg_tile

    @always_inline
    fn zero(self):
        _ = self.reg_tile.fill(0)


struct OutputRegisterBuffer[
    dtype: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    output_frag_size: Int,
](RegisterBuffer):
    comptime reg_dtype = Self.dtype

    comptime reg_tile_layout = Layout.row_major(
        Self.num_n_mmas * Self.num_m_mmas, Self.output_frag_size
    )
    comptime RegisterTileType = LocalLayoutTensor[
        Self.dtype,
        Self.reg_tile_layout,
    ]

    var reg_tile: Self.RegisterTileType

    @always_inline
    fn __init__(out self):
        self.reg_tile = Self.RegisterTileType.stack_allocation()

    @staticmethod
    @always_inline
    fn get_dtype() -> DType:
        return Self.reg_dtype

    @always_inline
    fn vectorize(
        self,
    ) -> Self.RegisterTileType.VectorizedType[1, Self.output_frag_size]:
        return self.reg_tile.vectorize[1, Self.output_frag_size]()

    @always_inline
    fn apply_softmax_denominator(self, rowsum: LayoutTensor[Self.dtype, ...]):
        comptime for m_mma in range(Self.num_m_mmas):
            var rowsum_inv = recip(rowsum[m_mma, 0])

            comptime for n_mma in range(Self.num_n_mmas):
                comptime for i in range(Self.output_frag_size):
                    self.reg_tile[n_mma * Self.num_m_mmas + m_mma, i] *= rebind[
                        Self.RegisterTileType.element_type
                    ](rowsum_inv)

    @always_inline
    fn zero(self):
        _ = self.reg_tile.fill(0)

    @always_inline
    fn get_reg_tile[stage: Int = 0](self) -> Self.RegisterTileType:
        return self.reg_tile


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
](RegisterMMABuffer):
    comptime reg_dtype = Self.accum_type_
    comptime mma_dtype = Self.dtype
    comptime mma_tile_layout = Layout.row_major(
        Self.num_m_mmas, simd_width_of[Self.dtype]()
    )
    comptime reg_tile_layout = Layout.row_major(
        Self.num_n_mmas * Self.num_m_mmas, Self.output_frag_size
    )

    comptime reg_tile_layout_ = Layout.row_major(
        Self.num_stages * Self.num_n_mmas * Self.num_m_mmas,
        Self.output_frag_size,
    )

    comptime RegisterTileType_ = LocalLayoutTensor[
        Self.accum_type_,
        Self.reg_tile_layout_,
    ]

    comptime RegisterTileType = LocalLayoutTensor[
        Self.accum_type_,
        Self.reg_tile_layout,
    ]
    comptime MMATileType = LocalLayoutTensor[
        Self.mma_dtype,
        Self.mma_tile_layout,
    ]

    var reg_tile: Self.RegisterTileType_

    comptime shared_memory_layout = blocked_product(
        Layout.row_major(Self.BM, Self.BK),
        Layout.row_major(1, Self.BN // Self.BK),
    )

    comptime SharedMemoryTileType = SharedLayoutTensor[
        Self.dtype,
        Self.shared_memory_layout,
    ]

    var shared_memory_tile: Self.SharedMemoryTileType

    @always_inline
    fn __init__(
        out self,
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
    ):
        self.reg_tile = Self.RegisterTileType_.stack_allocation()
        self.shared_memory_tile = Self.SharedMemoryTileType(shared_ptr)

    @always_inline
    fn get_mma_tile_reg[
        tile_idx: Int, k_idx: Int, stage: Int = 0
    ](self) -> Self.MMATileType:
        comptime OutputTileType = LocalLayoutTensor[
            Self.mma_dtype,
            Layout.row_major(Self.num_m_mmas, Self.output_frag_size),
        ]

        var out = OutputTileType.stack_allocation()

        var reg_tile = self.reg_tile.split[Self.num_stages]()[stage]

        comptime if Self.tr_load_enabled:
            # if tr loads are used then we don't need any packing logic
            # just convert the registers to bf16

            comptime if Self.mma_shape[0] == 32:
                comptime assert (
                    Self.output_frag_size == 16
                ), "output_frag_size must be 16 for 32x32 mma shape"

                comptime for j in range(Self.output_frag_size):
                    out[0, j] = reg_tile[tile_idx, j].cast[Self.mma_dtype]()
            elif Self.mma_shape[0] == 16:
                comptime assert (
                    Self.output_frag_size == 4
                ), "output_frag_size must be 4 for 16x16 mma shape"

                var mma_reg_tile = Self.MMATileType.stack_allocation()
                var reg_tile_split = reg_tile.split[Self.num_n_mmas // 2]()[
                    tile_idx
                ]

                comptime for m, j in itertools.product(
                    range(Self.num_m_mmas), range(Self.output_frag_size)
                ):
                    mma_reg_tile[m, j] = reg_tile_split[m, j].cast[
                        Self.mma_dtype
                    ]()
                    mma_reg_tile[m, Self.output_frag_size + j] = reg_tile_split[
                        m + Self.num_m_mmas, j
                    ].cast[Self.mma_dtype]()
                return mma_reg_tile
            else:
                constrained[
                    False,
                    String("Unsupported mma shape: ", Self.mma_shape[0]),
                ]()
        else:
            # this is special packing, the pattern here depends on how we load
            # and transpose the v tile when writing to the shared memory
            comptime for j in range(4):
                out[0, 2 * j] = reg_tile[tile_idx, j].cast[Self.mma_dtype]()
                out[0, 2 * j + 1] = reg_tile[tile_idx, 4 + j].cast[
                    Self.mma_dtype
                ]()
                out[0, 2 * j + 8] = reg_tile[tile_idx, 8 + j].cast[
                    Self.mma_dtype
                ]()
                out[0, 2 * j + 8 + 1] = reg_tile[tile_idx, 12 + j].cast[
                    Self.mma_dtype
                ]()
        return rebind[Self.MMATileType](
            out.tile[Self.num_n_mmas, simd_width_of[Self.mma_dtype]()](0, k_idx)
        )

    @always_inline
    fn get_mma_tile_shared[tile_idx: Int, k_idx: Int](self) -> Self.MMATileType:
        var mma_reg_tile = Self.MMATileType.stack_allocation()
        comptime num_warps_n = Self.WN // Self.BN
        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        var warp_tile = self.shared_memory_tile.tile[Self.WM, Self.BK](
            warp_row, tile_idx
        )

        comptime tensor_core_mma = TiledTensorCore[
            Self.accum_type_,
            Self.dtype,
            Self.mma_shape,
            group_size = Self.k_group_size,
            transpose_b=False,
        ]()

        tensor_core_mma.mma_op.load_a[swizzle=None](
            warp_tile,
            mma_reg_tile.vectorize[1, simd_width_of[Self.dtype]()](),
            UInt(k_idx),
        )
        return mma_reg_tile

    @always_inline
    fn get_mma_tile[tile_idx: Int, k_idx: Int](self) -> Self.MMATileType:
        return self.get_mma_tile[tile_idx, k_idx, 0]()

    @always_inline
    fn get_mma_tile[
        tile_idx: Int, k_idx: Int, stage: Int
    ](self) -> Self.MMATileType:
        return self.get_mma_tile_reg[
            tile_idx, k_idx, stage
        ]() if not Self.shared_memory_backed else self.get_mma_tile_shared[
            tile_idx, k_idx
        ]()

    @staticmethod
    @always_inline
    fn get_dtype() -> DType:
        return Self.mma_dtype

    @always_inline
    fn vectorize[
        stage: Int = 0
    ](
        self,
        out res: Self.RegisterTileType.VectorizedType[1, Self.output_frag_size],
    ):
        return rebind[type_of(res)](
            self.reg_tile.split[Self.num_stages]()[stage].vectorize[
                1, Self.output_frag_size
            ]()
        )

    @always_inline
    fn zero[stage: Int](self):
        _ = self.reg_tile.split[Self.num_stages]()[stage].fill(0)

    @always_inline
    fn zero(self):
        self.zero[0]()

    @always_inline
    fn get_reg_tile[stage: Int = 0](self) -> Self.RegisterTileType:
        return rebind[Self.RegisterTileType](
            self.reg_tile.split[Self.num_stages]()[stage]
        )

    @always_inline
    fn get_shared_memory_tile(
        self, tile_idx: Int
    ) -> Self.SharedMemoryTileType.TileType[Self.BM, Self.BK]:
        return self.shared_memory_tile.tile[Self.BM, Self.BK](0, tile_idx)

    @always_inline
    fn copy_to_shared(self):
        comptime warp_layout = get_warp_layout[Self.mma_shape]()
        comptime fragment_layout = get_fragment_layout[Self.mma_shape]()
        comptime num_warps_n = Self.BN // Self.WN
        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]
        comptime num_n_mmas_per_bk = Self.num_n_mmas // (Self.WN // Self.BK)

        # for the following indexing logic, WN must be equal to BN or BK
        comptime assert (
            Self.WN == Self.BK or Self.WN == Self.BN
        ), "WN must be equal to BN or BK"

        var p_reg_vectorized = self.vectorize()

        comptime for i in range(Self.WN // Self.BK):
            var p_smem_tile = self.get_shared_memory_tile(
                i + warp_col * (Self.WN // Self.BK)
            )
            var p_smem_warp_tile = p_smem_tile.tile[Self.WM, Self.BK](
                warp_row, i
            )

            comptime for m_mma, n_mma in itertools.product(
                range(Self.num_m_mmas), range(num_n_mmas_per_bk)
            ):
                var p_smem_mma_tile = p_smem_warp_tile.tile[
                    Self.mma_shape[0], Self.mma_shape[1]
                ](m_mma, n_mma)
                var p_reg_tile = p_reg_vectorized.tile[1, 1](
                    (n_mma + i * num_n_mmas_per_bk) * Self.num_m_mmas + m_mma,
                    0,
                )
                copy_local_to_shared[thread_layout=warp_layout](
                    p_smem_mma_tile.vectorize[
                        fragment_layout.shape[0].value(),
                        fragment_layout.shape[1].value(),
                    ](),
                    p_reg_tile,
                )
