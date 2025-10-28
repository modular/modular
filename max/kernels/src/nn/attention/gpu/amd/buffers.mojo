# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import OptionalReg
from math import ceildiv, recip
from math.constants import log2e
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
    convert_f32_to_bf16,
    get_fragment_layout,
    get_warp_coords,
    get_warp_layout,
    pad,
)


trait KVBuffer:
    alias _dtype: DType
    alias mma_tile_layout: Layout
    alias _num_stages: Int

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
    alias reg_dtype: DType
    alias reg_tile_layout: Layout

    @staticmethod
    fn get_dtype() -> DType:
        ...

    fn zero(self):
        ...

    fn get_reg_tile(
        self,
    ) -> LocalLayoutTensor[Self.reg_dtype, Self.reg_tile_layout,]:
        ...


trait RegisterMMABuffer(RegisterBuffer):
    alias mma_dtype: DType
    alias mma_tile_layout: Layout

    fn get_mma_tile[
        tile_idx: Int, k_idx: Int
    ](self,) -> LocalLayoutTensor[Self.mma_dtype, Self.mma_tile_layout,]:
        ...


trait KVBufferConfig:
    alias wsize: Int
    alias wtile_dim0: Int
    alias wtile_dim1: Int

    alias btile_dim0: Int
    alias btile_dim1: Int

    alias iterator_axis: Int

    @staticmethod
    @always_inline
    fn get_wtile_coord() -> IndexList[2]:
        ...


@fieldwise_init
struct KBufferConfig[BN: Int, BK: Int, WN: Int](KVBufferConfig):
    alias wsize = Self.wtile_dim0
    alias wtile_dim0 = WN
    alias wtile_dim1 = BK

    alias btile_dim0 = BN
    alias btile_dim1 = BK

    alias iterator_axis = 1

    @staticmethod
    @always_inline
    fn get_wtile_coord() -> IndexList[2]:
        var warp_col = get_warp_coords[BN, WN]()[1]
        return IndexList[2](Int(warp_col), 0)


@fieldwise_init
struct VBufferConfig[BN: Int, BK: Int, WN: Int, depth: Int](KVBufferConfig):
    alias wsize = Self.wtile_dim1
    alias wtile_dim0 = BK
    alias wtile_dim1 = depth // (BN // WN)

    alias btile_dim0 = BK
    alias btile_dim1 = depth

    alias iterator_axis = 0

    @staticmethod
    @always_inline
    fn get_wtile_coord() -> IndexList[2]:
        var warp_col = get_warp_coords[BN, WN]()[1]
        return IndexList[2](0, Int(warp_col))


struct KVBufferImpl[
    dtype: DType,
    layout: Layout,
    address_space: BaseAddressSpace,
    alignment: Int,
    origin: Origin,
    masked: Bool,
    layout_int_type: DType,
    linear_idx_type: DType, //,
    config: KVBufferConfig,
    tensor_core_mma: TiledTensorCore,
    swizzle: OptionalReg[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
    token_gen: Bool = False,
](KVBuffer):
    alias _dtype = dtype
    alias _num_stages = num_stages
    alias MMA_N = tensor_core_mma.shape[1]
    alias MMA_K = tensor_core_mma.shape[2]
    alias num_warps_n = BN // WN
    alias num_mmas = ceildiv(config.wsize, Self.MMA_N)

    alias num_k_tiles = ceildiv(BK, Self.MMA_K * tensor_core_mma.group_size)
    alias simd_width = simd_width_of[dtype]()

    alias num_repeats = config.btile_dim1 // Self.simd_width

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

    alias base_layout = Layout.row_major(config.btile_dim0, Self.simd_width)
    alias tiler_layout = Layout.row_major(1, Self.num_repeats)
    alias smem_layout = blocked_product(
        Self.base_layout,
        Self.tiler_layout,
        coalesce_output=True,
    ) if not token_gen else Layout.row_major(
        config.btile_dim0, config.btile_dim1
    )

    alias thread_layout = Layout.row_major(
        min(
            num_threads,
            (config.btile_dim0 * config.btile_dim1) // Self.simd_width,
        )
        * Self.simd_width
        // Self.smem_layout.stride[0].value(),
        Self.smem_layout.stride[0].value() // Self.simd_width,
    ) if token_gen else Layout.row_major(num_threads // 4, 4)

    alias LoadTileType = LocalLayoutTensor[
        dtype,
        Layout.row_major(
            num_stages * Self.num_mmas * Self.num_k_tiles,
            Self.simd_width,
        ),
    ]
    var load_tile: Self.LoadTileType

    alias mma_tile_layout = Layout.row_major(Self.num_mmas, Self.simd_width)

    alias MMATileType = LocalLayoutTensor[
        dtype,
        Self.mma_tile_layout,
    ]
    var mma_tile: Self.MMATileType

    alias wtile_dim0 = config.wtile_dim0
    alias wtile_dim1 = config.wtile_dim1

    alias SharedIterType = LayoutTensorIter[
        dtype,
        Self.smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]

    var smem_iter: Self.SharedIterType

    alias SharedTileType = Self.SharedIterType.LayoutTensorType
    alias SharedWarpTileType = Self.SharedTileType.TileType[
        Self.wtile_dim0, Self.wtile_dim1
    ]

    var bounds: Int
    var load_tile_id: Int

    alias GlobalTensorType = LayoutTensor[
        dtype,
        layout,
        origin,
        address_space=address_space,
        alignment=alignment,
        masked=masked,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]

    alias GlobalTiledIteratorType = Self.GlobalTensorType.TiledIteratorType[
        config.btile_dim0,
        config.btile_dim1,
        axis = config.iterator_axis,
    ]

    var global_iterator: Self.GlobalTiledIteratorType

    @always_inline
    fn __init__(
        out self,
        global_tile: Self.GlobalTensorType,
        num_b_rows: OptionalReg[Int],
        shared_ptr: UnsafePointer[
            Scalar[dtype],
            address_space = AddressSpace.SHARED, **_,
        ],
    ):
        # constrained[
        #     mma_shape[2] * k_group_size == 16,
        #     "mma_shape[2] * k_group_size must be 16",
        # ]()
        self.load_tile = type_of(self.load_tile).stack_allocation()
        self.mma_tile = type_of(self.mma_tile).stack_allocation()
        self.smem_iter = type_of(self.smem_iter)(shared_ptr, 0)
        alias stride = Self.GlobalTiledIteratorType.layout.stride[0].value()
        self.bounds = num_b_rows.value() * stride if num_b_rows else Int.MAX
        self.global_iterator = global_tile.tiled_iterator[
            config.btile_dim0,
            config.btile_dim1,
            axis = config.iterator_axis,
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
            self.load_tile.split[num_stages]()[self.load_tile_id].vectorize[
                1, Self.simd_width
            ](),
            self.global_iterator,
            self.bounds,
        )
        self.global_iterator._incr()
        self.load_tile_id = (self.load_tile_id + 1) % num_stages

    @always_inline
    fn get_mma_tile(self) -> Self.MMATileType:
        return self.mma_tile

    @always_inline
    fn copy_to_shared[
        tile_id: Int = 0
    ](self,):
        var smem_tile = self.smem_iter.next_unsafe(0)[]
        copy_local_to_shared[
            thread_layout = Self.thread_layout, swizzle=swizzle, row_major=True
        ](
            smem_tile.vectorize[1, Self.simd_width](),
            self.load_tile.split[num_stages]()[tile_id].vectorize[
                1, Self.simd_width
            ](),
        )

    @always_inline
    fn load_from_shared[
        k_mma: Int,
    ](self):
        alias num_warps_n = BN // WN
        var warp_col = get_warp_coords[BN, WN]()[1]
        var smem_tile = self.smem_iter.next_unsafe(0)[]

        var wtile_coord0 = config.get_wtile_coord()[0]
        var wtile_coord1 = config.get_wtile_coord()[1]
        var warp_tile = smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
            wtile_coord0, wtile_coord1
        )

        tensor_core_mma.mma_op.load_b[swizzle=swizzle](
            warp_tile,
            self.get_mma_tile().vectorize[1, Self.simd_width](),
            UInt(k_mma),
        )


alias KBuffer[
    tensor_core_mma: TiledTensorCore,
    swizzle: OptionalReg[Swizzle],
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

alias VBuffer[
    tensor_core_mma: TiledTensorCore,
    swizzle: OptionalReg[Swizzle],
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
    linear_idx_type: DType, //,
    tensor_core_mma: TiledTensorCore,
    BN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
](KVBuffer):
    alias _dtype = dtype
    alias _num_stages = num_stages
    alias simd_width = simd_width_of[dtype]()
    alias num_repeats = BK // Self.simd_width

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

    alias base_layout = Layout.row_major(
        Self.pad[depth](),
        Self.simd_width,
    )
    alias tiler_layout = Layout.row_major(1, Self.num_repeats)
    alias smem_layout = blocked_product(
        Self.base_layout,
        Self.tiler_layout,
        coalesce_output=True,
    )

    alias MMA_M = tensor_core_mma.shape[0]
    alias MMA_K = tensor_core_mma.shape[2]
    alias num_k_tiles = ceildiv(BK, Self.MMA_K * tensor_core_mma.group_size)
    alias num_depth_tiles = depth // Self.MMA_M

    alias depth_tile_size = min(depth, 128)

    # for depth = 64, we use 8B loads instead of 16B loads
    # this keeps the layout of the memory access the same but may not be optimal
    # can come back to this if perf becomes an issue
    alias load_width = 4 if depth == 64 else Self.simd_width
    alias loads_per_thread_per_depth_tile = (Self.depth_tile_size * BK) // (
        Self.load_width * Self.num_threads
    )

    alias LoadTileType = LocalLayoutTensor[
        dtype,
        Layout.row_major(
            (
                Self.loads_per_thread_per_depth_tile
                * (depth // Self.depth_tile_size)
            )
            * num_stages,
            Self.load_width,
        ),
    ]

    var load_tile: Self.LoadTileType

    alias mma_tile_layout = Layout.row_major(
        depth // Self.MMA_M, Self.simd_width
    )

    alias MMATileType = LocalLayoutTensor[
        dtype,
        Self.mma_tile_layout,
    ]

    var mma_tile: Self.MMATileType

    alias SharedIterType = LayoutTensorIter[
        dtype,
        Self.smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]

    var smem_iter: Self.SharedIterType

    alias SharedTileType = Self.SharedIterType.LayoutTensorType

    alias GlobalTensorType = LayoutTensor[
        dtype,
        layout,
        origin,
        address_space=address_space,
        alignment=alignment,
        masked=masked,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]

    alias GlobalTiledIteratorType = Self.GlobalTensorType.TiledIteratorType[
        BK,
        depth,
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
            Scalar[dtype],
            address_space = AddressSpace.SHARED, **_,
        ],
    ):
        constrained[depth in (64, 128, 256), "depth must be 64, 128, or 256"]()
        constrained[
            tensor_core_mma.shape[2] * tensor_core_mma.group_size == 16,
            "tensor_core_mma.shape[2] * tensor_core_mma.group_size must be 16",
        ]()

        self.global_base_tile = global_tile
        self.global_iterator = global_tile.tiled_iterator[BK, depth, axis=0](
            0, 0
        )

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
        return pad[dtype, depth, dim]()

    @always_inline
    fn load_from_dram(
        mut self,
    ):
        var global_tile = self.global_iterator[]
        var warp_id = get_warp_id()

        constrained[
            Self.loads_per_thread_per_depth_tile == 2,
            "loads_per_thread_per_depth_tile must be 2",
        ]()
        var load_tile = self.load_tile.split[Self.num_stages]()[
            self.current_stage
        ]

        @parameter
        for depth_idx in range(depth // Self.depth_tile_size):
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

            @parameter
            for i in range(Self.loads_per_thread_per_depth_tile):
                var warp_tile = (
                    global_tile.tile[16, depth](
                        Int(warp_id) // 2,
                        0,
                    )
                    .tile[8, depth](i, 0)
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

        @parameter
        for depth_idx in range(depth // Self.depth_tile_size):
            var smem_warp_tile = smem_iter_tensor.tile[
                Self.pad[depth](),
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

            @parameter
            for j in range(Self.load_width):
                # each thread loads 2x8 elements from gmem
                # they are interleaved and written to smem
                var reg_tile_0 = load_tile[0 + depth_idx * 2, j][0]
                var reg_tile_1 = load_tile[1 + depth_idx * 2, j][0]
                var reg_pair = SIMD[dtype, 2](reg_tile_0, reg_tile_1)
                lane_tile[j, 0] = rebind[lane_tile.element_type](reg_pair)

    @always_inline
    fn load_from_shared[
        k_mma: Int,
    ](self):
        # MMA
        # threads in 16x4 layout
        # each column loads depth x 8 elements from smem
        var col_idx = lane_id() // 32
        var lane = lane_id() % 32
        var smem_iter_tensor = self.smem_iter.next_unsafe(0)[]

        @parameter
        for depth_idx in range(Self.num_depth_tiles):
            # TODO: document and parameterize this magic
            var smem_fragment = (
                smem_iter_tensor.tile[Self.pad[depth](), 8](
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
    alias reg_dtype = dtype
    alias mma_dtype = dtype
    alias simd_width = simd_width_of[dtype]()
    alias MMA_M = mma_shape[0]
    alias MMA_K = mma_shape[2]
    alias num_mmas = ceildiv(WM, Self.MMA_M)
    alias num_k_tiles = ceildiv(BK, Self.MMA_K * k_group_size)

    alias MMATileType = Self.RegisterTileType.SplitElementType[
        Self.num_tiles
    ].SplitElementType[Self.num_k_tiles]
    alias mma_tile_layout = Self.MMATileType.layout

    alias num_tiles = depth // BK
    alias reg_tile_layout = Layout.row_major(
        Self.num_mmas * Self.num_k_tiles * Self.num_tiles, Self.simd_width
    )
    alias RegisterTileType = LocalLayoutTensor[
        dtype,
        Self.reg_tile_layout,
    ]

    var reg_tile: Self.RegisterTileType

    alias TiledIteratorType = Self.RegisterTileType.TiledIteratorType[
        Self.num_mmas * Self.num_k_tiles, Self.simd_width, axis=0
    ]

    # TODO: This is expensive, dereferencing q_gmem_warp_iter[] is expensive and
    # using its dim() is also expensive. Need to find a better way to do this.

    @staticmethod
    @always_inline
    fn get_dtype() -> DType:
        return Self.reg_dtype

    @always_inline
    fn __init__(out self, tensor: LayoutTensor[dtype, **_]):
        self.reg_tile = type_of(self.reg_tile).stack_allocation()

        alias num_warps_n = BN // WN
        var warp_row = get_warp_coords[BN, WN]()[0]
        var bounds = max(
            min(Int32(WM), Int32(tensor.dim[0]() - WM * warp_row))
            * tensor.stride[0](),
            0,
        )
        var gmem_warp_iter = tensor.tiled_iterator[WM, BK, axis=1](warp_row, 0)
        var mma_tiles = self.reg_tile.split[Self.num_tiles]()

        @parameter
        for i in range(Self.num_tiles):
            var reg_tile = mma_tiles[i]
            copy_dram_to_local[
                src_thread_layout=thread_layout,
                thread_scope = ThreadScope.WARP,
            ](
                reg_tile.vectorize[1, Self.simd_width](),
                gmem_warp_iter,
                Int(readfirstlane(Int32(bounds))),
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
    fn get_reg_tile(self) -> Self.RegisterTileType:
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
    alias reg_dtype = dtype

    alias reg_tile_layout = Layout.row_major(
        num_n_mmas * num_m_mmas, output_frag_size
    )
    alias RegisterTileType = LocalLayoutTensor[
        dtype,
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
    ) -> Self.RegisterTileType.VectorizedType[1, output_frag_size]:
        return self.reg_tile.vectorize[1, output_frag_size]()

    @always_inline
    fn apply_softmax_denominator(self, rowsum: LayoutTensor[dtype, **_]):
        @parameter
        for m_mma in range(num_m_mmas):
            var rowsum_inv = recip(rowsum[m_mma, 0])

            @parameter
            for n_mma in range(num_n_mmas):

                @parameter
                for i in range(output_frag_size):
                    self.reg_tile[n_mma * num_m_mmas + m_mma, i] *= rebind[
                        Self.RegisterTileType.element_type
                    ](rowsum_inv)

    @always_inline
    fn zero(self):
        _ = self.reg_tile.fill(0)

    @always_inline
    fn get_reg_tile(self) -> Self.RegisterTileType:
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
](RegisterMMABuffer):
    alias reg_dtype = accum_type_
    alias mma_dtype = dtype
    alias mma_tile_layout = Layout.row_major(num_m_mmas, simd_width_of[dtype]())
    alias reg_tile_layout = Layout.row_major(
        num_n_mmas * num_m_mmas, output_frag_size
    )

    alias RegisterTileType = LocalLayoutTensor[
        accum_type_,
        Self.reg_tile_layout,
    ]

    alias OutputTileType = LocalLayoutTensor[
        Self.mma_dtype,
        Layout.row_major(num_m_mmas, output_frag_size),
    ]
    alias MMATileType = LocalLayoutTensor[
        Self.mma_dtype,
        Self.mma_tile_layout,
    ]

    var reg_tile: Self.RegisterTileType

    alias shared_memory_layout = blocked_product(
        Layout.row_major(BM, BK), Layout.row_major(1, BN // BK)
    )

    alias SharedMemoryTileType = SharedLayoutTensor[
        dtype,
        Self.shared_memory_layout,
    ]

    var shared_memory_tile: Self.SharedMemoryTileType

    @always_inline
    fn __init__(
        out self,
        shared_ptr: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.SHARED, **_
        ],
    ):
        self.reg_tile = Self.RegisterTileType.stack_allocation()
        self.shared_memory_tile = Self.SharedMemoryTileType(shared_ptr)

    @always_inline
    fn get_mma_tile_reg[tile_idx: Int, k_idx: Int](self) -> Self.MMATileType:
        var out = Self.OutputTileType.stack_allocation()

        @parameter
        for j in range(4):
            out[0, 2 * j] = convert_f32_to_bf16[Self.mma_dtype](
                self.reg_tile[tile_idx, j]
            )

            out[0, 2 * j + 1] = convert_f32_to_bf16[Self.mma_dtype](
                self.reg_tile[tile_idx, 4 + j]
            )
            out[0, 2 * j + 8] = convert_f32_to_bf16[Self.mma_dtype](
                self.reg_tile[tile_idx, 8 + j]
            )
            out[0, 2 * j + 8 + 1] = convert_f32_to_bf16[Self.mma_dtype](
                self.reg_tile[tile_idx, 12 + j]
            )
        return rebind[Self.MMATileType](
            out.tile[num_n_mmas, simd_width_of[Self.mma_dtype]()](0, k_idx)
        )

    @always_inline
    fn get_mma_tile_shared[tile_idx: Int, k_idx: Int](self) -> Self.MMATileType:
        var mma_reg_tile = Self.MMATileType.stack_allocation()
        alias num_warps_n = WN // BN
        var warp_row = get_warp_coords[BN, WN]()[0]
        var warp_tile = self.shared_memory_tile.tile[WM, BK](warp_row, tile_idx)

        alias tensor_core_mma = TiledTensorCore[
            accum_type_,
            dtype,
            mma_shape,
            group_size=k_group_size,
            transpose_b=False,
        ]()

        tensor_core_mma.mma_op.load_a[swizzle=None](
            warp_tile,
            mma_reg_tile.vectorize[1, simd_width_of[dtype]()](),
            UInt(k_idx),
        )
        return mma_reg_tile

    @always_inline
    fn get_mma_tile[tile_idx: Int, k_idx: Int](self) -> Self.MMATileType:
        return self.get_mma_tile_reg[
            tile_idx, k_idx
        ]() if not Self.shared_memory_backed else self.get_mma_tile_shared[
            tile_idx, k_idx
        ]()

    @staticmethod
    @always_inline
    fn get_dtype() -> DType:
        return Self.mma_dtype

    @always_inline
    fn vectorize(
        self,
    ) -> Self.RegisterTileType.VectorizedType[1, output_frag_size]:
        return self.reg_tile.vectorize[1, output_frag_size]()

    @always_inline
    fn zero(self):
        _ = self.reg_tile.fill(0)

    @always_inline
    fn get_reg_tile(self) -> Self.RegisterTileType:
        return self.reg_tile

    @always_inline
    fn get_shared_memory_tile(
        self, tile_idx: Int
    ) -> Self.SharedMemoryTileType.TileType[BM, BK]:
        return self.shared_memory_tile.tile[BM, BK](0, tile_idx)

    @always_inline
    fn copy_to_shared(self):
        alias warp_layout = get_warp_layout[mma_shape]()
        alias fragment_layout = get_fragment_layout[mma_shape]()
        alias num_warps_n = Self.BN // Self.WN
        var warp_row = get_warp_coords[BN, WN]()[0]
        var warp_col = get_warp_coords[BN, WN]()[1]
        alias num_n_mmas_per_bk = Self.num_n_mmas // (Self.WN // Self.BK)

        # for the following indexing logic, WN must be equal to BN or BK
        constrained[
            Self.WN == Self.BK or Self.WN == Self.BN,
            "WN must be equal to BN or BK",
        ]()

        var p_reg_vectorized = self.vectorize()

        @parameter
        for i in range(Self.WN // Self.BK):
            var p_smem_tile = self.get_shared_memory_tile(
                i + warp_col * (Self.WN // Self.BK)
            )
            var p_smem_warp_tile = p_smem_tile.tile[Self.WM, Self.BK](
                warp_row, i
            )

            @parameter
            for m_mma in range(Self.num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas_per_bk):
                    var p_smem_mma_tile = p_smem_warp_tile.tile[
                        Self.mma_shape[0], Self.mma_shape[1]
                    ](m_mma, n_mma)
                    var p_reg_tile = p_reg_vectorized.tile[1, 1](
                        (n_mma + i * num_n_mmas_per_bk) * Self.num_m_mmas
                        + m_mma,
                        0,
                    )
                    copy_local_to_shared[thread_layout=warp_layout](
                        p_smem_mma_tile.vectorize[
                            fragment_layout.shape[0].value(),
                            fragment_layout.shape[1].value(),
                        ](),
                        p_reg_tile,
                    )
