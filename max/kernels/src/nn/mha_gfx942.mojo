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
from sys import align_of, simd_width_of, size_of
from sys.info import _cdna_4_or_newer
from sys.intrinsics import readfirstlane

from algorithm.functional import unswitch
from gpu import WARP_SIZE, barrier, block_idx, lane_id, thread_idx
from gpu import warp_id as get_warp_id
from gpu.memory import AddressSpace
from gpu.sync import AMDScheduleBarrierMask, schedule_barrier
from layout import IntTuple, Layout, LayoutTensor
from layout._utils import idx2crd, make_amd_buffer_resource
from layout.element import Element
from layout.int_tuple import UNKNOWN_VALUE
from layout.layout import blocked_product
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    copy_dram_to_local,
    copy_local_to_dram,
    copy_local_to_shared,
)
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.swizzle import Swizzle
from layout.tensor_core import (
    TensorCore,
    TiledTensorCore,
    get_mma_shape,
    num_matrix_reg,
)
from memory import AddressSpace as BaseAddressSpace
from memory import bitcast, stack_allocation
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_utils import (
    MHAConfig,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from nn.softmax import _online_softmax_iter_for_mma_output, softmax

from utils import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf


@always_inline
fn get_fragment_layout[mma_shape: IndexList[3]]() -> Layout:
    return Layout.row_major(1, num_matrix_reg[mma_shape[0], mma_shape[1]]())


@always_inline
fn get_nested_fragment_layout[mma_shape: IndexList[3]]() -> Layout:
    return (
        Layout(
            IntTuple(1, IntTuple(4, 4)), IntTuple(1, IntTuple(1, 8))
        ) if mma_shape[0]
        == 32 else get_fragment_layout[mma_shape]()
    )


@always_inline
fn get_warp_layout[mma_shape: IndexList[3]]() -> Layout:
    return Layout.col_major(32, 2) if mma_shape[0] == 32 else Layout.col_major(
        16, 4
    )


@always_inline
fn get_warp_coords[BN: Int, WN: Int]() -> IndexList[2]:
    alias num_warps_n = BN // WN
    var warp_row = get_warp_id() // UInt(num_warps_n)
    var warp_col = get_warp_id() % UInt(num_warps_n)
    return IndexList[2](Int(warp_row), Int(warp_col))


alias LocalLayoutTensor[dtype: DType, layout: Layout] = LayoutTensor[
    dtype,
    layout,
    MutableAnyOrigin,
    address_space = AddressSpace.LOCAL,
]

alias SharedLayoutTensor[dtype: DType, layout: Layout] = LayoutTensor[
    dtype,
    layout,
    MutableAnyOrigin,
    address_space = AddressSpace.SHARED,
]


@always_inline("nodebug")
fn copy_local_to_dram2[
    dst_thread_layout: Layout,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](dst: LayoutTensor, src: LayoutTensor, dst_base: LayoutTensor):
    # TODO: use copy_local_to_dram instead. This is a hack for hackathon :|.

    var worker_idx = (
        thread_idx.x if thread_scope == ThreadScope.BLOCK else lane_id()
    )
    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    var offset = (Int(dst.ptr) - Int(dst_base.ptr)) // size_of[dst.dtype]()
    var buffer = make_amd_buffer_resource(dst_base)
    var dst_frag_offset = dst_fragments.distance(dst.ptr) + offset

    alias M = src.layout.shape[0].value()
    alias N = src.layout.shape[1].value()

    @parameter
    for n in range(N):

        @parameter
        for m in range(M):
            alias src_idx = 4 * n + 16 * m
            alias i = 4 * m + n

            alias dst_static_idx = dst_fragments.layout(i)
            var dst_idx = dst_frag_offset

            @parameter
            if dst_fragments.layout.all_dims_known():
                dst_idx += dst_static_idx
            else:
                dst_idx += dst_fragments.runtime_layout(i)

            var src_element = Element[index_type = src.linear_idx_type].load(
                src.ptr.offset(src_idx),
                src.runtime_element_layout,
            )

            alias element_stride = dst_fragments.element_layout.stride[
                1
            ].value()

            @parameter
            if element_stride == 1:
                buffer.store(
                    Int32(dst_idx),
                    src_element.element_data.cast[dst.dtype](),
                )
            else:

                @parameter
                for i in range(dst_fragments.element_layout.size()):
                    alias element_offset = dst_fragments.element_layout(i)
                    var src = src_element.element_data[i].cast[dst.dtype]()
                    buffer.store(
                        Int32(dst_idx + element_offset),
                        src,
                    )


@always_inline
fn convert_f32_to_bf16[dtype: DType](x: SIMD, out res: SIMD[dtype, x.size]):
    # CK uses truncation for f32 to bf16 conversion but it's not accurate,
    # we only use it when benchmarking against CK otherwise in practice
    # we use the accurate conversion.
    alias use_truncation = False

    @parameter
    if use_truncation:
        res = type_of(res)(from_bits=(x.to_bits() >> 16).cast[DType.uint16]())
    else:
        res = x.cast[dtype]()


trait AttentionConfig(ImplicitlyCopyable):
    @staticmethod
    @always_inline
    fn q_head_idx() -> UInt:
        ...

    @staticmethod
    @always_inline
    fn q_tile_idx() -> UInt:
        ...

    @staticmethod
    @always_inline
    fn kv_head_idx() -> UInt:
        ...

    @staticmethod
    @always_inline
    fn get_mma_shape() -> IndexList[3]:
        ...

    @staticmethod
    @always_inline
    fn get_q_offset[q_depth: UInt]() -> UInt32:
        ...

    @staticmethod
    @always_inline
    fn get_output_offset[output_depth: UInt]() -> UInt32:
        ...


@fieldwise_init
struct MHAAttentionConfig[token_gen: Bool, config: MHAConfig, group: Int](
    AttentionConfig
):
    @staticmethod
    @always_inline
    fn q_head_idx() -> UInt:
        @parameter
        if token_gen:
            alias mma_shape = Self.get_mma_shape()
            var group_idx = lane_id() % UInt(mma_shape[0])
            return block_idx.y * UInt(group) + UInt(group_idx)
        else:
            return block_idx.y

    @staticmethod
    @always_inline
    fn q_tile_idx() -> UInt:
        return block_idx.x if not Self.token_gen else 0

    @staticmethod
    @always_inline
    fn kv_head_idx() -> UInt:
        return block_idx.y if Self.token_gen else block_idx.y // UInt(
            Self.group
        )

    @staticmethod
    @always_inline
    fn get_mma_shape() -> IndexList[3]:
        var mma_shape = (
            IndexList[3](32, 32, 16) if (
                _cdna_4_or_newer()
                and config.depth != 64
                # will deal with 64 later
            ) else IndexList[3](32, 32, 8)
        ) if not token_gen else IndexList[3](16, 16, 16)
        return mma_shape

    @staticmethod
    @always_inline
    fn get_q_offset[q_depth: UInt]() -> UInt32:
        return q_depth * (
            (Self.kv_head_idx() * UInt(group) if token_gen else block_idx.y)
            + config.num_heads * Self.q_tile_idx() * config.block_m()
        )

    @staticmethod
    @always_inline
    fn get_output_offset[output_depth: UInt]() -> UInt32:
        return output_depth * (
            (Self.kv_head_idx() * UInt(group) if token_gen else block_idx.y)
            + config.num_heads * Self.q_tile_idx() * config.block_m()
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
            (config.btile_dim0 * config.btile_dim1) // UInt(Self.simd_width),
        )
        * UInt(Self.simd_width)
        // UInt(Self.smem_layout.stride[0].value()),
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


@always_inline
fn pad[dtype: DType, depth: Int, size: Int]() -> Int:
    alias simd_width = simd_width_of[dtype]()
    alias padding = 0 if depth == 64 else size // simd_width
    return size + padding


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
                        warp_id // 2,
                        0,
                    )
                    .tile[8, depth](i, 0)
                    .tile[4, Self.depth_tile_size](warp_id % 2, depth_idx)
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
        var lane_coords = idx2crd[Layout.col_major(16, 4)](lane_id())
        var lane_row = lane_coords[0]
        var lane_col = lane_coords[1]

        var smem_iter_tensor = self.smem_iter.next_unsafe(0)[]
        var load_tile = self.load_tile.split[Self.num_stages]()[tile_id]

        @parameter
        for depth_idx in range(depth // Self.depth_tile_size):
            var smem_warp_tile = smem_iter_tensor.tile[
                Self.pad[depth](),
                Self.simd_width,
            ](0, warp_id).tile[
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
                    0, col_idx + UInt(k_mma * 2)
                )
                .vectorize[1, Self.simd_width]()
                .tile[Self.pad[Self.MMA_M](), 1](depth_idx, 0)
                .tile[Self.pad[Self.simd_width](), 1](
                    lane // UInt(Self.simd_width), 0
                )
                .slice[: Self.simd_width, :]()
                .tile[1, 1](lane % UInt(Self.simd_width), 0)
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
                i + warp_col * UInt(Self.WN // Self.BK)
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


@always_inline
fn _mask_apply[
    masked: Bool,
    accum_type: DType,
    token_gen: Bool,
    mma_shape: IndexList[3],
    num_m_mmas: Int,
    num_n_mmas: Int,
    mask_t: MHAMask,
    group: Int,
    fragment_layout: Layout,
    warp_layout: Layout,
    use_exp2: Bool = False,
](
    kv_tile_start_row: UInt32,
    kv_tile_num_rows: UInt32,
    start_pos: UInt32,
    seq_len: UInt32,
    num_keys: UInt32,
    mask_block_row: UInt32,
    mask_warp_row: UInt32,
    mask_warp_col: UInt32,
    scale: Float32,
    mask: mask_t,
    p_reg_vectorized: LayoutTensor[accum_type, **_],
    not_last_iter: Bool,
    cache_start_pos: UInt32 = 0,
):
    alias output_frag_size = fragment_layout.size()

    alias rowwise_stride = fragment_layout.shape[0].value()
    alias colwise_stride = fragment_layout.shape[1].value()
    alias frag_is_row_vector = rowwise_stride == 1
    constrained[
        frag_is_row_vector,
        "fragment layout is not a row vector",
    ]()

    var lane = lane_id()
    var scale_log2e: Scalar[accum_type] = scale.cast[accum_type]() * (
        log2e if use_exp2
        and not mask_t.apply_log2e_after_mask else Scalar[accum_type](1)
    )

    var coords = idx2crd[warp_layout](lane)
    var lane_row = coords[0] * rowwise_stride
    var lane_col = coords[1] * colwise_stride

    @parameter
    if token_gen:
        if lane_row >= group:
            return

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            alias mma_id = n_mma * num_m_mmas + m_mma
            p_reg_vectorized[mma_id, 0] = (
                p_reg_vectorized[mma_id, 0] * scale_log2e
            )
            # Coordinates in mask for current mma tile.
            var mask_frag_row = mask_warp_row + m_mma * mma_shape[0]
            var mask_frag_col = (
                mask_warp_col
                + n_mma * mma_shape[1]
                + (kv_tile_start_row if token_gen else 0)
            )
            mask_frag_row += lane_row
            mask_frag_col += lane_col
            # The row in score matrix of shape seq_len x num_keys.
            # Mask col is score col since we don't partition in col.
            var score_row = (
                num_keys - 1
            ) if token_gen else mask_block_row + mask_frag_row
            var score_col = mask_frag_col
            var score_col_with_cache_start_pos = score_col + cache_start_pos
            var score_row_with_start_pos = score_row + start_pos

            @parameter
            if masked:

                @parameter
                for j in range(output_frag_size):
                    alias fragment_col = fragment_layout(j)
                    var group_idx = lane_row
                    var q_head_idx = (
                        block_idx.y * UInt(group) + UInt(group_idx)
                    ) if token_gen else block_idx.y
                    p_reg_vectorized[mma_id, 0][j] = mask.mask(
                        IndexList[4, element_type = DType.uint32](
                            Int(block_idx.z),
                            Int(q_head_idx),
                            Int(score_row_with_start_pos),
                            Int(score_col_with_cache_start_pos + fragment_col),
                        ),
                        p_reg_vectorized[mma_id, 0][j],
                    )

            @parameter
            if mask_t.apply_log2e_after_mask:
                p_reg_vectorized[mma_id, 0] = (
                    p_reg_vectorized[mma_id, 0] * log2e
                )

            if (not not_last_iter or token_gen) and mask_t.mask_out_of_bound:
                var bound_y = (
                    kv_tile_start_row
                    + kv_tile_num_rows if token_gen else num_keys
                )

                @parameter
                for j in range(output_frag_size):
                    alias fragment_col = fragment_layout(j)

                    var bound_x = num_keys if token_gen else seq_len

                    p_reg_vectorized[mma_id, 0][j] = _kernel_mask(
                        IndexList[2, element_type = DType.uint32](
                            Int(score_row),
                            Int(score_col + fragment_col),
                        ),
                        IndexList[2, element_type = DType.uint32](
                            Int(bound_x), Int(bound_y)
                        ),
                        p_reg_vectorized[mma_id, 0][j],
                    )


struct SharedMemoryManager[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    num_rowwise_warps: Int,
    token_gen: Bool,
    depth_v: Int = depth,
](Defaultable):
    var p_smem: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]
    # p_smem is used for p
    var k_v_smem: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]
    # k_v_smem is used for k, v, and scratch
    alias alignment = align_of[SIMD[dtype, simd_width_of[dtype]()]]()
    alias accum_type = get_accum_type[dtype]()
    alias p_smem_size = BM * BN if token_gen else 0
    alias simd_width = simd_width_of[dtype]()
    # depth // simd_width is the padding

    alias k_smem_size = BN * BK
    alias v_smem_size = BK * pad[dtype, depth, depth]()
    alias k_v_smem_size = max(Self.k_smem_size, Self.v_smem_size)

    @always_inline
    fn __init__(out self):
        self.p_smem = stack_allocation[
            Self.p_smem_size,
            dtype,
            address_space = AddressSpace.SHARED,
            alignment = Self.alignment,
        ]()
        self.k_v_smem = stack_allocation[
            Self.k_v_smem_size,
            dtype,
            address_space = AddressSpace.SHARED,
            alignment = Self.alignment,
        ]()

    @always_inline
    fn get_k_ptr[
        _dtype: DType
    ](
        self,
    ) -> UnsafePointer[
        Scalar[_dtype],
        address_space = AddressSpace.SHARED,
    ]:
        return self.k_v_smem.bitcast[Scalar[_dtype]]()

    @always_inline
    fn get_v_ptr[
        _dtype: DType
    ](
        self,
    ) -> UnsafePointer[
        Scalar[_dtype],
        address_space = AddressSpace.SHARED,
    ]:
        return self.get_k_ptr[_dtype]()

    @always_inline
    fn get_p_ptr(
        self,
    ) -> UnsafePointer[Scalar[dtype], address_space = AddressSpace.SHARED,]:
        return self.p_smem.bitcast[Scalar[dtype]]()

    @always_inline
    fn get_warp_scratch_ptr(
        self,
    ) -> UnsafePointer[
        Scalar[Self.accum_type],
        address_space = AddressSpace.SHARED,
    ]:
        return self.k_v_smem.bitcast[
            Scalar[Self.accum_type]
        ]() if token_gen else {}


struct GlobalMemoryManager[
    dtype: DType,
    BM: UInt32,
    BN: UInt32,
    BK: UInt32,
    depth: UInt32,
    num_heads: UInt32,
    group: UInt32,
    token_gen: Bool,
    q_depth: UInt32 = depth,
    output_depth: UInt32 = depth,
]:
    alias kv_num_heads = num_heads // group
    # BHSD layout for q and kv cache
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(q_depth)),
        IntTuple(Int(num_heads * q_depth), 1),
    ) if not token_gen else Layout.row_major(Int(BM), Int(q_depth))

    alias output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(output_depth)),
        IntTuple(Int(num_heads * output_depth), 1),
    ) if not token_gen else Layout.row_major(Int(BM), Int(output_depth))

    alias kv_gmem_layout = Layout(
        IntTuple(Int(BN), Int(depth)),
        IntTuple(Int(Self.kv_num_heads * depth), 1),
    )

    var q_offset: UInt32
    var q_runtime_layout: RuntimeLayout[
        Self.q_gmem_layout,
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    var output_offset: UInt32
    var output_runtime_layout: RuntimeLayout[
        Self.output_gmem_layout,
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    @always_inline
    fn __init__(
        out self,
        q_tile_idx: UInt32,
        kv_head_idx: UInt32,
        seq_len: Int,
        q_offset: UInt32,
        output_offset: UInt32,
    ):
        var q_tile_num_rows = min(
            BM, UInt(seq_len) - q_tile_idx * BM
        ) if not token_gen else group

        self.q_offset = q_offset
        self.output_offset = output_offset

        self.q_runtime_layout = type_of(self.q_runtime_layout)(
            {Int(q_tile_num_rows), Int(q_depth)},
            {Int(num_heads * q_depth if not token_gen else q_depth), 1},
        )

        self.output_runtime_layout = type_of(self.output_runtime_layout)(
            {Int(q_tile_num_rows), Int(output_depth)},
            {
                Int(
                    num_heads * output_depth if not token_gen else output_depth
                ),
                1,
            },
        )

    @always_inline
    fn get_q_tensor[
        qtype: DType,
    ](
        self,
        ptr: UnsafePointer[Scalar[qtype]],
        out result: LayoutTensor[
            qtype,
            Self.q_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        return {ptr + Int(self.q_offset), self.q_runtime_layout}

    @always_inline
    fn get_output_tensor[
        out_type: DType,
    ](
        self,
        ptr: UnsafePointer[Scalar[out_type]],
        out result: LayoutTensor[
            out_type,
            Self.output_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        return {ptr + Int(self.output_offset), self.output_runtime_layout}

    @always_inline
    fn get_kv_tensor[
        kvtype: DType, //,
    ](
        self,
        ptr: UnsafePointer[Scalar[kvtype], **_],
        kv_tile_num_rows: UInt32,
        out result: LayoutTensor[
            kvtype,
            Self.kv_gmem_layout,
            ptr.origin,
            masked=True,
            address_space = ptr.address_space,
        ],
    ):
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = type_of(result.runtime_layout)(
            type_of(result.runtime_layout.shape)(
                Int(kv_tile_num_rows), Int(depth)
            ),
            type_of(result.runtime_layout.stride)(
                Int(Self.kv_num_heads * depth), 1
            ),
        )

        return {ptr, kv_runtime_layout}


@always_inline
fn mma[
    c_register_buffer_type: RegisterBuffer,
    a_register_buffer_type: RegisterMMABuffer,
    b_buffer_type: KVBuffer, //,
    tensor_core_mma: TiledTensorCore,
    BK: Int,
    prefetch_function: OptionalReg[fn () capturing -> None],
    swap_a_b: Bool = False,
    beg_iter: Int = 0,
    num_iters: Int = 1,
    prefetched_b_tile: Bool = False,
](
    c: c_register_buffer_type,
    mut a_tile: a_register_buffer_type,
    mut b_tile: b_buffer_type,
):
    constrained[b_buffer_type._num_stages == 2, "b_tile.num_stages must be 2"]()
    alias num_k_mmas2 = ceildiv(
        BK, UInt(tensor_core_mma.shape[2] * tensor_core_mma.group_size)
    )

    @parameter
    if not prefetched_b_tile:
        b_tile.load_from_dram()

    @parameter
    for i in range(beg_iter, beg_iter + num_iters):

        @parameter
        if i < beg_iter + num_iters - 1:
            b_tile.load_from_dram()

            @parameter
            if i == beg_iter + num_iters - 2:

                @parameter
                if prefetch_function:
                    alias prefetch_func = prefetch_function.value()
                    prefetch_func()

        b_tile.copy_to_shared[i % 2]()

        barrier()

        @parameter
        for k_mma in range(num_k_mmas2):
            var a_reg_tile = a_tile.get_mma_tile[i, k_mma]()

            b_tile.load_from_shared[k_mma,]()

            tensor_core_mma.mma[swap_a_b=swap_a_b](
                a_reg_tile, b_tile.get_mma_tile(), c.get_reg_tile()
            )

        barrier()


struct Attention[
    attention_config_t: AttentionConfig,
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask, //,
    config: MHAConfig,
    group: Int,
    token_gen: Bool,
    sink: Bool,
    q_depth: Int = config.depth,
    cache_depth: Int = config.depth,
    output_depth: Int = config.depth,
]:
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()
    alias WM = config.warp_m()
    alias WN = config.warp_n()
    alias num_threads = config.num_threads()
    alias num_heads = config.num_heads
    alias num_warps_n = Self.BN // Self.WN
    alias num_warps_m = Self.BM // Self.WM
    alias depth = config.depth
    alias accum_type = get_accum_type[q_type]()

    alias mma_shape = Self.attention_config_t.get_mma_shape()

    alias fragment_layout = get_fragment_layout[Self.mma_shape]()
    alias output_frag_size = Self.fragment_layout.size()
    alias fragment_layout_nested = get_nested_fragment_layout[Self.mma_shape]()

    alias num_m_mmas = ceildiv(Self.WM, UInt(Self.mma_shape[0]))
    alias num_n_mmas = ceildiv(Self.WN, UInt(Self.mma_shape[1]))
    alias num_n_mmas_output = ceildiv(
        Self.output_depth // Self.num_warps_n, UInt(Self.mma_shape[1])
    )

    alias swap_a_b = True
    alias use_exp2 = True
    alias k_group_size = 16 // Self.mma_shape[2] if not token_gen else 2
    alias num_k_mmas2 = ceildiv(
        Self.BK, UInt(Self.mma_shape[2] * Self.k_group_size)
    )

    alias warp_layout = get_warp_layout[Self.mma_shape]()

    alias num_stages = 2

    alias OutputRegisterBufferType = OutputRegisterBuffer[
        Self.accum_type,
        Self.num_m_mmas,
        Self.num_n_mmas_output,
        Self.output_frag_size,
    ]

    alias PRegisterBufferType = PRegisterBuffer[
        Self.accum_type,
        q_type,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.WM,
        Self.WN,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.output_frag_size,
        Self.token_gen,
        Self.mma_shape,
        Self.k_group_size,
    ]

    alias row_layout = Layout.row_major(
        Self.num_m_mmas, Self.fragment_layout.shape[0].value()
    )

    alias RowMaxTensorType = LocalLayoutTensor[
        Self.accum_type,
        Self.row_layout,
    ]

    alias RowSumTensorType = Self.RowMaxTensorType

    alias GlobalMemoryManagerType = GlobalMemoryManager[
        q_type,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.depth,
        Self.num_heads,
        Self.group,
        Self.token_gen,
        Self.q_depth,
        Self.output_depth,
    ]

    alias SharedMemoryManagerType = SharedMemoryManager[
        q_type,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.depth,
        Self.num_warps_n,
        Self.token_gen,
        Self.output_depth,
    ]

    alias QRegisterBufferType = QRegisterBuffer[
        dtype = Self.q_type,
        mma_shape = Self.mma_shape,
        k_group_size = Self.k_group_size,
        WM = Self.WM,
        WN = Self.WN,
        BN = Self.BN,
        BK = Self.BK,
        depth = Self.q_depth,
        thread_layout = Self.warp_layout,
    ]

    var out_reg_buffer: Self.OutputRegisterBufferType
    var p_reg_buffer: Self.PRegisterBufferType

    var rowmax: Self.RowMaxTensorType
    var rowsum: Self.RowSumTensorType

    var gmem_manager: Self.GlobalMemoryManagerType
    var smem_manager: Self.SharedMemoryManagerType

    var q_buffer: Self.QRegisterBufferType
    var output_ptr: UnsafePointer[Scalar[Self.output_type],]

    var batch_idx: Int

    var k: k_t
    var v: v_t
    var mask: mask_t

    var mask_block_row: UInt32
    var mask_warp_row: UInt32
    var mask_warp_col: UInt32

    var scale: Float32

    var seq_len: Int
    var num_keys: Int
    var start_pos: Int
    var cache_start_pos: Int

    var warp_scratch_tensor: SharedLayoutTensor[
        Self.accum_type,
        Layout.row_major(2 * Self.num_warps_n, Self.BM),
    ]

    @staticmethod
    @always_inline
    fn q_head_idx() -> UInt:
        return Self.attention_config_t.q_head_idx()

    @staticmethod
    @always_inline
    fn q_tile_idx() -> UInt:
        return Self.attention_config_t.q_tile_idx()

    @staticmethod
    @always_inline
    fn kv_head_idx() -> UInt:
        return Self.attention_config_t.kv_head_idx()

    @always_inline
    fn zero_p_buffer(self):
        self.p_reg_buffer.zero()

    @always_inline
    fn get_batch_idx(self) -> Int:
        return self.batch_idx

    @staticmethod
    @always_inline
    fn get_tensor_core_mma_qk(
        out result: TiledTensorCore[
            get_accum_type[q_type](),
            q_type,
            Self.mma_shape,
            group_size = Self.k_group_size,
            transpose_b=True,
        ],
    ):
        return type_of(result)()

    @staticmethod
    @always_inline
    fn get_tensor_core_mma_pv(
        out result: TiledTensorCore[
            get_accum_type[q_type](),
            q_type,
            Self.mma_shape,
            group_size = Self.k_group_size,
            transpose_b=False,
        ],
    ):
        return type_of(result)()

    @always_inline
    fn mma_qk[
        k_buffer_type: KVBuffer, //,
        prefetch_function: OptionalReg[fn () capturing -> None] = None,
        beg_iter: Int = 0,
        num_iters: Int = Int(Self.depth // Self.BK),
        prefetched_b_tile: Bool = False,
    ](mut self, mut k_buffer: k_buffer_type):
        mma[
            tensor_core_mma = Self.get_tensor_core_mma_qk(),
            BK = Self.BK,
            prefetch_function=prefetch_function,
            swap_a_b = Self.swap_a_b,
            beg_iter=beg_iter,
            num_iters=num_iters,
            prefetched_b_tile=prefetched_b_tile,
        ](
            self.p_reg_buffer,
            self.q_buffer,
            k_buffer,
        )

    @always_inline
    fn mma_pv[
        v_buffer_type: KVBuffer, //,
        prefetch_function: OptionalReg[fn () capturing -> None] = None,
        prefetched_b_tile: Bool = True,
    ](mut self, mut v_buffer: v_buffer_type):
        mma[
            tensor_core_mma = Self.get_tensor_core_mma_pv(),
            BK = Self.BK,
            prefetch_function=prefetch_function,
            swap_a_b = Self.swap_a_b,
            num_iters = Int(Self.BN // Self.BK),
            prefetched_b_tile=prefetched_b_tile,
        ](
            self.out_reg_buffer,
            self.p_reg_buffer,
            v_buffer,
        )

    @always_inline
    fn mask_status(
        self,
        kv_tile_start_row: UInt32,
    ) -> TileMaskStatus:
        @parameter
        if token_gen:
            # Decoding with mask checking: check single token at num_keys-1
            return self.mask.status(
                Index[dtype = DType.uint32](
                    Int(self.num_keys - 1),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.uint32](Int(1), Int(Self.BN)),
            )
        else:
            # Prefill or decoding without mask checking: check full tile
            return self.mask.status(
                Index[dtype = DType.uint32](
                    Int(self.mask_block_row + self.start_pos),
                    Int(kv_tile_start_row + self.cache_start_pos),
                ),
                Index[dtype = DType.uint32](Int(Self.BM), Int(Self.BN)),
            )

    @always_inline
    fn mask_advance(mut self):
        @parameter
        if not token_gen:
            self.mask_warp_col += Self.BN

    @always_inline
    fn mask_skip_tile(self, status: TileMaskStatus) -> Bool:
        return status == TileMaskStatus.FULL_MASK

    @always_inline
    fn mask_skip_and_advance(
        mut self,
        kv_tile_start_row: UInt32,
    ) -> Bool:
        @parameter
        if not token_gen or mask_t.check_mask_during_decoding:
            var status = self.mask_status(
                kv_tile_start_row,
            )
            if self.mask_skip_tile(status):
                self.mask_advance()
                return True
        return False

    @always_inline
    fn mask_apply(
        mut self,
        kv_tile_start_row: UInt32,
        kv_tile_num_rows: UInt32,
        not_last_iter: Bool,
    ):
        @always_inline
        @parameter
        fn _mask_apply_impl[masked: Bool]():
            _mask_apply[
                masked=masked,
                accum_type = Self.accum_type,
                token_gen = Self.token_gen,
                mma_shape = Self.mma_shape,
                num_m_mmas = Self.num_m_mmas,
                num_n_mmas = Self.num_n_mmas,
                mask_t = Self.mask_t,
                group = Self.group,
                fragment_layout = Self.fragment_layout_nested,
                warp_layout = Self.warp_layout,
                use_exp2 = Self.use_exp2,
            ](
                kv_tile_start_row,
                kv_tile_num_rows,
                self.start_pos,
                self.seq_len,
                self.num_keys,
                Int(self.mask_block_row),
                Int(self.mask_warp_row),
                self.mask_warp_col,
                self.scale,
                self.mask,
                self.p_reg_buffer.vectorize(),
                not_last_iter,
                self.cache_start_pos,
            )

        @parameter
        if not Self.token_gen or Self.mask_t.check_mask_during_decoding:
            var mask_status = self.mask_status(
                kv_tile_start_row,
            )
            unswitch[_mask_apply_impl](
                mask_status == TileMaskStatus.PARTIAL_MASK
            )
        else:
            _mask_apply_impl[masked=True]()
        self.mask_advance()

    @always_inline
    fn __init__(
        out self,
        attention_config: attention_config_t,
        output_ptr: UnsafePointer[Scalar[Self.output_type],],
        q: UnsafePointer[Scalar[Self.q_type]],
        k: k_t,
        v: v_t,
        mask: mask_t,
        sink_weights: OptionalReg[
            LayoutTensor[
                Self.q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin
            ]
        ],
        batch_idx: Int,
        scale: Float32,
        seq_len: Int,
        num_keys: Int,
        start_pos: Int,
        cache_start_pos: Int = 0,
    ):
        self.rowmax = Self.RowMaxTensorType.stack_allocation()
        self.rowsum = Self.RowSumTensorType.stack_allocation()
        self.out_reg_buffer = Self.OutputRegisterBufferType()
        self.out_reg_buffer.zero()

        self.gmem_manager = Self.GlobalMemoryManagerType(
            Self.q_tile_idx(),
            Self.kv_head_idx(),
            seq_len,
            Self.attention_config_t.get_q_offset[UInt(q_depth)](),
            Self.attention_config_t.get_output_offset[UInt(output_depth)](),
        )
        self.smem_manager = Self.SharedMemoryManagerType()

        self.warp_scratch_tensor = type_of(self.warp_scratch_tensor)(
            self.smem_manager.get_warp_scratch_ptr()
        )

        self.p_reg_buffer = Self.PRegisterBufferType(
            self.smem_manager.get_p_ptr()
        )

        var q_tile = self.gmem_manager.get_q_tensor(q)
        self.q_buffer = Self.QRegisterBufferType(q_tile)

        self.output_ptr = output_ptr

        self.k = k
        self.v = v
        self.mask = mask

        self.mask_block_row = self.q_tile_idx() * Self.BM
        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]
        self.mask_warp_row = warp_row * Self.WM
        self.mask_warp_col = warp_col * Self.WN

        self.batch_idx = batch_idx
        self.scale = scale

        self.seq_len = seq_len
        self.num_keys = num_keys
        self.start_pos = start_pos
        self.cache_start_pos = cache_start_pos

        @parameter
        if sink:
            debug_assert(
                Bool(sink_weights),
                "expect sink_weights to be non-null when sink=true",
            )
            var sink_weight = (
                sink_weights.value()[Int(self.q_head_idx())][0].cast[
                    Self.accum_type
                ]()
                * log2e
            )
            self.rowmax = self.rowmax.fill(sink_weight)
            self.rowsum = self.rowsum.fill(1)
        else:
            self.rowmax = self.rowmax.fill(min_or_neg_inf[Self.accum_type]())
            self.rowsum = self.rowsum.fill(0)

    @always_inline
    fn online_softmax(self):
        var warp_scratch = self.warp_scratch_tensor
        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]

        _online_softmax_iter_for_mma_output[
            Self.accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(Self.num_m_mmas, Self.num_n_mmas),
            # threads layout by warp
            Layout.row_major(Self.num_warps_m, Self.num_warps_n),
            Self.warp_layout,
            use_exp2 = Self.use_exp2,
            fragment_layout = Self.fragment_layout,
        ](
            self.out_reg_buffer.vectorize(),
            self.p_reg_buffer.vectorize(),
            warp_scratch.tile[2 * Self.num_warps_n, Self.WM](0, Int(warp_row)),
            self.rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            self.rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

    @always_inline
    fn store_output(self):
        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]
        var output_tile = self.gmem_manager.get_output_tensor(self.output_ptr)
        var output_warp_tile = output_tile.tile[
            Self.WM, Self.output_depth // Self.num_warps_n
        ](warp_row, warp_col)

        @parameter
        if Self.mma_shape[0] == 32:
            copy_local_to_dram2[
                dst_thread_layout = Self.warp_layout,
                thread_scope = ThreadScope.WARP,
            ](
                output_warp_tile.vectorize[
                    1,
                    4,
                ](),
                self.out_reg_buffer.reg_tile.vectorize[1, 4](),
                output_tile,
            )
        else:
            copy_local_to_dram[
                dst_thread_layout = Self.warp_layout,
                thread_scope = ThreadScope.WARP,
            ](
                output_warp_tile.vectorize[
                    Self.fragment_layout.shape[0].value(),
                    Self.fragment_layout.shape[1].value(),
                ](),
                self.out_reg_buffer.vectorize(),
                output_tile,
            )

    @always_inline
    fn copy_fragment_to_smem(self):
        @parameter
        if not Self.token_gen:
            return

        self.p_reg_buffer.copy_to_shared()

    @always_inline
    fn store_partition_info(
        self,
        num_partitions: Int,
        exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[Self.q_type]()]],
        qk_max_ptr: UnsafePointer[Scalar[get_accum_type[Self.q_type]()]],
    ):
        @parameter
        if not Self.token_gen:
            return

        var q_head_idx = self.q_head_idx()
        if num_partitions > 1:
            if thread_idx.x < UInt(group):
                var row_sum = self.rowsum[0, 0][0]
                var row_max = self.rowmax[0, 0][0]

                exp_sum_ptr[q_head_idx] = row_sum
                qk_max_ptr[q_head_idx] = row_max

    @always_inline
    fn prefill(
        mut self,
    ):
        constrained[Self.BK == 32, "BK must be 32"]()

        @always_inline
        @parameter
        fn loop_over_kvcache[
            tile_size: Int
        ](kv_tile_start_row: UInt32, end: UInt32, not_last_iter: Bool):
            if self.mask_skip_and_advance(
                kv_tile_start_row,
            ):
                return

            var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

            var k_tile = self.gmem_manager.get_kv_tensor(
                self.k.block_paged_ptr[Self.BN](
                    self.get_batch_idx(),
                    kv_tile_start_row,
                    Self.kv_head_idx(),
                    0,
                ),
                kv_tile_num_rows,
            )

            var v_tile = self.gmem_manager.get_kv_tensor(
                self.v.block_paged_ptr[Self.BN](
                    self.get_batch_idx(),
                    kv_tile_start_row,
                    Self.kv_head_idx(),
                    0,
                ),
                kv_tile_num_rows,
            )

            self.zero_p_buffer()

            var num_b_rows = Int(kv_tile_num_rows)

            var k_buffer = KBuffer[
                tensor_core_mma = Self.get_tensor_core_mma_qk(),
                swizzle=None,
                BN = Self.BN,
                WN = Self.WN,
                BK = Self.BK,
                depth = Self.depth,
                num_threads = Self.num_threads,
                num_stages = Self.num_stages,
            ](
                k_tile,
                num_b_rows,
                self.smem_manager.get_k_ptr[k_tile.dtype](),
            )

            var v_buffer = VBufferTransposeLoads[
                tensor_core_mma = Self.get_tensor_core_mma_pv(),
                BN = Self.BN,
                BK = Self.BK,
                depth = Self.depth,
                num_threads = Self.num_threads,
                num_stages = Self.num_stages,
            ](v_tile, self.smem_manager.get_v_ptr[v_tile.dtype]())

            @parameter
            @always_inline
            fn prefetch_function():
                v_buffer.load_from_dram()

            self.mma_qk[prefetch_function=prefetch_function](k_buffer)

            self.mask_apply(
                kv_tile_start_row,
                kv_tile_num_rows,
                not_last_iter,
            )
            # don't know why we need this barrier but i get random failures without it
            barrier()
            self.online_softmax()
            barrier()

            self.mma_pv(v_buffer)

        for i in range(UInt32(0), UInt32(self.num_keys), UInt32(Self.BN)):
            var end = min(i + Self.BN, self.num_keys)
            loop_over_kvcache[Self.BN](i, end, end != self.num_keys)

        self.out_reg_buffer.apply_softmax_denominator(self.rowsum)

        self.store_output()

    @always_inline
    fn decoding(
        mut self,
        exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[Self.q_type]()]],
        qk_max_ptr: UnsafePointer[Scalar[get_accum_type[Self.q_type]()]],
        num_partitions: Int,
    ):
        constrained[Self.BK == 32, "BK must be 32"]()

        @always_inline
        @parameter
        fn loop_over_kvcache[
            tile_size: Int
        ](kv_tile_start_row: Int, end: Int, not_last_iter: Bool):
            if self.mask_skip_and_advance(
                kv_tile_start_row,
            ):
                return

            var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

            var k_tile = self.gmem_manager.get_kv_tensor(
                self.k.block_paged_ptr[Self.BN](
                    self.get_batch_idx(),
                    kv_tile_start_row,
                    self.kv_head_idx(),
                    0,
                ),
                kv_tile_num_rows,
            )

            var v_tile = self.gmem_manager.get_kv_tensor(
                self.v.block_paged_ptr[Self.BN](
                    self.get_batch_idx(),
                    kv_tile_start_row,
                    self.kv_head_idx(),
                    0,
                ),
                kv_tile_num_rows,
            )

            self.zero_p_buffer()

            alias swizzle = Swizzle(2, 0, 2)

            var num_b_rows = OptionalReg[Int](
                kv_tile_num_rows
            ) if not not_last_iter else None

            var k_buffer = KBuffer[
                tensor_core_mma = Self.get_tensor_core_mma_qk(),
                swizzle=swizzle,
                BN = Self.BN,
                WN = Self.WN,
                BK = Self.BK,
                depth = Self.depth,
                num_threads = Self.num_threads,
                num_stages = Self.num_stages,
                token_gen = Self.token_gen,
            ](
                k_tile,
                num_b_rows,
                self.smem_manager.get_k_ptr[k_tile.dtype](),
            )
            var v_tile_slice = v_tile.slice[:, : Int(Self.output_depth)]()
            var v_buffer = VBuffer[
                tensor_core_mma = Self.get_tensor_core_mma_pv(),
                swizzle=None,
                BN = Self.BN,
                WN = Self.WN,
                BK = Self.BK,
                depth = Self.output_depth,
                num_threads = Self.num_threads,
                num_stages = Self.num_stages,
                token_gen = Self.token_gen,
            ](
                v_tile_slice,
                num_b_rows,
                self.smem_manager.get_v_ptr[v_tile.dtype](),
            )

            @parameter
            @always_inline
            fn prefetch_function():
                v_buffer.load_from_dram()

            self.mma_qk[prefetch_function=prefetch_function](k_buffer)

            self.mask_apply(
                kv_tile_start_row,
                kv_tile_num_rows,
                not_last_iter,
            )

            # Not sure why we need this barrier here, but the code hangs without it
            barrier()

            self.online_softmax()

            # warp scratch and p_smem are using the same smem space
            barrier()

            self.copy_fragment_to_smem()

            barrier()

            self.mma_pv(v_buffer)
            # ensure that smem for v is not required anymore
            barrier()

        start, end = get_start_and_end_for_partitions[Self.BN](
            self.num_keys, num_partitions, block_idx.x
        )

        for i in range(start, end, Self.BN):
            var end_ = min(i + Self.BN, end)
            loop_over_kvcache[Self.BN](i, end_, end_ != end)

        # Apply softmax denominator.
        self.out_reg_buffer.apply_softmax_denominator(self.rowsum)
        self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
        self.store_output()
