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
from sys.intrinsics import readfirstlane
from sys.info import _cdna_4_or_newer
from buffer import NDBuffer

from algorithm.functional import unswitch
from gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
)
from gpu import warp_id as get_warp_id
from gpu.intrinsics import buffer_store
from gpu.memory import AddressSpace
from gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
)
from memory import AddressSpace as BaseAddressSpace
from layout import IntTuple, Layout, LayoutTensor
from layout.layout import blocked_product
from layout._utils import get_amd_buffer_descriptor, idx2crd, TensorCoreKGroup
from layout.element import Element
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    copy_local_to_shared,
    copy_dram_to_local,
    copy_local_to_dram,
)
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.swizzle import Swizzle
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import TensorCore, get_mma_shape, num_matrix_reg
from memory import bitcast, stack_allocation
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_utils import (
    MHAConfig,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from nn.softmax import (
    _online_softmax_iter_for_mma_output,
    softmax,
)

from utils import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf


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
    var descriptor = get_amd_buffer_descriptor(dst_base)
    var dst_frag_offset = dst_fragments.distance(dst.ptr) + offset
    alias num_stores_per_thread = dst_fragments.layout.size()

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
                buffer_store(
                    descriptor,
                    Int32(dst_idx),
                    src_element.element_data.cast[dst.dtype](),
                )
            else:

                @parameter
                for i in range(dst_fragments.element_layout.size()):
                    alias element_offset = dst_fragments.element_layout(i)
                    var src = src_element.element_data[i].cast[dst.dtype]()
                    buffer_store(
                        descriptor,
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
        res = __type_of(res)(from_bits=(x.to_bits() >> 16).cast[DType.uint16]())
    else:
        res = x.cast[dtype]()


struct KBuffer[
    dtype: DType,
    layout: Layout,
    address_space: BaseAddressSpace,
    alignment: Int,
    origin: Origin,
    masked: Bool, //,
    mma_shape: IndexList[3],
    k_group_size: Int,
    swizzle: OptionalReg[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    num_threads: Int,
]:
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_mmas = ceildiv(WN, Self.MMA_N)
    alias num_k_tiles = ceildiv(BK, Self.MMA_K * k_group_size)
    alias simd_width = simd_width_of[dtype]()

    alias num_repeats = BK // Self.simd_width

    # Shared memory layout
    # Layout construction for standard memory access:
    # - base_layout: Layout.row_major(BN, simd_width) -> BN×simd_width tiles
    # - tiler_layout: Layout.row_major(1, num_repeats) -> repeat tiles num_repeats times horizontally
    # - smem_layout: blocked_product(base_layout, tiler_layout) -> tiled blocked layout
    #
    # Resulting shape: BN×(simd_width × num_repeats) = BN×BK tensor
    # Where BK = simd_width × num_repeats, typically simd_width=8, num_repeats=BK/8
    #
    # This creates num_repeats blocks of BN×simd_width arranged horizontally:
    # Within each simd_width-column block, elements are consecutive (stride 1)
    # Between blocks: stride = BN × simd_width
    #
    # ASCII diagram for BN=128, simd_width=8, BK=32 (showing first 2 of 4 blocks):
    # ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    # │        Block 0 (128×8)                     │        Block 1 (128×8)                     │     ... 2 more blocks           │
    # ├────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
    # │   0    1    2    3    4    5    6    7     │ 1024 1025 1026 1027 1028 1029 1030 1031    │ (Block 2: 2048-3071)            │
    # │   8    9   10   11   12   13   14   15     │ 1032 1033 1034 1035 1036 1037 1038 1039    │ (Block 3: 3072-4095)            │
    # │  16   17   18   19   20   21   22   23     │ 1040 1041 1042 1043 1044 1045 1046 1047    │                                 │
    # │  24   25   26   27   28   29   30   31     │ 1048 1049 1050 1051 1052 1053 1054 1055    │                                 │
    # │ ...                                        │  ...                                       │                                 │
    # │1016 1017 1018 1019 1020 1021 1022 1023     │ 2040 2041 2042 2043 2044 2045 2046 2047    │                                 │
    # └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    # stride between blocks = BN × simd_width = 128 × 8 = 1024

    alias base_layout = Layout.row_major(BN, Self.simd_width)
    alias tiler_layout = Layout.row_major(1, Self.num_repeats)
    alias smem_layout = blocked_product(
        Self.base_layout,
        Self.tiler_layout,
        coalesce_output=True,
    )

    alias thread_layout = Layout.row_major(num_threads // 4, 4)

    alias LoadTileType = LayoutTensor[
        dtype,
        Layout.row_major(
            Self.num_mmas * Self.num_k_tiles,
            Self.simd_width,
        ),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var load_tile: Self.LoadTileType

    alias MMATileType = LayoutTensor[
        dtype,
        Layout.row_major(Self.num_mmas, Self.simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var mma_tile: Self.MMATileType

    alias wtile_dim0 = WN
    alias wtile_dim1 = BK

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

    alias GlobalTensorType = LayoutTensor[
        dtype,
        layout,
        origin,
        address_space=address_space,
        alignment=alignment,
        masked=masked,
    ]

    alias GlobalTiledIteratorType = Self.GlobalTensorType.TiledIteratorType[
        BN,
        BK,
        axis=1,
    ]

    var global_iterator: Self.GlobalTiledIteratorType

    @always_inline
    fn __init__(
        out self,
        global_tile: LayoutTensor[
            dtype,
            layout,
            origin,
            address_space=address_space,
            alignment=alignment,
            masked=masked,
        ],
        num_b_rows: OptionalReg[Int],
        shared_ptr: UnsafePointer[
            Scalar[dtype],
            address_space = AddressSpace.SHARED, **_,
        ],
    ):
        constrained[
            mma_shape[2] * k_group_size == 16,
            "mma_shape[2] * k_group_size must be 16",
        ]()
        self.load_tile = __type_of(self.load_tile).stack_allocation()
        self.mma_tile = __type_of(self.mma_tile).stack_allocation()
        self.smem_iter = __type_of(self.smem_iter)(shared_ptr, 0)
        alias stride = Self.GlobalTiledIteratorType.layout.stride[0].value()
        self.bounds = num_b_rows.value() * stride if num_b_rows else Int.MAX
        self.global_iterator = global_tile.tiled_iterator[
            BN,
            BK,
            axis=1,
        ](0, 0)

    @always_inline
    fn load_from_dram(
        mut self,
    ):
        copy_dram_to_local[src_thread_layout = Self.thread_layout,](
            self.load_tile.vectorize[1, Self.simd_width](),
            self.global_iterator,
            self.bounds,
        )
        self.global_iterator._incr()

    @always_inline
    fn get_mma_tile(self) -> Self.MMATileType:
        return self.mma_tile

    @always_inline
    fn copy_to_shared(
        self,
    ):
        var smem_tile = self.smem_iter.next_unsafe(0)[]
        copy_local_to_shared[
            thread_layout = Self.thread_layout, swizzle=swizzle, row_major=True
        ](
            smem_tile.vectorize[1, Self.simd_width](),
            self.load_tile.vectorize[1, Self.simd_width](),
        )

    @always_inline
    fn load_from_shared[
        accum_type: DType,
        mma_input_type: DType,
        mma_shape: IndexList[3],
        k_group_size: Int,
        transpose_b: Bool,
        k_mma: Int,
    ](self):
        alias num_warps_n = BN // WN
        var warp_col = get_warp_id() % num_warps_n
        var smem_tile = self.smem_iter.next_unsafe(0)[]

        var wtile_coord0 = Int(warp_col)
        var wtile_coord1 = 0
        var warp_tile = smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
            wtile_coord0, wtile_coord1
        )
        alias tensor_core_mma = TensorCoreKGroup[
            accum_type,
            mma_input_type,
            mma_shape,
            k_group_size=k_group_size,
            transpose_b=transpose_b,
        ]()
        tensor_core_mma.mma_op.load_b[swizzle=swizzle](
            warp_tile,
            self.get_mma_tile().vectorize[1, Self.simd_width](),
            UInt(k_mma),
        )


@always_inline
fn pad[dtype: DType, depth: Int, size: Int]() -> Int:
    alias simd_width = simd_width_of[dtype]()
    alias padding = 0 if depth == 64 else size // simd_width
    return size + padding


struct VBuffer[
    dtype: DType,
    layout: Layout,
    address_space: BaseAddressSpace,
    alignment: Int,
    origin: Origin,
    masked: Bool, //,
    mma_shape: IndexList[3],
    k_group_size: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
]:
    alias simd_width = simd_width_of[dtype]()
    alias num_repeats = BK // Self.simd_width

    # V Buffer shared memory layout
    # - base_layout: Layout.row_major(depth + padding, simd_width) -> (depth+padding)×simd_width tiles
    # - tiler_layout: Layout.row_major(1, num_repeats) -> repeat tiles num_repeats times horizontally
    # - smem_layout: blocked_product(base_layout, tiler_layout) -> tiled blocked layout with padding
    #
    # Resulting shape: (depth + padding)×(simd_width × num_repeats) = (depth + depth//8)×BK tensor
    # Where padding = depth//8 helps avoid bank conflicts, BK = simd_width × num_repeats
    #
    # This creates num_repeats blocks of (depth+padding)×simd_width arranged horizontally:
    # Within each simd_width-column block, elements are consecutive (stride 1)
    # Between blocks: stride = (depth + padding) × simd_width
    #
    # ASCII diagram for depth=128, padding=16, simd_width=8, BK=32 (showing first 2 of 4 blocks):
    # ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    # │        Block 0 (144×8)                     │        Block 1 (144×8)                     │     ... 2 more blocks           │
    # ├────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
    # │   0    1    2    3    4    5    6    7     │ 1152 1153 1154 1155 1156 1157 1158 1159    │ (Block 2: 2304-3455)            │
    # │   8    9   10   11   12   13   14   15     │ 1160 1161 1162 1163 1164 1165 1166 1167    │ (Block 3: 3456-4607)            │
    # │  16   17   18   19   20   21   22   23     │ 1168 1169 1170 1171 1172 1173 1174 1175    │                                 │
    # │  24   25   26   27   28   29   30   31     │ 1176 1177 1178 1179 1180 1181 1182 1183    │                                 │
    # │ ...                                        │  ...                                       │                                 │
    # │1144 1145 1146 1147 1148 1149 1150 1151     │ 2296 2297 2298 2299 2300 2301 2302 2303    │                                 │
    # └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    # stride between blocks = (depth + padding) × simd_width = 144 × 8 = 1152

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

    alias MMA_M = mma_shape[0]
    alias MMA_K = mma_shape[2]
    alias num_k_tiles = ceildiv(BK, Self.MMA_K * k_group_size)
    alias num_depth_tiles = depth // Self.MMA_M

    alias depth_tile_size = min(depth, 128)

    # for depth = 64, we use 8B loads instead of 16B loads
    # this keeps the layout of the memory access the same but may not be optimal
    # can come back to this if perf becomes an issue
    alias load_width = 4 if depth == 64 else Self.simd_width
    alias loads_per_thread_per_depth_tile = (Self.depth_tile_size * BK) // (
        Self.load_width * Self.num_threads
    )

    alias LoadTileType = LayoutTensor[
        dtype,
        Layout.row_major(
            Self.loads_per_thread_per_depth_tile
            * (depth // Self.depth_tile_size),
            Self.load_width,
        ),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    var load_tile: Self.LoadTileType

    alias MMATileType = LayoutTensor[
        dtype,
        Layout.row_major(
            depth // Self.MMA_M * Self.num_k_tiles, Self.simd_width
        ),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
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
    ]

    alias GlobalTiledIteratorType = Self.GlobalTensorType.TiledIteratorType[
        BK,
        depth,
        axis=0,
    ]

    var global_iterator: Self.GlobalTiledIteratorType
    var global_base_tile: Self.GlobalTensorType

    @always_inline
    fn __init__(
        out self,
        global_tile: LayoutTensor[
            dtype,
            layout,
            origin,
            address_space=address_space,
            alignment=alignment,
            masked=masked,
        ],
        shared_ptr: UnsafePointer[
            Scalar[dtype],
            address_space = AddressSpace.SHARED, **_,
        ],
    ):
        constrained[depth in (64, 128, 256), "depth must be 64, 128, or 256"]()
        constrained[
            mma_shape[2] * k_group_size == 16,
            "mma_shape[2] * k_group_size must be 16",
        ]()

        self.global_base_tile = global_tile
        self.global_iterator = global_tile.tiled_iterator[BK, depth, axis=0](
            0, 0
        )

        self.load_tile = __type_of(self.load_tile).stack_allocation()
        self.mma_tile = __type_of(self.mma_tile).stack_allocation()
        self.smem_iter = __type_of(self.smem_iter)(shared_ptr, 0)

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
                    self.load_tile.tile[1, Self.load_width](
                        i + depth_idx * Self.loads_per_thread_per_depth_tile,
                        0,
                    ).vectorize[1, Self.load_width](),
                    warp_tile.vectorize[1, Self.load_width](),
                    self.global_base_tile,
                )
        self.global_iterator._incr()

    @always_inline
    fn get_mma_tile(self) -> Self.MMATileType:
        return self.mma_tile

    @always_inline
    fn copy_to_shared(
        self,
    ):
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
                var reg_tile_0 = self.load_tile[0 + depth_idx * 2, j][0]
                var reg_tile_1 = self.load_tile[1 + depth_idx * 2, j][0]
                var reg_pair = SIMD[dtype, 2](reg_tile_0, reg_tile_1)
                lane_tile[j, 0] = rebind[lane_tile.element_type](reg_pair)

    @always_inline
    fn load_from_shared(self):
        # MMA
        # threads in 16x4 layout
        # each column loads depth x 8 elements from smem
        var col_idx = lane_id() // 32
        var lane = lane_id() % 32
        var smem_iter_tensor = self.smem_iter.next_unsafe(0)[]

        @parameter
        for k_mma_idx in range(Self.num_k_tiles):

            @parameter
            for depth_idx in range(Self.num_depth_tiles):
                # TODO: document and parameterize this magic
                var smem_fragment = (
                    smem_iter_tensor.tile[Self.pad[depth](), 8](
                        0, col_idx + k_mma_idx * 2
                    )
                    .vectorize[1, Self.simd_width]()
                    .tile[Self.pad[Self.MMA_M](), 1](depth_idx, 0)
                    .tile[Self.pad[Self.simd_width](), 1](
                        lane // Self.simd_width, 0
                    )
                    .slice[: Self.simd_width, :]()
                    .tile[1, 1](lane % Self.simd_width, 0)
                )
                self.mma_tile.split[Self.num_k_tiles]()[k_mma_idx].vectorize[
                    1, Self.simd_width
                ]().tile[1, 1](depth_idx, 0).copy_from(smem_fragment)


struct QRegisterBuffer[
    dtype: DType,
    layout: Layout,
    address_space: BaseAddressSpace,
    alignment: Int,
    origin: Origin,
    masked: Bool,
    layout_int_type: DType,
    linear_idx_type: DType, //,
    mma_shape: IndexList[3],
    k_group_size: Int,
    WM: Int,
    WN: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    thread_layout: Layout,
]:
    alias simd_width = simd_width_of[dtype]()
    alias MMA_M = mma_shape[0]
    alias MMA_K = mma_shape[2]
    alias num_mmas = ceildiv(WM, Self.MMA_M)
    alias num_k_tiles = ceildiv(BK, Self.MMA_K * k_group_size)

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
    var gmem_tensor: Self.GlobalTensorType

    alias num_tiles = depth // BK
    alias RegisterTileType = LayoutTensor[
        dtype,
        Layout.row_major(
            Self.num_mmas * Self.num_k_tiles * Self.num_tiles, Self.simd_width
        ),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    var mma_tile: Self.RegisterTileType

    alias TiledIteratorType = Self.RegisterTileType.TiledIteratorType[
        Self.num_mmas * Self.num_k_tiles, Self.simd_width, axis=0
    ]

    # TODO: This is expensive, dereferencing q_gmem_warp_iter[] is expensive and
    # using its dim() is also expensive. Need to find a better way to do this.

    @always_inline
    fn __init__(
        out self,
        tensor: LayoutTensor[
            dtype,
            layout,
            origin,
            address_space=address_space,
            alignment=alignment,
            masked=masked,
            layout_int_type=layout_int_type,
            linear_idx_type=linear_idx_type,
        ],
    ):
        constrained[
            mma_shape[2] * k_group_size == 16,
            "mma_shape[2] * k_group_size must be 16",
        ]()
        self.gmem_tensor = tensor
        self.mma_tile = __type_of(self.mma_tile).stack_allocation()

    @always_inline
    fn load_from_dram(mut self):
        alias num_warps_n = BN // WN
        var warp_row = get_warp_id() // num_warps_n
        var bounds = max(
            min(Int32(WM), Int32(self.gmem_tensor.dim[0]() - WM * warp_row))
            * self.gmem_tensor.stride[0](),
            0,
        )
        var gmem_warp_iter = self.gmem_tensor.tiled_iterator[WM, BK, axis=1](
            warp_row, 0
        )
        var mma_tiles = self.mma_tile.split[Self.num_tiles]()

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
    fn get_mma_tile[
        tile_idx: Int, k_idx: Int
    ](self) -> Self.RegisterTileType.SplitElementType[
        Self.num_tiles
    ].SplitElementType[Self.num_k_tiles]:
        return self.mma_tile.split[Self.num_tiles]()[tile_idx].split[
            Self.num_k_tiles
        ]()[k_idx]


struct PRegisterBuffer[
    accum_type: DType,
    dtype: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    output_frag_size: Int,
]:
    alias RegisterTileType = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, output_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var reg_tile: Self.RegisterTileType

    alias OutputTileType = LayoutTensor[
        dtype,
        Layout.row_major(num_m_mmas, output_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    @always_inline
    fn __init__(out self):
        self.reg_tile = Self.RegisterTileType.stack_allocation().fill(0)

    @always_inline
    fn interleave[tile_idx: Int](self) -> Self.OutputTileType:
        var out = Self.OutputTileType.stack_allocation()

        @parameter
        for j in range(4):
            out[0, 2 * j] = convert_f32_to_bf16[dtype](
                self.reg_tile[tile_idx, j]
            )

            out[0, 2 * j + 1] = convert_f32_to_bf16[dtype](
                self.reg_tile[tile_idx, 4 + j]
            )
            out[0, 2 * j + 8] = convert_f32_to_bf16[dtype](
                self.reg_tile[tile_idx, 8 + j]
            )
            out[0, 2 * j + 8 + 1] = convert_f32_to_bf16[dtype](
                self.reg_tile[tile_idx, 12 + j]
            )
        return out


@always_inline
fn _apply_mask[
    masked: Bool,
    accum_type: DType,
    token_gen: Bool,
    MMA_M: Int,
    MMA_N: Int,
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
    var scale_log2e: SIMD[accum_type, 1] = scale.cast[accum_type]() * (
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
            var mask_frag_row = mask_warp_row + m_mma * MMA_M
            var mask_frag_col = (
                mask_warp_col
                + n_mma * MMA_N
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
            var score_row_with_start_pos = score_row + start_pos

            @parameter
            if masked:

                @parameter
                for j in range(output_frag_size):
                    alias fragment_col = fragment_layout(j)
                    var group_idx = lane_row
                    var q_head_idx = (
                        block_idx.y * group + group_idx
                    ) if token_gen else block_idx.y
                    p_reg_vectorized[mma_id, 0][j] = mask.mask(
                        IndexList[4, element_type = DType.uint32](
                            Int(block_idx.z),
                            Int(q_head_idx),
                            Int(score_row_with_start_pos),
                            Int(score_col + fragment_col),
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


@always_inline
fn apply_softmax_denominator[
    accum_type: DType, //,
    num_m_mmas: Int,
    num_n_mmas: Int,
    fragment_layout: Layout,
](
    out_reg_tile: LayoutTensor[accum_type, **_],
    rowsum: LayoutTensor[accum_type, **_],
):
    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv = recip(rowsum[m_mma, 0])

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(fragment_layout.size()):

                @parameter
                if fragment_layout.shape[0].value() > 1:
                    rowsum_inv = recip(rowsum[m_mma, i])
                out_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rebind[
                    out_reg_tile.element_type
                ](rowsum_inv)


struct SharedMemoryManager[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    num_rowwise_warps: Int,
    token_gen: Bool,
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
    alias k_v_smem_size = pad[dtype, depth, depth]() * BK

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
    fn get_kv_ptr[
        dtype: DType
    ](
        self,
    ) -> UnsafePointer[
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
        alignment2 = Self.alignment,
    ]:
        return self.k_v_smem.bitcast[Scalar[dtype]]()

    @always_inline
    fn get_p_ptr(
        self,
    ) -> UnsafePointer[
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
        alignment2 = Self.alignment,
    ]:
        return self.p_smem.bitcast[Scalar[dtype]]()

    @always_inline
    fn get_k_iter(
        self,
        out result: LayoutTensorIter[
            dtype,
            Layout.row_major(BN, BK),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            circular=True,
        ],
    ):
        constrained[token_gen, "this function is only used for token_gen"]()
        return {self.k_v_smem, BN * depth}

    @always_inline
    fn get_v_iter(
        self,
        out result: LayoutTensorIter[
            dtype,
            Layout.row_major(BK, BN),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            circular=True,
        ],
    ):
        constrained[token_gen, "this function is only used for token_gen"]()
        return {self.k_v_smem, BN * depth}

    @always_inline
    fn get_p_iter(
        self,
        out result: LayoutTensorIter[
            dtype,
            Layout.row_major(BM, BK),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            circular=True,
        ],
    ):
        return {self.p_smem, BM * BN}

    @always_inline
    fn get_warp_scratch_tensor(
        self,
        out result: LayoutTensor[
            Self.accum_type,
            Layout.row_major(2 * num_rowwise_warps, BM),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
    ):
        constrained[
            result.layout.size()
            * (size_of[Self.accum_type]() // size_of[dtype]())
            <= Self.k_v_smem_size,
            "warp_scratch_tile is too large",
        ]()
        var ptr = self.k_v_smem.bitcast[Scalar[Self.accum_type]]()
        return {ptr if token_gen else {}}


struct GlobalMemoryManager[
    dtype: DType,
    BM: UInt32,
    BN: UInt32,
    BK: UInt32,
    depth: UInt32,
    num_heads: UInt32,
    group: UInt32,
    token_gen: Bool,
]:
    alias kv_num_heads = num_heads // group
    # BHSD layout for q and kv cache
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)),
        IntTuple(Int(num_heads * depth), 1),
    ) if not token_gen else Layout.row_major(Int(BM), Int(depth))

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

    @always_inline
    fn __init__(
        out self, q_tile_idx: UInt32, kv_head_idx: UInt32, seq_len: Int
    ):
        var q_tile_num_rows = min(
            BM, UInt(seq_len) - q_tile_idx * BM
        ) if not token_gen else group

        self.q_offset = depth * (
            (kv_head_idx * group if token_gen else block_idx.y)
            + num_heads * q_tile_idx * BM
        )

        self.q_runtime_layout = __type_of(self.q_runtime_layout)(
            RuntimeTuple[
                Self.q_gmem_layout.shape,
                element_type = __type_of(self.q_runtime_layout).element_type,
            ](Int(q_tile_num_rows), Int(depth)),
            RuntimeTuple[
                Self.q_gmem_layout.stride,
                element_type = __type_of(self.q_runtime_layout).linear_idx_type,
            ](Int(num_heads * depth if not token_gen else depth), 1),
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
            Self.q_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        return self.get_q_tensor(ptr)

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
            alignment = ptr.alignment2,
        ],
    ):
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = __type_of(result.runtime_layout)(
            __type_of(result.runtime_layout.shape)(
                Int(kv_tile_num_rows), Int(depth)
            ),
            __type_of(result.runtime_layout.stride)(
                Int(Self.kv_num_heads * depth), 1
            ),
        )

        return {ptr, kv_runtime_layout}


@always_inline
fn mha_single_batch_amd[
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    group: Int,
    config: MHAConfig,
    sink: Bool = False,
    sink_type: DType = output_type,
](
    output: UnsafePointer[Scalar[output_type],],
    q: UnsafePointer[Scalar[q_type],],
    k: k_t,
    v: v_t,
    seq_len: Int,
    num_keys: Int,
    scale: Float32,
    batch_idx: Int,
    start_pos: Int,
    mask: mask_t,
    sink_weights: OptionalReg[NDBuffer[q_type, 1, MutableAnyOrigin]],
):
    alias token_gen = False
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    alias num_heads = config.num_heads
    alias BK = config.block_k()
    constrained[BN == depth, "BN must be equal to depth"]()
    alias simd_width = simd_width_of[q_type]()

    alias mma_shape = IndexList[3](32, 32, 16) if (
        _cdna_4_or_newer()
        and depth != 64
        # will deal with 64 later
    ) else IndexList[3](32, 32, 8)

    alias fragment_layout = Layout.row_major(1, 16)
    alias fragment_layout_nested = Layout(
        IntTuple(1, IntTuple(4, 4)), IntTuple(1, IntTuple(1, 8))
    )
    alias warp_layout = Layout.col_major(32, 2)
    alias swap_a_b = True
    alias k_group_size = 16 // mma_shape[2]

    alias output_frag_size = fragment_layout.size()
    alias accum_type = get_accum_type[q_type]()

    alias WM = config.warp_m()
    alias WN = config.warp_n()
    alias num_m_mmas = ceildiv(WM, UInt(mma_shape[0]))
    alias num_n_mmas = ceildiv(WN, UInt(mma_shape[1]))
    alias num_k_mmas2 = ceildiv(BK, UInt(mma_shape[2] * k_group_size))
    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    var out_reg_tile = (
        tb[accum_type]()
        .row_major[num_m_mmas * num_n_mmas, output_frag_size]()
        .local()
        .alloc()
        .fill(0)
    )

    var warp_id = get_warp_id()

    var warp_row = warp_id // num_warps_n
    var warp_col = warp_id % num_warps_n

    var kv_head_idx = block_idx.y // group

    var q_tile_idx = block_idx.x

    var q_head_idx = block_idx.y

    var gmem_manager = GlobalMemoryManager[
        q_type, BM, BN, BK, depth, num_heads, group, token_gen
    ](q_tile_idx, kv_head_idx, seq_len)

    var q_tile = gmem_manager.get_q_tensor(q)

    var output_tile = gmem_manager.get_output_tensor(output)

    var rowmax = (
        tb[accum_type]()
        .row_major[num_m_mmas, fragment_layout.shape[0].value()]()
        .local()
        .alloc()
    )
    var rowsum = (
        tb[accum_type]()
        .row_major[num_m_mmas, fragment_layout.shape[0].value()]()
        .local()
        .alloc()
    )

    @parameter
    if sink:
        debug_assert(
            Bool(sink_weights),
            "expect sink_weights to be non-null when sink=true",
        )
        rowmax = rowmax.fill(
            sink_weights.value()[Int(q_head_idx)].cast[accum_type]()
        )
        rowsum = rowsum.fill(1)
    else:
        rowmax = rowmax.fill(min_or_neg_inf[accum_type]())
        rowsum = rowsum.fill(0)

    var smem_manager = SharedMemoryManager[
        q_type, BM, BN, BK, depth, num_warps_n, token_gen
    ]()

    var warp_scratch = smem_manager.get_warp_scratch_tensor()

    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_row * WM
    var mask_warp_col = warp_col * WN

    constrained[BK == 32, "BK must be 32"]()

    var q_buffer = QRegisterBuffer[
        mma_shape=mma_shape,
        k_group_size=k_group_size,
        WM=WM,
        WN=WN,
        BN=BN,
        BK=BK,
        depth=depth,
        thread_layout=warp_layout,
    ](q_tile)

    q_buffer.load_from_dram()

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int
    ](kv_tile_start_row: UInt32, end: UInt32, not_last_iter: Bool):
        var mask_status = mask.status(
            Index[dtype = DType.uint32](
                Int(q_tile_idx * BM + start_pos),
                Int(kv_tile_start_row),
            ),
            Index[dtype = DType.uint32](Int(BM), Int(BN)),
        )

        if mask_status == TileMaskStatus.FULL_MASK:
            mask_warp_col += BN
            return

        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        var k_tile = gmem_manager.get_kv_tensor(
            k.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )

        var v_tile = gmem_manager.get_kv_tensor(
            v.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )

        var p_buffer = PRegisterBuffer[
            accum_type,
            q_type,
            num_m_mmas,
            num_n_mmas,
            output_frag_size,
        ]()

        var num_b_rows = Int(kv_tile_num_rows)
        alias num_threads = config.num_threads()

        var v_buffer = VBuffer[
            mma_shape=mma_shape,
            k_group_size=k_group_size,
            BN=BN,
            BK=BK,
            depth=depth,
            num_threads=num_threads,
        ](v_tile, smem_manager.get_kv_ptr[v_tile.dtype]())

        var k_buffer = KBuffer[
            mma_shape=mma_shape,
            k_group_size=k_group_size,
            swizzle=None,
            BN=BN,
            WN=BN,
            BK=BK,
            num_threads=num_threads,
        ](
            k_tile,
            num_b_rows,
            smem_manager.get_kv_ptr[k_tile.dtype](),
        )

        alias tensor_core_mma = TensorCoreKGroup[
            accum_type,
            q_type,
            mma_shape,
            k_group_size=k_group_size,
            transpose_b=True,
        ]()

        # calculate k q ^T
        k_buffer.load_from_dram()

        @parameter
        for i in range(depth // BK):
            k_buffer.copy_to_shared()

            barrier()

            @parameter
            if i < depth // BK - 1:
                k_buffer.load_from_dram()

                @parameter
                if i == depth // BK - 2:
                    # prefetch v from dram
                    v_buffer.load_from_dram()

            @parameter
            for k_mma in range(num_k_mmas2):
                var q_mma_tile = q_buffer.get_mma_tile[i, k_mma]()
                k_buffer.load_from_shared[
                    # TODO: I should be able to use tensor_core_mma here
                    # but getting compiler errors
                    accum_type,
                    q_type,
                    mma_shape,
                    k_group_size,
                    True,
                    k_mma,
                ]()
                var k_mma_tile = k_buffer.get_mma_tile()
                tensor_core_mma.mma[swap_a_b=swap_a_b](
                    q_mma_tile, k_mma_tile, p_buffer.reg_tile
                )

            barrier()

        var p_reg_vectorized = p_buffer.reg_tile.vectorize[
            1, output_frag_size
        ]()

        alias use_exp2 = True

        @always_inline
        @parameter
        fn _apply_mask_impl[masked: Bool]():
            _apply_mask[
                masked=masked,
                accum_type=accum_type,
                token_gen=token_gen,
                MMA_M = mma_shape[0],
                MMA_N = mma_shape[1],
                num_m_mmas=num_m_mmas,
                num_n_mmas=num_n_mmas,
                mask_t=mask_t,
                group=group,
                fragment_layout=fragment_layout_nested,
                warp_layout=warp_layout,
                use_exp2=use_exp2,
            ](
                kv_tile_start_row,
                kv_tile_num_rows,
                start_pos,
                seq_len,
                num_keys,
                Int(mask_block_row),
                Int(mask_warp_row),
                mask_warp_col,
                scale,
                mask,
                p_reg_vectorized,
                not_last_iter,
            )

        unswitch[_apply_mask_impl](mask_status == TileMaskStatus.PARTIAL_MASK)

        mask_warp_col += BN
        alias reg_layout_by_mma_unit = Layout.row_major(
            num_m_mmas * num_n_mmas, output_frag_size
        )
        # don't know why we need this barrier but i get random failures without it
        barrier()
        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            warp_layout,
            use_exp2=use_exp2,
            fragment_layout=fragment_layout,
        ](
            out_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            p_buffer.reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            warp_scratch.tile[2 * num_warps_n, WM](0, Int(warp_row)),
            rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

        barrier()

        # calculate v^T p^T

        @parameter
        for i in range(BN // BK):
            # v has been prefetched from dram during the last mma
            v_buffer.copy_to_shared()

            @parameter
            if i < (BN // BK) - 1:
                v_buffer.load_from_dram()

            # ensure that shared memory is filled
            barrier()

            v_buffer.load_from_shared()

            var p_mma_tile_interleaved = p_buffer.interleave[i]()

            @parameter
            for k_mma_idx in range(v_buffer.num_k_tiles):
                tensor_core_mma.mma[swap_a_b=swap_a_b](
                    p_mma_tile_interleaved.tile[1, simd_width](0, k_mma_idx),
                    v_buffer.mma_tile.tile[depth // mma_shape[0], simd_width](
                        k_mma_idx, 0
                    ),
                    out_reg_tile,
                )

            barrier()

    for i in range(UInt32(0), UInt32(num_keys), UInt32(BN)):
        var end = min(i + BN, num_keys)
        loop_over_kvcache[BN](i, end, end != num_keys)

    # Apply softmax denominator.
    apply_softmax_denominator[
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas,
        fragment_layout=fragment_layout,
    ](out_reg_tile, rowsum)

    var output_warp_tile = output_tile.tile[WM, WN](warp_row, warp_col)

    copy_local_to_dram2[
        dst_thread_layout=warp_layout,
        thread_scope = ThreadScope.WARP,
    ](
        output_warp_tile.vectorize[
            1,
            4,
        ](),
        out_reg_tile.vectorize[1, 4](),
        output_tile,
    )


@always_inline
fn mma[
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    transpose_b: Bool,
    k_group_size: Int,
    config: MHAConfig,
    prefetch_function: fn[Int] () capturing -> None,
    swizzle: OptionalReg[Swizzle] = None,
    swap_a_b: Bool = False,
    num_iters: Int = 1,
    token_gen: Bool = False,
](
    c: LayoutTensor,
    mut a_iter: LayoutTensorIter,
    a_smem_iter: LayoutTensorIter,
    mut b_iter: LayoutTensorIter,
    b_smem_iter: LayoutTensorIter[*_, address_space = AddressSpace.SHARED, **_],
    num_b_rows: OptionalReg[Int] = None,
):
    alias BK = config.block_k()
    # a can be either bfloat16 or float32 but b is always the same type as mma_input_type
    alias mma_input_type = b_iter.dtype
    alias simd_width = simd_width_of[mma_input_type]()
    alias accum_type = get_accum_type[mma_input_type]()
    alias WM = config.warp_m()
    alias WN = config.warp_n()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    var warp_id = get_warp_id()
    alias num_warps = config.num_threads() // WARP_SIZE
    alias num_threads = config.num_threads()
    alias num_warps_n = BN // WN

    var warp_row = warp_id // num_warps_n
    var warp_col = warp_id % num_warps_n

    alias thread_layout_b = Layout.row_major(
        min(num_threads, BN * BK // simd_width)
        * simd_width
        // b_smem_iter.layout.stride[0].value(),
        b_smem_iter.layout.stride[0].value() // simd_width,
    ) if token_gen else Layout.row_major(num_threads // 4, 4)

    alias tensor_core_mma = TensorCoreKGroup[
        accum_type,
        mma_input_type,
        IndexList[3](MMA_M, MMA_N, MMA_K),
        k_group_size=k_group_size,
        transpose_b=transpose_b,
    ]()

    alias num_m_mmas = ceildiv(WM, UInt(MMA_M))
    alias num_n_mmas = ceildiv(WN, UInt(MMA_N))
    alias num_k_mmas2 = ceildiv(BK, UInt(MMA_K * k_group_size))

    alias a_frag_size = num_matrix_reg[MMA_M, MMA_K]()
    alias b_frag_size = num_matrix_reg[MMA_N, MMA_K]()
    alias c_frag_size = num_matrix_reg[MMA_M, MMA_N]()

    var b_load_tile = (
        tb[mma_input_type]()
        .row_major[2 * num_n_mmas * num_k_mmas2, b_frag_size * k_group_size]()
        .local()
        .alloc()
        .split[2]()
    )

    @parameter
    @always_inline
    fn copy_dram_to_local_b[reg_tile_id: Int]():
        @parameter
        if b_iter.address_space != AddressSpace.SHARED:
            alias b_stride = b_iter.layout.stride[0].value()
            copy_dram_to_local[src_thread_layout=thread_layout_b,](
                b_load_tile[reg_tile_id].vectorize[1, simd_width](),
                b_iter,
                num_b_rows.value() * b_stride if num_b_rows else Int.MAX,
            )
            b_iter._incr()

    copy_dram_to_local_b[0]()

    @parameter
    for i in range(num_iters):

        @parameter
        if i < num_iters - 1:
            copy_dram_to_local_b[(i + 1) % 2]()

            @parameter
            if i == num_iters - 2:
                prefetch_function[0]()

        var b_smem_tile = b_smem_iter.next_unsafe(0)[]

        copy_local_to_shared[
            thread_layout=thread_layout_b, swizzle=swizzle, row_major=True
        ](
            b_smem_tile.vectorize[1, simd_width](),
            b_load_tile[i % 2].vectorize[1, simd_width](),
        )

        barrier()

        var a_reg_tile = (
            tb[mma_input_type]()
            .row_major[num_m_mmas, a_frag_size * k_group_size]()
            .local()
            .alloc()
        )
        var b_reg_tile = (
            tb[mma_input_type]()
            .row_major[num_n_mmas, b_frag_size * k_group_size]()
            .local()
            .alloc()
        )

        alias b_wtile_dim0 = WN if transpose_b else BK
        alias b_wtile_dim1 = BK if transpose_b else WN
        var b_wtile_coord0 = Int(warp_col) if transpose_b else 0
        var b_wtile_coord1 = 0 if transpose_b else Int(warp_col)
        var b_warp_tile = b_smem_tile.tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0, b_wtile_coord1
        )

        @parameter
        for k_mma in range(num_k_mmas2):

            @parameter
            if a_iter.address_space != AddressSpace.LOCAL:
                var a_warp_tile = a_smem_iter.next_unsafe(i)[].tile[WM, BK](
                    warp_row, 0
                )
                tensor_core_mma.mma_op.load_a[swizzle=swizzle](
                    a_warp_tile,
                    a_reg_tile.vectorize[1, a_frag_size * k_group_size](),
                    k_mma,
                )
            else:
                var a_reg_tile_input = a_iter.next_unsafe(i)[]
                a_reg_tile.vectorize[1, a_frag_size]().copy_from(
                    a_reg_tile_input.tile[1, simd_width](k_mma, 0).vectorize[
                        1, a_frag_size
                    ]()
                )

            tensor_core_mma.mma_op.load_b[swizzle=swizzle](
                b_warp_tile,
                b_reg_tile.vectorize[1, b_frag_size * k_group_size](),
                k_mma,
            )

            tensor_core_mma.mma[swap_a_b=swap_a_b](a_reg_tile, b_reg_tile, c)

        barrier()


@always_inline
fn mha_decoding_single_batch_amd[
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    group: Int,
    config: MHAConfig,
    sink: Bool = False,
](
    output: UnsafePointer[Scalar[output_type],],
    q: UnsafePointer[Scalar[q_type],],
    k: k_t,
    v: v_t,
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    seq_len: Int,
    num_keys: Int,
    num_partitions: Int,
    scale: Float32,
    batch_idx: Int,
    start_pos: Int,
    mask: mask_t,
    sink_weights: OptionalReg[NDBuffer[q_type, 1, MutableAnyOrigin]],
):
    alias token_gen = True

    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    alias num_heads = config.num_heads
    alias kv_num_heads = num_heads // group
    alias BK = config.block_k()
    constrained[BN == depth, "BN must be equal to depth"]()
    alias simd_width = simd_width_of[q_type]()

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias use_transposed_layout = True
    alias fragment_layout = Layout.row_major(
        1, 4
    ) if use_transposed_layout else Layout.row_major(4, 1)
    alias warp_layout = Layout.col_major(
        16, 4
    ) if use_transposed_layout else Layout.row_major(4, 16)
    alias swap_a_b = use_transposed_layout
    alias k_group_size = 2

    alias output_frag_size = fragment_layout.size()
    alias accum_type = get_accum_type[q_type]()

    alias WM = config.warp_m()
    alias WN = config.warp_n()
    alias num_m_mmas = ceildiv(WM, UInt(MMA_M))
    alias num_n_mmas = ceildiv(WN, UInt(MMA_N))
    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    var out_reg_tile = (
        tb[accum_type]()
        .row_major[num_m_mmas * num_n_mmas, output_frag_size]()
        .local()
        .alloc()
        .fill(0)
    )

    var warp_id = get_warp_id()

    var warp_row = warp_id // num_warps_n
    var warp_col = warp_id % num_warps_n

    var kv_head_idx = block_idx.y

    alias rowwise_stride = fragment_layout.shape[0].value()
    var q_tile_idx = 0
    var lane = lane_id()
    var coords = idx2crd[warp_layout](lane)
    var group_idx = coords[0] * rowwise_stride
    var q_head_idx = block_idx.y * group + group_idx

    var gmem_manager = GlobalMemoryManager[
        q_type, BM, BN, BK, depth, num_heads, group, token_gen
    ](q_tile_idx, kv_head_idx, seq_len)

    var q_tile = gmem_manager.get_q_tensor(q)

    var output_tile = gmem_manager.get_output_tensor(output)

    var rowmax = (
        tb[accum_type]()
        .row_major[num_m_mmas, fragment_layout.shape[0].value()]()
        .local()
        .alloc()
    )
    var rowsum = (
        tb[accum_type]()
        .row_major[num_m_mmas, fragment_layout.shape[0].value()]()
        .local()
        .alloc()
    )

    @parameter
    if sink:
        debug_assert(
            Bool(sink_weights),
            "expect sink_weights to be non-null when sink=true",
        )
        rowmax = rowmax.fill(
            sink_weights.value()[Int(q_head_idx)].cast[accum_type]()
        )
        rowsum = rowsum.fill(1)
    else:
        rowmax = rowmax.fill(min_or_neg_inf[accum_type]())
        rowsum = rowsum.fill(0)

    var smem_manager = SharedMemoryManager[
        q_type, BM, BN, BK, depth, num_warps_n, token_gen
    ]()

    var p_smem_iter = smem_manager.get_p_iter()
    var k_smem_iter = smem_manager.get_k_iter()
    var v_smem_iter = smem_manager.get_v_iter()

    var warp_scratch = smem_manager.get_warp_scratch_tensor()

    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_row * WM
    var mask_warp_col = warp_col * WN

    constrained[BK == 32, "BK must be 32"]()

    # the following assumes BK == 32, i.e. simd_width = 2*frag_size
    alias q_reg_size = (depth // BK) * num_m_mmas * simd_width

    var q_reg_data = stack_allocation[
        q_reg_size,
        q_type,
        address_space = AddressSpace.LOCAL,
    ]()

    var q_reg_tile_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(num_m_mmas, simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ](q_reg_data, q_reg_size)

    var q_gmem_warp_iter = q_tile.tiled_iterator[WM, BK, axis=1](warp_row, 0)

    @parameter
    for i in range(depth // BK):
        var q_reg_tile = q_reg_tile_iter.next_unsafe(i)[]
        copy_dram_to_local[
            src_thread_layout = Layout.col_major(16, 4),
            thread_scope = ThreadScope.WARP,
        ](
            q_reg_tile.vectorize[1, simd_width](),
            q_gmem_warp_iter,
            q_tile.dim[0]() * q_tile.stride[0](),
        )
        q_gmem_warp_iter._incr()

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int
    ](kv_tile_start_row: Int, end: Int, not_last_iter: Bool):
        @parameter
        if mask_t.check_mask_during_decoding:
            var mask_status = mask.status(
                Index[dtype = DType.uint32](
                    Int(num_keys - 1),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.uint32](Int(1), Int(BN)),
            )

            if mask_status == TileMaskStatus.FULL_MASK:
                return

        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        var k_tile = gmem_manager.get_kv_tensor(
            k.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )
        var k_global_iterator = k_tile.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_tile = gmem_manager.get_kv_tensor(
            v.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )

        var v_global_iterator = v_tile.tiled_iterator[BK, BN, axis=0](0, 0)

        var p_reg_tile = (
            tb[accum_type]()
            .row_major[num_m_mmas * num_n_mmas, output_frag_size]()
            .local()
            .alloc()
            .fill(0)
        )

        alias swizzle = Swizzle(2, 0, 2)

        var num_b_rows = OptionalReg[Int](
            kv_tile_num_rows
        ) if not not_last_iter else None

        # TODO (KERN-1708):this is just a dummy iterator to satisfy the interface
        # will fix it with better interface later
        var q_smem_iter = LayoutTensorIter[
            q_type,
            Layout.row_major(num_m_mmas, simd_width),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
        ](
            UnsafePointer[
                Scalar[q_type], address_space = AddressSpace.SHARED
            ](),
            q_reg_size,
        )

        @parameter
        @always_inline
        fn prefetch_function[tile_id: Int]():
            ...

        mma[
            MMA_M=MMA_M,
            MMA_N=MMA_N,
            MMA_K=MMA_K,
            transpose_b=True,
            k_group_size=k_group_size,
            config=config,
            prefetch_function=prefetch_function,
            swizzle=swizzle,
            swap_a_b=swap_a_b,
            num_iters = Int(depth // BK),
            token_gen=token_gen,
        ](
            p_reg_tile,
            q_reg_tile_iter,
            q_smem_iter,
            k_global_iterator,
            k_smem_iter,
            num_b_rows,
        )

        var p_reg_vectorized = p_reg_tile.vectorize[1, output_frag_size]()

        alias use_exp2 = True

        @always_inline
        @parameter
        fn _apply_mask_impl[masked: Bool]():
            _apply_mask[
                masked=masked,
                accum_type=accum_type,
                token_gen=token_gen,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                num_m_mmas=num_m_mmas,
                num_n_mmas=num_n_mmas,
                mask_t=mask_t,
                group=group,
                fragment_layout=fragment_layout,
                warp_layout=warp_layout,
                use_exp2=use_exp2,
            ](
                kv_tile_start_row,
                kv_tile_num_rows,
                start_pos,
                seq_len,
                num_keys,
                Int(mask_block_row),
                Int(mask_warp_row),
                mask_warp_col,
                scale,
                mask,
                p_reg_vectorized,
                not_last_iter,
            )

        @parameter
        if mask_t.check_mask_during_decoding:
            var mask_status = mask.status(
                Index[dtype = DType.uint32](
                    Int(num_keys - 1),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.uint32](Int(1), Int(BN)),
            )
            unswitch[_apply_mask_impl](
                mask_status == TileMaskStatus.PARTIAL_MASK
            )
        else:
            _apply_mask_impl[masked=True]()

        alias reg_layout_by_mma_unit = Layout.row_major(
            num_m_mmas * num_n_mmas, output_frag_size
        )

        # Not sure why we need this barrier here, but the code hangs without it
        barrier()

        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            warp_layout,
            use_exp2=use_exp2,
            fragment_layout=fragment_layout,
        ](
            out_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            warp_scratch.tile[2 * num_warps_n, WM](0, Int(warp_row)),
            rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

        # warp scratch and p_smem are using the same smem space
        barrier()

        copy_fragment_to_smem[
            BM,
            BN,
            BK,
            WM,
            WN,
            MMA_M,
            MMA_N,
            num_m_mmas,
            num_n_mmas,
            fragment_layout,
            warp_layout,
        ](
            p_smem_iter,
            p_reg_vectorized,
            warp_row,
            warp_col,
        )

        barrier()

        mma[
            MMA_M=MMA_M,
            MMA_N=MMA_N,
            MMA_K=MMA_K,
            transpose_b=False,
            k_group_size=k_group_size,
            config=config,
            prefetch_function=prefetch_function,
            swizzle=None,
            swap_a_b=swap_a_b,
            num_iters = Int(BN // BK),
            token_gen=token_gen,
        ](
            out_reg_tile,
            p_smem_iter,
            p_smem_iter,
            v_global_iterator,
            v_smem_iter,
            num_b_rows,
        )
        # ensure that smem for v is not required anymore
        barrier()

    start, end = get_start_and_end_for_partitions[BN](
        num_keys, num_partitions, block_idx.x
    )

    for i in range(start, end, BN):
        var end_ = min(i + BN, end)
        loop_over_kvcache[BN](i, end_, end_ != end)

    # Apply softmax denominator.
    apply_softmax_denominator[
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas,
        fragment_layout=fragment_layout,
    ](out_reg_tile, rowsum)

    if num_partitions > 1:
        if thread_idx.x < UInt(group):
            var row_sum = rowsum[0, 0][0]
            var row_max = rowmax[0, 0][0]

            exp_sum_ptr[q_head_idx] = row_sum
            qk_max_ptr[q_head_idx] = row_max

    var output_warp_tile = output_tile.tile[WM, WN](warp_row, warp_col)
    copy_local_to_dram[
        dst_thread_layout=warp_layout,
        thread_scope = ThreadScope.WARP,
    ](
        output_warp_tile.vectorize[
            fragment_layout.shape[0].value(),
            fragment_layout.shape[1].value(),
        ](),
        out_reg_tile.vectorize[1, output_frag_size](),
        output_tile,
    )


@always_inline
fn copy_fragment_to_smem[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    fragment_layout: Layout,
    warp_layout: Layout,
](
    p_smem_iter: LayoutTensorIter[*_, address_space = AddressSpace.SHARED, **_],
    p_reg_vectorized: LayoutTensor[*_, address_space = AddressSpace.LOCAL, **_],
    warp_row: Int,
    warp_col: Int,
):
    alias num_n_mmas_per_bk = num_n_mmas // (WN // BK)

    # for the following indexing logic, WN must be equal to BN or BK
    constrained[WN == BK or WN == BN, "WN must be equal to BN or BK"]()

    @parameter
    for i in range(WN // BK):
        var p_smem_tile = p_smem_iter.next_unsafe(i + warp_col * (WN // BK))[]
        var p_smem_warp_tile = p_smem_tile.tile[WM, BK](warp_row, i)

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas_per_bk):
                var p_smem_mma_tile = p_smem_warp_tile.tile[MMA_M, MMA_N](
                    m_mma, n_mma
                )
                var p_reg_tile = p_reg_vectorized.tile[1, 1](
                    (n_mma + i * num_n_mmas_per_bk) * num_m_mmas + m_mma,
                    0,
                )
                copy_local_to_shared[thread_layout=warp_layout](
                    p_smem_mma_tile.vectorize[
                        fragment_layout.shape[0].value(),
                        fragment_layout.shape[1].value(),
                    ](),
                    p_reg_tile,
                )
