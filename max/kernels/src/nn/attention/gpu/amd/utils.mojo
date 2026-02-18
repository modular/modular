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

from sys import align_of, simd_width_of, size_of

from gpu import lane_id, thread_idx, block_idx
from gpu import warp_id as get_warp_id, WARP_SIZE
from layout import IntTuple, Layout, LayoutTensor
from layout._utils import idx2crd, make_amd_buffer_resource
from layout.element import Element
from layout.layout_tensor import ThreadScope
from layout.runtime_layout import RuntimeLayout
from layout.tensor_core import num_matrix_reg
from memory import AddressSpace as BaseAddressSpace
from memory import stack_allocation

from utils import IndexList
from utils.numerics import get_accum_type
from layout.swizzle import Swizzle
from gpu._utils import to_i32, to_llvm_shared_mem_ptr, to_i64
from itertools import product
from sys._assembly import inlined_assembly


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
    comptime num_warps_n = BN // WN
    var warp_row = get_warp_id() // UInt(num_warps_n)
    var warp_col = get_warp_id() % UInt(num_warps_n)
    return IndexList[2](Int(warp_row), Int(warp_col))


@always_inline
fn pad[dtype: DType, depth: Int, size: Int]() -> Int:
    comptime simd_width = simd_width_of[dtype]()
    comptime padding = 0 if depth == 64 else size // simd_width
    return size + padding


comptime LocalLayoutTensor[dtype: DType, layout: Layout] = LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
    address_space = AddressSpace.LOCAL,
]

comptime SharedLayoutTensor[dtype: DType, layout: Layout] = LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
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
    var dst_frag_offset = dst_fragments.distance(dst.ptr) + Scalar[
        dst.linear_idx_type
    ](offset)

    comptime M = src.layout.shape[0].value()
    comptime N = src.layout.shape[1].value()

    @parameter
    for n in range(N):

        @parameter
        for m in range(M):
            comptime src_idx = 4 * n + 16 * m
            comptime i = 4 * m + n

            comptime dst_static_idx = dst_fragments.layout(i)
            var dst_idx = dst_frag_offset

            @parameter
            if dst_fragments.layout.all_dims_known():
                dst_idx += Scalar[dst.linear_idx_type](dst_static_idx)
            else:
                dst_idx += dst_fragments.runtime_layout(i)

            var src_element = Element[index_type = src.linear_idx_type].load(
                src.ptr + src_idx,
                src.runtime_element_layout,
            )

            comptime element_stride = dst_fragments.element_layout.stride[
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
                    comptime element_offset = dst_fragments.element_layout(i)
                    var src = src_element.element_data[i].cast[dst.dtype]()
                    buffer.store(
                        Int32(
                            dst_idx
                            + Scalar[dst.linear_idx_type](element_offset)
                        ),
                        src,
                    )


struct SharedMemoryManager[
    shared_kv: Bool,
    full_kv: Bool,
    depth_padded: Bool,
    double_buffer: Bool,
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    token_gen: Bool,
]:
    var p_smem: UnsafePointer[
        Scalar[Self.dtype],
        MutExternalOrigin,
        address_space = AddressSpace.SHARED,
    ]
    # p_smem is used for p
    var k_smem: UnsafePointer[
        Scalar[Self.dtype],
        MutExternalOrigin,
        address_space = AddressSpace.SHARED,
    ]
    var v_smem: UnsafePointer[
        Scalar[Self.dtype],
        MutExternalOrigin,
        address_space = AddressSpace.SHARED,
    ]
    # k_v_smem is used for k, v, and scratch
    comptime alignment = align_of[
        SIMD[Self.dtype, simd_width_of[Self.dtype]()]
    ]()
    comptime accum_type = get_accum_type[Self.dtype]()
    comptime p_smem_size = Self.BM * Self.BN if Self.token_gen else 0
    comptime simd_width = simd_width_of[Self.dtype]()
    # depth // simd_width is the padding
    comptime k_smem_size = Self.BN * (
        Self.depth if Self.full_kv else Self.BK
    ) * (2 if Self.double_buffer else 1)
    comptime v_smem_size = (Self.BN if Self.full_kv else Self.BK) * (
        pad[
            Self.dtype, Self.depth, Self.depth
        ]() if Self.depth_padded else Self.depth
    ) * (2 if Self.double_buffer else 1)

    @always_inline
    fn __init__(out self):
        self.p_smem = stack_allocation[
            Self.p_smem_size,
            Self.dtype,
            address_space = AddressSpace.SHARED,
            alignment = Self.alignment,
        ]()

        comptime kv_smem_size = max(Self.k_smem_size, Self.v_smem_size)

        self.k_smem = stack_allocation[
            kv_smem_size if Self.shared_kv else Self.k_smem_size,
            Self.dtype,
            address_space = AddressSpace.SHARED,
            alignment = Self.alignment,
        ]()

        self.v_smem = self.k_smem if Self.shared_kv else stack_allocation[
            Self.v_smem_size,
            Self.dtype,
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
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ]:
        return self.k_smem.bitcast[Scalar[_dtype]]()

    @always_inline
    fn get_v_ptr[
        _dtype: DType
    ](
        self,
    ) -> UnsafePointer[
        Scalar[_dtype],
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ]:
        return self.v_smem.bitcast[Scalar[_dtype]]()

    @always_inline
    fn get_p_ptr[
        _dtype: DType
    ](
        self,
    ) -> UnsafePointer[
        Scalar[_dtype],
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ]:
        return self.p_smem.bitcast[Scalar[_dtype]]()

    @always_inline
    fn get_warp_scratch_ptr[
        _dtype: DType
    ](
        self,
    ) -> UnsafePointer[
        Scalar[_dtype],
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ]:
        return self.k_smem.bitcast[Scalar[_dtype]]() if Self.token_gen else {}


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
    comptime kv_num_heads = Self.num_heads // Self.group
    # BHSD layout for q and kv cache
    comptime q_gmem_layout = Layout(
        IntTuple(Int(Self.BM), Int(Self.q_depth)),
        IntTuple(Int(Self.num_heads * Self.q_depth), 1),
    ) if not Self.token_gen else Layout.row_major(
        Int(Self.BM), Int(Self.q_depth)
    )

    comptime output_gmem_layout = Layout(
        IntTuple(Int(Self.BM), Int(Self.output_depth)),
        IntTuple(Int(Self.num_heads * Self.output_depth), 1),
    ) if not Self.token_gen else Layout.row_major(
        Int(Self.BM), Int(Self.output_depth)
    )

    comptime kv_gmem_layout = Layout(
        IntTuple(Int(Self.BN), Int(Self.depth)),
        IntTuple(Int(Self.kv_num_heads * Self.depth), 1),
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
            Self.BM, UInt32(seq_len) - q_tile_idx * Self.BM
        ) if not Self.token_gen else Self.group

        self.q_offset = q_offset
        self.output_offset = output_offset

        self.q_runtime_layout = type_of(self.q_runtime_layout)(
            {Int(q_tile_num_rows), Int(Self.q_depth)},
            {
                Int(
                    Self.num_heads
                    * Self.q_depth if not Self.token_gen else Self.q_depth
                ),
                1,
            },
        )

        self.output_runtime_layout = type_of(self.output_runtime_layout)(
            {Int(q_tile_num_rows), Int(Self.output_depth)},
            {
                Int(
                    Self.num_heads
                    * Self.output_depth if not Self.token_gen else Self.output_depth
                ),
                1,
            },
        )

    @always_inline
    fn get_q_tensor[
        qtype: DType,
    ](
        self,
        ptr: UnsafePointer[Scalar[qtype], ImmutAnyOrigin],
        out result: LayoutTensor[
            qtype,
            Self.q_gmem_layout,
            ImmutAnyOrigin,
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
        ptr: UnsafePointer[Scalar[out_type], MutAnyOrigin],
        out result: LayoutTensor[
            out_type,
            Self.output_gmem_layout,
            MutAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        return {ptr + Int(self.output_offset), self.output_runtime_layout}

    @always_inline
    fn get_kv_tensor[
        kvtype: DType,
        //,
    ](
        self,
        ptr: UnsafePointer[Scalar[kvtype], ImmutAnyOrigin],
        kv_tile_num_rows: UInt32,
        out result: LayoutTensor[
            kvtype,
            Self.kv_gmem_layout,
            ImmutAnyOrigin,
            masked=True,
            address_space = ptr.address_space,
        ],
    ):
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = type_of(result.runtime_layout)(
            type_of(result.runtime_layout.shape)(
                Int(kv_tile_num_rows), Int(Self.depth)
            ),
            type_of(result.runtime_layout.stride)(
                Int(Self.kv_num_heads * Self.depth), 1
            ),
        )

        return {ptr, kv_runtime_layout}


comptime _alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.AsyncCopies", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`
comptime _no_alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.LocalLoads", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`


@always_inline
fn _load_tr16_b64_row(
    tile: LayoutTensor[_, _, address_space = AddressSpace.SHARED, ...]
) -> SIMD[tile.dtype, 4]:
    # ds_read_tr16_b64 uses a set of 4x4 lanes (amd calls 16 lanes a "row")
    # to load a 4x16 tile. Each lane loads 4 contiguous elements from the tile.
    # Then they are exchanged such that at the end of this operation you get a
    # SIMD[tile.dtype, 4], with each lane containing a column of the 4x16 tile.
    comptime assert size_of[tile.dtype]() == 2, String(
        "Expected tile.dtype to be DType.bfloat16, but got ", tile.dtype
    )
    comptime assert tile.shape[0]() == 4, String(
        "Expected tile.shape[0]() to be 4, but got ", tile.shape[0]()
    )
    comptime assert tile.shape[1]() == 16, String(
        "Expected tile.shape[1]() to be 16, but got ", tile.shape[1]()
    )

    comptime thread_layout = Layout.row_major(4, 4)
    var lane_in_row = lane_id() % 16
    var dist_result = tile.vectorize[1, 4]().distribute_with_offset[
        thread_layout
    ](lane_in_row)
    var offset = dist_result[2]
    var ptr = tile.ptr + offset

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr<3>`
    ](ptr)

    var llvm_res = __mlir_op.`rocdl.ds.read.tr16.b64`[
        _type = __mlir_type.`vector<4 x bf16>`,
        noalias_scopes=_alias_scope_attr,
        alias_scopes=_no_alias_scope_attr,
    ](
        shared_ptr3,
    )

    return rebind[SIMD[tile.dtype, 4]](
        __mlir_op.`pop.cast_from_builtin`[
            _type = SIMD[tile.dtype, 4]._mlir_type
        ](llvm_res)
    )
    # return ds_read_tr16_b64(ptr)


@always_inline
fn _load_tr16_b64_warp[
    mma_shape: IndexList[3],
](tile: LayoutTensor[_, _, address_space = AddressSpace.SHARED, ...]) -> SIMD[
    tile.dtype, 4
]:
    # for 8x32 we need 2x2 distribution of rows (16 lanes), 2x2 x 4x16 = 8x32
    # for 16x16 we need 4x1 distribution of rows (16 lanes), 4x1 x 4x16 = 16x16
    comptime row_layout = Layout.row_major(2, 2) if mma_shape[
        0
    ] == 32 else Layout.row_major(4, 1)
    comptime assert tile.dtype == DType.bfloat16, String(
        "Expected tile.dtype to be DType.bfloat16, but got ", tile.dtype
    )
    comptime assert tile.shape[0]() == row_layout.shape[0].value() * 4, String(
        "Expected tile.shape[0]() to be ",
        row_layout.shape[0].value() * 4,
        ", but got ",
        tile.shape[0](),
    )
    comptime assert tile.shape[1]() == row_layout.shape[1].value() * 16, String(
        "Expected tile.shape[1]() to be ",
        row_layout.shape[1].value() * 16,
        ", but got ",
        tile.shape[1](),
    )

    var coords = idx2crd[row_layout](Int(lane_id() // 16))
    var shared_b_tile = tile.tile[4, 16](coords[0], coords[1])
    return _load_tr16_b64_row(shared_b_tile)


@always_inline
fn load_b_tr[
    mma_shape: IndexList[3]
](tile: LayoutTensor[_, _, address_space = AddressSpace.SHARED, ...]) -> SIMD[
    tile.dtype, 8
]:
    """Loads the b operand tile for AMD tensor core MFMA instructions using transposed memory access.

    This function supports double-rate MFMA shapes (32x32x16, 16x16x32) with bfloat16 input.
    The input tile (shape = (mma_shape[2], mma_shape[1])) is split along the K dimension into
    two halves of shape (MMA_K//2, MMA_N). Each half is loaded using `_load_tr16_b64_warp`, which
    performs a transposed (column-major) load from shared memory. The resulting two 4-element SIMD
    vectors are concatenated into a single `SIMD[tile.dtype, 8]` vector.

    Parameters:
        mma_shape: The MMA instruction tile shape (only 32x32x16 or 16x16x32 supported).

    Args:
        tile:      A `LayoutTensor`, residing in shared memory, with shape (mma_shape[2], mma_shape[1])
                   and dtype `DType.bfloat16`.

    Returns:
        SIMD[tile.dtype, 8]: Concatenated transposed SIMD loads from both halves of the tile.
    """
    # only support double-rate mfma shapes for now
    comptime assert mma_shape in (
        IndexList[3](32, 32, 16),
        IndexList[3](16, 16, 32),
    ), String(
        "Unsupported mma_shape: ",
        mma_shape[0],
        "x",
        mma_shape[1],
        "x",
        mma_shape[2],
        ". Supported shapes: 32x32x16, 16x16x32",
    )
    comptime assert tile.dtype == DType.bfloat16, String(
        "Expected tile.dtype to be DType.bfloat16, but got ", tile.dtype
    )
    comptime assert tile.shape[0]() == mma_shape[2], String(
        "Expected tile.shape[0]() to be mma_shape[2]=",
        mma_shape[2],
        ", but got ",
        tile.shape[0](),
    )
    comptime assert tile.shape[1]() == mma_shape[1], String(
        "Expected tile.shape[1]() to be mma_shape[1]=",
        mma_shape[1],
        ", but got ",
        tile.shape[1](),
    )
    # Loads the input tile as two halves along the K dimension, each of shape
    # (MMA_K//2, MMA_N), and concatenates the resulting 4-element vectors.
    # This is designed for use in multi-head attention (MHA) kernels where
    # the output fragment of a previous MFMA serves as the input to the next.
    #
    # For example, with MMA shape (32, 32, 16), this function splits a tile of
    # shape (16, 32) into two (8, 32) tiles, loads 4 values from each, and
    # joins them. This follows the MFMA output pattern on AMD GPUs where output
    # fragments are organized in 4-element vectors.
    #
    # Typical usage: when fusing two MMAs, you can efficiently pass the
    # accumulator of the first (after downcasting to 2 bytes) as part of the input to the next.
    var tiles = tile.split[2]()
    var part_1 = _load_tr16_b64_warp[mma_shape](tiles[0])
    var part_2 = _load_tr16_b64_warp[mma_shape](tiles[1])
    return part_1.join(part_2)


@always_inline
fn copy_dram_to_sram_lds[
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](dst: LayoutTensor, src: LayoutTensor, lds_base_ptr: UInt32 = 0):
    comptime thread_layout = Layout.row_major(16, 4)
    var worker_idx = lane_id()

    var bc = make_amd_buffer_resource(src)

    comptime M = src.shape[0]()
    comptime N = src.shape[1]()
    # we use 16x4 thread layout to load 16x32 tile from dram to sram
    # but we need to load 32x16 tiles from sram for mma, so swizzle need to be applied to
    # the whole 32x32 tile.
    comptime BM = 32
    comptime BN = 32
    comptime BM_SUB = thread_layout.shape[0].value()

    comptime aux = 0  # _cache_operation_to_amd_aux[cache_policy]()

    var lds_ptr = lds_base_ptr

    @parameter
    for n_tile, m_tile, m_sub_tile in product(
        range(N // BN), range(M // BM), range(BM // BM_SUB)
    ):
        var dst_partitions = dst.tile[BM, BN](m_tile, n_tile).tile[BM_SUB, BN](
            m_sub_tile, 0
        )
        var src_partitions = src.tile[BM, BN](m_tile, n_tile).tile[BM_SUB, BN](
            m_sub_tile, 0
        )
        comptime dst_layout = dst_partitions.layout
        # dst need to be contiguous
        comptime assert dst_layout.stride[1].value() == 1, String(dst_layout)
        comptime assert dst_layout.stride[0].value() == 32, String(dst_layout)
        var worker_idx_with_offset = worker_idx + UInt(m_sub_tile * WARP_SIZE)
        var src_dist = src_partitions.vectorize[
            1, simd_width_of[src.dtype]()
        ]().distribute[thread_layout](
            (
                UInt(
                    swizzle.value()(Int(worker_idx_with_offset))
                ) if swizzle else worker_idx_with_offset
            )
            % UInt(WARP_SIZE)
        )
        comptime dtype = src.dtype
        var ptr = dst_partitions.ptr
        var dst_ptr = ptr.address_space_cast[AddressSpace.SHARED]()
        # bc.load_to_lds[width = simd_width_of[src.dtype]()](
        #     Int32(src_offset + src_load_offset),
        #     dst_ptr,
        #     scalar_offset=0,
        # )

        var desc_ptr_ = UnsafePointer[
            Scalar[DType.bfloat16],
            MutAnyOrigin,
            address_space = AddressSpace.BUFFER_RESOURCE,
        ]()

        var ptr_to_ptr = UnsafePointer(to=desc_ptr_)
        var ptr_to_simd = UnsafePointer(to=bc.desc)
        ptr_to_ptr[0] = ptr_to_simd.bitcast[
            UnsafePointer[
                Scalar[DType.bfloat16],
                MutAnyOrigin,
                address_space = AddressSpace.BUFFER_RESOURCE,
            ]
        ]()[0]
        var desc_ptr_llvm = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type = __mlir_type.`!llvm.ptr<8>`
        ](desc_ptr_)

        var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type = __mlir_type.`!llvm.ptr<3>`
        ](type_of(dst_ptr)())

        comptime num_bytes_per_lane = size_of[dtype]() * simd_width_of[dtype]()
        var vector_offset_bytes = Int(src_dist.ptr) - Int(src_partitions.ptr)
        var scalar_offset_bytes = Int(src_partitions.ptr) - Int(src.ptr)

        inlined_assembly[
            "s_mov_b32 m0, $0", NoneType, has_side_effect=True, constraints="s"
        ](lds_ptr)
        __mlir_op.`rocdl.raw.ptr.buffer.load.lds`[
            alias_scopes=_alias_scope_attr,
            _type=None,
        ](
            desc_ptr_llvm,
            shared_ptr3,
            to_i32(Int32(num_bytes_per_lane)),
            to_i32(Int32(vector_offset_bytes)),
            to_i32(Int32(scalar_offset_bytes)),
            to_i32(0),
            to_i32(aux),
        )
        comptime num_bytes_per_warp = UInt32(
            thread_layout.size() * num_bytes_per_lane
        )
        lds_ptr += num_bytes_per_warp


@always_inline
fn load_b_[
    mma_shape: IndexList[3], swizzle: Optional[Swizzle], k_tile_idx: Int
](src: LayoutTensor) -> SIMD[src.dtype, simd_width_of[src.dtype]()]:
    comptime MMA_M = mma_shape[0]
    comptime MMA_K = mma_shape[2]
    comptime assert src.shape[0]() == MMA_M
    comptime simd_width = simd_width_of[src.dtype]()
    var tile = src.tile[MMA_M, MMA_K](0, k_tile_idx)
    comptime thread_layout = Layout.col_major(32, 2) if mma_shape[
        0
    ] == 32 else Layout.col_major(16, 4)
    var dist = tile.vectorize[1, simd_width]().distribute[thread_layout,](
        lane_id()
    )
    var offset = dist.distance(src.ptr)

    @parameter
    if swizzle:
        offset = swizzle.value()(
            offset // Scalar[src.linear_idx_type](simd_width)
        ) * Scalar[src.linear_idx_type](simd_width)

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr<3>`
    ](src.ptr + offset)

    var llvm_res = __mlir_op.`llvm.load`[
        _type = __mlir_type.`vector<8 x bf16>`,
        alignment = to_i64(16),
        noalias_scopes=_alias_scope_attr,
        alias_scopes=_no_alias_scope_attr,
    ](
        shared_ptr3,
    )

    return rebind[SIMD[src.dtype, simd_width]](
        __mlir_op.`pop.cast_from_builtin`[
            _type = SIMD[src.dtype, simd_width]._mlir_type
        ](llvm_res)
    )


@always_inline
fn load_b[
    mma_shape: IndexList[3], swizzle: Optional[Swizzle]
](
    src: LayoutTensor,
    out res: LayoutTensor[
        src.dtype,
        Layout.row_major(src.layout.size() // (WARP_SIZE * 8), 8),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ],
):
    var output = type_of(res).stack_allocation()
    comptime MMA_M = mma_shape[0]
    comptime MMA_K = mma_shape[2]
    comptime M = src.shape[0]() // MMA_M
    comptime N = src.shape[1]() // MMA_K
    var output_vectorized = output.vectorize[1, 8]()

    @parameter
    for i, j in product(range(M), range(N)):
        var out_reg = load_b_[mma_shape, swizzle, j](
            src.tile[MMA_M, src.shape[1]()](i, 0)
        )
        output_vectorized[i + j * M, 0] = rebind[
            type_of(output_vectorized[i + j * M, 0])
        ](out_reg)
    return output
