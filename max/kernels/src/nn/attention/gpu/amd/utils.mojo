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

from memory import LegacyUnsafePointer as UnsafePointer
from sys import align_of, simd_width_of, size_of

from gpu import lane_id, thread_idx, block_idx
from gpu import warp_id as get_warp_id
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


@always_inline
fn pad[dtype: DType, depth: Int, size: Int]() -> Int:
    alias simd_width = simd_width_of[dtype]()
    alias padding = 0 if depth == 64 else size // simd_width
    return size + padding


alias LocalLayoutTensor[dtype: DType, layout: Layout] = LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
    address_space = AddressSpace.LOCAL,
]

alias SharedLayoutTensor[dtype: DType, layout: Layout] = LayoutTensor[
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
        Scalar[Self.dtype], address_space = AddressSpace.SHARED
    ]
    # p_smem is used for p
    var k_smem: UnsafePointer[
        Scalar[Self.dtype], address_space = AddressSpace.SHARED
    ]
    var v_smem: UnsafePointer[
        Scalar[Self.dtype], address_space = AddressSpace.SHARED
    ]
    # k_v_smem is used for k, v, and scratch
    alias alignment = align_of[SIMD[Self.dtype, simd_width_of[Self.dtype]()]]()
    alias accum_type = get_accum_type[Self.dtype]()
    alias p_smem_size = Self.BM * Self.BN if Self.token_gen else 0
    alias simd_width = simd_width_of[Self.dtype]()
    # depth // simd_width is the padding
    alias k_smem_size = Self.BN * (Self.depth if Self.full_kv else Self.BK) * (
        2 if Self.double_buffer else 1
    )
    alias v_smem_size = (Self.BN if Self.full_kv else Self.BK) * (
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

        alias kv_smem_size = max(Self.k_smem_size, Self.v_smem_size)

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
    alias kv_num_heads = Self.num_heads // Self.group
    # BHSD layout for q and kv cache
    alias q_gmem_layout = Layout(
        IntTuple(Int(Self.BM), Int(Self.q_depth)),
        IntTuple(Int(Self.num_heads * Self.q_depth), 1),
    ) if not Self.token_gen else Layout.row_major(
        Int(Self.BM), Int(Self.q_depth)
    )

    alias output_gmem_layout = Layout(
        IntTuple(Int(Self.BM), Int(Self.output_depth)),
        IntTuple(Int(Self.num_heads * Self.output_depth), 1),
    ) if not Self.token_gen else Layout.row_major(
        Int(Self.BM), Int(Self.output_depth)
    )

    alias kv_gmem_layout = Layout(
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
            Self.BM, UInt(seq_len) - q_tile_idx * Self.BM
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
        ptr: UnsafePointer[Scalar[qtype]],
        out result: LayoutTensor[
            qtype,
            Self.q_gmem_layout,
            MutAnyOrigin,
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
            MutAnyOrigin,
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
                Int(kv_tile_num_rows), Int(Self.depth)
            ),
            type_of(result.runtime_layout.stride)(
                Int(Self.kv_num_heads * Self.depth), 1
            ),
        )

        return {ptr, kv_runtime_layout}
