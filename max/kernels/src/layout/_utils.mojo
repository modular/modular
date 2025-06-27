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

from collections import Optional
from sys import sizeof
from sys.intrinsics import readfirstlane

from buffer import NDBuffer
from gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from gpu.intrinsics import _buffer_resource, make_buffer_resource
from layout import *
from layout.layout_tensor import LayoutTensor, LayoutTensorIter
from layout.tensor_core import TensorCore
from utils import IndexList
from gpu.mma import mma
from .int_tuple import _get_index_type, _get_layout_type, product


struct ManagedLayoutTensor[
    dtype: DType,
    layout: Layout,
    *,
]:
    alias index_type: DType = _get_index_type(layout, AddressSpace.GENERIC)
    alias element_type: DType = _get_layout_type(layout, AddressSpace.GENERIC)
    alias layout_tensor_type = LayoutTensor[
        dtype,
        layout,
        MutableAnyOrigin,
        layout_int_type = Self.element_type,
        linear_idx_type = Self.index_type,
    ]

    var device_data: Optional[DeviceBuffer[dtype]]
    var host_data: HostBuffer[dtype]
    var runtime_layout: RuntimeLayout[
        layout,
        element_type = Self.element_type,
        linear_idx_type = Self.index_type,
    ]
    var ctx: DeviceContext

    @always_inline
    fn __init__(out self) raises:
        self.ctx = DeviceContext(api="cpu")
        self.runtime_layout = {}
        self.device_data = None
        self.host_data = self.ctx.enqueue_create_host_buffer[dtype](
            self.runtime_layout.size()
        )
        self.ctx.synchronize()

    @always_inline
    fn __init__(out self, runtime_layout: RuntimeLayout[layout, **_]) raises:
        self.ctx = DeviceContext(api="cpu")

        constrained[
            runtime_layout.linear_idx_type == Self.index_type,
            "Mismatch of index type for RuntimeLayout: ",
            String(runtime_layout.linear_idx_type),
            " and LayoutTensor: ",
            String(Self.index_type),
            ".",
        ]()

        self.runtime_layout = rebind[__type_of(self.runtime_layout)](
            runtime_layout
        )
        self.device_data = None
        self.host_data = self.ctx.enqueue_create_host_buffer[dtype](
            self.runtime_layout.size()
        )
        self.ctx.synchronize()

    @always_inline
    fn __init__(out self, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.runtime_layout = {}
        self.device_data = ctx.enqueue_create_buffer[dtype](
            self.runtime_layout.size()
        )
        self.host_data = self.ctx.enqueue_create_host_buffer[dtype](
            self.runtime_layout.size()
        )
        self.ctx.synchronize()

    @always_inline
    fn __init__(
        out self, runtime_layout: RuntimeLayout[layout, **_], ctx: DeviceContext
    ) raises:
        constrained[
            runtime_layout.element_type == Self.element_type,
            String(
                "Mismatch of element type for RuntimeLayout:",
                runtime_layout.element_type,
                "and LayoutTensor:",
                Self.element_type,
                ".",
                sep=" ",
            ),
        ]()

        constrained[
            runtime_layout.linear_idx_type == Self.index_type,
            "Mismatch of index type for RuntimeLayout: ",
            String(runtime_layout.linear_idx_type),
            " and LayoutTensor: ",
            String(Self.index_type),
        ]()

        self.ctx = ctx

        self.runtime_layout = rebind[__type_of(self.runtime_layout)](
            runtime_layout
        )
        self.device_data = ctx.enqueue_create_buffer[dtype](
            self.runtime_layout.size()
        )
        self.host_data = self.ctx.enqueue_create_host_buffer[dtype](
            self.runtime_layout.size()
        )
        self.ctx.synchronize()

    fn device_tensor[
        update: Bool = True
    ](self) raises -> Self.layout_tensor_type:
        debug_assert(
            Bool(self.ctx.api() != "cpu"),
            "device_tensor cannot be constructed for host only tensor.",
        )

        @parameter
        if update:
            self._update_device()

        @parameter
        if layout.all_dims_known():
            return Self.layout_tensor_type(
                self.device_data.value()._unsafe_ptr(),
            )
        else:
            return Self.layout_tensor_type(
                self.device_data.value()._unsafe_ptr(),
                self.runtime_layout,
            )

    fn device_buffer[
        update: Bool = True
    ](self) raises -> NDBuffer[dtype, 2, MutableAnyOrigin]:
        @parameter
        if update:
            self._update_device()

        constrained[layout.rank() == 2, "Only support exporting 2D NDBuffer."]()

        M = self.runtime_layout.dim(0)
        N = self.runtime_layout.dim(1)

        return NDBuffer[dtype, 2](
            self.device_data.value()._unsafe_ptr(), (M, N)
        )

    fn tensor[update: Bool = True](self) raises -> Self.layout_tensor_type:
        @parameter
        if update:
            self._update_host()

        @parameter
        if layout.all_dims_known():
            return Self.layout_tensor_type(
                self.host_data.unsafe_ptr(),
            )
        else:
            return Self.layout_tensor_type(
                self.host_data.unsafe_ptr(),
                self.runtime_layout,
            )

    fn buffer[
        update: Bool = True
    ](self) raises -> NDBuffer[dtype, 2, MutableAnyOrigin]:
        @parameter
        if update:
            self._update_host()

        constrained[layout.rank() == 2, "Only support exporting 2D NDBuffer."]()

        M = self.runtime_layout.dim(0)
        N = self.runtime_layout.dim(1)

        return NDBuffer[dtype, 2](self.host_data.unsafe_ptr(), (M, N))

    fn _update_device(self) raises:
        if self.ctx.api() != "cpu":
            self.ctx.enqueue_copy(self.device_data.value(), self.host_data)
            self.ctx.synchronize()

    fn _update_host(self) raises:
        if self.ctx.api() != "cpu":
            self.ctx.enqueue_copy(self.host_data, self.device_data.value())
            self.ctx.synchronize()

    @always_inline
    fn __del__(owned self):
        pass


fn load_to_simd(
    tensor: LayoutTensor,
    out res: SIMD[tensor.dtype, product(tensor.layout.shape)],
):
    constrained[
        tensor.layout.all_dims_known(),
        "load_to_simd is supported only for tensors with known layout",
    ]()
    alias size = __type_of(res).size
    return rebind[__type_of(res)](
        tensor.reshape[Layout(size)]().vectorize[size]()[0]
    )


@always_inline
fn _get_bounds(tensor: LayoutTensor) -> Int:
    constrained[
        tensor.element_layout.all_dims_known(),
        "Element layout must be known for _get_bounds",
    ]()
    constrained[
        tensor.element_layout.size() == 1, "Element layout must be a scalar"
    ]()

    if tensor.dim[0]() == 0 or tensor.dim[1]() == 0:
        return 0

    alias element_layout = tensor.element_layout
    alias element_offset = element_layout(element_layout.size() - 1)
    var strides = tensor.runtime_layout.stride.value
    var offset = tensor._get_offset(
        strides, IndexList[2](tensor.dim[0]() - 1, tensor.dim[1]() - 1)
    )
    return offset + 1


@always_inline
fn get_amd_buffer_descriptor(tensor: LayoutTensor) -> _buffer_resource:
    var ptr = tensor.ptr
    var size = _get_bounds(tensor)
    return make_buffer_resource(readfirstlane(ptr), readfirstlane(size))


@always_inline
fn get_amd_buffer_descriptor(
    tensor_iter: LayoutTensorIter, bound: Int
) -> _buffer_resource:
    return make_buffer_resource(
        readfirstlane(tensor_iter.ptr), readfirstlane(bound)
    )


@always_inline
fn idx2crd[layout: Layout](idx: Int) -> IndexList[layout.rank()]:
    constrained[layout.all_dims_known(), "Layout must be known for idx2crd"]()
    var res = IndexList[layout.rank()]()

    @parameter
    for i in range(layout.rank()):
        alias stride = layout.stride[i].value()
        alias shape = layout.shape[i].value()
        res[i] = (Int(idx) // stride) % shape
    return res


@always_inline
fn hash(tensor: LayoutTensor) -> Int:
    # Calculate hash of the content of the layout tensor, it can be useful for debugging
    constrained[
        sizeof[tensor.dtype]() == 2, "Only support 2 byte types for hash"
    ]()
    var hash_value: Int = 0
    alias size = tensor.layout.size()

    for i in range(tensor.dim[0]()):
        for j in range(tensor.dim[1]()):
            var val = tensor[i, j]
            var addr = UnsafePointer(to=val)
            var addr_int = addr.bitcast[Int16]()
            var val_int = addr_int[0]
            hash_value = ((hash_value << 5) + hash_value) + Int(val_int)
    return hash_value


@fieldwise_init
struct TensorCoreKGroup[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    k_group_size: Int,
    transpose_b: Bool = False,
]:
    """TensorCoreKGroup provides a wrapper around TensorCore to support multiple MMAs along the K dimension.

    Enables larger K dimension operations by decomposing them into multiple smaller MMA operations.
    Currently only being used for AMD GPUs to enable 16×16×32 operations using two 16×16×16 MMAs.

    Parameters:
        out_type: The data type for output/accumulation operations.
        in_type: The data type for input matrix elements.
        shape: The shape parameters for individual MMA operations [M, N, K].
        k_group_size: Number of MMA operations along the K dimension.
        transpose_b: Whether to transpose the b matrix. Defaults to False.
    """

    alias mma_op = TensorCore[out_type, in_type, shape, transpose_b]()

    @staticmethod
    @always_inline
    fn _mma_inner[
        c_linear_map: Layout
    ](a_frag: LayoutTensor, b_frag: LayoutTensor, c_frag: LayoutTensor,):
        alias num_m_mmas = a_frag.shape[0]()
        alias num_n_mmas = b_frag.shape[0]()

        constrained[
            c_frag.shape[0]() == num_m_mmas * num_n_mmas,
            "Fragments size mismatch. Expected c_frag shape[0] to be num_m_mmas"
            " * num_n_mmas = "
            + String(num_m_mmas * num_n_mmas)
            + ", got "
            + String(c_frag.shape[0]()),
        ]()

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                alias c_idx = c_linear_map(IntTuple(m_mma, n_mma))
                mma(
                    c_frag[c_idx, 0],
                    a_frag[m_mma, 0],
                    b_frag[n_mma, 0],
                    c_frag[c_idx, 0],
                )

    @staticmethod
    @always_inline
    fn mma[
        swap_a_b: Bool = False
    ](
        a_reg_tile: LayoutTensor,
        b_reg_tile: LayoutTensor,
        c_reg_tile: LayoutTensor,
    ):
        """Perform multiple matrix multiply-accumulate operations along the K dimension.

        Executes k_group_size MMA operations, processing slices of the K dimension
        and accumulating results in c_reg_tile.

        Parameters:
            swap_a_b: Whether to swap a and b operands. Defaults to False.

        Args:
            a_reg_tile: Input matrix a fragments [num_m_mmas, k_group_size * a_frag_size].
            b_reg_tile: Input matrix b fragments [num_n_mmas, k_group_size * b_frag_size].
            c_reg_tile: Accumulation matrix c fragments, modified in-place.
        """
        alias num_m_mmas = a_reg_tile.shape[0]()
        alias num_n_mmas = b_reg_tile.shape[0]()

        alias a_frag_size = Self.mma_op.a_reg_type.size
        alias b_frag_size = Self.mma_op.b_reg_type.size
        alias c_frag_size = Self.mma_op.c_reg_type.size

        constrained[k_group_size > 0, "k_group_size must be greater than 0"]()

        constrained[
            c_reg_tile.shape[1]() == c_frag_size,
            "c_reg_tile.shape[1]() must be equal to c_frag_size",
        ]()
        constrained[
            a_reg_tile.shape[1]() == k_group_size * a_frag_size,
            "a_reg_tile.shape[1]() must be equal to k_group_size * a_frag_size",
        ]()
        constrained[
            b_reg_tile.shape[1]() == k_group_size * b_frag_size,
            "b_reg_tile.shape[1]() must be equal to k_group_size * b_frag_size",
        ]()

        alias c_linear_map = Layout.row_major(
            num_n_mmas, num_m_mmas
        ) if swap_a_b else Layout.col_major(num_m_mmas, num_n_mmas)

        @parameter
        for k in range(k_group_size):
            var a_reg_k = a_reg_tile.tile[num_m_mmas, a_frag_size](0, k)
            var b_reg_k = b_reg_tile.tile[num_n_mmas, b_frag_size](0, k)
            Self._mma_inner[c_linear_map](
                b_reg_k.vectorize[1, b_frag_size](),
                a_reg_k.vectorize[1, a_frag_size](),
                c_reg_tile.vectorize[1, c_frag_size](),
            )
