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

from collections import Optional
from sys import size_of
from sys.intrinsics import readfirstlane

from gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from gpu.intrinsics import AMDBufferResource
from gpu.compute.mma import mma
from layout import *
from layout.layout_tensor import LayoutTensor, LayoutTensorIter
from memory.unsafe import bitcast

from utils import IndexList

from .int_tuple import _get_index_type, _get_layout_type, product


struct ManagedLayoutTensor[
    dtype: DType,
    layout: Layout,
    *,
]:
    comptime index_type: DType = _get_index_type(
        Self.layout, AddressSpace.GENERIC
    )
    comptime element_type: DType = _get_layout_type(
        Self.layout, AddressSpace.GENERIC
    )
    comptime layout_tensor_type = LayoutTensor[
        Self.dtype,
        Self.layout,
        MutAnyOrigin,
        layout_int_type = Self.element_type,
        linear_idx_type = Self.index_type,
    ]

    var device_data: Optional[DeviceBuffer[Self.dtype]]
    var host_data: HostBuffer[Self.dtype]
    var runtime_layout: RuntimeLayout[
        Self.layout,
        element_type = Self.element_type,
        linear_idx_type = Self.index_type,
    ]
    var ctx: DeviceContext

    @always_inline
    fn __init__(out self) raises:
        self.ctx = DeviceContext(api="cpu")
        self.runtime_layout = {}
        self.device_data = None
        self.host_data = self.ctx.enqueue_create_host_buffer[Self.dtype](
            self.runtime_layout.size()
        )
        self.ctx.synchronize()

    @always_inline
    fn __init__(
        out self, runtime_layout: RuntimeLayout[Self.layout, ...]
    ) raises:
        self.ctx = DeviceContext(api="cpu")

        comptime assert runtime_layout.linear_idx_type == Self.index_type, (
            "Mismatch of index type for RuntimeLayout: "
            + String(runtime_layout.linear_idx_type)
            + " and LayoutTensor: "
            + String(Self.index_type)
            + "."
        )

        self.runtime_layout = rebind[type_of(self.runtime_layout)](
            runtime_layout
        )
        self.device_data = None
        self.host_data = self.ctx.enqueue_create_host_buffer[Self.dtype](
            self.runtime_layout.size()
        )
        self.ctx.synchronize()

    @always_inline
    fn __init__(out self, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.runtime_layout = {}
        self.device_data = ctx.enqueue_create_buffer[Self.dtype](
            self.runtime_layout.size()
        )
        self.host_data = self.ctx.enqueue_create_host_buffer[Self.dtype](
            self.runtime_layout.size()
        )
        self.ctx.synchronize()

    @always_inline
    fn __init__(
        out self,
        runtime_layout: RuntimeLayout[Self.layout, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            runtime_layout.element_type == Self.element_type
        ), String(
            "Mismatch of element type for RuntimeLayout:",
            runtime_layout.element_type,
            "and LayoutTensor:",
            Self.element_type,
            ".",
            sep=" ",
        )

        comptime assert runtime_layout.linear_idx_type == Self.index_type, (
            "Mismatch of index type for RuntimeLayout: "
            + String(runtime_layout.linear_idx_type)
            + " and LayoutTensor: "
            + String(Self.index_type)
        )

        self.ctx = ctx

        self.runtime_layout = rebind[type_of(self.runtime_layout)](
            runtime_layout
        )
        self.device_data = ctx.enqueue_create_buffer[Self.dtype](
            self.runtime_layout.size()
        )
        self.host_data = self.ctx.enqueue_create_host_buffer[Self.dtype](
            self.runtime_layout.size()
        )
        self.ctx.synchronize()

    fn device_tensor[
        update: Bool = True
    ](self) raises -> Self.layout_tensor_type:
        debug_assert(
            self.ctx.api() != "cpu",
            "device_tensor cannot be constructed for host only tensor.",
        )

        comptime if update:
            self._update_device()

        comptime if Self.layout.all_dims_known():
            return Self.layout_tensor_type(
                self.device_data.value().unsafe_ptr(),
            )
        else:
            return Self.layout_tensor_type(
                self.device_data.value().unsafe_ptr(),
                self.runtime_layout,
            )

    fn tensor[update: Bool = True](self) raises -> Self.layout_tensor_type:
        comptime if update:
            self._update_host()

        comptime if Self.layout.all_dims_known():
            return Self.layout_tensor_type(
                self.host_data.unsafe_ptr(),
            )
        else:
            return Self.layout_tensor_type(
                self.host_data.unsafe_ptr(),
                self.runtime_layout,
            )

    fn _update_device(self) raises:
        if self.ctx.api() != "cpu":
            self.ctx.enqueue_copy(self.device_data.value(), self.host_data)
            self.ctx.synchronize()

    fn _update_host(self) raises:
        if self.ctx.api() != "cpu":
            self.ctx.enqueue_copy(self.host_data, self.device_data.value())
            self.ctx.synchronize()

    @always_inline
    fn __del__(deinit self):
        pass


fn load_to_simd(
    tensor: LayoutTensor,
    out res: SIMD[tensor.dtype, product(tensor.layout.shape)],
):
    comptime assert (
        tensor.layout.all_dims_known()
    ), "load_to_simd is supported only for tensors with known layout"
    comptime size = type_of(res).size
    return rebind[type_of(res)](
        tensor.reshape[Layout(size)]().vectorize[size]()[0]
    )


@always_inline
fn _get_bounds(tensor: LayoutTensor) -> Int:
    comptime assert (
        tensor.element_layout.all_dims_known()
    ), "Element layout must be known for _get_bounds"
    comptime assert (
        tensor.element_layout.size() == 1
    ), "Element layout must be a scalar"

    if tensor.dim[0]() == 0 or tensor.dim[1]() == 0:
        return 0

    comptime element_layout = tensor.element_layout
    comptime element_offset = element_layout(element_layout.size() - 1)
    comptime tensor_t = type_of(tensor)
    var strides = tensor.runtime_layout.stride.value
    var offset = tensor._get_offset(
        strides,
        tensor_t.idx_list_t[2](tensor.dim[0]() - 1, tensor.dim[1]() - 1),
    )
    return offset + 1


@always_inline
fn make_amd_buffer_resource(
    tensor: LayoutTensor,
) -> AMDBufferResource:
    var ptr = tensor.ptr
    var size = _get_bounds(tensor)
    return AMDBufferResource(readfirstlane(ptr), readfirstlane(size))


@always_inline
fn make_amd_buffer_resource(
    tensor_iter: LayoutTensorIter, bound: Int
) -> AMDBufferResource:
    return AMDBufferResource(
        readfirstlane(tensor_iter.ptr), readfirstlane(bound)
    )


@always_inline
fn idx2crd[layout: Layout](idx: Int) -> IndexList[layout.rank()]:
    comptime assert layout.all_dims_known(), "Layout must be known for idx2crd"
    var res = IndexList[layout.rank()]()

    comptime for i in range(layout.rank()):
        comptime stride = layout.stride[i].value()
        comptime shape = layout.shape[i].value()
        res[i] = (idx // stride) % shape
    return res


@always_inline
fn hash(tensor: LayoutTensor) -> Int:
    # Calculate hash of the content of the layout tensor, it can be useful for debugging
    comptime assert (
        size_of[tensor.dtype]() == 2
    ), "Only support 2 byte types for hash"
    var hash_value: Int = 0
    comptime size = tensor.layout.size()

    for i in range(tensor.dim[0]()):
        for j in range(tensor.dim[1]()):
            var val = tensor[i, j]
            var addr = UnsafePointer(to=val)
            var addr_int = addr.bitcast[Int16]()
            var val_int = addr_int[0]
            hash_value = ((hash_value << 5) + hash_value) + Int(val_int)
    return hash_value
