# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional
from sys import bitwidthof

from buffer import NDBuffer
from gpu.host import DeviceBuffer, DeviceContext
from layout import *
from layout.layout_tensor import _get_index_type, LayoutTensor
from memory import UnsafePointer

from utils.index import Index

from .int_tuple import product
from gpu.intrinsics import make_buffer_resource, _buffer_resource


struct ManagedLayoutTensor[
    dtype: DType,
    layout: Layout,
    *,
]:
    alias layout_bitwidth = bitwidthof[_get_index_type(AddressSpace.GENERIC)]()
    var device_data: Optional[DeviceBuffer[dtype]]
    var host_data: UnsafePointer[Scalar[dtype]]
    var runtime_layout: RuntimeLayout[layout, bitwidth = Self.layout_bitwidth]
    var ctx: Optional[DeviceContext]

    @always_inline
    fn __init__(out self):
        self.ctx = None
        self.device_data = None
        self.host_data = __type_of(self.host_data).alloc(layout.size())
        self.runtime_layout = __type_of(self.runtime_layout)()

    @always_inline
    fn __init__(out self, runtime_layout: RuntimeLayout[layout, **_]):
        self.ctx = None
        self.device_data = None
        self.host_data = __type_of(self.host_data).alloc(runtime_layout.size())
        self.runtime_layout = rebind[__type_of(self.runtime_layout)](
            runtime_layout
        )

    @always_inline
    fn __init__(out self, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.device_data = ctx.create_buffer_sync[dtype](layout.size())
        self.host_data = __type_of(self.host_data).alloc(layout.size())
        self.runtime_layout = __type_of(self.runtime_layout)()

    @always_inline
    fn __init__(
        out self, runtime_layout: RuntimeLayout[layout, **_], ctx: DeviceContext
    ) raises:
        self.ctx = ctx
        self.device_data = ctx.create_buffer_sync[dtype](runtime_layout.size())
        self.host_data = __type_of(self.host_data).alloc(runtime_layout.size())
        self.runtime_layout = rebind[__type_of(self.runtime_layout)](
            runtime_layout
        )

    fn device_tensor[
        update: Bool = True
    ](self) raises -> LayoutTensor[dtype, layout]:
        debug_assert(
            Bool(self.ctx),
            "device_tensor cannot be constructed for host only tensor.",
        )

        @parameter
        if update:
            self._update_device()

        @parameter
        if layout.all_dims_known():
            return LayoutTensor[dtype, layout](
                self.device_data.value(),
            )
        else:
            return LayoutTensor[dtype, layout](
                self.device_data.value().unsafe_ptr(),
                self.runtime_layout,
            )

    fn device_buffer[update: Bool = True](self) raises -> NDBuffer[dtype, 2]:
        @parameter
        if update:
            self._update_device()

        constrained[layout.rank() == 2, "Only support exporting 2D NDBuffer."]()

        M = self.runtime_layout.dim(0)
        N = self.runtime_layout.dim(1)

        return NDBuffer[dtype, 2](self.device_data.value().unsafe_ptr(), (M, N))

    fn tensor[update: Bool = True](self) raises -> LayoutTensor[dtype, layout]:
        @parameter
        if update:
            self._update_host()

        @parameter
        if layout.all_dims_known():
            return LayoutTensor[dtype, layout](
                self.host_data,
            )
        else:
            return LayoutTensor[dtype, layout](
                self.host_data,
                self.runtime_layout,
            )

    fn buffer[update: Bool = True](self) raises -> NDBuffer[dtype, 2]:
        @parameter
        if update:
            self._update_host()

        constrained[layout.rank() == 2, "Only support exporting 2D NDBuffer."]()

        M = self.runtime_layout.dim(0)
        N = self.runtime_layout.dim(1)

        return NDBuffer[dtype, 2](self.host_data, (M, N))

    fn _update_device(self) raises:
        if self.ctx:
            self.ctx.value().copy(self.device_data.value(), self.host_data)

    fn _update_host(self) raises:
        if self.ctx:
            self.ctx.value().copy(self.host_data, self.device_data.value())

    @always_inline
    fn __del__(owned self):
        self.host_data.free()


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
fn _get_size(tensor: LayoutTensor) -> Int:
    @parameter
    if tensor.layout.all_dims_known():
        alias size = tensor.layout.size()
        return size
    else:
        return tensor.runtime_layout.size()


@always_inline
fn get_amd_buffer_descriptor(tensor: LayoutTensor) -> _buffer_resource:
    var ptr = tensor.ptr
    var size = _get_size(tensor)
    return make_buffer_resource(ptr, size)
