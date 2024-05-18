# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from gpu.host.cuda_instance import CudaInstance
from gpu.host.device import Device
from gpu.host.context import Context


trait DeviceBufferBase:
    fn get_ptr(self) -> Pointer[NoneType]:
        pass


@register_passable
struct DeviceBuffer[type: AnyRegType](DeviceBufferBase, Sized):
    var ptr: Pointer[type]
    var ctx_ptr: Pointer[DeviceContext]
    var size: Int
    var owning: Bool

    fn __init__(inout self, ctx: DeviceContext, size: Int) raises:
        self.ctx_ptr = Pointer[DeviceContext].address_of(ctx)
        self.ptr = self.ctx_ptr[].cuda_context.malloc[type](size)
        self.size = size
        self.owning = True

    fn __copyinit__(inout self, existing: Self):
        self.ctx_ptr = existing.ctx_ptr
        self.ptr = existing.ptr
        self.size = existing.size
        self.owning = False

    fn __del__(owned self):
        try:
            if self.owning and self.ptr:
                self.ctx_ptr[].cuda_context.free(self.ptr)
                self.ptr = Pointer[type]()
                self.size = 0
        except e:
            print("something went wrong", e)

    fn __len__(self) -> Int:
        return self.size

    fn get_ptr(self) -> Pointer[NoneType]:
        return Pointer[NoneType]()


struct DeviceContext:
    var cuda_instance: CudaInstance
    var cuda_context: Context

    fn __init__(inout self) raises:
        self.cuda_instance = CudaInstance()
        self.cuda_context = Context(Device(self.cuda_instance))

    fn __copyinit__(inout self, existing: Self):
        self.cuda_instance = existing.cuda_instance
        self.cuda_context = existing.cuda_context

    fn create_buffer[
        type: AnyRegType
    ](self, size: Int) raises -> DeviceBuffer[type]:
        return DeviceBuffer[type](self, size)

    fn enqueue[func_type: AnyRegType, func: func_type](self) raises:
        var f = Function[func_type, func](self.cuda_context)

    fn enqueue_copy_to_device[
        type: AnyRegType
    ](self, buf: DeviceBuffer[type], ptr: Pointer[type]) raises:
        self.cuda_context.copy_host_to_device(buf.ptr, ptr, len(buf))

    fn enqueue_copy_from_device[
        type: AnyRegType
    ](self, ptr: Pointer[type], buf: DeviceBuffer[type]) raises:
        self.cuda_context.copy_device_to_host(ptr, buf.ptr, len(buf))
