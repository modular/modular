# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from gpu.host.cuda_instance import CudaInstance
from gpu.host.device import Device
from gpu.host.context import Context
from gpu.host.function import Function
from gpu.host.stream import Stream


@register_passable
struct DeviceBuffer[type: AnyTrivialRegType](Sized):
    var ptr: Pointer[type]
    var ctx_ptr: UnsafePointer[DeviceContext]
    var size: Int
    var owning: Bool

    fn __init__(inout self, ctx: DeviceContext, size: Int) raises:
        self.ctx_ptr = UnsafePointer[DeviceContext].address_of(ctx)
        self.ptr = self.ctx_ptr[].cuda_context.malloc[type](size)
        self.size = size
        self.owning = True

    fn __init__(inout self):
        self.ctx_ptr = UnsafePointer[DeviceContext]()
        self.ptr = Pointer[type]()
        self.size = 0
        self.owning = False

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

    fn create_sub_buffer[
        view_type: AnyTrivialRegType
    ](self, offset: Int, size: Int) raises -> DeviceBuffer[view_type]:
        if sizeof[view_type]() * (offset + size) > sizeof[type]() * self.size:
            raise Error("offset and size exceed original buffer size")
        var sub_buffer = DeviceBuffer[view_type]()
        sub_buffer.ctx_ptr = self.ctx_ptr
        sub_buffer.ptr = rebind[Pointer[view_type]](self.ptr).offset(offset)
        sub_buffer.size = size
        sub_buffer.owning = False
        return sub_buffer


struct DeviceFunction[func_type: AnyTrivialRegType, //, func: func_type]:
    var ctx_ptr: UnsafePointer[DeviceContext]
    var cuda_function: Function[func]

    fn __init__(inout self, ctx: DeviceContext) raises:
        self.ctx_ptr = UnsafePointer[DeviceContext].address_of(ctx)
        self.cuda_function = Function[func](self.ctx_ptr[].cuda_context)


struct DeviceContext:
    # NOTE: The fields' declaration order matches the destruction order
    var cuda_stream: Stream
    var cuda_context: Context
    var cuda_instance: CudaInstance

    fn __init__(inout self) raises:
        self.cuda_instance = CudaInstance()
        self.cuda_context = Context(Device(self.cuda_instance))
        self.cuda_stream = Stream(self.cuda_context)

    fn __copyinit__(inout self, existing: Self):
        self.cuda_instance = existing.cuda_instance
        self.cuda_context = existing.cuda_context
        self.cuda_stream = existing.cuda_stream

    fn create_buffer[
        type: AnyTrivialRegType
    ](self, size: Int) raises -> DeviceBuffer[type]:
        return DeviceBuffer[type](self, size)

    fn compile_function[
        func_type: AnyTrivialRegType, //, func: func_type
    ](self) raises -> DeviceFunction[func]:
        return DeviceFunction[func](self)

    fn enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
    ) raises:
        f.cuda_function._call_pack(
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            stream=self.cuda_stream,
        )

    fn enqueue_copy_to_device[
        type: AnyTrivialRegType
    ](self, buf: DeviceBuffer[type], ptr: Pointer[type]) raises:
        self.cuda_context.copy_host_to_device_async(
            buf.ptr, ptr, len(buf), self.cuda_stream
        )

    fn enqueue_copy_from_device[
        type: AnyTrivialRegType
    ](self, ptr: Pointer[type], buf: DeviceBuffer[type]) raises:
        self.cuda_context.copy_device_to_host_async(
            ptr, buf.ptr, len(buf), self.cuda_stream
        )

    fn copy_to_device_sync[
        type: AnyTrivialRegType
    ](self, buf: DeviceBuffer[type], ptr: Pointer[type]) raises:
        self.cuda_context.copy_host_to_device(buf.ptr, ptr, len(buf))

    fn copy_from_device_sync[
        type: AnyTrivialRegType
    ](self, ptr: Pointer[type], buf: DeviceBuffer[type]) raises:
        self.cuda_context.copy_device_to_host(ptr, buf.ptr, len(buf))

    fn synchronize(self) raises:
        self.cuda_stream.synchronize()
