# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .device import Device, _get_driver_path, _CDevice
from max._utils import call_dylib_func
from sys.ffi import DLHandle
from ._driver_library import DriverLibrary, ManagedDLHandle
from gpu.host import DeviceContext, KernelProfilingInfo


fn alloc_device_context() -> UnsafePointer[DeviceContext]:
    try:
        var ctx_ptr = UnsafePointer[DeviceContext].alloc(1)
        ctx_ptr.init_pointee_move(DeviceContext())
        return ctx_ptr
    except e:
        return abort[UnsafePointer[DeviceContext]](e)


fn alloc_device_buffer(
    ctx: UnsafePointer[DeviceContext], bytes: Int
) -> DTypePointer[DType.uint8]:
    try:
        var ret = ctx[].cuda_context.malloc_async[DType.uint8](
            bytes, ctx[].cuda_stream
        )
        ctx[].cuda_stream.synchronize()
        return ret
    except e:
        return abort[Pointer[UInt8]]()


fn copy_device_to_host(
    ctx: UnsafePointer[DeviceContext],
    dev_ptr: DTypePointer[DType.uint8],
    host_ptr: DTypePointer[DType.uint8],
    size: Int,
):
    try:
        ctx[].cuda_context.copy_device_to_host_async(
            host_ptr, dev_ptr, size, ctx[].cuda_stream
        )
        ctx[].cuda_stream.synchronize()
    except e:
        abort(e)


fn copy_host_to_device(
    ctx: UnsafePointer[DeviceContext],
    dev_ptr: DTypePointer[DType.uint8],
    host_ptr: DTypePointer[DType.uint8],
    size: Int,
):
    try:
        ctx[].cuda_context.copy_host_to_device_async(
            dev_ptr, host_ptr, size, ctx[].cuda_stream
        )
        ctx[].cuda_stream.synchronize()
    except e:
        abort(e)


fn copy_device_to_device(
    ctx: UnsafePointer[DeviceContext],
    dst_ptr: DTypePointer[DType.uint8],
    src_ptr: DTypePointer[DType.uint8],
    size: Int,
):
    try:
        ctx[].cuda_context.copy_device_to_device_async(
            dst_ptr, src_ptr, size, ctx[].cuda_stream
        )
        ctx[].cuda_stream.synchronize()
    except e:
        abort(e)


fn free_buffer(
    ctx: UnsafePointer[DeviceContext], ptr: DTypePointer[DType.uint8]
):
    try:
        ctx[].cuda_context.free_async(ptr, ctx[].cuda_stream)
        ctx[].cuda_stream.synchronize()
    except e:
        abort(e)


fn free_context(ctx: UnsafePointer[DeviceContext]):
    ctx.destroy_pointee()


@value
@register_passable("trivial")
struct ContextAPIFuncPtrs:
    var alloc_device_context: fn () -> UnsafePointer[DeviceContext]
    var alloc_device_buffer: fn (
        UnsafePointer[DeviceContext], Int
    ) -> DTypePointer[DType.uint8]
    var copy_device_to_host: fn (
        UnsafePointer[DeviceContext],
        DTypePointer[DType.uint8],
        DTypePointer[DType.uint8],
        Int,
    ) -> None
    var copy_host_to_device: fn (
        UnsafePointer[DeviceContext],
        DTypePointer[DType.uint8],
        DTypePointer[DType.uint8],
        Int,
    ) -> None
    var copy_device_to_device: fn (
        UnsafePointer[DeviceContext],
        DTypePointer[DType.uint8],
        DTypePointer[DType.uint8],
        Int,
    ) -> None
    var free_buffer: fn (
        UnsafePointer[DeviceContext],
        DTypePointer[DType.uint8],
    ) -> None
    var free_context: fn (UnsafePointer[DeviceContext],) -> None

    fn __init__(inout self):
        self.alloc_device_context = alloc_device_context
        self.alloc_device_buffer = alloc_device_buffer
        self.copy_device_to_host = copy_device_to_host
        self.copy_host_to_device = copy_host_to_device
        self.copy_device_to_device = copy_device_to_device
        self.free_buffer = free_buffer
        self.free_context = free_context


fn cuda_device(gpu_id: Int = 0) raises -> Device:
    var lib = ManagedDLHandle(DLHandle(_get_driver_path()))
    alias func_name_create = "M_createCUDADevice"

    var device_context_func_ptrs = ContextAPIFuncPtrs()

    var _cdev = _CDevice(
        call_dylib_func[UnsafePointer[NoneType]](
            lib.get_handle(),
            func_name_create,
            gpu_id,
            UnsafePointer[ContextAPIFuncPtrs].address_of(
                device_context_func_ptrs
            ),
        )
    )
    _ = device_context_func_ptrs
    return Device(lib^, owned_ptr=_cdev)
