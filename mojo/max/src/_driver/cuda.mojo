# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .device import Device, _get_driver_path, _CDevice
from max._utils import call_dylib_func
from sys.ffi import DLHandle
from ._driver_library import DriverLibrary, ManagedDLHandle
from gpu.host import DeviceContext, DeviceBuffer


fn alloc_device_context() -> UnsafePointer[DeviceContext]:
    try:
        var ctx_ptr = UnsafePointer[DeviceContext].alloc(1)
        ctx_ptr.init_pointee_move(DeviceContext())
        return ctx_ptr
    except e:
        return abort[UnsafePointer[DeviceContext]](e)


fn alloc_device_buffer(
    ctx: UnsafePointer[DeviceContext], bytes: Int
) -> Pointer[UInt8]:
    try:
        # FIXME: create_buffer returns DType now, we need to update things downstream
        var buf = ctx[].create_buffer[DType.uint8](bytes)
        return Pointer[UInt8](address=int(buf^.take_ptr()))
    except e:
        return abort[Pointer[UInt8]](e)


fn copy_device_to_host(
    ctx: UnsafePointer[DeviceContext],
    dev_buf: Pointer[UInt8],
    host_buf: Pointer[UInt8],
    size: Int,
):
    try:
        ctx[].copy_from_device_sync(
            host_buf, DeviceBuffer(ctx[], dev_buf, size, owning=False)
        )
    except e:
        abort(e)


fn copy_host_to_device(
    ctx: UnsafePointer[DeviceContext],
    dev_buf: Pointer[UInt8],
    host_buf: Pointer[UInt8],
    size: Int,
):
    try:
        ctx[].copy_to_device_sync(
            DeviceBuffer(ctx[], dev_buf, size, owning=False), host_buf
        )
    except e:
        abort(e)


fn copy_device_to_device(
    ctx: UnsafePointer[DeviceContext],
    dst_buf: Pointer[UInt8],
    src_buf: Pointer[UInt8],
    size: Int,
):
    try:
        ctx[].copy_device_to_device_sync(
            DeviceBuffer(ctx[], dst_buf, size, owning=False),
            DeviceBuffer(ctx[], src_buf, size, owning=False),
        )
    except e:
        abort(e)


fn free_buffer(ctx: UnsafePointer[DeviceContext], ptr: Pointer[UInt8]):
    _ = DeviceBuffer[DType.uint8](ctx[], ptr, 0, owning=True)


fn free_context(ctx: UnsafePointer[DeviceContext]):
    ctx.destroy_pointee()


@value
@register_passable("trivial")
struct ContextAPIFuncPtrs:
    var alloc_device_context: fn () -> UnsafePointer[DeviceContext]
    var alloc_device_buffer: fn (UnsafePointer[DeviceContext], Int) -> Pointer[
        UInt8
    ]
    var copy_device_to_host: fn (
        UnsafePointer[DeviceContext],
        Pointer[UInt8],
        Pointer[UInt8],
        Int,
    ) -> None
    var copy_host_to_device: fn (
        UnsafePointer[DeviceContext],
        Pointer[UInt8],
        Pointer[UInt8],
        Int,
    ) -> None
    var copy_device_to_device: fn (
        UnsafePointer[DeviceContext],
        Pointer[UInt8],
        Pointer[UInt8],
        Int,
    ) -> None
    var free_buffer: fn (
        UnsafePointer[DeviceContext],
        Pointer[UInt8],
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
