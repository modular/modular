# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections import Optional
from sys.ffi import DLHandle
from max._utils import call_dylib_func, get_lib_path_from_cfg
from pathlib import Path
from tensor import TensorSpec
from ._driver_library import DriverLibrary, ManagedDLHandle
from .device_memory import DeviceMemory, DeviceTensor


struct CPUDescriptor:
    var numa_id: Int

    fn __init__(inout self, *, numa_id: Optional[Int] = None):
        self.numa_id = numa_id.value() if numa_id else -1


fn _get_driver_path() raises -> String:
    return get_lib_path_from_cfg(".driver_lib", "MAX Driver")


@value
@register_passable("trivial")
struct _CDevice:
    var _ptr: UnsafePointer[NoneType]

    fn copy(self, lib: DriverLibrary) -> Self:
        alias func_name_copy = "M_copyDevice"
        return call_dylib_func[UnsafePointer[NoneType]](
            lib.get_handle(), func_name_copy, self
        )

    fn free_data(self, lib: DriverLibrary, data: DTypePointer[DType.uint8]):
        alias func_free_data = "M_freeDeviceData"
        call_dylib_func(lib.get_handle(), func_free_data, self, data)

    fn __eq__(self, other: Self) -> Bool:
        return self._ptr == other._ptr


struct Device(Stringable):
    var lib: DriverLibrary
    var _cdev: _CDevice

    fn __init__(inout self):
        """Creates a default-initialized Device."""

        self.lib = ManagedDLHandle(DLHandle(DTypePointer[DType.int8]()))
        self._cdev = _CDevice(UnsafePointer[NoneType]())

    fn __init__(
        inout self, lib: DriverLibrary, *, owned owned_ptr: _CDevice
    ) raises:
        """Creates a Device from an opaque _CDevice object. Not intended for
        external use."""

        self.lib = lib
        self._cdev = owned_ptr

    fn __copyinit__(inout self, existing: Self):
        """Copy constructor for device.
        Args:
            existing(Device): Instance from which to copy.
        """

        self.lib = existing.lib
        self._cdev = existing._cdev.copy(self.lib)

    fn __moveinit__(inout self, owned existing: Self):
        self.lib = existing.lib^
        self._cdev = existing._cdev

    fn allocate(
        self, spec: TensorSpec, name: Optional[String] = None
    ) raises -> DeviceTensor:
        """Returns a DeviceMemory allocated in the Device's memory space.
        DeviceMemory holds a reference count to Device.
        """

        return DeviceTensor(spec, self, name)

    fn allocate(
        self, bytecount: Int, name: Optional[String] = None
    ) raises -> DeviceMemory:
        """Returns a DeviceMemory allocated in the Device's memory space.
        DeviceMemory holds a reference count to Device.
        """

        return DeviceMemory(bytecount, self, name)

    fn _free(self, data: DTypePointer[DType.uint8]):
        self._cdev.free_data(self.lib, data)

    fn __str__(self) -> String:
        """Returns a descriptor of the device."""

        alias func_name_desc = "M_getDeviceDesc"
        return StringRef(
            call_dylib_func[DTypePointer[DType.uint8]](
                self.lib.get_handle(), func_name_desc, self._cdev
            )
        )

    fn __del__(owned self):
        """Decrements the refcount to Device and destroys it if this object holds
        the only reference."""

        alias func_name_destroy = "M_destroyDevice"
        call_dylib_func[NoneType](
            self.lib.get_handle(), func_name_destroy, self._cdev
        )
        # Extend lifetime of library until C function returns.
        _ = self.lib^

    fn __eq__(self, other: Self) -> Bool:
        return self._cdev == other._cdev


fn cpu_device(descriptor: CPUDescriptor = CPUDescriptor()) raises -> Device:
    """Creates a CPU Device from a CPUDescriptor."""
    alias func_name_create = "M_createCPUDevice"
    var lib = ManagedDLHandle(DLHandle(_get_driver_path()))
    var _cdev = _CDevice(
        call_dylib_func[UnsafePointer[NoneType]](
            lib.get_handle(),
            func_name_create,
            Int32(descriptor.numa_id),
        )
    )
    return Device(lib^, owned_ptr=_cdev)
