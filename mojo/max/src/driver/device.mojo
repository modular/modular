# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from builtin._startup import _get_current_or_global_runtime
from collections import Optional
from sys.ffi import DLHandle
from max.engine._utils import call_dylib_func
from max.engine._utils import get_lib_path_from_cfg
from pathlib import Path
from tensor import TensorSpec, Tensor


struct CPUDescriptor:
    var numa_id: Int

    fn __init__(inout self, *, numa_id: Optional[Int] = None):
        self.numa_id = numa_id.value()[] if numa_id else -1


fn _get_driver_path() raises -> String:
    return get_lib_path_from_cfg(".driver_lib", "MAX Driver")


struct DeviceTensor(StringableRaising):
    var _ptr: DTypePointer[DType.invalid]
    var lib: DLHandle
    var spec: TensorSpec

    fn __init__(inout self, spec: TensorSpec, device: Device) raises:
        """Allocates a DeviceTensor from the Device's address space."""

        self.lib = DLHandle(_get_driver_path())
        alias func_name_create_tensor = "M_createDeviceTensor"
        self._ptr = call_dylib_func[DTypePointer[DType.invalid]](
            self.lib,
            func_name_create_tensor,
            spec.bytecount(),
            device._cdev,
            _get_current_or_global_runtime(),
        )
        self.spec = spec

    fn __del__(owned self):
        alias func_name_destroy = "M_destroyDeviceTensor"
        call_dylib_func[NoneType](self.lib, func_name_destroy, self._ptr)
        self.lib.close()

    fn get_device(self) raises -> Device:
        """Returns the device on which the DeviceTensor was allocated."""

        alias func_name_get_dev = "M_getDeviceFromDeviceTensor"
        return Device(
            _CDevice(
                call_dylib_func[
                    DTypePointer[DType.invalid], DTypePointer[DType.invalid]
                ](self.lib, func_name_get_dev, self._ptr)
            )
        )

    fn __str__(self) raises -> String:
        """Returns a description of the DeviceTensor."""

        return (
            "DeviceTensor("
            + str(self.get_device())
            + ",Spec("
            + str(self.spec)
            + "))"
        )

    fn get_tensor[dtype: DType](owned self) -> Tensor[dtype]:
        """Consumes the DeviceTensor and converts it to a Tensor."""

        # TODO: This is dangerous because DeviceTensor's allocator may not match
        # Tensor's deallocator. Currently they do match, but we should not rely
        # on this. Eventually the new Tensor type replacing stdlib Tensor will
        # need to know how to free data it consumes.
        alias func_name_steal_data = "M_takeDataFromDeviceTensor"
        var data = call_dylib_func[DTypePointer[dtype]](
            self.lib, func_name_steal_data, self._ptr
        )
        return Tensor[dtype](self.spec, data)


@value
@register_passable
struct _CDevice:
    var _ptr: DTypePointer[DType.invalid]


struct Device(Stringable):
    var lib: DLHandle
    var _cdev: _CDevice

    fn __init__(inout self, descriptor: CPUDescriptor = CPUDescriptor()) raises:
        """Creates a CPU Device from a CPUDescriptor."""

        alias func_name_create = "M_createCPUDevice"
        self.lib = DLHandle(_get_driver_path())
        self._cdev = _CDevice(
            call_dylib_func[DTypePointer[DType.invalid]](
                self.lib, func_name_create, descriptor.numa_id
            )
        )

    fn __init__(inout self, _cdev: _CDevice) raises:
        """Creates a CPU Device from an opaque _CDevice object. Not intended for
        external use."""

        self.lib = DLHandle(_get_driver_path())
        alias func_name_copy = "M_copyDevice"
        self._cdev = _CDevice(
            call_dylib_func[DTypePointer[DType.invalid]](
                self.lib, func_name_copy, _cdev
            )
        )

    fn allocate(inout self, spec: TensorSpec) raises -> DeviceTensor:
        """Returns a DeviceTensor allocated in the Device's memory space.
        DeviceTensor holds a reference count to Device.
        """

        return DeviceTensor(spec, self)

    fn __str__(self) -> String:
        """Returns a descriptor of the device."""

        alias func_name_desc = "M_getDeviceDesc"
        return StringRef(
            call_dylib_func[DTypePointer[DType.uint8]](
                self.lib, func_name_desc, self._cdev
            )
        )

    fn __del__(owned self):
        """Decrements the refcount to Device and destroys it if this object holds
        the only reference."""

        alias func_name_destroy = "M_destroyDevice"
        call_dylib_func[NoneType](self.lib, func_name_destroy, self._cdev)
        self.lib.close()
