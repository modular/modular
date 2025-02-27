# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Defines the types and functions to interact with hardware devices.

For example, you can create a CPU device like this:

```mojo
from max.driver import cpu_device

def main():
    device = cpu_device()
```
"""


from collections import Optional
from collections.string import StaticString
from pathlib import Path

from max._utils import call_dylib_func, get_lib_path_from_cfg
from max.tensor import TensorSpec
from memory import UnsafePointer

from ._driver_library import DriverLibrary
from ._status import Status, _CStatus
from .device_memory import DeviceMemory, DeviceTensor


struct _CPUDescriptor:
    var numa_id: Int

    fn __init__(out self, *, numa_id: Optional[Int] = None):
        self.numa_id = numa_id.value() if numa_id else -1


fn _get_driver_path() raises -> String:
    return get_lib_path_from_cfg(".driver_lib", "MAX Driver")


@value
@register_passable("trivial")
struct _CDevice:
    var _ptr: UnsafePointer[NoneType]

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self._ptr = ptr

    fn copy(self, lib: Optional[DriverLibrary]) -> Self:
        if not lib:
            return self
        return lib.value().copy_device_fn(self._ptr)

    fn free_data(self, lib: DriverLibrary, data: UnsafePointer[UInt8]) raises:
        var status = Status(lib)
        lib.free_device_data_fn(self._ptr, data, status.impl)
        if status:
            raise String(status)

    fn __eq__(self, other: Self) -> Bool:
        return self._ptr == other._ptr


struct Device(Stringable):
    """Represents a logical instance of a device, for eg: CPU. This
    can be used to allocate and manage memory in a device's address space,
    and to compile and execute models and graphs on a device.
    """

    var _lib: Optional[DriverLibrary]
    var _cdev: _CDevice

    fn __init__(out self):
        """Constructs a default initialized Device in a state that is only valid
        for deletion. Can be used to represent a 'moved from' state.


        Use cpu_device() or accelerator_device() to create a CPU or GPU Device.
        """

        self._lib = None
        self._cdev = _CDevice(UnsafePointer[NoneType]())

    @doc_private
    fn __init__(
        mut self, lib: DriverLibrary, *, owned owned_ptr: _CDevice
    ) raises:
        self._lib = lib
        self._cdev = owned_ptr

    fn __copyinit__(out self, existing: Self):
        """Create a copy of the Device (bumping a refcount on the underlying Device).

        Args:
            existing: Instance from which to copy.
        """

        self._lib = existing._lib
        self._cdev = existing._cdev.copy(existing._lib)

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn __moveinit__(out self, owned existing: Self):
        """Create a new Device and consume `existing`.

        Args:
            existing: Instance from which to move from.
        """
        self._lib = existing._lib^
        self._cdev = existing._cdev

    fn allocate(
        self, spec: TensorSpec, name: Optional[String] = None
    ) raises -> DeviceTensor:
        """Creates tensor allocated in the Device's address space.

        Args:
            spec: TensorSpec descripting the shape and type of the tensor to allocate.
            name: An optional name for the DeviceTensor.
        Returns:
            DeviceTensor allocated in Device's address space.
        """

        return DeviceTensor(spec, self, name)

    fn allocate(
        self, bytecount: Int, name: Optional[String] = None
    ) raises -> DeviceMemory:
        """Allocates a DeviceMemory object in the Device's address space.

        Args:
            bytecount: The size of the memory to allocate in bytes.
            name: An optional name for the DeviceMemory.

        Returns:
            A DeviceMemory object allocated in the Device's address space.
        """

        return DeviceMemory(bytecount, self, name)

    fn unsafe_ptr(self) -> UnsafePointer[NoneType]:
        """Gets the underlying pointer to the Device.

        Returns:
          The underlying pointer of the Device.
        """
        return self._cdev._ptr

    fn _free(self, data: UnsafePointer[UInt8]) raises:
        self._cdev.free_data(self._lib.value(), data)

    fn __str__(self) -> String:
        """Returns a descriptor of the device.

        Returns:
            String representation of device.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this Device to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write(
            StaticString(
                unsafe_from_utf8_ptr=self._lib.value().get_device_desc_fn(
                    self._cdev._ptr
                )
            )
        )

    fn __del__(owned self):
        """Destroys the device.

        Note that any DeviceBuffer allocated on the Device will contain a reference
        to the Device, and the Device will only be de-allocated when all of its
        DeviceBuffers have also been destroyed.
        """
        if not self._cdev._ptr:
            return
        self._lib.value().destroy_device_fn(self._cdev._ptr)

    fn __eq__(self, other: Self) -> Bool:
        """Check if `self` and `other` point to the same underlying Device.

        Args:
            other: Instance to compare against.
        Returns:
            True if they are the same logical device.
        """
        return self._cdev == other._cdev


fn cpu_device() raises -> Device:
    """Creates a CPU Device.

    Returns:
        A logical device representing CPU.
    """
    var lib = DriverLibrary()
    var descriptor = _CPUDescriptor()
    var status = Status(lib)
    var device = lib.create_cpu_device_fn(descriptor.numa_id, status.impl)
    if status:
        raise String(status)
    return Device(lib, owned_ptr=device)
