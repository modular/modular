# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements device operations."""

from sys.ffi import DLHandle

from memory import stack_allocation, UnsafePointer

from utils import StringRef

from ._utils_v1 import _check_error, _human_memory
from .cuda_instance_v1 import *
from .dim import Dim

# ===----------------------------------------------------------------------===#
# Device Information
# ===----------------------------------------------------------------------===#


fn device_count() raises -> Int:
    """
    Returns the number of devices with compute capability greater than or equal
    to 2.0 that are available for execution.
    """

    var cuDeviceGetCount = cuDeviceGetCount.load()
    var res: Int32 = 0
    _check_error(cuDeviceGetCount(UnsafePointer.address_of(res)))
    return int(res)


struct DeviceV1:
    var id: Int32
    var cuda_dll: CudaDLL

    @implicit
    fn __init__(out self, id: Int = 0):
        self.id = id
        self.cuda_dll = CudaDLL()

    fn __init__(out self, cuda_instance: CudaInstance, id: Int = 0):
        self.id = id
        self.cuda_dll = cuda_instance.cuda_dll

    fn __copyinit__(out self, existing: Self):
        self.id = existing.id
        self.cuda_dll = existing.cuda_dll

    fn cuda_version(self) raises -> (Int, Int):
        var res: Int32 = 0
        _check_error(
            self.cuda_dll.cuDriverGetVersion(UnsafePointer.address_of(res))
        )

        var major = res // 1000
        var minor = (res % 1000) // 10
        return (int(major), int(minor))

    fn _query(self, attr: DeviceAttribute) raises -> Int:
        """Returns information about a particular device attribute."""
        var res: Int32 = 0
        _check_error(
            self.cuda_dll.cuDeviceGetAttribute(
                UnsafePointer.address_of(res), attr, self.id
            )
        )
        return int(res)

    fn compute_capability(self) raises -> Int:
        """Returns the device compute capability version."""
        return self._query(
            DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
        ) * 10 + self._query(DeviceAttribute.COMPUTE_CAPABILITY_MINOR)
