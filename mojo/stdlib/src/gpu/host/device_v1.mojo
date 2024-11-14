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


struct DeviceContextInfo(StringableRaising):
    var _ctx: DeviceContext

    fn __init__(out self, ctx: DeviceContext):
        self._ctx = ctx

    @no_inline
    fn __str__(self) raises -> String:
        var res = "name: " + self._name() + "\n"
        res += "driver_version:" + str(self.driver_version()) + "\n"
        res += "memory: " + _human_memory(self._total_memory()) + "\n"
        res += (
            "compute_capability: "
            + str(self._query(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR))
            + "."
            + str(self._query(DeviceAttribute.COMPUTE_CAPABILITY_MINOR))
            + "\n"
        )
        res += (
            "clock_rate: " + str(self._query(DeviceAttribute.CLOCK_RATE)) + "\n"
        )
        res += (
            "warp_size: " + str(self._query(DeviceAttribute.WARP_SIZE)) + "\n"
        )
        res += (
            "max_threads_per_block: "
            + str(self._query(DeviceAttribute.MAX_THREADS_PER_BLOCK))
            + "\n"
        )
        res += (
            "max_shared_memory: "
            + _human_memory(
                self._query(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK)
            )
            + "\n"
        )
        res += (
            "max_block: "
            + str(
                Dim(
                    self._query(DeviceAttribute.MAX_BLOCK_DIM_X),
                    self._query(DeviceAttribute.MAX_BLOCK_DIM_Y),
                    self._query(DeviceAttribute.MAX_BLOCK_DIM_Z),
                )
            )
            + "\n"
        )
        res += (
            "max_grid: "
            + str(
                Dim(
                    self._query(DeviceAttribute.MAX_GRID_DIM_X),
                    self._query(DeviceAttribute.MAX_GRID_DIM_Y),
                    self._query(DeviceAttribute.MAX_GRID_DIM_Z),
                )
            )
            + "\n"
        )
        # Attribute not supported on AMD
        try:
            res += (
                "max_access_window_size: "
                + str(
                    self._query(DeviceAttribute.MAX_ACCESS_POLICY_WINDOW_SIZE)
                )
                + "\n"
            )
        except:
            pass
        res += "SM count: " + str(self.multiprocessor_count()) + "\n"
        res += "Max threads per SM: " + str(self.max_threads_per_sm()) + "\n"

        return res

    fn _name(self) raises -> String:
        """Get an identifier string for the device."""
        return self._ctx.name()

    fn driver_version(self) raises -> Int:
        return self._ctx.get_driver_version()

    fn _total_memory(self) raises -> Int:
        """Returns the total amount of memory on the device."""
        free, total = self._ctx.get_memory_info()
        return total

    fn _query(self, attr: DeviceAttribute) raises -> Int:
        """Returns information about a particular device attribute."""
        return self._ctx.get_attribute(attr)

    fn multiprocessor_count(self) raises -> Int:
        """Returns the number of multiprocessors on this device."""
        return self._query(DeviceAttribute.MULTIPROCESSOR_COUNT)

    fn max_registers_per_block(self) raises -> Int:
        """Returns the maximum number of 32-bit registers available per block.
        """
        return self._query(DeviceAttribute.MAX_REGISTERS_PER_BLOCK)

    fn max_threads_per_sm(self) raises -> Int:
        """Returns the maximum resident threads per multiprocessor."""
        return self._query(DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR)

    fn compute_capability(self) raises -> Int:
        """Returns the device compute capability version."""
        return self._query(
            DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
        ) * 10 + self._query(DeviceAttribute.COMPUTE_CAPABILITY_MINOR)


struct DeviceV1:
    var id: Int32
    var cuda_dll: CudaDLL

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
