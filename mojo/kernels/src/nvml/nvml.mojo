# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements wrappers around the NVIDIA Management Library (nvml)."""

from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from memory.unsafe import Pointer
from ._utils import _check_error
from pathlib import Path
from utils.vector import DynamicVector
from debug import trap

# ===----------------------------------------------------------------------===#
# Constants
# ===----------------------------------------------------------------------===#

alias CUDA_NVML_LIBRARY_PATH = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so"


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    if not Path(CUDA_NVML_LIBRARY_PATH).exists():
        print("the CUDA NVML library was not found at", CUDA_NVML_LIBRARY_PATH)
        trap()
    let ptr = Pointer[DLHandle].alloc(1)
    let handle = DLHandle(CUDA_NVML_LIBRARY_PATH)
    _ = handle.get_function[fn () -> Result]("nvmlInit_v2")()
    __get_address_as_lvalue(ptr.address) = handle
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: Pointer[NoneType]):
    __get_address_as_lvalue(ptr.bitcast[DLHandle]().address)._del_old()
    ptr.free()


@always_inline
fn _get_dylib_function[
    result_type: AnyRegType
](name: StringRef) raises -> result_type:
    return _ffi_get_dylib_function[
        "CUDA_NVML_LIBRARY", _init_dylib, _destroy_dylib, result_type
    ](name)


# ===----------------------------------------------------------------------===#
# Result
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Result:
    var code: Int32

    alias SUCCESS = Result(0)
    """The operation was successful"""

    alias UNINITIALIZED = Result(1)
    """NVML was not first initialized with nvmlInit()"""

    alias INVALID_ARGUMENT = Result(2)
    """A supplied argument is invalid"""

    alias NOT_SUPPORTED = Result(3)
    """The requested operation is not available on target device"""

    alias NO_PERMISSION = Result(4)
    """The current user does not have permission for operation"""

    alias ALREADY_INITIALIZED = Result(5)
    """Deprecated: Multiple initializations are now allowed through ref
    counting"""

    alias NOT_FOUND = Result(6)
    """A query to find an object was unsuccessful"""

    alias INSUFFICIENT_SIZE = Result(7)
    """An input argument is not large enough"""

    alias INSUFFICIENT_POWER = Result(8)
    """A device's external power cables are not properly attached"""

    alias DRIVER_NOT_LOADED = Result(9)
    """NVIDIA driver is not loaded"""

    alias TIMEOUT = Result(10)
    """User provided timeout passed"""

    alias IRQ_ISSUE = Result(11)
    """NVIDIA Kernel detected an interrupt issue with a GPU"""

    alias LIBRARY_NOT_FOUND = Result(12)
    """NVML Shared Library couldn't be found or loaded"""

    alias FUNCTION_NOT_FOUND = Result(13)
    """Local version of NVML doesn't implement this function"""

    alias CORRUPTED_INFOROM = Result(14)
    """infoROM is corrupted"""

    alias GPU_IS_LOST = Result(15)
    """The GPU has fallen off the bus or has otherwise become inaccessible"""

    alias RESET_REQUIRED = Result(16)
    """The GPU requires a reset before it can be used again"""

    alias OPERATING_SYSTEM = Result(17)
    """The GPU control device has been blocked by the operating system/cgroups"""

    alias LIB_RM_VERSION_MISMATCH = Result(18)
    """RM detects a driver/library version mismatch"""

    alias IN_USE = Result(19)
    """An operation cannot be performed because the GPU is currently in use"""

    alias MEMORY = Result(20)
    """Insufficient memory"""

    alias NO_DATA = Result(21)
    """No data"""

    alias VGPU_ECC_NOT_SUPPORTED = Result(22)
    """The requested vgpu operation is not available on target device, becasue
    ECC is enabled"""

    alias INSUFFICIENT_RESOURCES = Result(23)
    """Ran out of critical resources, other than memory"""

    alias FREQ_NOT_SUPPORTED = Result(24)
    """Ran out of critical resources, other than memory"""

    alias ARGUMENT_VERSION_MISMATCH = Result(25)
    """The provided version is invalid/unsupported"""

    alias DEPRECATED = Result(26)
    """The requested functionality has been deprecated"""

    alias NOT_READY = Result(27)
    """The system is not ready for the request"""

    alias GPU_NOT_FOUND = Result(28)
    """No GPUs were found"""

    alias UNKNOWN = Result(999)
    """An internal driver error occurred"""

    @always_inline("nodebug")
    fn __init__(code: Int32) -> Self:
        return Self {code: code}

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        if self == Result.SUCCESS:
            return "SUCCESS"

        if self == Result.UNINITIALIZED:
            return "NVML_UNINITIALIZED"

        if self == Result.INVALID_ARGUMENT:
            return "NVML_INVALID_ARGUMENT"

        if self == Result.NOT_SUPPORTED:
            return "NVML_NOT_SUPPORTED"

        if self == Result.NO_PERMISSION:
            return "NVML_NO_PERMISSION"

        if self == Result.ALREADY_INITIALIZED:
            return "NVML_ALREADY_INITIALIZED"

        if self == Result.NOT_FOUND:
            return "NVML_NOT_FOUND"

        if self == Result.INSUFFICIENT_SIZE:
            return "NVML_INSUFFICIENT_SIZE"

        if self == Result.INSUFFICIENT_POWER:
            return "NVML_INSUFFICIENT_POWER"

        if self == Result.DRIVER_NOT_LOADED:
            return "NVML_DRIVER_NOT_LOADED"

        if self == Result.TIMEOUT:
            return "NVML_TIMEOUT"

        if self == Result.IRQ_ISSUE:
            return "NVML_IRQ_ISSUE"

        if self == Result.LIBRARY_NOT_FOUND:
            return "NVML_LIBRARY_NOT_FOUND"

        if self == Result.FUNCTION_NOT_FOUND:
            return "NVML_FUNCTION_NOT_FOUND"

        if self == Result.CORRUPTED_INFOROM:
            return "NVML_CORRUPTED_INFOROM"

        if self == Result.GPU_IS_LOST:
            return "NVML_GPU_IS_LOST"

        if self == Result.RESET_REQUIRED:
            return "NVML_RESET_REQUIRED"

        if self == Result.OPERATING_SYSTEM:
            return "NVML_OPERATING_SYSTEM"

        if self == Result.LIB_RM_VERSION_MISMATCH:
            return "NVML_LIB_RM_VERSION_MISMATCH"

        if self == Result.IN_USE:
            return "NVML_IN_USE"

        if self == Result.MEMORY:
            return "NVML_MEMORY"

        if self == Result.NO_DATA:
            return "NVML_NO_DATA"

        if self == Result.VGPU_ECC_NOT_SUPPORTED:
            return "NVML_VGPU_ECC_NOT_SUPPORTED"

        if self == Result.INSUFFICIENT_RESOURCES:
            return "NVML_INSUFFICIENT_RESOURCES"

        if self == Result.FREQ_NOT_SUPPORTED:
            return "NVML_FREQ_NOT_SUPPORTED"

        if self == Result.ARGUMENT_VERSION_MISMATCH:
            return "NVML_ARGUMENT_VERSION_MISMATCH"

        if self == Result.DEPRECATED:
            return "NVML_DEPRECATED"

        if self == Result.NOT_READY:
            return "NVML_NOT_READY"

        if self == Result.GPU_NOT_FOUND:
            return "NVML_GPU_NOT_FOUND"

        return "NVML_UNKNOWN"


# ===----------------------------------------------------------------------===#
# EnableState
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct EnableState:
    var code: Int32

    alias DISENABLED = Result(0)
    """Feature disabled"""

    alias ENABLED = Result(1)
    """Feature enabled"""

    @always_inline("nodebug")
    fn __init__(code: Int32) -> Self:
        return Self {code: code}

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


# ===----------------------------------------------------------------------===#
# Device
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _DeviceImpl:
    var handle: DTypePointer[DType.invalid]

    @always_inline
    fn __init__() -> Self:
        return Self {handle: DTypePointer[DType.invalid]()}

    @always_inline
    fn __init__(handle: DTypePointer[DType.invalid]) -> Self:
        return Self {handle: handle}

    @always_inline
    fn __bool__(self) -> Bool:
        return self.handle.__bool__()


struct Device:
    var device: _DeviceImpl

    fn __init__(inout self, idx: Int = 0) raises:
        var device = _DeviceImpl()
        _check_error(
            _get_dylib_function[fn (UInt32, Pointer[_DeviceImpl]) -> Result](
                "nvmlDeviceGetHandleByIndex_v2"
            )(UInt32(idx), Pointer.address_of(device))
        )
        self.device = device

    fn mem_clocks(self) raises -> DynamicVector[Int]:
        var num_clocks = UInt32()

        let result = _get_dylib_function[
            fn (_DeviceImpl, Pointer[UInt32], Pointer[UInt32]) -> Result
        ]("nvmlDeviceGetSupportedMemoryClocks")(
            self.device, Pointer.address_of(num_clocks), Pointer[UInt32]()
        )
        if result != Result.INSUFFICIENT_SIZE:
            _check_error(result)

        var clocks = DynamicVector[UInt32]()
        clocks.resize(int(num_clocks))

        _check_error(
            _get_dylib_function[
                fn (_DeviceImpl, Pointer[UInt32], Pointer[UInt32]) -> Result
            ]("nvmlDeviceGetSupportedMemoryClocks")(
                self.device, Pointer.address_of(num_clocks), clocks.data
            )
        )

        var res = DynamicVector[Int]()
        res.resize(int(num_clocks))
        for i in range(int(num_clocks)):
            res[i] = int(clocks[i])

        clocks._del_old()

        return res
