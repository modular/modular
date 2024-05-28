# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements wrappers around the NVIDIA Management Library (nvml)."""

from collections import List
from os import abort
from pathlib import Path
from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function

from memory.unsafe import Pointer


# ===----------------------------------------------------------------------===#
# Constants
# ===----------------------------------------------------------------------===#

alias CUDA_NVML_LIBRARY_PATH = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so"


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib(ignored: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
    if not Path(CUDA_NVML_LIBRARY_PATH).exists():
        print("the CUDA NVML library was not found at", CUDA_NVML_LIBRARY_PATH)
        abort()
    var ptr = Pointer[DLHandle].alloc(1)
    var handle = DLHandle(CUDA_NVML_LIBRARY_PATH)
    _ = handle.get_function[fn () -> Result]("nvmlInit_v2")()
    ptr[] = handle
    return ptr.bitcast[NoneType]().address


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn _get_dylib_function[
    func_name: StringLiteral, result_type: AnyRegType
]() raises -> result_type:
    return _ffi_get_dylib_function[
        "CUDA_NVML_LIBRARY", func_name, _init_dylib, _destroy_dylib, result_type
    ]()


# ===----------------------------------------------------------------------===#
# Result
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Result(Stringable, EqualityComparable):
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


@always_inline
fn _check_error(err: Result) raises:
    if err != Result.SUCCESS:
        raise str(err)


# ===----------------------------------------------------------------------===#
# EnableState
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct EnableState(EqualityComparable):
    var code: Int32

    alias DISABLED = Result(0)
    """Feature disabled"""

    alias ENABLED = Result(1)
    """Feature enabled"""

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
    fn __init__(inout self):
        self.handle = DTypePointer[DType.invalid]()

    @always_inline
    fn __bool__(self) -> Bool:
        return self.handle.__bool__()


struct Device:
    var idx: Int
    var device: _DeviceImpl

    fn __init__(inout self, idx: Int = 0) raises:
        var device = _DeviceImpl()
        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetHandleByIndex_v2",
                fn (UInt32, Pointer[_DeviceImpl]) -> Result,
            ]()(UInt32(idx), Pointer.address_of(device))
        )
        self.idx = idx
        self.device = device

    fn __copyinit__(inout self: Self, existing: Self):
        self.idx = existing.idx
        self.device = existing.device

    fn mem_clocks(self) raises -> List[Int]:
        var num_clocks = UInt32()

        var result = _get_dylib_function[
            "nvmlDeviceGetSupportedMemoryClocks",
            fn (_DeviceImpl, Pointer[UInt32], UnsafePointer[UInt32]) -> Result,
        ]()(
            self.device, Pointer.address_of(num_clocks), UnsafePointer[UInt32]()
        )
        if result != Result.INSUFFICIENT_SIZE:
            _check_error(result)

        var clocks = List[UInt32]()
        clocks.resize(int(num_clocks))

        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetSupportedMemoryClocks",
                fn (
                    _DeviceImpl, Pointer[UInt32], UnsafePointer[UInt32]
                ) -> Result,
            ]()(self.device, Pointer.address_of(num_clocks), clocks.data)
        )

        var res = List[Int]()
        for clock in clocks:
            res.append(int(clock[]))

        return res

    fn graphics_clocks(self, memory_clock_mhz: Int) raises -> List[Int]:
        var num_clocks = UInt32()

        var result = _get_dylib_function[
            "nvmlDeviceGetSupportedGraphicsClocks",
            fn (
                _DeviceImpl, UInt32, Pointer[UInt32], UnsafePointer[UInt32]
            ) -> Result,
        ]()(
            self.device,
            UInt32(memory_clock_mhz),
            Pointer.address_of(num_clocks),
            UnsafePointer[UInt32](),
        )
        if result != Result.INSUFFICIENT_SIZE:
            _check_error(result)

        var clocks = List[UInt32]()
        clocks.resize(int(num_clocks))

        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetSupportedGraphicsClocks",
                fn (
                    _DeviceImpl, UInt32, Pointer[UInt32], UnsafePointer[UInt32]
                ) -> Result,
            ]()(
                self.device,
                UInt32(memory_clock_mhz),
                Pointer.address_of(num_clocks),
                clocks.data,
            )
        )

        var res = List[Int]()
        for clock in clocks:
            res.append(int(clock[]))

        return res

    fn set_clock(self, mem_clock: Int, graphics_clock: Int) raises:
        _check_error(
            _get_dylib_function[
                "nvmlDeviceSetApplicationsClocks",
                fn (_DeviceImpl, UInt32, UInt32) -> Result,
            ]()(self.device, UInt32(mem_clock), UInt32(graphics_clock))
        )

    fn gpu_turbo_enabled(self) raises -> Bool:
        """Returns True if the gpu turbo is enabled."""
        var is_enabled = _EnableState.DISABLED
        var default_is_enabled = _EnableState.DISABLED
        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetAutoBoostedClocksEnabled",
                fn (
                    _DeviceImpl, Pointer[_EnableState], Pointer[_EnableState]
                ) -> Result,
            ]()(
                self.device,
                Pointer.address_of(is_enabled),
                Pointer.address_of(default_is_enabled),
            )
        )
        return is_enabled == _EnableState.ENABLED

    fn set_gpu_turbo(self, enabled: Bool = True) raises:
        """Sets the GPU turbo state."""
        _check_error(
            _get_dylib_function[
                "nvmlDeviceSetAutoBoostedClocksEnabled",
                fn (_DeviceImpl, _EnableState) -> Result,
            ]()(
                self.device,
                _EnableState.ENABLED if enabled else _EnableState.DISABLED,
            )
        )

    fn get_persistence_mode(self) raises -> Bool:
        """Returns True if the gpu persistence mode is enabled."""
        var is_enabled = _EnableState.DISABLED
        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetPersistenceMode",
                fn (_DeviceImpl, Pointer[_EnableState]) -> Result,
            ]()(
                self.device,
                Pointer.address_of(is_enabled),
            )
        )
        return is_enabled == _EnableState.ENABLED

    fn set_persistence_mode(self, enabled: Bool = True) raises:
        """Sets the persistence mode."""
        _check_error(
            _get_dylib_function[
                "nvmlDeviceSetPersistenceMode",
                fn (_DeviceImpl, _EnableState) -> Result,
            ]()(
                self.device,
                _EnableState.ENABLED if enabled else _EnableState.DISABLED,
            )
        )

    fn __str__(self) -> String:
        return "Device(" + str(self.idx) + ")"


@value
@register_passable("trivial")
struct _EnableState:
    var state: Int32

    alias DISABLED = _EnableState(0)  # Feature disabled
    alias ENABLED = _EnableState(1)  # Feature enabled

    fn __eq__(self, other: Self) -> Bool:
        return self.state == other.state
