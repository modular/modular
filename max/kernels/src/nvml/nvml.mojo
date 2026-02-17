# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements wrappers around the NVIDIA Management Library (nvml)."""

from collections.string.string_slice import _to_string_list
from os import abort
from pathlib import Path
from ffi import _get_dylib_function as _ffi_get_dylib_function
from ffi import _Global, OwnedDLHandle, _try_find_dylib, c_char

from memory import stack_allocation, LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
comptime OpaquePointer = LegacyUnsafePointer[
    mut=True, NoneType, origin=MutAnyOrigin
]

# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

comptime CUDA_NVML_LIBRARY_DIR = Path("/usr/lib/x86_64-linux-gnu")
comptime CUDA_NVML_LIBRARY_BASE_NAME = "libnvidia-ml"
comptime CUDA_NVML_LIBRARY_EXT = ".so"

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#


fn _get_nvml_library_paths() raises -> List[Path]:
    var paths = List[Path]()
    var lib_name = CUDA_NVML_LIBRARY_BASE_NAME + CUDA_NVML_LIBRARY_EXT
    # Look for libnvidia-ml.so
    paths.append(CUDA_NVML_LIBRARY_DIR / lib_name)
    # Look for libnvida-ml.so.1
    paths.append(CUDA_NVML_LIBRARY_DIR / (lib_name + ".1"))
    # Look for libnvidia-ml.so.<driver>.<major>.<minor>
    for fd in CUDA_NVML_LIBRARY_DIR.listdir():
        var path = CUDA_NVML_LIBRARY_DIR / fd
        if CUDA_NVML_LIBRARY_BASE_NAME in String(fd):
            paths.append(path)
    return paths^


comptime CUDA_NVML_LIBRARY = _Global["CUDA_NVML_LIBRARY", _init_dylib]


fn _init_dylib() -> OwnedDLHandle:
    try:
        var dylib = _try_find_dylib(_get_nvml_library_paths())
        _check_error(
            dylib._handle.get_function[fn() -> Result]("nvmlInit_v2")()
        )
        return dylib^
    except e:
        abort(String("CUDA NVML library initialization failed: ", e))


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: __TypeOfAllTypes
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_NVML_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# NVIDIA Driver Version
# ===-----------------------------------------------------------------------===#


struct DriverVersion(ImplicitlyCopyable, Stringable):
    var _value: List[String]

    fn __init__(out self, var value: List[String]):
        self._value = value^

    fn __copyinit__(out self, copy: Self):
        self._value = copy._value.copy()

    fn major(self) raises -> Int:
        return Int(self._value[0])

    fn minor(self) raises -> Int:
        return Int(self._value[1])

    fn patch(self) raises -> Int:
        return Int(self._value[2]) if len(self._value) > 2 else 0

    fn __str__(self) -> String:
        var patch = self._value[2] if len(self._value) > 2 else ""
        return String(self._value[0], ".", self._value[1], ".", patch)


# ===-----------------------------------------------------------------------===#
# Result
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Result(Equatable, Stringable, TrivialRegisterPassable, Writable):
    var code: Int32

    comptime SUCCESS = Self(0)
    """The operation was successful."""

    comptime UNINITIALIZED = Self(1)
    """NVML was not first initialized with `nvmlInit()`."""

    comptime INVALID_ARGUMENT = Self(2)
    """A supplied argument is invalid."""

    comptime NOT_SUPPORTED = Self(3)
    """The requested operation is not available on target device."""

    comptime NO_PERMISSION = Self(4)
    """The current user does not have permission for operation."""

    comptime ALREADY_INITIALIZED = Self(5)
    """Deprecated: Multiple initializations are now allowed through ref
    counting.
    """

    comptime NOT_FOUND = Self(6)
    """A query to find an object was unsuccessful."""

    comptime INSUFFICIENT_SIZE = Self(7)
    """An input argument is not large enough."""

    comptime INSUFFICIENT_POWER = Self(8)
    """A device's external power cables are not properly attached."""

    comptime DRIVER_NOT_LOADED = Self(9)
    """NVIDIA driver is not loaded."""

    comptime TIMEOUT = Self(10)
    """User provided timeout passed."""

    comptime IRQ_ISSUE = Self(11)
    """NVIDIA Kernel detected an interrupt issue with a GPU."""

    comptime LIBRARY_NOT_FOUND = Self(12)
    """NVML Shared Library couldn't be found or loaded."""

    comptime FUNCTION_NOT_FOUND = Self(13)
    """Local version of NVML doesn't implement this function."""

    comptime CORRUPTED_INFOROM = Self(14)
    """The infoROM is corrupted."""

    comptime GPU_IS_LOST = Self(15)
    """The GPU has fallen off the bus or has otherwise become inaccessible."""

    comptime RESET_REQUIRED = Self(16)
    """The GPU requires a reset before it can be used again."""

    comptime OPERATING_SYSTEM = Self(17)
    """The GPU control device has been blocked by the operating system/cgroups."""

    comptime LIB_RM_VERSION_MISMATCH = Self(18)
    """RM detects a driver/library version mismatch."""

    comptime IN_USE = Self(19)
    """An operation cannot be performed because the GPU is currently in use."""

    comptime MEMORY = Self(20)
    """Insufficient memory."""

    comptime NO_DATA = Self(21)
    """No data."""

    comptime VGPU_ECC_NOT_SUPPORTED = Self(22)
    """The requested vgpu operation is not available on target device, because
    ECC is enabled.
    """

    comptime INSUFFICIENT_RESOURCES = Self(23)
    """Ran out of critical resources, other than memory."""

    comptime FREQ_NOT_SUPPORTED = Self(24)
    """Ran out of critical resources, other than memory."""

    comptime ARGUMENT_VERSION_MISMATCH = Self(25)
    """The provided version is invalid/unsupported."""

    comptime DEPRECATED = Self(26)
    """The requested functionality has been deprecated."""

    comptime NOT_READY = Self(27)
    """The system is not ready for the request."""

    comptime GPU_NOT_FOUND = Self(28)
    """No GPUs were found."""

    comptime UNKNOWN = Self(999)
    """An internal driver error occurred."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    fn write_to(self, mut writer: Some[Writer]):
        if self == Result.SUCCESS:
            writer.write("SUCCESS")
        elif self == Result.UNINITIALIZED:
            writer.write("NVML_UNINITIALIZED")
        elif self == Result.INVALID_ARGUMENT:
            writer.write("NVML_INVALID_ARGUMENT")
        elif self == Result.NOT_SUPPORTED:
            writer.write("NVML_NOT_SUPPORTED")
        elif self == Result.NO_PERMISSION:
            writer.write("NVML_NO_PERMISSION")
        elif self == Result.ALREADY_INITIALIZED:
            writer.write("NVML_ALREADY_INITIALIZED")
        elif self == Result.NOT_FOUND:
            writer.write("NVML_NOT_FOUND")
        elif self == Result.INSUFFICIENT_SIZE:
            writer.write("NVML_INSUFFICIENT_SIZE")
        elif self == Result.INSUFFICIENT_POWER:
            writer.write("NVML_INSUFFICIENT_POWER")
        elif self == Result.DRIVER_NOT_LOADED:
            writer.write("NVML_DRIVER_NOT_LOADED")
        elif self == Result.TIMEOUT:
            writer.write("NVML_TIMEOUT")
        elif self == Result.IRQ_ISSUE:
            writer.write("NVML_IRQ_ISSUE")
        elif self == Result.LIBRARY_NOT_FOUND:
            writer.write("NVML_LIBRARY_NOT_FOUND")
        elif self == Result.FUNCTION_NOT_FOUND:
            writer.write("NVML_FUNCTION_NOT_FOUND")
        elif self == Result.CORRUPTED_INFOROM:
            writer.write("NVML_CORRUPTED_INFOROM")
        elif self == Result.GPU_IS_LOST:
            writer.write("NVML_GPU_IS_LOST")
        elif self == Result.RESET_REQUIRED:
            writer.write("NVML_RESET_REQUIRED")
        elif self == Result.OPERATING_SYSTEM:
            writer.write("NVML_OPERATING_SYSTEM")
        elif self == Result.LIB_RM_VERSION_MISMATCH:
            writer.write("NVML_LIB_RM_VERSION_MISMATCH")
        elif self == Result.IN_USE:
            writer.write("NVML_IN_USE")
        elif self == Result.MEMORY:
            writer.write("NVML_MEMORY")
        elif self == Result.NO_DATA:
            writer.write("NVML_NO_DATA")
        elif self == Result.VGPU_ECC_NOT_SUPPORTED:
            writer.write("NVML_VGPU_ECC_NOT_SUPPORTED")
        elif self == Result.INSUFFICIENT_RESOURCES:
            writer.write("NVML_INSUFFICIENT_RESOURCES")
        elif self == Result.FREQ_NOT_SUPPORTED:
            writer.write("NVML_FREQ_NOT_SUPPORTED")
        elif self == Result.ARGUMENT_VERSION_MISMATCH:
            writer.write("NVML_ARGUMENT_VERSION_MISMATCH")
        elif self == Result.DEPRECATED:
            writer.write("NVML_DEPRECATED")
        elif self == Result.NOT_READY:
            writer.write("NVML_NOT_READY")
        elif self == Result.GPU_NOT_FOUND:
            writer.write("NVML_GPU_NOT_FOUND")
        else:
            writer.write("NVML_UNKNOWN")

    fn __str__(self) -> String:
        return String(self)


@always_inline
fn _check_error(err: Result) raises:
    if err != Result.SUCCESS:
        raise Error(err)


# ===-----------------------------------------------------------------------===#
# EnableState
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct EnableState(Equatable, TrivialRegisterPassable):
    var code: Int32

    comptime DISABLED = Self(0)
    """Feature disabled."""

    comptime ENABLED = Self(1)
    """Feature enabled."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code


# ===-----------------------------------------------------------------------===#
# ClockType
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct ClockType(Equatable, TrivialRegisterPassable):
    var code: Int32

    comptime GRAPHICS = Self(0)
    """Graphics clock domain."""

    comptime SM = Self(1)
    """SM clock domain."""

    comptime MEM = Self(2)
    """Memory clock domain."""

    comptime VIDEO = Self(2)
    """Video clock domain."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code


# ===-----------------------------------------------------------------------===#
# Device
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _DeviceImpl(Defaultable, TrivialRegisterPassable):
    var handle: OpaquePointer

    @always_inline
    fn __init__(out self):
        self.handle = OpaquePointer()

    @always_inline
    fn __bool__(self) -> Bool:
        return self.handle.__bool__()


struct Device(Writable):
    var idx: Int
    var device: _DeviceImpl

    fn __init__(out self, idx: Int = 0) raises:
        var device = _DeviceImpl()
        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetHandleByIndex_v2",
                fn(UInt32, UnsafePointer[_DeviceImpl]) -> Result,
            ]()(UInt32(idx), UnsafePointer(to=device))
        )
        self.idx = idx
        self.device = device

    fn get_driver_version(self) raises -> DriverVersion:
        """Returns NVIDIA driver version."""
        comptime max_length = 16
        var driver_version_buffer = stack_allocation[max_length, c_char]()

        _check_error(
            _get_dylib_function[
                "nvmlSystemGetDriverVersion",
                fn(UnsafePointer[c_char], UInt32) -> Result,
            ]()(driver_version_buffer, UInt32(max_length))
        )
        var driver_version_list = StringSlice(
            unsafe_from_utf8_ptr=driver_version_buffer
        ).split(".")
        return DriverVersion(_to_string_list(driver_version_list))

    fn _max_clock(self, clock_type: ClockType) raises -> Int:
        var clock = UInt32()
        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetMaxClockInfo",
                fn(_DeviceImpl, ClockType, UnsafePointer[UInt32]) -> Result,
            ]()(self.device, clock_type, UnsafePointer(to=clock))
        )
        return Int(clock)

    fn max_mem_clock(self) raises -> Int:
        return self._max_clock(ClockType.MEM)

    fn max_graphics_clock(self) raises -> Int:
        return self._max_clock(ClockType.GRAPHICS)

    fn mem_clocks(self) raises -> List[Int]:
        var num_clocks = UInt32()

        var result = _get_dylib_function[
            "nvmlDeviceGetSupportedMemoryClocks",
            fn(
                _DeviceImpl, UnsafePointer[UInt32], UnsafePointer[UInt32]
            ) -> Result,
        ]()(
            self.device,
            UnsafePointer(to=num_clocks),
            UnsafePointer[UInt32](),
        )
        if result != Result.INSUFFICIENT_SIZE:
            _check_error(result)

        var clocks = List[UInt32](length=Int(num_clocks), fill=0)

        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetSupportedMemoryClocks",
                fn(
                    _DeviceImpl, UnsafePointer[UInt32], UnsafePointer[UInt32]
                ) -> Result,
            ]()(self.device, UnsafePointer(to=num_clocks), clocks.unsafe_ptr())
        )

        var res = List[Int](capacity=len(clocks))
        for clock in clocks:
            res.append(Int(clock))

        return res^

    fn graphics_clocks(self, memory_clock_mhz: Int) raises -> List[Int]:
        var num_clocks = UInt32()

        var result = _get_dylib_function[
            "nvmlDeviceGetSupportedGraphicsClocks",
            fn(
                _DeviceImpl,
                UInt32,
                UnsafePointer[UInt32],
                UnsafePointer[UInt32],
            ) -> Result,
        ]()(
            self.device,
            UInt32(memory_clock_mhz),
            UnsafePointer(to=num_clocks),
            UnsafePointer[UInt32](),
        )

        if result == Result.SUCCESS:
            return List[Int]()

        if result != Result.INSUFFICIENT_SIZE:
            _check_error(result)

        var clocks = List[UInt32](length=Int(num_clocks), fill=0)

        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetSupportedGraphicsClocks",
                fn(
                    _DeviceImpl,
                    UInt32,
                    UnsafePointer[UInt32],
                    UnsafePointer[UInt32],
                ) -> Result,
            ]()(
                self.device,
                UInt32(memory_clock_mhz),
                UnsafePointer(to=num_clocks),
                clocks.unsafe_ptr(),
            )
        )

        var res = List[Int](capacity=len(clocks))
        for clock in clocks:
            res.append(Int(clock))

        return res^

    fn set_clock(self, mem_clock: Int, graphics_clock: Int) raises:
        _check_error(
            _get_dylib_function[
                "nvmlDeviceSetApplicationsClocks",
                fn(_DeviceImpl, UInt32, UInt32) -> Result,
            ]()(self.device, UInt32(mem_clock), UInt32(graphics_clock))
        )

    fn gpu_turbo_enabled(self) raises -> Bool:
        """Returns True if the gpu turbo is enabled."""
        var is_enabled = _EnableState.DISABLED
        var default_is_enabled = _EnableState.DISABLED
        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetAutoBoostedClocksEnabled",
                fn(
                    _DeviceImpl,
                    UnsafePointer[_EnableState],
                    UnsafePointer[_EnableState],
                ) -> Result,
            ]()(
                self.device,
                UnsafePointer(to=is_enabled),
                UnsafePointer(to=default_is_enabled),
            )
        )
        return is_enabled == _EnableState.ENABLED

    fn set_gpu_turbo(self, enabled: Bool = True) raises:
        """Sets the GPU turbo state."""
        _check_error(
            _get_dylib_function[
                "nvmlDeviceSetAutoBoostedClocksEnabled",
                fn(_DeviceImpl, _EnableState) -> Result,
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
                fn(_DeviceImpl, UnsafePointer[_EnableState]) -> Result,
            ]()(
                self.device,
                UnsafePointer(to=is_enabled),
            )
        )
        return is_enabled == _EnableState.ENABLED

    fn set_persistence_mode(self, enabled: Bool = True) raises:
        """Sets the persistence mode."""
        _check_error(
            _get_dylib_function[
                "nvmlDeviceSetPersistenceMode",
                fn(_DeviceImpl, _EnableState) -> Result,
            ]()(
                self.device,
                _EnableState.ENABLED if enabled else _EnableState.DISABLED,
            )
        )

    fn set_max_gpu_clocks(device: Device) raises:
        var max_mem_clock = device.mem_clocks()
        sort(max_mem_clock)

        var max_graphics_clock = device.graphics_clocks(max_mem_clock[-1])
        sort(max_graphics_clock)

        for clock_val in reversed(max_graphics_clock):
            try:
                device.set_clock(max_mem_clock[-1], clock_val)
                print(
                    "the device clocks for device=",
                    device,
                    " were set to mem=",
                    max_mem_clock[-1],
                    " and graphics=",
                    clock_val,
                    sep="",
                )
                return
            except:
                pass

        raise Error("unable to set max gpu clock for ", device)

    @no_inline
    fn __str__(self) -> String:
        return self.__repr__()

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        writer.write("Device(", self.idx, ")")

    @no_inline
    fn __repr__(self) -> String:
        return String.write(self)


@fieldwise_init
struct _EnableState(TrivialRegisterPassable):
    var state: Int32

    comptime DISABLED = _EnableState(0)  # Feature disabled
    comptime ENABLED = _EnableState(1)  # Feature enabled

    fn __eq__(self, other: Self) -> Bool:
        return self.state == other.state
