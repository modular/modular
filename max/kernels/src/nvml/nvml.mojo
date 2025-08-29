# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import _Global, _OwnedDLHandle, _try_find_dylib, c_char

from memory import stack_allocation

# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

alias CUDA_NVML_LIBRARY_DIR = Path("/usr/lib/x86_64-linux-gnu")
alias CUDA_NVML_LIBRARY_BASE_NAME = "libnvidia-ml"
alias CUDA_NVML_LIBRARY_EXT = ".so"

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#


fn _get_nvml_library_paths() raises -> List[Path]:
    var paths = List[Path]()
    var common_path = CUDA_NVML_LIBRARY_DIR / (
        CUDA_NVML_LIBRARY_BASE_NAME + CUDA_NVML_LIBRARY_EXT
    )
    paths.append(common_path)
    for fd in CUDA_NVML_LIBRARY_DIR.listdir():
        var path = CUDA_NVML_LIBRARY_DIR / fd
        if CUDA_NVML_LIBRARY_BASE_NAME in String(fd):
            paths.append(path)
    return paths


alias CUDA_NVML_LIBRARY = _Global[
    "CUDA_NVML_LIBRARY", _OwnedDLHandle, _init_dylib
]


fn _init_dylib() -> _OwnedDLHandle:
    try:
        var dylib = _try_find_dylib(_get_nvml_library_paths())
        _check_error(
            dylib._handle.get_function[fn () -> Result]("nvmlInit_v2")()
        )
        return dylib^
    except e:
        return abort[_OwnedDLHandle](
            String("CUDA NVML library initialization failed: ", e)
        )


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_NVML_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# NVIDIA Driver Version
# ===-----------------------------------------------------------------------===#


struct DriverVersion(Copyable, Movable, StringableRaising):
    var _value: List[String]

    fn __init__(out self, value: List[String]):
        self._value = value

    fn major(self) raises -> Int:
        return Int(self._value[0])

    fn minor(self) raises -> Int:
        return Int(self._value[1])

    fn patch(self) raises -> Int:
        return Int(self._value[2]) if len(self._value) > 2 else 0

    fn __str__(self) raises -> String:
        return String(self.major(), ".", self.minor(), ".", self.patch())


# ===-----------------------------------------------------------------------===#
# Result
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct Result(Copyable, EqualityComparable, Movable, Stringable, Writable):
    var code: Int32

    alias SUCCESS = Self(0)
    """The operation was successful"""

    alias UNINITIALIZED = Self(1)
    """NVML was not first initialized with nvmlInit()"""

    alias INVALID_ARGUMENT = Self(2)
    """A supplied argument is invalid"""

    alias NOT_SUPPORTED = Self(3)
    """The requested operation is not available on target device"""

    alias NO_PERMISSION = Self(4)
    """The current user does not have permission for operation"""

    alias ALREADY_INITIALIZED = Self(5)
    """Deprecated: Multiple initializations are now allowed through ref
    counting"""

    alias NOT_FOUND = Self(6)
    """A query to find an object was unsuccessful"""

    alias INSUFFICIENT_SIZE = Self(7)
    """An input argument is not large enough"""

    alias INSUFFICIENT_POWER = Self(8)
    """A device's external power cables are not properly attached"""

    alias DRIVER_NOT_LOADED = Self(9)
    """NVIDIA driver is not loaded"""

    alias TIMEOUT = Self(10)
    """User provided timeout passed"""

    alias IRQ_ISSUE = Self(11)
    """NVIDIA Kernel detected an interrupt issue with a GPU"""

    alias LIBRARY_NOT_FOUND = Self(12)
    """NVML Shared Library couldn't be found or loaded"""

    alias FUNCTION_NOT_FOUND = Self(13)
    """Local version of NVML doesn't implement this function"""

    alias CORRUPTED_INFOROM = Self(14)
    """infoROM is corrupted"""

    alias GPU_IS_LOST = Self(15)
    """The GPU has fallen off the bus or has otherwise become inaccessible"""

    alias RESET_REQUIRED = Self(16)
    """The GPU requires a reset before it can be used again"""

    alias OPERATING_SYSTEM = Self(17)
    """The GPU control device has been blocked by the operating system/cgroups"""

    alias LIB_RM_VERSION_MISMATCH = Self(18)
    """RM detects a driver/library version mismatch"""

    alias IN_USE = Self(19)
    """An operation cannot be performed because the GPU is currently in use"""

    alias MEMORY = Self(20)
    """Insufficient memory"""

    alias NO_DATA = Self(21)
    """No data"""

    alias VGPU_ECC_NOT_SUPPORTED = Self(22)
    """The requested vgpu operation is not available on target device, because
    ECC is enabled"""

    alias INSUFFICIENT_RESOURCES = Self(23)
    """Ran out of critical resources, other than memory"""

    alias FREQ_NOT_SUPPORTED = Self(24)
    """Ran out of critical resources, other than memory"""

    alias ARGUMENT_VERSION_MISMATCH = Self(25)
    """The provided version is invalid/unsupported"""

    alias DEPRECATED = Self(26)
    """The requested functionality has been deprecated"""

    alias NOT_READY = Self(27)
    """The system is not ready for the request"""

    alias GPU_NOT_FOUND = Self(28)
    """No GPUs were found"""

    alias UNKNOWN = Self(999)
    """An internal driver error occurred"""

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
@register_passable("trivial")
struct EnableState(Copyable, EqualityComparable, Movable):
    var code: Int32

    alias DISABLED = Self(0)
    """Feature disabled"""

    alias ENABLED = Self(1)
    """Feature enabled"""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code


# ===-----------------------------------------------------------------------===#
# ClockType
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct ClockType(Copyable, EqualityComparable, Movable):
    var code: Int32

    alias GRAPHICS = Self(0)
    """Graphics clock domain"""

    alias SM = Self(1)
    """SM clock domain"""

    alias MEM = Self(2)
    """Memory clock domain"""

    alias VIDEO = Self(2)
    """Video clock domain"""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code


# ===-----------------------------------------------------------------------===#
# Device
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct _DeviceImpl(Copyable, Defaultable, Movable):
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
                fn (UInt32, UnsafePointer[_DeviceImpl]) -> Result,
            ]()(UInt32(idx), UnsafePointer(to=device))
        )
        self.idx = idx
        self.device = device

    fn __copyinit__(out self, existing: Self):
        self.idx = existing.idx
        self.device = existing.device

    fn get_driver_version(self) raises -> DriverVersion:
        """Returns NVIDIA driver version."""
        alias max_length = 16
        var driver_version_buffer = stack_allocation[max_length, c_char]()

        _check_error(
            _get_dylib_function[
                "nvmlSystemGetDriverVersion",
                fn (UnsafePointer[c_char], UInt32) -> Result,
            ]()(driver_version_buffer, UInt32(max_length))
        )
        var driver_version_list = StaticString(
            unsafe_from_utf8_ptr=driver_version_buffer
        ).split(".")
        return DriverVersion(_to_string_list(driver_version_list))

    fn _max_clock(self, clock_type: ClockType) raises -> Int:
        var clock = UInt32()
        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetMaxClockInfo",
                fn (_DeviceImpl, ClockType, UnsafePointer[UInt32]) -> Result,
            ]()(self.device, clock_type, UnsafePointer(to=clock))
        )
        return Int(clock)

    fn max_mem_clock(self) raises -> Int:
        return self._max_clock(ClockType.MEM)

    fn max_graphics_clock(self) raises -> Int:
        return self._max_clock(ClockType.GRAPHICS)

    fn mem_clocks(self) raises -> List[Int, hint_trivial_type=True]:
        var num_clocks = UInt32()

        var result = _get_dylib_function[
            "nvmlDeviceGetSupportedMemoryClocks",
            fn (
                _DeviceImpl, UnsafePointer[UInt32], UnsafePointer[UInt32]
            ) -> Result,
        ]()(
            self.device,
            UnsafePointer(to=num_clocks),
            UnsafePointer[UInt32](),
        )
        if result != Result.INSUFFICIENT_SIZE:
            _check_error(result)

        var clocks = List[UInt32](length=UInt(num_clocks), fill=0)

        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetSupportedMemoryClocks",
                fn (
                    _DeviceImpl, UnsafePointer[UInt32], UnsafePointer[UInt32]
                ) -> Result,
            ]()(self.device, UnsafePointer(to=num_clocks), clocks.unsafe_ptr())
        )

        var res = List[Int, hint_trivial_type=True](capacity=len(clocks))
        for clock in clocks:
            res.append(Int(clock))

        return res

    fn graphics_clocks(
        self, memory_clock_mhz: Int
    ) raises -> List[Int, hint_trivial_type=True]:
        var num_clocks = UInt32()

        var result = _get_dylib_function[
            "nvmlDeviceGetSupportedGraphicsClocks",
            fn (
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
            return List[Int, hint_trivial_type=True]()

        if result != Result.INSUFFICIENT_SIZE:
            _check_error(result)

        var clocks = List[UInt32](length=UInt(num_clocks), fill=0)

        _check_error(
            _get_dylib_function[
                "nvmlDeviceGetSupportedGraphicsClocks",
                fn (
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

        var res = List[Int, hint_trivial_type=True](capacity=len(clocks))
        for clock in clocks:
            res.append(Int(clock))

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
                fn (_DeviceImpl, UnsafePointer[_EnableState]) -> Result,
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
                fn (_DeviceImpl, _EnableState) -> Result,
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

        for i in reversed(range(len(max_graphics_clock))):
            try:
                device.set_clock(max_mem_clock[-1], max_graphics_clock[i])
                print(
                    "the device clocks for device=",
                    device,
                    " were set to mem=",
                    max_mem_clock[-1],
                    " and graphics=",
                    max_graphics_clock[i],
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
@register_passable("trivial")
struct _EnableState(Copyable, Movable):
    var state: Int32

    alias DISABLED = _EnableState(0)  # Feature disabled
    alias ENABLED = _EnableState(1)  # Feature enabled

    fn __eq__(self, other: Self) -> Bool:
        return self.state == other.state
