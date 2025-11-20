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
"""Implements basic utils for working with time.

You can import these APIs from the `time` package. For example:

```mojo
from time import perf_counter_ns
```
"""

from math import floor
from memory import LegacyUnsafePointer as UnsafePointer
from os import abort
from sys import (
    CompilationTarget,
    external_call,
    is_amd_gpu,
    is_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
)

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#

# Enums used in time.h 's glibc
comptime _CLOCK_REALTIME = 0
comptime _CLOCK_MONOTONIC = 1 if CompilationTarget.is_linux() else 6
comptime _CLOCK_PROCESS_CPUTIME_ID = 2 if CompilationTarget.is_linux() else 12
comptime _CLOCK_THREAD_CPUTIME_ID = 3 if CompilationTarget.is_linux() else 16
comptime _CLOCK_MONOTONIC_RAW = 4

# Constants
comptime _NSEC_PER_USEC = 1000
comptime _NSEC_PER_MSEC = 1_000_000
comptime _USEC_PER_MSEC = 1000
comptime _MSEC_PER_SEC = 1000
comptime _NSEC_PER_SEC = _NSEC_PER_USEC * _USEC_PER_MSEC * _MSEC_PER_SEC


@fieldwise_init
@register_passable("trivial")
struct _CTimeSpec(
    Defaultable, ImplicitlyCopyable, Movable, Stringable, Writable
):
    var tv_sec: Int  # Seconds
    var tv_subsec: Int  # subsecond (nanoseconds on linux and usec on mac)

    fn __init__(out self):
        self.tv_sec = 0
        self.tv_subsec = 0

    fn as_nanoseconds(self) -> UInt:
        @parameter
        if CompilationTarget.is_linux():
            return UInt(self.tv_sec * _NSEC_PER_SEC + self.tv_subsec)
        else:
            return UInt(
                self.tv_sec * _NSEC_PER_SEC + self.tv_subsec * _NSEC_PER_USEC
            )

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_nanoseconds(), "ns")


@always_inline
fn _clock_gettime(clockid: Int) -> _CTimeSpec:
    """Low-level call to the clock_gettime libc function"""
    var ts = _CTimeSpec()

    # Call libc's clock_gettime.
    _ = external_call["clock_gettime", Int32](Int32(clockid), Pointer(to=ts))

    return ts


@always_inline
fn _gettime_as_nsec_unix(clockid: Int) -> UInt:
    @parameter
    if CompilationTarget.is_linux():
        var ts = _clock_gettime(clockid)
        return ts.as_nanoseconds()
    else:
        return UInt(
            Int(external_call["clock_gettime_nsec_np", Int64](Int32(clockid)))
        )


@always_inline
fn _gpu_clock() -> UInt:
    """Returns a 64-bit unsigned cycle counter."""
    comptime asm = _gpu_clock_inst()
    return UInt(Int(llvm_intrinsic[asm, Int64]()))


fn _gpu_clock_inst() -> StaticString:
    @parameter
    if is_nvidia_gpu():
        return "llvm.nvvm.read.ptx.sreg.clock64"
    elif is_amd_gpu():
        return "llvm.amdgcn.s.memtime"
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString,
            operation="_gpu_clock",
        ]()


@always_inline
fn _realtime_nanoseconds() -> UInt:
    """Returns the current realtime time in nanoseconds"""
    return _gettime_as_nsec_unix(_CLOCK_REALTIME)


@always_inline
fn _monotonic_nanoseconds() -> UInt:
    """Returns the current monotonic time in nanoseconds"""

    @parameter
    if is_gpu():
        return _gpu_clock()
    else:
        return _gettime_as_nsec_unix(_CLOCK_MONOTONIC)


@always_inline
fn _monotonic_raw_nanoseconds() -> UInt:
    """Returns the current monotonic time in nanoseconds"""
    return _gettime_as_nsec_unix(_CLOCK_MONOTONIC_RAW)


@always_inline
fn _process_cputime_nanoseconds() -> UInt:
    """Returns the high-resolution per-process timer from the CPU"""

    return _gettime_as_nsec_unix(_CLOCK_PROCESS_CPUTIME_ID)


@always_inline
fn _thread_cputime_nanoseconds() -> UInt:
    """Returns the thread-specific CPU-time clock"""

    return _gettime_as_nsec_unix(_CLOCK_THREAD_CPUTIME_ID)


# ===-----------------------------------------------------------------------===#
# perf_counter
# ===-----------------------------------------------------------------------===#


@always_inline
fn perf_counter() -> Float64:
    """Return the value (in fractional seconds) of a performance counter, i.e.
    a clock with the highest available resolution to measure a short duration.
    It does include time elapsed during sleep and is system-wide. The reference
    point of the returned value is undefined, so that only the difference
    between the results of two calls is valid.

    Returns:
        The current time in ns.
    """
    return Float64(_monotonic_nanoseconds()) / _NSEC_PER_SEC


# ===-----------------------------------------------------------------------===#
# perf_counter_ns
# ===-----------------------------------------------------------------------===#


@always_inline
fn perf_counter_ns() -> UInt:
    """Return the value (in nanoseconds) of a performance counter, i.e.
    a clock with the highest available resolution to measure a short duration.
    It does include time elapsed during sleep and is system-wide. The reference
    point of the returned value is undefined, so that only the difference
    between the results of two calls is valid.

    Returns:
        The current time in ns.
    """
    return _monotonic_nanoseconds()


# ===-----------------------------------------------------------------------===#
# global perf_counter_ns
# ===-----------------------------------------------------------------------===#


@always_inline
fn global_perf_counter_ns() -> SIMD[DType.uint64, 1]:
    """Returns the current value in the global nanosecond resolution timer. This value
    is common across all SM's. Currently, this is only supported on NVIDIA GPUs, on
    non-NVIDIA GPUs, this function returns the same value as perf_counter_ns().

    Returns:
        The current time in ns.
    """

    @parameter
    if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.read.ptx.sreg.globaltimer",
            UInt64,
            has_side_effect=True,
        ]()

    return perf_counter_ns()


# ===-----------------------------------------------------------------------===#
# monotonic
# ===-----------------------------------------------------------------------===#


@always_inline
fn monotonic() -> UInt:
    """
    Returns the current monotonic time time in nanoseconds. This function
    queries the current platform's monotonic clock, making it useful for
    measuring time differences, but the significance of the returned value
    varies depending on the underlying implementation.

    Returns:
        The current time in ns.
    """
    return perf_counter_ns()


# ===-----------------------------------------------------------------------===#
# time_function
# ===-----------------------------------------------------------------------===#


@always_inline
@parameter
fn time_function[func: fn () raises capturing [_] -> None]() raises -> UInt:
    """Measures the time spent in the function.

    Parameters:
        func: The function to time.

    Returns:
        The time elapsed in the function in ns.

    Raises:
        If the operation fails.
    """
    var tic = perf_counter_ns()
    func()
    var toc = perf_counter_ns()
    return toc - tic


@always_inline
@parameter
fn time_function[func: fn () capturing [_] -> None]() -> UInt:
    """Measures the time spent in the function.

    Parameters:
        func: The function to time.

    Returns:
        The time elapsed in the function in ns.
    """

    @parameter
    fn raising_func() raises:
        func()

    try:
        return time_function[raising_func]()
    except err:
        return abort[UInt](String(err))


# ===-----------------------------------------------------------------------===#
# sleep
# ===-----------------------------------------------------------------------===#


fn sleep(sec: Float64):
    """Suspends the current thread for the seconds specified.

    Args:
        sec: The number of seconds to sleep for.
    """

    @parameter
    if is_gpu():
        var nsec = sec * 1.0e9
        comptime intrinsic = _gpu_sleep_inst()
        llvm_intrinsic[intrinsic, NoneType](nsec.cast[DType.int32]())
        return

    comptime NANOSECONDS_IN_SECOND = 1_000_000_000
    var total_secs = floor(sec)
    var tv_spec = _CTimeSpec(
        Int(total_secs),
        Int((sec - total_secs) * NANOSECONDS_IN_SECOND),
    )
    var req = UnsafePointer[_CTimeSpec](to=tv_spec)
    var rem = UnsafePointer[_CTimeSpec]()
    _ = external_call["nanosleep", Int32](req, rem)
    _ = tv_spec
    _ = req
    _ = rem


fn _gpu_sleep_inst() -> StaticString:
    @parameter
    if is_nvidia_gpu():
        return "llvm.nvvm.nanosleep"
    elif is_amd_gpu():
        return "llvm.amdgcn.s.sleep"
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString,
            operation="sleep",
        ]()


fn sleep(sec: UInt):
    """Suspends the current thread for the seconds specified.

    Args:
        sec: The number of seconds to sleep for.
    """

    @parameter
    if is_gpu():
        return sleep(Float64(sec))

    external_call["sleep", NoneType](Int32(sec))
