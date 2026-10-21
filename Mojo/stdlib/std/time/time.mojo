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
"""Implements basic utils for working with time.

You can import these APIs from the `time` package. For example:

```mojo
from std.time import perf_counter_ns
```
"""

from std._plugin import CurrentPlugin
from std.math import ceil, floor
from std.ffi import external_call
from std.sys import CompilationTarget, is_gpu

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
struct _CTimeSpec(Defaultable, TrivialRegisterPassable, Writable):
    var tv_sec: Int  # Seconds
    var tv_nsec: Int  # Nanoseconds

    def as_nanoseconds(self) -> Int:
        return self.tv_sec * _NSEC_PER_SEC + self.tv_nsec

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_nanoseconds(), "ns")


@inline(.always)
def _clock_gettime(clockid: Int) -> _CTimeSpec:
    """Low-level call to the clock_gettime libc function"""
    var ts = _CTimeSpec()

    # Call libc's clock_gettime.
    _ = external_call["clock_gettime", Int32](Int32(clockid), Pointer(to=ts))

    return ts


@inline(.always)
def _gettime_as_nsec_unix(clockid: Int) -> Int:
    comptime if CompilationTarget.is_linux():
        var ts = _clock_gettime(clockid)
        return ts.as_nanoseconds()
    else:
        return Int(
            external_call["clock_gettime_nsec_np", Int64](Int32(clockid))
        )


@inline(.always)
def _realtime_nanoseconds() -> Int:
    """Returns the current realtime time in nanoseconds"""
    return _gettime_as_nsec_unix(_CLOCK_REALTIME)


@inline(.always)
def _monotonic_nanoseconds() -> Int:
    """Returns the current monotonic time in nanoseconds"""

    comptime if is_gpu():
        return Int(global_perf_counter_ns())
    else:
        return _gettime_as_nsec_unix(_CLOCK_MONOTONIC)


@inline(.always)
def _monotonic_raw_nanoseconds() -> Int:
    """Returns the current monotonic time in nanoseconds"""
    return _gettime_as_nsec_unix(_CLOCK_MONOTONIC_RAW)


@inline(.always)
def _process_cputime_nanoseconds() -> Int:
    """Returns the high-resolution per-process timer from the CPU"""

    return _gettime_as_nsec_unix(_CLOCK_PROCESS_CPUTIME_ID)


@inline(.always)
def _thread_cputime_nanoseconds() -> Int:
    """Returns the thread-specific CPU-time clock"""

    return _gettime_as_nsec_unix(_CLOCK_THREAD_CPUTIME_ID)


# ===-----------------------------------------------------------------------===#
# perf_counter
# ===-----------------------------------------------------------------------===#


@inline(.always)
def perf_counter() -> Float64:
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


@inline(.always)
def perf_counter_ns() -> Int:
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


comptime _GlobalPerfCounterNsPluginHookFnType = def() thin -> UInt64
"""Plugin-hook signature for `PluginHooks.global_perf_counter_ns_fn`."""


@inline(.always)
def global_perf_counter_ns() -> UInt64:
    """Returns the current value in the global nanosecond resolution timer. This value
    is common across all SM's.

    On GPUs, this reads the timer that the target's stdlib plugin provides,
    such as the globaltimer register on NVIDIA GPUs or the realtime counter
    (a constant-speed clock) on AMD GPUs. On other platforms, this falls back
    to perf_counter_ns().

    Returns:
        The current time in ns.
    """

    comptime if case .Some(_read) = CurrentPlugin.global_perf_counter_ns_fn:
        return _read()
    elif is_gpu():
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()
    else:
        return UInt64(perf_counter_ns())


# ===-----------------------------------------------------------------------===#
# monotonic
# ===-----------------------------------------------------------------------===#


@inline(.always)
def monotonic() -> Int:
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


@inline(.always)
def time_function[FuncType: def() raises -> None](func: FuncType) raises -> Int:
    """Measures the time spent in the function.

    Parameters:
        FuncType: The function type to time.

    Args:
        func: The closure carrying the captured state of the timed function.

    Returns:
        The time elapsed in the function in ns.

    Raises:
        If the operation fails.
    """
    var tic = perf_counter_ns()
    func()
    var toc = perf_counter_ns()
    return toc - tic


@inline(.always)
def time_function[FuncType: def() -> None](func: FuncType) -> Int:
    """Measures the time spent in the function.

    Parameters:
        FuncType: The function type to time.

    Args:
        func: The closure carrying the captured state of the timed function.

    Returns:
        The time elapsed in the function in ns.
    """
    var tic = perf_counter_ns()
    func()
    var toc = perf_counter_ns()
    return toc - tic


# ===-----------------------------------------------------------------------===#
# sleep
# ===-----------------------------------------------------------------------===#


comptime _SleepPluginHookFnType = def(sec: Float64) thin -> None
"""Plugin-hook signature for `PluginHooks.sleep_fn`."""


def sleep(sec: Float64):
    """Suspends the current thread for the seconds specified.

    Args:
        sec: The number of seconds to sleep for. Values <= 0 return immediately.
    """
    # Guard against non-positive sleep durations.
    if sec <= 0.0:
        return

    comptime if case .Some(_sleep) = CurrentPlugin.sleep_fn:
        _sleep(sec)
        return
    elif is_gpu():
        CompilationTarget.unsupported_target_error[operation="time.sleep()"]()

    comptime NANOSECONDS_IN_SECOND = 1_000_000_000
    var total_secs = floor(sec)
    var nsecs = Int(ceil((sec - total_secs) * NANOSECONDS_IN_SECOND))
    # Handle floating-point rounding edge case where the nanosecond value
    # rounds to exactly one second (e.g., 0.9999999999 seconds).
    if nsecs >= NANOSECONDS_IN_SECOND:
        total_secs += 1
        nsecs -= NANOSECONDS_IN_SECOND
    var tv_spec = _CTimeSpec(
        Int(total_secs),
        nsecs,
    )
    var req = Pointer(to=tv_spec)
    var rem = OptionalPointer[_CTimeSpec, MutUntrackedOrigin]()
    _ = external_call["nanosleep", Int32](req, rem)


def sleep(sec: Int):
    """Suspends the current thread for the seconds specified.

    Args:
        sec: The number of seconds to sleep for.
    """

    comptime if is_gpu():
        return sleep(Float64(sec))

    external_call["sleep", NoneType](Int32(sec))
