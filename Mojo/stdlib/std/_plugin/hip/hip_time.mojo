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
"""AMD lowerings of `std.time` functions, for `HIPPluginHooks`."""

from std.sys import llvm_intrinsic


@inline(.always)
def _amd_gpu_realtime() -> UInt64:
    """Returns the AMD GPU real-time counter (constant-speed clock).

    This reads the realtime register which provides a constant-speed 64-bit
    clock counter, independent of GPU core clock frequency scaling. This is
    suitable for wall-clock timing measurements.

    The counter frequency is 100 MHz (10ns per tick) on MI355 (gfx950) and
    likely other modern AMD GPUs, though this may vary by architecture.
    Use _AMD_GPU_REALTIME_FREQ_HZ for conversions.
    """
    return llvm_intrinsic["llvm.readsteadycounter", UInt64]()


# AMD GPU real-time counter frequency in Hz.
# MI355 (gfx950) and likely other modern AMD GPUs use 100 MHz.
# This can be queried at runtime via hipDeviceAttributeWallClockRate if needed.
comptime _AMD_GPU_REALTIME_FREQ_HZ: UInt64 = 100_000_000


@inline(.always)
def _global_perf_counter_ns() -> UInt64:
    """Converts the realtime counter's ticks to nanoseconds."""
    # At 100 MHz, each tick is 10ns (1e9 / 100e6 = 10).
    var ticks = _amd_gpu_realtime()
    return (ticks * 1_000_000_000) // _AMD_GPU_REALTIME_FREQ_HZ


@inline(.always)
def _sleep(sec: Float64):
    """Sleeps in `s_sleep` steps until the realtime counter shows `sec`
    elapsed.

    This approach is based on ROCm's ockl rtcwait implementation. The
    `s_sleep` instruction accepts values 0-127 and sleeps for approximately
    that many cycles.
    """
    # The tiered approach (127 -> 15 -> 1) balances power efficiency
    # (longer sleeps when far from target) with timing accuracy
    # (shorter sleeps as we approach the target).
    var total_ticks = UInt64(sec * Float64(_AMD_GPU_REALTIME_FREQ_HZ))
    var start = _amd_gpu_realtime()
    var end = start + total_ticks
    var now = start

    # Thresholds are approximate tick counts where each sleep level
    # is appropriate (based on 10ns per tick at 100 MHz).
    while now < end:
        var remaining = end - now
        if remaining > 3000:  # > ~30us remaining
            llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(127))
        elif remaining > 400:  # > ~4us remaining
            llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(15))
        else:
            llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))
        now = _amd_gpu_realtime()
