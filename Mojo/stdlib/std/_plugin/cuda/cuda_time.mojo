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
"""NVIDIA lowerings of `std.time` functions, for `CUDAPluginHooks`."""

from std.sys import llvm_intrinsic


@inline(.always)
def _global_perf_counter_ns() -> UInt64:
    """Reads the `globaltimer` special register, a nanosecond timer shared by
    every SM."""
    return llvm_intrinsic[
        "llvm.nvvm.read.ptx.sreg.globaltimer",
        UInt64,
        has_side_effect=True,
    ]()


@inline(.always)
def _sleep(sec: Float64):
    """Sleeps in `nanosleep` steps until `globaltimer` shows `sec` elapsed."""
    # NVIDIA's nanosleep has a max duration of 1ms (1,000,000 ns).
    # Loop to handle longer sleep durations.
    comptime MAX_SLEEP_NS = 1_000_000  # 1ms in nanoseconds
    var total_ns = UInt64(sec * 1.0e9)
    var start = _global_perf_counter_ns()
    var elapsed = _global_perf_counter_ns() - start
    while elapsed < total_ns:
        var remaining = total_ns - elapsed
        var sleep_ns = Int32(min(remaining, UInt64(MAX_SLEEP_NS)))
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](sleep_ns)
        elapsed = _global_perf_counter_ns() - start
