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
"""Runtime test for time.sleep() on GPU.

This test verifies that time.sleep() works correctly on NVIDIA GPUs, including:
1. Durations longer than 1ms (requires looping since nanosleep has 1ms max)
2. Durations shorter than 1ms (single nanosleep call)
3. Edge cases like zero/negative durations
"""

from time import global_perf_counter_ns, sleep

from gpu.host import DeviceContext
from testing import assert_true


fn sleep_kernel_100ms(result_ptr: UnsafePointer[UInt64, MutExternalOrigin]):
    """GPU kernel that sleeps for 100ms and stores elapsed time."""
    # Use global_perf_counter_ns() which returns actual nanoseconds on NVIDIA
    # GPUs, unlike perf_counter_ns() which returns cycle counts.
    var start = global_perf_counter_ns()
    sleep(0.1)
    var end = global_perf_counter_ns()
    result_ptr[] = end - start


fn sleep_kernel_500us(result_ptr: UnsafePointer[UInt64, MutExternalOrigin]):
    """GPU kernel that sleeps for 500 microseconds (sub-1ms)."""
    var start = global_perf_counter_ns()
    sleep(0.0005)
    var end = global_perf_counter_ns()
    result_ptr[] = end - start


fn sleep_kernel_zero(result_ptr: UnsafePointer[UInt64, MutExternalOrigin]):
    """GPU kernel that sleeps for zero duration (should return immediately)."""
    var start = global_perf_counter_ns()
    sleep(0.0)
    var end = global_perf_counter_ns()
    result_ptr[] = end - start


def test_sleep_100ms(ctx: DeviceContext):
    """Test 100ms sleep (requires loop since nanosleep max is 1ms)."""
    var result_host = ctx.enqueue_create_host_buffer[DType.uint64](1)
    var result_device = ctx.enqueue_create_buffer[DType.uint64](1)

    result_host[0] = 0
    ctx.enqueue_function[sleep_kernel_100ms, sleep_kernel_100ms](
        result_device, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    var elapsed_ms_gpu = Float64(result_host[0]) / 1_000_000.0

    # The implementation loops until elapsed >= total_ns using global_perf_counter_ns(),
    # so the measured time should be at least 100ms. NVIDIA's nanosleep can oversleep
    # up to 2x per call, so we allow up to 300ms for the upper bound.
    assert_true(
        elapsed_ms_gpu >= 95.0 and elapsed_ms_gpu <= 300.0,
        "100ms sleep outside expected bounds [95ms, 300ms]",
    )


def test_sleep_500us(ctx: DeviceContext):
    """Test 500 microsecond sleep (sub-1ms, single nanosleep call)."""
    var result_host = ctx.enqueue_create_host_buffer[DType.uint64](1)
    var result_device = ctx.enqueue_create_buffer[DType.uint64](1)

    result_host[0] = 0
    ctx.enqueue_function[sleep_kernel_500us, sleep_kernel_500us](
        result_device, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    var elapsed_us_gpu = Float64(result_host[0]) / 1_000.0

    # The implementation loops until elapsed >= total_ns using global_perf_counter_ns(),
    # so the measured time should be at least 500us. Allow up to 3ms for oversleep.
    assert_true(
        elapsed_us_gpu >= 450.0 and elapsed_us_gpu <= 3000.0,
        "500us sleep outside expected bounds [450us, 3ms]",
    )


def test_sleep_zero(ctx: DeviceContext):
    """Test zero duration sleep (should return immediately)."""
    var result_host = ctx.enqueue_create_host_buffer[DType.uint64](1)
    var result_device = ctx.enqueue_create_buffer[DType.uint64](1)

    result_host[0] = 0
    ctx.enqueue_function[sleep_kernel_zero, sleep_kernel_zero](
        result_device, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    var elapsed_us_gpu = Float64(result_host[0]) / 1_000.0

    # Zero sleep should complete very quickly (< 1ms).
    assert_true(
        elapsed_us_gpu < 1000.0,
        "Zero sleep took too long (> 1ms)",
    )


def main():
    with DeviceContext() as ctx:
        test_sleep_100ms(ctx)
        test_sleep_500us(ctx)
        test_sleep_zero(ctx)
