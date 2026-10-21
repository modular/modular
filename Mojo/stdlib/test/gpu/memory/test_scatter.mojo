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
"""Tests for scatter operations on GPU with different address spaces.

Verifies that scatter works correctly for both global memory (using the
llvm.masked.scatter intrinsic) and shared memory (using the scalar fallback
that avoids CUDA_ERROR_ILLEGAL_ADDRESS).
"""

from std.gpu import sync, thread_idx
from std.gpu.host import DeviceContext
from std.memory import stack_allocation
from std.testing import assert_equal, TestSuite


# ===-----------------------------------------------------------------------===#
# Global memory scatter
# ===-----------------------------------------------------------------------===#


def scatter_global_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    """Scatters values into global memory using reversed offsets."""
    var tid = Int(thread_idx.x)
    comptime N = 4

    # Each thread scatters its thread ID to the reversed position.
    var offset = SIMD[DType.int32, 1](N - 1 - tid)
    output.scatter(offset, SIMD[DType.float32, 1](Float32(tid + 1)))


def test_scatter_global() raises:
    comptime N = 4

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[DType.float32](N)
        ctx.enqueue_function_experimental[scatter_global_kernel](
            buf, grid_dim=1, block_dim=N
        )
        var result = ctx.enqueue_create_host_buffer[DType.float32](N)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        # Thread 0 writes 1.0 to index 3, thread 1 writes 2.0 to index 2, etc.
        assert_equal(result[0], Float32(4.0))
        assert_equal(result[1], Float32(3.0))
        assert_equal(result[2], Float32(2.0))
        assert_equal(result[3], Float32(1.0))


# ===-----------------------------------------------------------------------===#
# Shared memory scatter
# ===-----------------------------------------------------------------------===#


def scatter_shared_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    """Scatters values into shared memory, then copies to global output."""
    var tid = Int(thread_idx.x)
    comptime N = 4

    var sram = stack_allocation[
        N, Float32, address_space=AddressSpace.SHARED
    ]()

    # Initialize shared memory to zero.
    sram[tid] = 0.0
    sync.barrier()

    # Scatter: each thread writes its value to the reversed position
    # in shared memory.
    var offset = SIMD[DType.int32, 1](N - 1 - tid)
    sram.scatter(offset, SIMD[DType.float32, 1](Float32(tid + 1)))
    sync.barrier()

    # Copy shared memory result to global output.
    output[tid] = sram[tid]


def test_scatter_shared() raises:
    comptime N = 4

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[DType.float32](N)
        ctx.enqueue_function_experimental[scatter_shared_kernel](
            buf, grid_dim=1, block_dim=N
        )
        var result = ctx.enqueue_create_host_buffer[DType.float32](N)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        # Same result: reversed mapping.
        assert_equal(result[0], Float32(4.0))
        assert_equal(result[1], Float32(3.0))
        assert_equal(result[2], Float32(2.0))
        assert_equal(result[3], Float32(1.0))


# ===-----------------------------------------------------------------------===#
# Masked scatter in shared memory
# ===-----------------------------------------------------------------------===#


def scatter_shared_masked_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    """Scatters values into shared memory with a mask applied."""
    var tid = Int(thread_idx.x)
    comptime N = 4

    var sram = stack_allocation[
        N, Float32, address_space=AddressSpace.SHARED
    ]()

    # Initialize shared memory to -1 so we can verify masked-off lanes.
    sram[tid] = -1.0
    sync.barrier()

    # Only even-indexed threads scatter their values.
    var masked = (tid % 2) == 0
    var offset = SIMD[DType.int32, 1](tid)
    sram.scatter(
        offset,
        SIMD[DType.float32, 1](Float32(tid + 1)),
        SIMD[DType.bool, 1](masked),
    )
    sync.barrier()

    output[tid] = sram[tid]


def test_scatter_shared_masked() raises:
    comptime N = 4

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[DType.float32](N)
        ctx.enqueue_function_experimental[scatter_shared_masked_kernel](
            buf, grid_dim=1, block_dim=N
        )
        var result = ctx.enqueue_create_host_buffer[DType.float32](N)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        # Thread 0 (even) writes 1.0 to index 0.
        assert_equal(result[0], Float32(1.0))
        # Thread 1 (odd) is masked off, so index 1 stays -1.0.
        assert_equal(result[1], Float32(-1.0))
        # Thread 2 (even) writes 3.0 to index 2.
        assert_equal(result[2], Float32(3.0))
        # Thread 3 (odd) is masked off, so index 3 stays -1.0.
        assert_equal(result[3], Float32(-1.0))


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
