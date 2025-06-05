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

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    ThroughputMeasure,
    BenchMetric,
)
from bit import log2_floor
from math import ceildiv
from memory import UnsafePointer, stack_allocation
from os.atomic import Atomic
from random import randint
from sys import has_accelerator, sizeof
from testing import assert_equal

from gpu import thread_idx, block_idx, block_dim, grid_dim, warp, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace, load

# Initialize parameters
# To archieve high bandwidth increase SIZE to large value
alias TPB = 512
alias LOG_TPB = log2_floor(TPB)
alias BATCH_SIZE = 8  # needs to be power of 2
alias SIZE = 1 << 29
alias NUM_BLOCKS = ceildiv(SIZE, TPB * BATCH_SIZE)
alias WARP_SIZE = 32
alias dtype = DType.int32


fn sum_kernel[
    size: Int, batch_size: Int
](output: UnsafePointer[Int32], a: UnsafePointer[Int32],):
    """Efficent reduction of the vector a."""
    sums = stack_allocation[
        TPB,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()
    global_tid = block_idx.x * block_dim.x + thread_idx.x
    tid = thread_idx.x
    threads_in_grid = TPB * grid_dim.x
    var sum: Int32 = 0

    for i in range(global_tid, size, threads_in_grid):
        idx = i * batch_size
        # Load in a vectorized fashion and reduce the loaded SIMD vector
        if idx < size:
            sum += load[width=batch_size](a, idx).reduce_add()
    sums[tid] = sum
    barrier()

    # Reduce until the first warp
    active_threads = TPB

    @parameter
    for power in range(1, LOG_TPB - 4):
        active_threads >>= 1
        if tid < active_threads:
            sums[tid] += sums[tid + active_threads]
        barrier()

    # Reduce the warp and accumulate via atomic addition
    if tid < WARP_SIZE:
        var warp_sum: Int32 = sums[tid][0]
        warp_sum = warp.sum(warp_sum)

        if tid == 0:
            _ = Atomic.fetch_add(output, warp_sum)


# Benchmark function for sum_kernel
@parameter
@always_inline
fn sum_kernel_benchmark(
    mut b: Bencher, input_data: (UnsafePointer[Int32], UnsafePointer[Int32])
) capturing raises:
    @parameter
    @always_inline
    fn kernel_launch_sum(ctx: DeviceContext) raises:
        var out_ptr = input_data[0]
        var a_ptr = input_data[1]
        ctx.enqueue_function[sum_kernel[SIZE, BATCH_SIZE]](
            out_ptr,
            a_ptr,
            grid_dim=NUM_BLOCKS,
            block_dim=TPB,
        )

    var bench_ctx = DeviceContext()
    b.iter_custom[kernel_launch_sum](bench_ctx)


def main():
    constrained[
        has_accelerator(),
        "This example requires a supported GPU",
    ]()

    with DeviceContext() as ctx:
        # Allocate memory on the device
        out = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

        # Initialise a with random integers between 0 and 10
        with a.map_to_host() as a_host:
            randint[dtype](a_host.unsafe_ptr(), SIZE, 0, 10)

        # Get unsafe pointers to device
        out_ptr = out.unsafe_ptr()
        a_ptr = a.unsafe_ptr()

        # Call the kernel
        ctx.enqueue_function[sum_kernel[SIZE, BATCH_SIZE]](
            out_ptr,
            a_ptr,
            grid_dim=NUM_BLOCKS,
            block_dim=TPB,
        )
        ctx.synchronize()

        # Calculate the sum in a sequential fashion on the host
        # for correctness check
        expected = ctx.enqueue_create_host_buffer[dtype](1).enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                expected[0] += a_host[i]

        # Assert the correctness of the kernel
        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            assert_equal(out_host[0], expected[0])

        # Benchmark performance
        var bench = Bench(BenchConfig(max_iters=50000))
        bench.bench_with_input[
            (UnsafePointer[Int32], UnsafePointer[Int32]), sum_kernel_benchmark
        ](
            BenchId("sum_kernel_benchmark", "gpu"),
            (out_ptr, a_ptr),
            ThroughputMeasure(BenchMetric.bytes, SIZE * sizeof[dtype]()),
        )
        # Pretty print in table format
        print(bench)
