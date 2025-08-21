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
# REQUIRES: NVIDIA-GPU

# RUN: NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# RUN: %mojo-build %s -o %t
# RUN: %mpirun -n $NUM_GPUS %t

from shmem import *
from os import listdir, getenv, setenv, abort
from gpu import block_dim, grid_dim, block_idx, thread_idx, barrier
from sys.ffi import c_int
from sys import sizeof
from math import iota


alias min_size = 1024 * 1024 * 32
alias max_size = min_size * 16
alias num_blocks = 32
alias threads_per_block = 512
alias iters = 4
alias warmup_iters = 1
alias step_factor = 2
alias chunk_size = 1024 * 256


fn ring_reduce(
    var dst: UnsafePointer[c_int],
    var src: UnsafePointer[c_int],
    var nreduce: Int,
    var signal: UnsafePointer[UInt64],
    chunk_size: Int,
):
    """Perform Allreduce using ring algorithm.

    This implements a ring-based allreduce that consists of two phases:
    1. Reduce phase: Data flows in ring pattern accumulating values
    2. Broadcast phase: Final result is broadcast back through the ring

    Args:
        dst: Destination buffer for reduced data.
        src: Source buffer with input data.
        nreduce: Number of elements to reduce.
        signal: Signaling buffer for synchronization.
        chunk_size: Size of each chunk in bytes.
    """
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes

    var thread_id = thread_idx.x
    var num_threads = block_dim.x
    var num_blocks = grid_dim.x
    var block_idx = block_idx.x
    var elems_per_block = nreduce // num_blocks

    if elems_per_block * (block_idx + 1) > nreduce:
        return

    src += block_idx * elems_per_block
    dst += block_idx * elems_per_block
    nreduce = elems_per_block
    signal += block_idx

    var chunk_elems = chunk_size // sizeof[DType.int32]()
    var num_chunks = nreduce // chunk_elems

    # Reduce phase - data flows through ring accumulating values
    for chunk in range(num_chunks):
        # Wait for data from previous PE (except PE 0 which starts)
        if mype != 0:
            if thread_id == 0:
                shmem_signal_wait_until(signal, SHMEM_CMP_GE, chunk + 1)

            barrier()

            var i = thread_id
            while i < chunk_elems:
                dst[i] = dst[i] + src[i]
                i += num_threads
            barrier()

        if thread_id == 0:
            shmem_put_signal_nbi(
                dst,
                src if mype == 0 else dst,
                chunk_elems,
                signal,
                1,
                SHMEM_SIGNAL_ADD,
                peer,
            )
        src += chunk_elems
        dst += chunk_elems

    # Broadcast phase - final result flows back through ring
    dst -= num_chunks * chunk_elems
    if thread_id == 0:
        for chunk in range(num_chunks):
            if mype < npes - 1:
                shmem_signal_wait_until(
                    signal,
                    SHMEM_CMP_GE,
                    chunk + 1 if mype == 0 else num_chunks + chunk + 1,
                )
            if mype < npes - 2:
                shmem_put_signal_nbi(
                    dst, dst, chunk_elems, signal, 1, SHMEM_SIGNAL_ADD, peer
                )
            dst += chunk_elems
        signal[0] = 0


def main():
    var min_ints = min_size // sizeof[DType.int32]()
    debug_assert(
        min_ints % num_blocks == 0, "min_size must be divisible by num_blocks"
    )

    with SHMEMContext() as ctx:
        var mype = shmem_my_pe()
        var npes = shmem_n_pes()

        # Allocate buffers
        var max_ints = max_size // sizeof[DType.int32]()

        var dst = ctx.enqueue_create_buffer[DType.int32](max_ints)
        var src = ctx.enqueue_create_buffer[DType.int32](max_ints)
        var data_h = ctx.enqueue_create_host_buffer[DType.int32](max_ints)
        var signal = shmem_calloc[DType.uint64](1)

        # Initialize test data - each element has value equal to its index
        iota(data_h.unsafe_ptr(), max_ints)

        # Copy test data to source buffer
        src.enqueue_copy_from(data_h)
        ctx.barrier_all()

        var dev_ctx = ctx.get_device_context()
        var dst_ptr = dst.unsafe_ptr()
        var src_ptr = src.unsafe_ptr()

        # Test different sizes
        var size = min_size
        while size <= max_size:
            var num_ints = size // sizeof[DType.int32]()

            # Warmup iterations
            for i in range(warmup_iters):
                ctx.enqueue_function_collective[ring_reduce](
                    dst_ptr,
                    src_ptr,
                    num_ints,
                    signal,
                    chunk_size,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
                ctx.barrier_all()
            ctx.synchronize()

            @parameter
            def benchmark():
                ctx.enqueue_function_collective[ring_reduce](
                    dst_ptr,
                    src_ptr,
                    num_ints,
                    signal,
                    chunk_size,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
                ctx.barrier_all()

            var elapsed_ns = dev_ctx.execution_time[benchmark](iters) / iters
            var elapsed_ms = elapsed_ns / 1e6

            ctx.synchronize()

            # Validate results - copy back and check
            dst.enqueue_copy_to(data_h)
            ctx.synchronize()

            # Each element should be i * npes after allreduce
            for i in range(num_ints):
                var expected = Int32(i * npes)
                if data_h[i] != expected:
                    # Avoid assert_equal overhead on these large buffers
                    abort(
                        String(
                            "PE: ",
                            mype,
                            " unexpected value at data_h[",
                            i,
                            "]",
                            " expected: ",
                            expected,
                            " actual: ",
                            data_h[i],
                        )
                    )

            if mype == 0:
                print(
                    "Test passed on size:",
                    size / 1024 / 1024,
                    "MB",
                    "\telapsed:",
                    elapsed_ms,
                    "ms",
                )

            size *= step_factor
