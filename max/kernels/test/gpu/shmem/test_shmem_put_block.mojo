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
from testing import assert_equal
from gpu import global_idx, block_dim, block_idx
from os import abort


fn set_and_shift_kernel(
    send_data: UnsafePointer[Float32],
    recv_data: UnsafePointer[Float32],
    num_elems: UInt,
    mype: Int32,
    npes: Int32,
    use_nbi: Bool,
):
    var thread_idx = global_idx.x

    # set the corresponding element of send_data
    if thread_idx < num_elems:
        send_data[thread_idx] = Float32(mype)

    var peer = (mype + 1) % npes
    var block_offset = block_idx.x * block_dim.x

    # Every thread in block 0 calls shmem_put_block. shmem_p over IB will use
    # one RMA message for every element, and it cannot leverage multiple threads
    # to copy the data to the destination GPU.

    if use_nbi:
        shmem_put_nbi[SHMEMScope.block](
            recv_data + block_offset,
            send_data + block_offset,
            min(block_dim.x, num_elems - block_offset),
            peer,
        )
    else:
        shmem_put[SHMEMScope.block](
            recv_data + block_offset,
            send_data + block_offset,
            min(block_dim.x, num_elems - block_offset),
            peer,
        )


fn test_shmem_put[use_nbi: Bool](ctx: SHMEMContext) raises:
    alias num_elems: UInt = 8192
    alias threads_per_block: UInt = 1024
    debug_assert(
        num_elems % threads_per_block == 0,
        "num_elems must be divisible by threads_per_block",
    )
    alias num_blocks = num_elems // threads_per_block

    var mype = shmem_my_pe()
    var npes = shmem_n_pes()

    var send_data = ctx.enqueue_create_buffer[DType.float32](num_elems)
    var recv_data = ctx.enqueue_create_buffer[DType.float32](num_elems)

    ctx.barrier_all()

    ctx.enqueue_function[set_and_shift_kernel](
        send_data.unsafe_ptr(),
        recv_data.unsafe_ptr(),
        num_elems,
        mype,
        npes,
        use_nbi,
        grid_dim=num_blocks,
        block_dim=threads_per_block,
    )

    var host = ctx.enqueue_create_host_buffer[DType.float32](num_elems)
    recv_data.enqueue_copy_to(host)

    # The completion of the non-blocking version of `shmem_put` is
    # guaranteed by the `nvshmem_barrier_all_on_stream` call.
    ctx.barrier_all()
    ctx.synchronize()

    # Verify the result
    var expected = Float32((mype - 1 + npes) % npes)

    for i in range(num_elems):
        assert_equal(
            host[i],
            expected,
            String("unexpected value on PE: ", mype, " at idx: ", i),
        )

    print("[", mype, "of", npes, "] run complete. use_nbi=", use_nbi)


def main():
    with SHMEMContext() as ctx:
        test_shmem_put[False](ctx)

        # Test the non-blocking version of `shmem_put` primitive, which returns
        # after initiating the operation.
        test_shmem_put[True](ctx)
