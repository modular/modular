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

from gpu.host import DeviceContext
from gpu.id import block_idx, grid_dim, thread_idx
from gpu import NamedBarrierSemaphore
from testing import assert_equal

alias NUM_BLOCKS = 32
alias NUM_THREADS = 64


fn test_named_barrier_semaphore_equal_kernel(
    locks_ptr: UnsafePointer[Int32],
    shared_ptr: UnsafePointer[Int32],
):
    var sema = NamedBarrierSemaphore[Int32(NUM_THREADS), 4, 1](
        locks_ptr, thread_idx.x
    )

    sema.wait_eq(0, block_idx.x)

    if thread_idx.x == 0:
        shared_ptr[block_idx.x] = locks_ptr[0]

    sema.arrive_set(0, block_idx.x + 1)


fn test_named_barrier_semaphore_equal(ctx: DeviceContext) raises:
    print("== test_named_barrier_semaphore_equal")

    var a_host = UnsafePointer[Scalar[DType.int32]].alloc(NUM_BLOCKS)

    var locks_data = ctx.enqueue_create_buffer[DType.int32](1)
    var shared_data = ctx.enqueue_create_buffer[DType.int32](NUM_BLOCKS)
    ctx.enqueue_memset(locks_data, 0)
    ctx.enqueue_memset(shared_data, NUM_BLOCKS)

    ctx.enqueue_function[test_named_barrier_semaphore_equal_kernel](
        locks_data,
        shared_data,
        grid_dim=(NUM_BLOCKS),
        block_dim=(NUM_THREADS),
    )
    ctx.synchronize()

    ctx.enqueue_copy(a_host, shared_data)

    for i in range(NUM_BLOCKS):
        assert_equal(a_host[i], i)

    _ = shared_data^
    _ = locks_data^


fn test_named_barrier_semaphore_less_than_kernel(
    locks_ptr: UnsafePointer[Int32],
    shared_ptr: UnsafePointer[Int32],
):
    var sema = NamedBarrierSemaphore[Int32(NUM_THREADS), 4, 1](
        locks_ptr, thread_idx.x
    )

    sema.wait_lt(0, block_idx.x)

    if thread_idx.x == 0:
        shared_ptr[block_idx.x] = locks_ptr[0]

    sema.arrive_set(0, block_idx.x + 1)


fn test_named_barrier_semaphore_less_than(ctx: DeviceContext) raises:
    print("== test_named_barrier_semaphore_less_than")

    var a_host = UnsafePointer[Scalar[DType.int32]].alloc(NUM_BLOCKS)

    var locks_data = ctx.enqueue_create_buffer[DType.int32](1)
    var shared_data = ctx.enqueue_create_buffer[DType.int32](NUM_BLOCKS)
    ctx.enqueue_memset(locks_data, 0)
    ctx.enqueue_memset(shared_data, NUM_BLOCKS)

    ctx.enqueue_function[test_named_barrier_semaphore_less_than_kernel](
        locks_data,
        shared_data,
        grid_dim=(NUM_BLOCKS),
        block_dim=(NUM_THREADS),
    )
    ctx.synchronize()

    ctx.enqueue_copy(a_host, shared_data)

    for i in range(NUM_BLOCKS):
        assert_equal(a_host[i], i)

    _ = shared_data^
    _ = locks_data^


def main():
    with DeviceContext() as ctx:
        test_named_barrier_semaphore_equal(ctx)
        test_named_barrier_semaphore_less_than(ctx)
