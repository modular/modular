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

from gpu import NamedBarrierSemaphore
from gpu.host import DeviceContext
from gpu import block_idx, grid_dim, thread_idx
from layout import Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from testing import assert_equal
from utils import IndexList

comptime NUM_BLOCKS = 32
comptime NUM_THREADS = 64


fn test_named_barrier_semaphore_equal_kernel(
    locks_ptr: UnsafePointer[Int32, MutAnyOrigin],
    shared_ptr: UnsafePointer[Int32, MutAnyOrigin],
):
    var sema = NamedBarrierSemaphore[Int32(NUM_THREADS), 4, 1](
        locks_ptr, Int(thread_idx.x)
    )

    sema.wait_eq(0, Int32(block_idx.x))

    if thread_idx.x == 0:
        shared_ptr[block_idx.x] = locks_ptr[0]

    sema.arrive_set(0, Int32(block_idx.x + 1))


fn test_named_barrier_semaphore_equal(ctx: DeviceContext) raises:
    print("== test_named_barrier_semaphore_equal")

    var locks_data = ManagedLayoutTensor[DType.int32, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(IndexList[1](1)), ctx
    )
    var shared_data = ManagedLayoutTensor[DType.int32, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            IndexList[1](NUM_BLOCKS)
        ),
        ctx,
    )
    var locks_host = locks_data.tensor[update=False]()
    var shared_host = shared_data.tensor[update=False]()
    locks_host[0] = Int32(0)
    for i in range(NUM_BLOCKS):
        shared_host[i] = Int32(NUM_BLOCKS)

    comptime kernel = test_named_barrier_semaphore_equal_kernel
    ctx.enqueue_function_experimental[kernel](
        locks_data.device_tensor().ptr,
        shared_data.device_tensor().ptr,
        grid_dim=(NUM_BLOCKS),
        block_dim=(NUM_THREADS),
    )
    shared_host = shared_data.tensor()

    for i in range(NUM_BLOCKS):
        assert_equal(shared_host[i], Int32(i))


fn test_named_barrier_semaphore_less_than_kernel(
    locks_ptr: UnsafePointer[Int32, MutAnyOrigin],
    shared_ptr: UnsafePointer[Int32, MutAnyOrigin],
):
    var sema = NamedBarrierSemaphore[Int32(NUM_THREADS), 4, 1](
        locks_ptr, Int(thread_idx.x)
    )

    sema.wait_lt(0, Int32(block_idx.x))

    if thread_idx.x == 0:
        shared_ptr[block_idx.x] = locks_ptr[0]

    sema.arrive_set(0, Int32(block_idx.x + 1))


fn test_named_barrier_semaphore_less_than(ctx: DeviceContext) raises:
    print("== test_named_barrier_semaphore_less_than")

    var locks_data = ManagedLayoutTensor[DType.int32, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(IndexList[1](1)), ctx
    )
    var shared_data = ManagedLayoutTensor[DType.int32, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            IndexList[1](NUM_BLOCKS)
        ),
        ctx,
    )
    var locks_host = locks_data.tensor[update=False]()
    var shared_host = shared_data.tensor[update=False]()
    locks_host[0] = Int32(0)
    for i in range(NUM_BLOCKS):
        shared_host[i] = Int32(NUM_BLOCKS)

    comptime kernel = test_named_barrier_semaphore_less_than_kernel
    ctx.enqueue_function_experimental[kernel](
        locks_data.device_tensor().ptr,
        shared_data.device_tensor().ptr,
        grid_dim=(NUM_BLOCKS),
        block_dim=(NUM_THREADS),
    )
    shared_host = shared_data.tensor()

    for i in range(NUM_BLOCKS):
        assert_equal(shared_host[i], Int32(i))


def main():
    with DeviceContext() as ctx:
        test_named_barrier_semaphore_equal(ctx)
        test_named_barrier_semaphore_less_than(ctx)
