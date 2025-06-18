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

import time

from gpu import AddressSpace, memory, sync, thread_idx
from gpu.host import DeviceContext
from memory import stack_allocation


fn copy_via_shared(
    src: UnsafePointer[Float32],
    dst: UnsafePointer[Float32],
):
    var thId = Int(thread_idx.x)
    var mem_buff: UnsafePointer[
        Float32, address_space = AddressSpace.SHARED
    ] = stack_allocation[16, Float32, address_space = AddressSpace.SHARED]()
    var src_global: UnsafePointer[
        Float32, address_space = AddressSpace.GLOBAL
    ] = src.address_space_cast[AddressSpace.GLOBAL]()

    memory.async_copy[4](
        src_global.offset(thId),
        mem_buff.offset(thId),
    )

    var m_barrier = stack_allocation[
        1, DType.int32, address_space = AddressSpace.SHARED
    ]()
    sync.mbarrier_init(m_barrier, 16)
    sync.mbarrier(m_barrier)
    var state = sync.mbarrier_arrive(m_barrier)
    var not_wait = False
    while not not_wait:
        time.sleep(100 * 1e-6)
        not_wait = sync.mbarrier_test_wait(m_barrier, state)

    dst[thId] = mem_buff[thId]


# CHECK-LABEL: run_copy_via_shared
fn run_copy_via_shared(ctx: DeviceContext) raises:
    print("== run_copy_via_shared")
    var in_data = UnsafePointer[Float32].alloc(16)
    var out_data = UnsafePointer[Float32].alloc(16)

    for i in range(16):
        in_data[i] = i + 1
        out_data[i] = 0

    var in_device = ctx.enqueue_create_buffer[DType.float32](16)
    var out_device = ctx.enqueue_create_buffer[DType.float32](16)

    ctx.enqueue_copy(in_device, in_data)
    ctx.enqueue_copy(out_device, out_data)

    ctx.enqueue_function[copy_via_shared](
        in_device,
        out_device,
        grid_dim=(1,),
        block_dim=(16,),
    )

    ctx.enqueue_copy(out_data, out_device)

    ctx.synchronize()

    # CHECK: 1.0
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: 6.0
    # CHECK: 7.0
    # CHECK: 8.0
    # CHECK: 9.0
    # CHECK: 10.0
    # CHECK: 11.0
    # CHECK: 12.0
    # CHECK: 13.0
    # CHECK: 14.0
    # CHECK: 15.0
    # CHECK: 16.0
    for i in range(16):
        print(out_data.load(i))

    in_data.free()
    out_data.free()
    _ = in_device
    _ = out_device
    _ = copy_via_shared_gpu^


def main():
    with DeviceContext() as ctx:
        run_copy_via_shared(ctx)
