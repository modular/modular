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

from sys import sizeof

from gpu import thread_idx
from gpu.host import DeviceContext
from gpu.memory import (
    AddressSpace,
    Fill,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
)
from memory import UnsafePointer, stack_allocation
from testing import assert_equal


fn copy_via_shared(
    src: UnsafePointer[Float32],
    dst: UnsafePointer[Float32],
):
    var thread_id = Int(thread_idx.x)
    var mem_buff: UnsafePointer[
        Float32, address_space = AddressSpace.SHARED
    ] = stack_allocation[16, Float32, address_space = AddressSpace.SHARED]()
    var src_global: UnsafePointer[
        Float32, address_space = AddressSpace.GLOBAL
    ] = src.address_space_cast[AddressSpace.GLOBAL]()

    async_copy[4](
        src_global.offset(thread_id),
        mem_buff.offset(thread_id),
    )

    async_copy_commit_group()
    async_copy_wait_group(0)

    dst[thread_id] = mem_buff[thread_id]


# CHECK-LABEL: run_copy_via_shared
fn run_copy_via_shared(ctx: DeviceContext) raises:
    print("== run_copy_via_shared")
    var in_data = UnsafePointer[Float32].alloc(16)
    var out_data = UnsafePointer[Float32].alloc(16)
    var in_data_device = ctx.enqueue_create_buffer[DType.float32](16)
    var out_data_device = ctx.enqueue_create_buffer[DType.float32](16)

    for i in range(16):
        in_data[i] = i + 1
        out_data[i] = 0

    ctx.enqueue_copy(in_data_device, in_data)
    ctx.enqueue_copy(out_data_device, out_data)

    ctx.enqueue_function[copy_via_shared](
        in_data_device,
        out_data_device,
        grid_dim=(1,),
        block_dim=(16),
    )

    ctx.enqueue_copy(out_data, out_data_device)

    ctx.synchronize()

    for i in range(16):
        assert_equal(out_data[i], Float32(i + 1))

    _ = in_data_device
    _ = out_data_device
    in_data.free()
    out_data.free()


fn copy_with_src_size(
    src: UnsafePointer[Float32, address_space = AddressSpace.GLOBAL],
    dst: UnsafePointer[Float32, address_space = AddressSpace.GLOBAL],
    src_size: Int,
):
    var smem = stack_allocation[
        8, DType.float32, address_space = AddressSpace.SHARED
    ]()

    for i in range(8):
        smem[i] = -1.0

    # src[0: 4] are valid addresses, this copies `src_size` elements.
    async_copy[16, fill = Float32(0)](src, smem, src_size)
    # src[4: 8] are OOB, this should ignore src and set dst to zero.
    # See https://github.com/NVIDIA/cutlass/blob/5b283c872cae5f858ab682847181ca9d54d97377/include/cute/arch/copy_sm80.hpp#L101-L127.
    # Use `mojo build <this test>; compute-sanitizer <this test>` to verify there
    # is no OOB access.
    async_copy[16, fill = Float32(0)](src + 4, smem + 4, 0)
    async_copy_wait_all()

    for i in range(8):
        dst[i] = smem[i]


fn copy_with_non_zero_fill[
    smem_size: Int
](
    src: UnsafePointer[BFloat16, address_space = AddressSpace.GLOBAL],
    dst: UnsafePointer[BFloat16, address_space = AddressSpace.GLOBAL],
):
    var smem = stack_allocation[
        smem_size, DType.bfloat16, address_space = AddressSpace.SHARED
    ]()

    for i in range(smem_size):
        smem[i] = 0

    var offset = smem_size // 2

    async_copy[16, fill = BFloat16(32)](src, smem, predicate=True)

    async_copy[16, fill = BFloat16(32)](
        src + offset, smem + offset, predicate=False
    )
    async_copy_wait_all()

    for i in range(smem_size):
        dst[i] = smem[i]


fn test_copy_with_src_size(ctx: DeviceContext) raises:
    alias size = 4

    # Allocate arrays of different sizes to trigger an OOB address in test.
    var a_host = UnsafePointer[Float32].alloc(size)
    var b_host = UnsafePointer[Float32].alloc(2 * size)

    for i in range(size):
        a_host[i] = i + 1

    for i in range(2 * size):
        b_host[i] = i + 1

    var a_device = ctx.enqueue_create_buffer[DType.float32](size)
    var b_device = ctx.enqueue_create_buffer[DType.float32](2 * size)

    ctx.enqueue_copy(a_device, a_host)

    alias kernel = copy_with_src_size
    alias src_size = 3 * sizeof[DType.float32]()

    ctx.enqueue_function[kernel](
        a_device,
        b_device,
        src_size,
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )

    ctx.enqueue_copy(b_host, b_device)

    ctx.synchronize()

    assert_equal(b_host[0], 1)
    assert_equal(b_host[1], 2)
    assert_equal(b_host[2], 3)
    assert_equal(b_host[3], 0)
    assert_equal(b_host[4], 0)
    assert_equal(b_host[5], 0)
    assert_equal(b_host[6], 0)
    assert_equal(b_host[7], 0)

    _ = a_device
    _ = b_device
    a_host.free()
    b_host.free()


fn test_copy_with_non_zero_fill(ctx: DeviceContext) raises:
    alias size = 8

    # Allocate arrays of different sizes to trigger an OOB address in test.
    var a_host = UnsafePointer[BFloat16].alloc(size)
    var b_host = UnsafePointer[BFloat16].alloc(2 * size)

    for i in range(size):
        a_host[i] = i + 1

    for i in range(2 * size):
        b_host[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.bfloat16](size)
    var b_device = ctx.enqueue_create_buffer[DType.bfloat16](2 * size)

    ctx.enqueue_copy(a_device, a_host)

    alias kernel = copy_with_non_zero_fill[2 * size]

    alias src_size = 3 * sizeof[DType.bfloat16]()

    ctx.enqueue_function[kernel](
        a_device,
        b_device,
        src_size,
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )

    ctx.enqueue_copy(b_host, b_device)

    ctx.synchronize()

    assert_equal(b_host[0], 1)
    assert_equal(b_host[1], 2)
    assert_equal(b_host[2], 3)
    assert_equal(b_host[3], 4)
    assert_equal(b_host[4], 5)
    assert_equal(b_host[5], 6)
    assert_equal(b_host[6], 7)
    assert_equal(b_host[7], 8)
    assert_equal(b_host[8], 32)
    assert_equal(b_host[9], 32)
    assert_equal(b_host[10], 32)
    assert_equal(b_host[11], 32)
    assert_equal(b_host[12], 32)
    assert_equal(b_host[13], 32)
    assert_equal(b_host[14], 32)
    assert_equal(b_host[15], 32)

    _ = a_device
    _ = b_device
    a_host.free()
    b_host.free()


def main():
    with DeviceContext() as ctx:
        run_copy_via_shared(ctx)
        test_copy_with_src_size(ctx)
        test_copy_with_non_zero_fill(ctx)
