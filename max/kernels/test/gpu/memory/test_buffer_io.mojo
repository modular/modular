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

from math import ceildiv

from gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from gpu.host import DeviceContext
from gpu.intrinsics import (
    _buffer_load_store_lds_nowait,
    _waitcnt,
    buffer_load,
    buffer_load_store_lds,
    buffer_store,
    make_buffer_resource,
)
from gpu.memory import AddressSpace
from memory import stack_allocation
from testing import assert_equal

alias size = 257
alias size_clip = size - 3


fn kernel[type: DType, width: Int](a: UnsafePointer[Scalar[type]]):
    var t0 = block_idx.x * block_dim.x + thread_idx.x
    var size2 = size // width
    var bc = make_buffer_resource(a, size_clip)
    for i in range(t0, size2, block_dim.x * grid_dim.x):
        var v = buffer_load[type, width](bc, width * i)
        buffer_store[type, width](bc, width * i, 2 * v)
    for i in range(width * size2, size, block_dim.x * grid_dim.x):
        var v = buffer_load[type, 1](bc, i)
        buffer_store[type, 1](bc, i, 2 * v)


fn kernel_lds[type: DType, nowait: Bool](a: UnsafePointer[Scalar[type]]):
    var a_shared = stack_allocation[
        size, type, address_space = AddressSpace.SHARED
    ]()

    var idx = thread_idx.x

    var t0 = block_idx.x * block_dim.x + thread_idx.x
    var bc = make_buffer_resource(a, size_clip)
    for i in range(t0, size, block_dim.x * grid_dim.x):
        a_shared[i] = 0
    barrier()

    @parameter
    if nowait:
        for i in range(t0, size, block_dim.x * grid_dim.x):
            _buffer_load_store_lds_nowait[type](bc, i, a_shared, i)
        _waitcnt()
        for i in range(t0, size, block_dim.x * grid_dim.x):
            a[i] = 2 * a_shared[i]
    else:
        for i in range(t0, size, block_dim.x * grid_dim.x):
            buffer_load_store_lds[type](bc, i, a_shared, i)
            a[i] = 2 * a_shared[i]


def test_buffer[type: DType, width: Int](ctx: DeviceContext):
    a_host_buf = UnsafePointer[Scalar[type]].alloc(size)
    a_device_buf = ctx.enqueue_create_buffer[type](size)

    for i in range(size):
        a_host_buf[i] = i + 1

    ctx.enqueue_copy(a_device_buf, a_host_buf)

    ctx.enqueue_function[kernel[type, width], dump_asm=False](
        a_device_buf,
        grid_dim=(1, 1),
        block_dim=(64),
    )
    ctx.enqueue_copy(a_host_buf, a_device_buf)

    ctx.synchronize()
    for i in range(size_clip):
        assert_equal(a_host_buf[i], 2 * (i + 1))
    for i in range(size_clip, size):
        assert_equal(a_host_buf[i], i + 1)


def test_buffer_lds[nowait: Bool](ctx: DeviceContext):
    alias type = DType.float32
    a_host_buf = UnsafePointer[Scalar[type]].alloc(size)
    a_device_buf = ctx.enqueue_create_buffer[type](size)

    for i in range(size):
        a_host_buf[i] = i + 1

    ctx.enqueue_copy(a_device_buf, a_host_buf)

    ctx.enqueue_function[kernel_lds[type, nowait], dump_asm=False](
        a_device_buf,
        grid_dim=ceildiv(size, 256),
        block_dim=256,
    )
    ctx.enqueue_copy(a_host_buf, a_device_buf)

    ctx.synchronize()
    for i in range(size_clip):
        assert_equal(a_host_buf[i], 2 * (i + 1))
    for i in range(size_clip, size):
        assert_equal(a_host_buf[i], 0)


def main():
    with DeviceContext() as ctx:
        test_buffer[DType.bfloat16, 1](ctx)
        test_buffer[DType.bfloat16, 2](ctx)
        test_buffer[DType.bfloat16, 4](ctx)
        test_buffer[DType.bfloat16, 8](ctx)
        test_buffer_lds[nowait=False](ctx)
        test_buffer_lds[nowait=True](ctx)
