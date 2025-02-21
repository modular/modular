# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s

from memory import UnsafePointer, stack_allocation
from testing import assert_equal
from math import ceildiv
from gpu import barrier, thread_idx, block_idx, block_dim, grid_dim
from gpu.memory import AddressSpace
from gpu.host import DeviceContext
from gpu.intrinsics import (
    buffer_load,
    buffer_store,
    buffer_load_store_lds,
    _buffer_load_store_lds_nowait,
    _waitcnt,
)

alias size = 257
alias size_clip = size - 0


fn kernel[type: DType, width: Int](a: UnsafePointer[Scalar[type]]):
    var t0 = block_idx.x * block_dim.x + thread_idx.x
    var size2 = size // width
    for i in range(t0, size2, block_dim.x * grid_dim.x):
        var v = buffer_load[type, width](a, width * i, size_clip)
        buffer_store[type, width](a, width * i, size_clip, 2 * v)
    for i in range(width * size2, size, block_dim.x * grid_dim.x):
        var v = buffer_load[type, 1](a, i, size_clip)
        buffer_store[type, 1](a, i, size_clip, 2 * v)


fn kernel_lds[type: DType, nowait: Bool](a: UnsafePointer[Scalar[type]]):
    var a_shared = stack_allocation[
        size, type, address_space = AddressSpace.SHARED
    ]()

    var idx = thread_idx.x

    var t0 = block_idx.x * block_dim.x + thread_idx.x
    for i in range(t0, size, block_dim.x * grid_dim.x):
        a_shared[i] = 0
    barrier()

    for i in range(t0, size, block_dim.x * grid_dim.x):

        @parameter
        if nowait:
            _buffer_load_store_lds_nowait(a, i, a_shared, i, size_clip)
            _waitcnt()
        else:
            buffer_load_store_lds(a, i, a_shared, i, size_clip)
        a[i] = 2 * a_shared[i]


def test_buffer[type: DType, width: Int](ctx: DeviceContext):
    a_host_buf = UnsafePointer[Scalar[type]].alloc(size)
    a_device_buf = ctx.enqueue_create_buffer[type](size)

    for i in range(size):
        a_host_buf[i] = i + 1

    ctx.enqueue_copy_to_device(a_device_buf, a_host_buf)

    ctx.enqueue_function[kernel[type, width], dump_asm=False](
        a_device_buf,
        grid_dim=(1, 1),
        block_dim=(64),
    )
    ctx.enqueue_copy_from_device(a_host_buf, a_device_buf)

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

    ctx.enqueue_copy_to_device(a_device_buf, a_host_buf)

    ctx.enqueue_function[kernel_lds[type, nowait], dump_asm=False](
        a_device_buf,
        grid_dim=ceildiv(size, 256),
        block_dim=256,
    )
    ctx.enqueue_copy_from_device(a_host_buf, a_device_buf)

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
