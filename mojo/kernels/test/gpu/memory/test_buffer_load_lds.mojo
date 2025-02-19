# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s

from memory import UnsafePointer, stack_allocation
from testing import assert_equal
from gpu import barrier, thread_idx
from gpu.memory import AddressSpace
from gpu.host import DeviceContext
from gpu.intrinsics import (
    buffer_load_lds,
    _buffer_load_lds_nowait,
    _wait_cnt_amd,
)


alias type = DType.float32
alias size = 256
# ignore the last two values
alias size_clip = 256 - 3


fn kernel[nowait: Bool](ptr: UnsafePointer[Scalar[type]]):
    var a_shared = stack_allocation[
        size, type, address_space = AddressSpace.SHARED
    ]()

    var idx = thread_idx.x

    for i in range(4):
        a_shared[64 * i + idx] = -1
    barrier()

    @parameter
    if nowait:
        for i in range(4):
            _buffer_load_lds_nowait(
                ptr, 64 * i + idx, a_shared, 64 * i + idx, size_clip
            )
        _wait_cnt_amd()
    else:
        for i in range(4):
            buffer_load_lds(
                ptr, 64 * i + idx, a_shared, 64 * i + idx, size_clip
            )
    for i in range(4):
        ptr[64 * i + idx] = 2 * a_shared[64 * i + idx]


def test_buffer_load_lds[nowait: Bool](ctx: DeviceContext):
    a_host_buf = UnsafePointer[Scalar[type]].alloc(size)
    a_device_buf = ctx.enqueue_create_buffer[type](size)

    for i in range(size):
        a_host_buf[i] = i + 1

    ctx.enqueue_copy_to_device(a_device_buf, a_host_buf)

    ctx.enqueue_function[kernel[nowait], dump_asm=False](
        a_device_buf,
        grid_dim=(1, 1),
        block_dim=(64),
    )
    ctx.enqueue_copy_from_device(a_host_buf, a_device_buf)

    ctx.synchronize()
    for i in range(size_clip):
        assert_equal(a_host_buf[i], 2 * (i + 1))
    for i in range(size_clip, size):
        assert_equal(a_host_buf[i], 0)


def main():
    with DeviceContext() as ctx:
        test_buffer_load_lds[nowait=False](ctx)
        test_buffer_load_lds[nowait=True](ctx)
