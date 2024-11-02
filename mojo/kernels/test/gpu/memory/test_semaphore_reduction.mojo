# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import ceildiv, isclose
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceBuffer, DeviceContext
from gpu.semaphore import Semaphore
from memory import UnsafePointer
from testing import assert_almost_equal
from gpu import BlockDim, BlockIdx, ThreadIdx, GridDim
from gpu.intrinsics import threadfence
from utils.index import Index


fn semaphore_block_reduce[
    type: DType,
](
    c_ptr: UnsafePointer[Scalar[type]],
    a_ptr: UnsafePointer[Scalar[type]],
    locks: UnsafePointer[Int32],
    n: Int,
):
    var tid = ThreadIdx.x()
    var sema = Semaphore(locks.offset(BlockIdx.x()), tid)
    sema.fetch()
    sema.wait(0)
    c_ptr[tid] += a_ptr[tid]
    threadfence()
    sema.release(1)


fn run_block_reduction[
    type: DType,
    N: Int,
](ctx: DeviceContext,) raises:
    print("== run_semaphore reduction kernel => ", str(type), N)

    var a_host = UnsafePointer[Scalar[type]].alloc(N)
    var c_host = UnsafePointer[Scalar[type]].alloc(N)

    var init_a = Float32(1)
    var init_c = Float32(0)
    for i in range(N):
        a_host[i] = init_a.cast[type]()
        c_host[i] = init_c.cast[type]()

    alias shape = DimList(N)
    alias num_parts = 16

    var a_device = ctx.create_buffer[type](N)
    var c_device = ctx.create_buffer[type](N)
    var lock_dev = ctx.create_buffer[type](num_parts)
    var a_buf = NDBuffer[type, 1, shape](a_device.ptr, Index(N))
    var c_buf = NDBuffer[type, 1, shape](c_device.ptr, Index(N))

    ctx.enqueue_copy_to_device(a_device, a_host)

    var func_red = ctx.compile_function[
        semaphore_block_reduce[type,],
        dump_ptx=False,
    ]()

    @always_inline
    @parameter
    fn run_func_red() raises:
        ctx.enqueue_function(
            func_red,
            c_buf.data,
            a_buf.data,
            lock_dev.ptr,
            N,
            grid_dim=num_parts,
            block_dim=N,
        )

    run_func_red()

    ctx.enqueue_copy_from_device(c_host, c_device)
    ctx.synchronize()

    print("counter[0] value => ", c_host[0])

    _ = a_device
    _ = c_device
    _ = lock_dev

    a_host.free()
    c_host.free()


def main():
    with DeviceContext() as ctx:
        run_block_reduction[DType.float32, 32](ctx)
