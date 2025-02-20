# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s

from math import ceildiv
from os.atomic import Atomic

from buffer import DimList, NDBuffer
from gpu import *
from gpu.host import DeviceContext
from memory import UnsafePointer
from testing import assert_equal


@value
struct FillStrategy:
    var value: Int

    alias LINSPACE = Self(0)
    alias NEG_LINSPACE = Self(1)
    alias SYMMETRIC_LINSPACE = Self(2)
    alias ZEROS = Self(3)
    alias ONES = Self(4)

    fn __is__(self, other: Self) -> Bool:
        return self.value == other.value


fn reduce(
    res_add: UnsafePointer[Float32],
    res_min: UnsafePointer[Float32],
    res_max: UnsafePointer[Float32],
    vec: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x

    if tid >= len:
        return

    _ = Atomic._fetch_add(res_add, vec[tid])

    Atomic.min(res_min, vec[tid])
    Atomic.max(res_max, vec[tid])


fn run_reduce(fill_strategy: FillStrategy, ctx: DeviceContext) raises:
    alias BLOCK_SIZE = 32
    alias n = 1024

    var vec_host = NDBuffer[DType.float32, 1, DimList(n)].stack_allocation()

    if fill_strategy is FillStrategy.LINSPACE:
        for i in range(n):
            vec_host[i] = i
    elif fill_strategy is FillStrategy.NEG_LINSPACE:
        for i in range(n):
            vec_host[i] = -i
    elif fill_strategy is FillStrategy.SYMMETRIC_LINSPACE:
        for i in range(n):
            vec_host[i] = i - n // 2
    elif fill_strategy is FillStrategy.ZEROS:
        for i in range(n):
            vec_host[i] = 0
    elif fill_strategy is FillStrategy.ONES:
        for i in range(n):
            vec_host[i] = 1

    var vec_device = ctx.enqueue_create_buffer[DType.float32](n)
    var res_add_device = ctx.enqueue_create_buffer[DType.float32](1)

    ctx.enqueue_copy_to_device(vec_device, vec_host.data)
    ctx.memset(res_add_device, 0)

    var res_min_device = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.memset(res_min_device, 0)

    var res_max_device = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.memset(res_max_device, 0)

    ctx.enqueue_function[reduce](
        res_add_device,
        res_min_device,
        res_max_device,
        vec_device,
        n,
        grid_dim=ceildiv(n, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )

    var res = Float32(0)
    ctx.enqueue_copy_from_device(UnsafePointer.address_of(res), res_add_device)

    var res_min = Float32(0)
    ctx.enqueue_copy_from_device(
        UnsafePointer.address_of(res_min), res_min_device
    )

    var res_max = Float32(0)
    ctx.enqueue_copy_from_device(
        UnsafePointer.address_of(res_max), res_max_device
    )

    ctx.synchronize()

    if fill_strategy is FillStrategy.LINSPACE:
        assert_equal(res, n * (n - 1) // 2)
        assert_equal(res_min, 0)
        assert_equal(res_max, n - 1)
    elif fill_strategy is FillStrategy.NEG_LINSPACE:
        assert_equal(res, -n * (n - 1) // 2)
        assert_equal(res_min, -n + 1)
        assert_equal(res_max, 0)
    elif fill_strategy is FillStrategy.SYMMETRIC_LINSPACE:
        assert_equal(res, -n // 2)
        assert_equal(res_min, -n // 2)
        assert_equal(res_max, (n - 1) // 2)
    elif fill_strategy is FillStrategy.ZEROS:
        assert_equal(res, 0)
        assert_equal(res_min, 0)
        assert_equal(res_max, 0)
    elif fill_strategy is FillStrategy.ONES:
        assert_equal(res, n)
        assert_equal(res_min, 0)
        assert_equal(res_max, 1)

    _ = vec_device


def main():
    with DeviceContext() as ctx:
        run_reduce(FillStrategy.LINSPACE, ctx)
        run_reduce(FillStrategy.NEG_LINSPACE, ctx)
        run_reduce(FillStrategy.SYMMETRIC_LINSPACE, ctx)
        run_reduce(FillStrategy.ZEROS, ctx)
        run_reduce(FillStrategy.ONES, ctx)
