# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(KERN-1652): Reenable this test once the issue is fixed.
# REQUIRES: DISABLED
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug %s | FileCheck %s

from builtin.io import _printf
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.id import thread_idx
from gpu.mma import mma


fn mma_sync_16x8x32_E4M3():
    a = SIMD[DType.float8_e4m3fn, 16](1.0)
    b = SIMD[DType.float8_e4m3fn, 8](2.0)
    c = SIMD[DType.float32, 4](0.0)
    d = SIMD[DType.float32, 4](0.0)
    mma(d, a, b, c)

    _printf["thread %d : %g %g %g %g\n"](
        thread_idx.x,
        d[0].cast[DType.float64](),
        d[1].cast[DType.float64](),
        d[2].cast[DType.float64](),
        d[3].cast[DType.float64](),
    )


def test_mma_sync_16x8x32_E4M3(ctx: DeviceContext):
    print("== test_mma_sync_16x8x32_E4M3")
    ctx.enqueue_function[mma_sync_16x8x32_E4M3](
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()


fn mma_sync_16x8x32_E4M2():
    a = SIMD[DType.float8_e5m2, 16](2.0)
    b = SIMD[DType.float8_e5m2, 8](3.0)
    c = SIMD[DType.float32, 4](0.0)
    d = SIMD[DType.float32, 4](0.0)
    mma(d, a, b, c)

    _printf["thread %d : %g %g %g %g\n"](
        thread_idx.x,
        d[0].cast[DType.float64](),
        d[1].cast[DType.float64](),
        d[2].cast[DType.float64](),
        d[3].cast[DType.float64](),
    )


def test_mma_sync_16x8x32_E5M2(ctx: DeviceContext):
    print("== test_mma_sync_16x8x32_E5M2")
    ctx.enqueue_function[mma_sync_16x8x32_E4M2](
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()


def main():
    with DeviceContext() as ctx:
        # CHECK-LABEL: test_mma_sync_16x8x32_E4M3
        # CHECK-DAG: thread 0 : 64 64 64 64
        # CHECK-DAG: thread 1 : 64 64 64 64
        # CHECK-DAG: thread 2 : 64 64 64 64
        # CHECK-DAG: thread 3 : 64 64 64 64
        # CHECK-DAG: thread 4 : 64 64 64 64
        # CHECK-DAG: thread 5 : 64 64 64 64
        # CHECK-DAG: thread 6 : 64 64 64 64
        # CHECK-DAG: thread 7 : 64 64 64 64
        # CHECK-DAG: thread 8 : 64 64 64 64
        # CHECK-DAG: thread 9 : 64 64 64 64
        # CHECK-DAG: thread 10 : 64 64 64 64
        # CHECK-DAG: thread 11 : 64 64 64 64
        # CHECK-DAG: thread 12 : 64 64 64 64
        # CHECK-DAG: thread 13 : 64 64 64 64
        # CHECK-DAG: thread 14 : 64 64 64 64
        # CHECK-DAG: thread 15 : 64 64 64 64
        # CHECK-DAG: thread 16 : 64 64 64 64
        # CHECK-DAG: thread 17 : 64 64 64 64
        # CHECK-DAG: thread 18 : 64 64 64 64
        # CHECK-DAG: thread 19 : 64 64 64 64
        # CHECK-DAG: thread 20 : 64 64 64 64
        # CHECK-DAG: thread 21 : 64 64 64 64
        # CHECK-DAG: thread 22 : 64 64 64 64
        # CHECK-DAG: thread 23 : 64 64 64 64
        # CHECK-DAG: thread 24 : 64 64 64 64
        # CHECK-DAG: thread 25 : 64 64 64 64
        # CHECK-DAG: thread 26 : 64 64 64 64
        # CHECK-DAG: thread 27 : 64 64 64 64
        # CHECK-DAG: thread 28 : 64 64 64 64
        # CHECK-DAG: thread 29 : 64 64 64 64
        # CHECK-DAG: thread 30 : 64 64 64 64
        # CHECK-DAG: thread 31 : 64 64 64 64
        test_mma_sync_16x8x32_E4M3(ctx)
        # CHECK-LABEL: test_mma_sync_16x8x32_E5M2
        # CHECK-DAG: thread 0 : 192 192 192 192
        # CHECK-DAG: thread 1 : 192 192 192 192
        # CHECK-DAG: thread 2 : 192 192 192 192
        # CHECK-DAG: thread 3 : 192 192 192 192
        # CHECK-DAG: thread 4 : 192 192 192 192
        # CHECK-DAG: thread 5 : 192 192 192 192
        # CHECK-DAG: thread 6 : 192 192 192 192
        # CHECK-DAG: thread 7 : 192 192 192 192
        # CHECK-DAG: thread 8 : 192 192 192 192
        # CHECK-DAG: thread 9 : 192 192 192 192
        # CHECK-DAG: thread 10 : 192 192 192 192
        # CHECK-DAG: thread 11 : 192 192 192 192
        # CHECK-DAG: thread 12 : 192 192 192 192
        # CHECK-DAG: thread 13 : 192 192 192 192
        # CHECK-DAG: thread 14 : 192 192 192 192
        # CHECK-DAG: thread 15 : 192 192 192 192
        # CHECK-DAG: thread 16 : 192 192 192 192
        # CHECK-DAG: thread 17 : 192 192 192 192
        # CHECK-DAG: thread 18 : 192 192 192 192
        # CHECK-DAG: thread 19 : 192 192 192 192
        # CHECK-DAG: thread 20 : 192 192 192 192
        # CHECK-DAG: thread 21 : 192 192 192 192
        # CHECK-DAG: thread 22 : 192 192 192 192
        # CHECK-DAG: thread 23 : 192 192 192 192
        # CHECK-DAG: thread 24 : 192 192 192 192
        # CHECK-DAG: thread 25 : 192 192 192 192
        # CHECK-DAG: thread 26 : 192 192 192 192
        # CHECK-DAG: thread 27 : 192 192 192 192
        # CHECK-DAG: thread 28 : 192 192 192 192
        # CHECK-DAG: thread 29 : 192 192 192 192
        # CHECK-DAG: thread 30 : 192 192 192 192
        # CHECK-DAG: thread 31 : 192 192 192 192
        test_mma_sync_16x8x32_E5M2(ctx)
