# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: GPU-H100
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.host import DeviceContext
from gpu.id import ThreadIdx
from gpu.mma import mma
from gpu.host._compile import _get_nvptx_target

from builtin.io import _printf


fn mma_sync_16x8x32_E4M3():
    a = SIMD[DType.float8e4m3, 16](1.0)
    b = SIMD[DType.float8e4m3, 8](2.0)
    c = SIMD[DType.float32, 4](0.0)
    d = SIMD[DType.float32, 4](0.0)
    mma(d, a, b, c)

    _printf["thread %d : %g %g %g %g\n"](
        ThreadIdx.x(),
        d[0].cast[DType.float64](),
        d[1].cast[DType.float64](),
        d[2].cast[DType.float64](),
        d[3].cast[DType.float64](),
    )


def test_mma_sync_16x8x32_E4M3(ctx: DeviceContext):
    print("== test_mma_sync_16x8x32_E4M3")
    var func = ctx.compile_function[
        mma_sync_16x8x32_E4M3, target = _get_nvptx_target["sm_90"]()
    ]()
    ctx.enqueue_function(
        func,
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()


fn mma_sync_16x8x32_E4M2():
    a = SIMD[DType.float8e5m2, 16](2.0)
    b = SIMD[DType.float8e5m2, 8](3.0)
    c = SIMD[DType.float32, 4](0.0)
    d = SIMD[DType.float32, 4](0.0)
    mma(d, a, b, c)

    _printf["thread %d : %g %g %g %g\n"](
        ThreadIdx.x(),
        d[0].cast[DType.float64](),
        d[1].cast[DType.float64](),
        d[2].cast[DType.float64](),
        d[3].cast[DType.float64](),
    )


def test_mma_sync_16x8x32_E5M2(ctx: DeviceContext):
    print("== test_mma_sync_16x8x32_E5M2")
    var func = ctx.compile_function[
        mma_sync_16x8x32_E4M2, target = _get_nvptx_target["sm_90"]()
    ]()
    ctx.enqueue_function(
        func,
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()


def main():
    with DeviceContext() as ctx:
        # CHECK-LABEL: test_mma_sync_16x8x32_E4M3
        # CHECK-DAG: thread 0 : 16 16 16 16
        # CHECK-DAG: thread 1 : 16 16 16 16
        # CHECK-DAG: thread 2 : 16 16 16 16
        # CHECK-DAG: thread 3 : 16 16 16 16
        # CHECK-DAG: thread 4 : 16 16 16 16
        # CHECK-DAG: thread 5 : 16 16 16 16
        # CHECK-DAG: thread 6 : 16 16 16 16
        # CHECK-DAG: thread 7 : 16 16 16 16
        # CHECK-DAG: thread 8 : 16 16 16 16
        # CHECK-DAG: thread 9 : 16 16 16 16
        # CHECK-DAG: thread 10 : 16 16 16 16
        # CHECK-DAG: thread 11 : 16 16 16 16
        # CHECK-DAG: thread 12 : 16 16 16 16
        # CHECK-DAG: thread 13 : 16 16 16 16
        # CHECK-DAG: thread 14 : 16 16 16 16
        # CHECK-DAG: thread 15 : 16 16 16 16
        # CHECK-DAG: thread 16 : 16 16 16 16
        # CHECK-DAG: thread 17 : 16 16 16 16
        # CHECK-DAG: thread 18 : 16 16 16 16
        # CHECK-DAG: thread 19 : 16 16 16 16
        # CHECK-DAG: thread 20 : 16 16 16 16
        # CHECK-DAG: thread 21 : 16 16 16 16
        # CHECK-DAG: thread 22 : 16 16 16 16
        # CHECK-DAG: thread 23 : 16 16 16 16
        # CHECK-DAG: thread 24 : 16 16 16 16
        # CHECK-DAG: thread 25 : 16 16 16 16
        # CHECK-DAG: thread 26 : 16 16 16 16
        # CHECK-DAG: thread 27 : 16 16 16 16
        # CHECK-DAG: thread 28 : 16 16 16 16
        # CHECK-DAG: thread 29 : 16 16 16 16
        # CHECK-DAG: thread 30 : 16 16 16 16
        # CHECK-DAG: thread 31 : 16 16 16 16
        test_mma_sync_16x8x32_E4M3(ctx)
        # CHECK-LABEL: test_mma_sync_16x8x32_E5M2
        # CHECK-DAG: thread 0 : 48 48 48 48
        # CHECK-DAG: thread 1 : 48 48 48 48
        # CHECK-DAG: thread 2 : 48 48 48 48
        # CHECK-DAG: thread 3 : 48 48 48 48
        # CHECK-DAG: thread 4 : 48 48 48 48
        # CHECK-DAG: thread 5 : 48 48 48 48
        # CHECK-DAG: thread 6 : 48 48 48 48
        # CHECK-DAG: thread 7 : 48 48 48 48
        # CHECK-DAG: thread 8 : 48 48 48 48
        # CHECK-DAG: thread 9 : 48 48 48 48
        # CHECK-DAG: thread 10 : 48 48 48 48
        # CHECK-DAG: thread 11 : 48 48 48 48
        # CHECK-DAG: thread 12 : 48 48 48 48
        # CHECK-DAG: thread 13 : 48 48 48 48
        # CHECK-DAG: thread 14 : 48 48 48 48
        # CHECK-DAG: thread 15 : 48 48 48 48
        # CHECK-DAG: thread 16 : 48 48 48 48
        # CHECK-DAG: thread 17 : 48 48 48 48
        # CHECK-DAG: thread 18 : 48 48 48 48
        # CHECK-DAG: thread 19 : 48 48 48 48
        # CHECK-DAG: thread 20 : 48 48 48 48
        # CHECK-DAG: thread 21 : 48 48 48 48
        # CHECK-DAG: thread 22 : 48 48 48 48
        # CHECK-DAG: thread 23 : 48 48 48 48
        # CHECK-DAG: thread 24 : 48 48 48 48
        # CHECK-DAG: thread 25 : 48 48 48 48
        # CHECK-DAG: thread 26 : 48 48 48 48
        # CHECK-DAG: thread 27 : 48 48 48 48
        # CHECK-DAG: thread 28 : 48 48 48 48
        # CHECK-DAG: thread 29 : 48 48 48 48
        # CHECK-DAG: thread 30 : 48 48 48 48
        # CHECK-DAG: thread 31 : 48 48 48 48
        test_mma_sync_16x8x32_E5M2(ctx)
