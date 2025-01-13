# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# REQUIRES: DISABLED
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from builtin.io import _printf
from gpu import barrier, lane_id
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.id import thread_idx
from gpu.intrinsics import threadfence
from gpu.memory import AddressSpace
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import Layout, LayoutTensor
from layout.fillers import arange

from utils.index import Index


fn wgmma_tf32_tf32_f32_fill_kernel[M: Int, N: Int, K: Int]():
    var smem_operand_a = LayoutTensor[
        DType.float32,
        Layout.row_major(M, K),
        address_space = AddressSpace.SHARED,
    ].stack_allocation().fill(1)

    var smem_operand_b = LayoutTensor[
        DType.float32,
        Layout.row_major(N, K),
        address_space = AddressSpace.SHARED,
    ].stack_allocation().fill(1)

    if thread_idx.x == 0:
        _ = smem_operand_a.fill(2)
        _ = smem_operand_b.fill(3)

    barrier()

    var mat_a_desc = WGMMADescriptor.create[8, 64](smem_operand_a.ptr)
    var mat_b_desc = WGMMADescriptor.create[8, 1](smem_operand_b.ptr)

    wgmma_fence_aligned()
    var c_reg = SIMD[DType.float32, 4](0)
    c_reg = wgmma_async[
        M, N, K, a_type = DType.tensor_float32, b_type = DType.tensor_float32
    ](mat_a_desc, mat_b_desc, c_reg)
    wgmma_commit_group_sync()
    wgmma_wait_group_sync()
    threadfence()
    wgmma_fence_aligned()
    res = c_reg.cast[DType.float64]()

    _printf["%lu : %g %g %g %g\n"](thread_idx.x, res[0], res[1], res[2], res[3])


# CHECK-LABEL: test_wgmma_64x64x8_f32_f32_f32_fill
# CHECK-DAG: 0 : 48 48 48 48
# CHECK-DAG: 1 : 48 48 48 48
# CHECK-DAG: 2 : 48 48 48 48
# CHECK-DAG: 3 : 48 48 48 48
# CHECK-DAG: 4 : 48 48 48 48
# CHECK-DAG: 5 : 48 48 48 48
# CHECK-DAG: 6 : 48 48 48 48
# CHECK-DAG: 7 : 48 48 48 48
# CHECK-DAG: 8 : 48 48 48 48
# CHECK-DAG: 9 : 48 48 48 48
# CHECK-DAG: 10 : 48 48 48 48
# CHECK-DAG: 11 : 48 48 48 48
# CHECK-DAG: 12 : 48 48 48 48
# CHECK-DAG: 13 : 48 48 48 48
# CHECK-DAG: 14 : 48 48 48 48
# CHECK-DAG: 15 : 48 48 48 48
# CHECK-DAG: 16 : 48 48 48 48
# CHECK-DAG: 17 : 48 48 48 48
# CHECK-DAG: 18 : 48 48 48 48
# CHECK-DAG: 19 : 48 48 48 48
# CHECK-DAG: 20 : 48 48 48 48
# CHECK-DAG: 21 : 48 48 48 48
# CHECK-DAG: 22 : 48 48 48 48
# CHECK-DAG: 23 : 48 48 48 48
# CHECK-DAG: 24 : 48 48 48 48
# CHECK-DAG: 25 : 48 48 48 48
# CHECK-DAG: 26 : 48 48 48 48
# CHECK-DAG: 27 : 48 48 48 48
# CHECK-DAG: 28 : 48 48 48 48
# CHECK-DAG: 29 : 48 48 48 48
# CHECK-DAG: 30 : 48 48 48 48
# CHECK-DAG: 31 : 48 48 48 48
# CHECK-DAG: 32 : 48 48 48 48
# CHECK-DAG: 33 : 48 48 48 48
# CHECK-DAG: 34 : 48 48 48 48
# CHECK-DAG: 35 : 48 48 48 48
# CHECK-DAG: 36 : 48 48 48 48
# CHECK-DAG: 37 : 48 48 48 48
# CHECK-DAG: 38 : 48 48 48 48
# CHECK-DAG: 39 : 48 48 48 48
# CHECK-DAG: 40 : 48 48 48 48
# CHECK-DAG: 41 : 48 48 48 48
# CHECK-DAG: 42 : 48 48 48 48
# CHECK-DAG: 43 : 48 48 48 48
# CHECK-DAG: 44 : 48 48 48 48
# CHECK-DAG: 45 : 48 48 48 48
# CHECK-DAG: 46 : 48 48 48 48
# CHECK-DAG: 47 : 48 48 48 48
# CHECK-DAG: 48 : 48 48 48 48
# CHECK-DAG: 49 : 48 48 48 48
# CHECK-DAG: 50 : 48 48 48 48
# CHECK-DAG: 51 : 48 48 48 48
# CHECK-DAG: 52 : 48 48 48 48
# CHECK-DAG: 53 : 48 48 48 48
# CHECK-DAG: 54 : 48 48 48 48
# CHECK-DAG: 55 : 48 48 48 48
# CHECK-DAG: 56 : 48 48 48 48
# CHECK-DAG: 57 : 48 48 48 48
# CHECK-DAG: 58 : 48 48 48 48
# CHECK-DAG: 59 : 48 48 48 48
# CHECK-DAG: 60 : 48 48 48 48
# CHECK-DAG: 61 : 48 48 48 48
# CHECK-DAG: 62 : 48 48 48 48
# CHECK-DAG: 63 : 48 48 48 48
# CHECK-DAG: 64 : 48 48 48 48
# CHECK-DAG: 65 : 48 48 48 48
# CHECK-DAG: 66 : 48 48 48 48
# CHECK-DAG: 67 : 48 48 48 48
# CHECK-DAG: 68 : 48 48 48 48
# CHECK-DAG: 69 : 48 48 48 48
# CHECK-DAG: 70 : 48 48 48 48
# CHECK-DAG: 71 : 48 48 48 48
# CHECK-DAG: 72 : 48 48 48 48
# CHECK-DAG: 73 : 48 48 48 48
# CHECK-DAG: 74 : 48 48 48 48
# CHECK-DAG: 75 : 48 48 48 48
# CHECK-DAG: 76 : 48 48 48 48
# CHECK-DAG: 77 : 48 48 48 48
# CHECK-DAG: 78 : 48 48 48 48
# CHECK-DAG: 79 : 48 48 48 48
# CHECK-DAG: 80 : 48 48 48 48
# CHECK-DAG: 81 : 48 48 48 48
# CHECK-DAG: 82 : 48 48 48 48
# CHECK-DAG: 83 : 48 48 48 48
# CHECK-DAG: 84 : 48 48 48 48
# CHECK-DAG: 85 : 48 48 48 48
# CHECK-DAG: 86 : 48 48 48 48
# CHECK-DAG: 87 : 48 48 48 48
# CHECK-DAG: 88 : 48 48 48 48
# CHECK-DAG: 89 : 48 48 48 48
# CHECK-DAG: 90 : 48 48 48 48
# CHECK-DAG: 91 : 48 48 48 48
# CHECK-DAG: 92 : 48 48 48 48
# CHECK-DAG: 93 : 48 48 48 48
# CHECK-DAG: 94 : 48 48 48 48
# CHECK-DAG: 95 : 48 48 48 48
# CHECK-DAG: 96 : 48 48 48 48
# CHECK-DAG: 97 : 48 48 48 48
# CHECK-DAG: 98 : 48 48 48 48
# CHECK-DAG: 99 : 48 48 48 48
# CHECK-DAG: 100 : 48 48 48 48
# CHECK-DAG: 101 : 48 48 48 48
# CHECK-DAG: 102 : 48 48 48 48
# CHECK-DAG: 103 : 48 48 48 48
# CHECK-DAG: 104 : 48 48 48 48
# CHECK-DAG: 105 : 48 48 48 48
# CHECK-DAG: 106 : 48 48 48 48
# CHECK-DAG: 107 : 48 48 48 48
# CHECK-DAG: 108 : 48 48 48 48
# CHECK-DAG: 109 : 48 48 48 48
# CHECK-DAG: 110 : 48 48 48 48
# CHECK-DAG: 111 : 48 48 48 48
# CHECK-DAG: 112 : 48 48 48 48
# CHECK-DAG: 113 : 48 48 48 48
# CHECK-DAG: 114 : 48 48 48 48
# CHECK-DAG: 115 : 48 48 48 48
# CHECK-DAG: 116 : 48 48 48 48
# CHECK-DAG: 117 : 48 48 48 48
# CHECK-DAG: 118 : 48 48 48 48
# CHECK-DAG: 119 : 48 48 48 48
# CHECK-DAG: 120 : 48 48 48 48
# CHECK-DAG: 121 : 48 48 48 48
# CHECK-DAG: 122 : 48 48 48 48
# CHECK-DAG: 123 : 48 48 48 48
# CHECK-DAG: 124 : 48 48 48 48
# CHECK-DAG: 125 : 48 48 48 48
# CHECK-DAG: 126 : 48 48 48 48
# CHECK-DAG: 127 : 48 48 48 48
def test_wgmma_64x64x8_f32_f32_f32_fill(ctx: DeviceContext):
    print("== test_wgmma_64x64x8_f32_f32_f32_fill")
    alias M = 64
    alias N = 8
    alias K = 8

    var func = ctx.compile_function[
        wgmma_tf32_tf32_f32_fill_kernel[M, N, K],
        _target = _get_gpu_target["sm_90"](),
    ]()
    ctx.enqueue_function(
        func,
        grid_dim=(1),
        block_dim=(32 * 4),
    )
    ctx.synchronize()

    _ = func^


def main():
    with DeviceContext() as ctx:
        test_wgmma_64x64x8_f32_f32_f32_fill(ctx)
