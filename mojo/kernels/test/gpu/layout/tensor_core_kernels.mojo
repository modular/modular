# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from builtin.io import _printf

from layout import LayoutTensor, Layout
from layout.tensor_core import TensorCore
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc

from gpu.host import DeviceContext
from gpu.id import ThreadIdx

from utils.index import StaticIntTuple, Index


fn mma_load_and_print_operands_kenrel[
    dst_dtype: DType,
    dtype: DType,
    lhs_layout: Layout,
    rhs_layout: Layout,
    inst_shape: StaticIntTuple[3],
](lhs: LayoutTensor[dtype, lhs_layout], rhs: LayoutTensor[dtype, rhs_layout]):
    var mma = TensorCore[dst_dtype, dtype, inst_shape]()
    var a_frags = mma.load_a(lhs)
    var b_frags = mma.load_a(rhs)
    _printf["thread %u a_vals=[%g %g %g %g], b_vals=[%g %g]\n"](
        ThreadIdx.x(),
        a_frags[0].cast[DType.float64](),
        a_frags[1].cast[DType.float64](),
        a_frags[2].cast[DType.float64](),
        a_frags[3].cast[DType.float64](),
        b_frags[0].cast[DType.float64](),
        b_frags[1].cast[DType.float64](),
    )


def test_load_operands[
    dst_dtype: DType, dtype: DType, shape: StaticIntTuple[3]
](ctx: DeviceContext):
    print("== test_load_f32_16x8x8")
    alias M = shape[0]
    alias N = shape[1]
    alias K = shape[2]

    var lhs = ManagedLayoutTensor[
        dst_dtype, Layout.row_major(M, K), gpu_managed_alloc, gpu_free
    ]()
    _ = lhs.tensor.linspace()
    var rhs = ManagedLayoutTensor[
        dst_dtype, Layout.row_major(K, N), gpu_managed_alloc, gpu_free
    ]()
    _ = rhs.tensor.linspace()
    alias mma_load_and_print_kenrel_fn = mma_load_and_print_operands_kenrel[
        dst_dtype, dtype, lhs.layout, rhs.layout, shape
    ]
    var func = ctx.compile_function[mma_load_and_print_kenrel_fn]()
    ctx.enqueue_function(
        func, lhs.tensor, rhs.tensor, grid_dim=(1, 1), block_dim=(32)
    )
    ctx.synchronize()


# CHECK-LABEL: test_load_f32_16x8x8
# CHECK-DAG: thread 0 a_vals=[0 64 4 68], b_vals=[0 1]
# CHECK-DAG: thread 1 a_vals=[1 65 5 69], b_vals=[2 3]
# CHECK-DAG: thread 2 a_vals=[2 66 6 70], b_vals=[4 5]
# CHECK-DAG: thread 3 a_vals=[3 67 7 71], b_vals=[6 7]
# CHECK-DAG: thread 4 a_vals=[8 72 12 76], b_vals=[8 9]
# CHECK-DAG: thread 5 a_vals=[9 73 13 77], b_vals=[10 11]
# CHECK-DAG: thread 6 a_vals=[10 74 14 78], b_vals=[12 13]
# CHECK-DAG: thread 7 a_vals=[11 75 15 79], b_vals=[14 15]
# CHECK-DAG: thread 8 a_vals=[16 80 20 84], b_vals=[16 17]
# CHECK-DAG: thread 9 a_vals=[17 81 21 85], b_vals=[18 19]
# CHECK-DAG: thread 10 a_vals=[18 82 22 86], b_vals=[20 21]
# CHECK-DAG: thread 11 a_vals=[19 83 23 87], b_vals=[22 23]
# CHECK-DAG: thread 12 a_vals=[24 88 28 92], b_vals=[24 25]
# CHECK-DAG: thread 13 a_vals=[25 89 29 93], b_vals=[26 27]
# CHECK-DAG: thread 14 a_vals=[26 90 30 94], b_vals=[28 29]
# CHECK-DAG: thread 15 a_vals=[27 91 31 95], b_vals=[30 31]
# CHECK-DAG: thread 16 a_vals=[32 96 36 100], b_vals=[32 33]
# CHECK-DAG: thread 17 a_vals=[33 97 37 101], b_vals=[34 35]
# CHECK-DAG: thread 18 a_vals=[34 98 38 102], b_vals=[36 37]
# CHECK-DAG: thread 19 a_vals=[35 99 39 103], b_vals=[38 39]
# CHECK-DAG: thread 20 a_vals=[40 104 44 108], b_vals=[40 41]
# CHECK-DAG: thread 21 a_vals=[41 105 45 109], b_vals=[42 43]
# CHECK-DAG: thread 22 a_vals=[42 106 46 110], b_vals=[44 45]
# CHECK-DAG: thread 23 a_vals=[43 107 47 111], b_vals=[46 47]
# CHECK-DAG: thread 24 a_vals=[48 112 52 116], b_vals=[48 49]
# CHECK-DAG: thread 25 a_vals=[49 113 53 117], b_vals=[50 51]
# CHECK-DAG: thread 26 a_vals=[50 114 54 118], b_vals=[52 53]
# CHECK-DAG: thread 27 a_vals=[51 115 55 119], b_vals=[54 55]
# CHECK-DAG: thread 28 a_vals=[56 120 60 124], b_vals=[56 57]
# CHECK-DAG: thread 29 a_vals=[57 121 61 125], b_vals=[58 59]
# CHECK-DAG: thread 30 a_vals=[58 122 62 126], b_vals=[60 61]
# CHECK-DAG: thread 31 a_vals=[59 123 63 127], b_vals=[62 63]
def test_load_tf32_tf32_16x8x8(ctx: DeviceContext):
    print("== test_load_f32_16x8x8")
    test_load_operands[DType.float32, DType.float32, Index(16, 8, 8)](ctx)


def main():
    with DeviceContext() as ctx:
        test_load_tf32_tf32_16x8x8(ctx)
