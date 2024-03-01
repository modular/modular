# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: cuda
# RUN: %mojo %s | FileCheck %s

from gpu.host import Function, Context, synchronize
from gpu.id import BlockDim, ThreadIdx, BlockIdx

from gpu.host.memory import _malloc_managed, _free

from builtin.io import _printf

from kernel_utils.layout_tensor import LayoutTensor

from kernel_utils.layout import Layout
from kernel_utils.int_tuple import IntTuple

from builtin.io import _printf


# FIXME: Make LayoutTensor register_passable to so we can use this
# type as an argument.
fn naive_matmul[
    layout_dst: Layout,
    layout_lhs: Layout,
    layout_rhs: Layout,
    BM: Int,
    BN: Int,
](
    dst_ptr: DTypePointer[DType.float32],
    lhs_ptr: DTypePointer[DType.float32],
    rhs_ptr: DTypePointer[DType.float32],
):
    var dst = LayoutTensor[layout_dst, DType.float32](dst_ptr)
    var lhs = LayoutTensor[layout_dst, DType.float32](lhs_ptr)
    var rhs = LayoutTensor[layout_dst, DType.float32](rhs_ptr)

    var dst_tile = dst.view[BM, BN](BlockIdx.y(), BlockIdx.x())

    dst_tile[ThreadIdx.y(), ThreadIdx.x()] = 0
    for k in range(dst.shape[0]()):
        var lhs_tile = rhs.view[BM, 1](BlockIdx.y(), k)
        var rhs_tile = lhs.view[1, BN](k, BlockIdx.x())
        dst_tile[ThreadIdx.y(), ThreadIdx.x()] += (
            lhs_tile[ThreadIdx.y(), k] * rhs_tile[k, ThreadIdx.x()]
        )


@always_inline
fn tensor_malloc_managed[
    layout: Layout, dtype: DType
]() raises -> LayoutTensor[layout, dtype]:
    var ptr = _malloc_managed[dtype](layout.size())
    return LayoutTensor[layout, dtype](ptr)


fn test_naive_matmul_kernel() raises:
    print("=== test_naive_matmul_kernel")
    alias M = 8
    alias N = 8
    alias K = 8
    alias BM = 4
    alias BN = 4

    alias layout_a = Layout(IntTuple(M, K), IntTuple(K, 1))
    alias layout_b = Layout(IntTuple(K, N), IntTuple(N, 1))
    alias layout_c = Layout(IntTuple(M, N), IntTuple(N, 1))

    var mat_a = tensor_malloc_managed[layout_a, DType.float32]()
    var mat_b = tensor_malloc_managed[layout_b, DType.float32]()
    var mat_c = tensor_malloc_managed[layout_c, DType.float32]()

    mat_a.linspace()
    mat_b.linspace()
    mat_c.fill(0)

    alias naive_matmul_kernel = naive_matmul[
        layout_c, layout_a, layout_b, BM, BN
    ]

    var kernel = Function[__type_of(naive_matmul_kernel), naive_matmul_kernel]()
    kernel(
        mat_c.ptr,
        mat_a.ptr,
        mat_b.ptr,
        grid_dim=(M // BM, N // BN),
        block_dim=(BM, BN),
    )

    synchronize()

    mat_c.print()

    _free(mat_c.ptr)
    _free(mat_a.ptr)
    _free(mat_b.ptr)


fn main() raises:
    with Context() as ctx:
        # CHECK: === test_naive_matmul_kernel
        # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
        # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
        # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
        # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
        # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
        # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
        # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
        # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
        test_naive_matmul_kernel()
