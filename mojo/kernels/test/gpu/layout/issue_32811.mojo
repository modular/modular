# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s


from gpu.host import Function, Context
from gpu.id import BlockDim, ThreadIdx, BlockIdx

from gpu.host.memory import _malloc_managed

from builtin.io import _printf

from kernel_utils.layout_tensor import LayoutTensor
from kernel_utils.layout import Layout
from kernel_utils.int_tuple import IntTuple


fn gpu_kernel(
    dst: DTypePointer[DType.float32],
    rhs: DTypePointer[DType.float32],
    lhs: DTypePointer[DType.float32],
):
    dst[BlockIdx.x() * 4 + ThreadIdx.x()] = (
        rhs[BlockIdx.x() * 4 + ThreadIdx.x()]
        + lhs[BlockIdx.x() * 4 + ThreadIdx.x()]
    )

    var dst_tensor = LayoutTensor[DType.float32, 16, 1](
        Layout(IntTuple(16, 1), IntTuple(1, 1)), dst
    )


def main():
    with Context() as ctx:
        var vec_a = _malloc_managed[DType.float32](16)
        var vec_b = _malloc_managed[DType.float32](16)
        var vec_c = _malloc_managed[DType.float32](16)
        for i in range(16):
            vec_a[i] = i
            vec_b[i] = i
            vec_c[i] = 0
        var kernel = Function[__type_of(gpu_kernel), gpu_kernel]()
        kernel(vec_c, vec_a, vec_b, block_dim=(4), grid_dim=(4))

        for i in range(16):
            print(vec_a[i], "+", vec_b[i], "=", vec_c[i])
