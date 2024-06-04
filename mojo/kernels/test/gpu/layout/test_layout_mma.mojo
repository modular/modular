# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv
from random import random_float64

from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    GridDim,
    ThreadIdx,
    barrier,
    lane_id,
)
from gpu.host import Context, Dim, Function, Stream, synchronize
from layout import *
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from layout.layout_tensor import outer_product_acc
from layout.tensor_core import *
from testing import *


fn mma_layout_tc[
    out_type: DType,
    in_type: DType,
    shape: StaticIntTuple[3],
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
](
    mat_c: LayoutTensor[out_type, layout_c],
    mat_a: LayoutTensor[in_type, layout_a],
    mat_b: LayoutTensor[in_type, layout_b],
):
    var tc = TensorCore[out_type, in_type, shape]()
    # alias shapes = TensorCore().get_shapes[out_type, in_type]()
    var a = tc.load_a(mat_a)
    var b = tc.load_b(mat_b)
    var c = tc.load_c(mat_c)
    var d = tc.mma(a, b, c)
    tc.store_d[layout_c](mat_c, d)


fn matmul_naive[
    out_type: DType,
    in_type: DType,
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
](
    mat_c: LayoutTensor[out_type, layout_c],
    mat_a: LayoutTensor[in_type, layout_a],
    mat_b: LayoutTensor[in_type, layout_b],
):
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    if x >= mat_c.shape[0]() or y >= mat_c.shape[1]():
        return

    var accum = mat_c[x, y]
    for i in range(mat_a.shape[1]()):
        accum += mat_a[x, i].cast[out_type]() * mat_b[i, y].cast[out_type]()
    mat_c[x, y] = accum


fn test_layout_mma[
    out_type: DType,
    in_type: DType,
    shape: StaticIntTuple[3],
    M: Int,
    N: Int,
    K: Int,
](
    rtol: Scalar[out_type] = 1e-05,
    rng_width: Float64 = Float64(10.0),
    debug: Bool = False,
) raises:
    print("== run layout mma => ", str(out_type), str(in_type), M, N, K)

    var stream = Stream()

    alias layout_a = Layout(IntTuple(M, K), IntTuple(K, 1))
    alias layout_b = Layout(IntTuple(K, N), IntTuple(N, 1))
    alias layout_c = Layout(IntTuple(M, N), IntTuple(N, 1))

    var mat_a = ManagedLayoutTensor[
        in_type, layout_a, gpu_managed_alloc, gpu_free
    ]()
    var mat_b = ManagedLayoutTensor[
        in_type, layout_b, gpu_managed_alloc, gpu_free
    ]()
    var mat_c = ManagedLayoutTensor[
        out_type, layout_c, gpu_managed_alloc, gpu_free
    ]()
    var mat_a_n = ManagedLayoutTensor[
        in_type, layout_a, gpu_managed_alloc, gpu_free
    ]()
    var mat_b_n = ManagedLayoutTensor[
        in_type, layout_b, gpu_managed_alloc, gpu_free
    ]()
    var mat_c_n = ManagedLayoutTensor[
        out_type, layout_c, gpu_managed_alloc, gpu_free
    ]()

    var rand_min = -1 * rng_width
    var rand_max = rng_width

    for i in range(M):
        for j in range(K):
            var val = random_float64(rand_min, rand_max).cast[DType.float32]()
            mat_a.tensor[i, j] = val.cast[in_type]()
            mat_a_n.tensor[i, j] = mat_a.tensor[i, j]
    for i in range(K):
        for j in range(N):
            var val = random_float64(rand_min, rand_max).cast[DType.float32]()
            mat_b.tensor[i, j] = val.cast[in_type]()
            mat_b_n.tensor[i, j] = mat_b.tensor[i, j]
    for i in range(M):
        for j in range(N):
            var val = Float32(0)
            mat_c.tensor[i, j] = val.cast[out_type]()
            mat_c_n.tensor[i, j] = mat_c.tensor[i, j]

    alias mma_func = mma_layout_tc[
        out_type, in_type, shape, layout_c, layout_a, layout_b
    ]

    var mma_kernel = Function[mma_func]()
    mma_kernel(
        mat_c.tensor,
        mat_a.tensor,
        mat_b.tensor,
        grid_dim=(1, 1),
        block_dim=(32),
        stream=stream,
    )

    synchronize()

    alias warps_per_block = 16
    alias naive_func = matmul_naive[
        out_type, in_type, layout_c, layout_a, layout_b
    ]
    var naive_kernel = Function[naive_func]()
    naive_kernel(
        mat_c_n,
        mat_a_n,
        mat_b_n,
        grid_dim=(ceildiv(M, warps_per_block), ceildiv(N, warps_per_block)),
        block_dim=(warps_per_block, warps_per_block),
    )

    synchronize()

    for i in range(M):
        for j in range(N):
            var out_val = mat_c.tensor[i, j]
            var out_ref = mat_c_n.tensor[i, j]
            if debug:
                if not math.isclose(out_val, out_ref, rtol=rtol):
                    print(i, out_val, out_ref)
            testing.assert_true(math.isclose(out_val, out_ref, rtol=rtol))

    _ = mat_a^
    _ = mat_b^
    _ = mat_c^
    _ = mat_a_n^
    _ = mat_b_n^
    _ = mat_c_n^

    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    alias shape_1684 = StaticIntTuple[3](16, 8, 4)
    alias shape_1688 = StaticIntTuple[3](16, 8, 8)
    alias shape_16816 = StaticIntTuple[3](16, 8, 16)
    try:
        with Context() as ctx:
            test_layout_mma[DType.float32, DType.float32, shape_1684, 16, 8, 4](
                rtol=1e-01
            )
            test_layout_mma[DType.float32, DType.float32, shape_1688, 16, 8, 8](
                rtol=1e-01
            )
            test_layout_mma[
                DType.float32, DType.bfloat16, shape_1688, 16, 8, 8
            ](rtol=1e-01)
            test_layout_mma[DType.float32, DType.float16, shape_1688, 16, 8, 8](
                rtol=1e-01
            )
    except e:
        print("CUDA_ERROR:", e)
