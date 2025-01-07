# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s

from math import ceildiv, isclose
from random import random_float64

from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    GridDim,
    ThreadIdx,
    GlobalIdx,
    barrier,
    lane_id,
)
from gpu.host import DeviceContext, Dim
from gpu.host.memory_v1 import _make_ctx_current
from gpu.host.nvidia_cuda import CUDA
from layout import *
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from layout.math import outer_product_acc
from layout.tensor_core import *
from testing import *


fn mma_layout_tc[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
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
    var d = tc.mma_op(a, b, c)
    tc.store_d(mat_c, d)


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
    var x = GlobalIdx.x
    var y = GlobalIdx.y

    if int(x) >= mat_c.shape[0]() or int(y) >= mat_c.shape[1]():
        return

    var accum = mat_c[int(x), int(y)]
    for i in range(mat_a.shape[1]()):
        accum += (
            mat_a[int(x), i].cast[out_type]()
            * mat_b[i, int(y)].cast[out_type]()
        )
    mat_c[int(x), int(y)] = accum


fn test_layout_mma[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    M: Int,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    rtol: Scalar[out_type] = 1e-05,
    rng_width: Float64 = Float64(10.0),
    debug: Bool = False,
) raises:
    print("== run layout mma => ", str(out_type), str(in_type), M, N, K)

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

    var mma_kernel = ctx.compile_function[mma_func]()
    ctx.enqueue_function(
        mma_kernel,
        mat_c.tensor,
        mat_a.tensor,
        mat_b.tensor,
        grid_dim=(1, 1),
        block_dim=(32),
    )

    ctx.synchronize()

    alias warps_per_block = 16
    alias naive_func = matmul_naive[
        out_type, in_type, layout_c, layout_a, layout_b
    ]
    var naive_kernel = ctx.compile_function[naive_func]()
    ctx.enqueue_function(
        naive_kernel,
        mat_c_n,
        mat_a_n,
        mat_b_n,
        grid_dim=(ceildiv(M, warps_per_block), ceildiv(N, warps_per_block)),
        block_dim=(warps_per_block, warps_per_block),
    )

    ctx.synchronize()

    for i in range(M):
        for j in range(N):
            var out_val = mat_c.tensor[i, j]
            var out_ref = mat_c_n.tensor[i, j]
            if debug:
                if not isclose(out_val, out_ref, rtol=rtol):
                    print(i, out_val, out_ref)
            testing.assert_true(isclose(out_val, out_ref, rtol=rtol))

    _ = mat_a^
    _ = mat_b^
    _ = mat_c^
    _ = mat_a_n^
    _ = mat_b_n^
    _ = mat_c_n^


def main():
    alias shape_1684 = IndexList[3](16, 8, 4)
    alias shape_1688 = IndexList[3](16, 8, 8)
    alias shape_16816 = IndexList[3](16, 8, 16)

    with DeviceContext() as ctx:
        var prev_ctx = _make_ctx_current(CUDA(ctx))
        test_layout_mma[DType.float32, DType.float32, shape_1684, 16, 8, 4](
            ctx, rtol=1e-01
        )
        test_layout_mma[DType.float32, DType.float32, shape_1688, 16, 8, 8](
            ctx, rtol=1e-01
        )
        test_layout_mma[DType.float32, DType.bfloat16, shape_1688, 16, 8, 8](
            ctx, rtol=1e-01
        )
        test_layout_mma[DType.float32, DType.float16, shape_1688, 16, 8, 8](
            ctx, rtol=1e-01
        )
        _ = _make_ctx_current(prev_ctx)
