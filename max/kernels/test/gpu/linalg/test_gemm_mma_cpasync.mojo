# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for the non-swapAB GEMM tensor-core kernel."""

from std.math import ceildiv
from std.random import random_float64

from max.gpu.host import DeviceContext
from std.memory import alloc

from linalg.gemv import gemm_mma_cpasync
import linalg.matmul.vendor.blas as vendor_blas

from internal_utils import assert_almost_equal
from std.testing import assert_equal

from layout import TileTensor, Coord, Idx, row_major
from std.utils import IndexList


def run_gemm_mma_cpasync[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    tile_k: Int = 128,
    swapAB: Bool = False,
](gemm_m: Int, gemm_k: Int, gemm_n: Int, *, ctx: DeviceContext,) raises:
    """Run the GEMM TC kernel and compare against cuBLAS.

    act:    (gemm_m, gemm_k).
    weight: (gemm_n, gemm_k).
    output: (gemm_m, gemm_n), row-major.
    C[M, N] = act[M, K] * weight[N, K]^T.

    `swapAB` only changes the kernel's internal tiling/epilogue; the caller-facing
    shapes and the row-major `[M, N]` output are identical, so the same cuBLAS
    reference applies either way.
    """
    print(
        "== gemm_tc  M=",
        gemm_m,
        " K=",
        gemm_k,
        " N=",
        gemm_n,
        " tile_k=",
        tile_k,
        " swapAB=",
        swapAB,
    )
    print("dtypes: act=", a_type, " weight=", b_type, " out=", c_type)

    var act_size = gemm_m * gemm_k
    var weight_size = gemm_n * gemm_k
    var out_size = gemm_m * gemm_n

    # Host buffers.
    var act_host = alloc[Scalar[a_type]](act_size)
    var weight_host = alloc[Scalar[b_type]](weight_size)
    var out_host = alloc[Scalar[c_type]](out_size)
    var ref_host = alloc[Scalar[c_type]](out_size)

    for i in range(act_size):
        act_host[i] = random_float64(min=-0.5, max=0.5).cast[a_type]()

    for i in range(weight_size):
        weight_host[i] = random_float64(min=-0.5, max=0.5).cast[b_type]()

    for i in range(out_size):
        out_host[i] = Scalar[c_type](0)
        ref_host[i] = Scalar[c_type](0)

    # Device buffers.
    var act_dev = ctx.enqueue_create_buffer[a_type](act_size)
    var weight_dev = ctx.enqueue_create_buffer[b_type](weight_size)
    var out_dev = ctx.enqueue_create_buffer[c_type](out_size)
    var ref_dev = ctx.enqueue_create_buffer[c_type](out_size)

    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(weight_dev, weight_host)
    ctx.enqueue_copy(out_dev, out_host)
    ctx.enqueue_copy(ref_dev, ref_host)

    # --- Run our kernel ---
    var a_shape = row_major((gemm_m, gemm_k))
    var w_shape = row_major((gemm_n, gemm_k))
    var c_shape = row_major((gemm_m, gemm_n))

    var a_tensor = TileTensor(act_dev, a_shape)
    var w_tensor = TileTensor(weight_dev, w_shape)
    var c_tensor = TileTensor(out_dev, c_shape)

    gemm_mma_cpasync[tile_k=tile_k, swapAB=swapAB](
        c_tensor,
        a_tensor,
        w_tensor,
        gemm_m,
        gemm_k,
        gemm_n,
        1,
        ctx,
    )
    ctx.synchronize()

    var ref_tensor = TileTensor(ref_dev, c_shape)
    vendor_blas.matmul(
        ctx,
        ref_tensor,
        a_tensor,
        w_tensor,
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    # Copy back.
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.synchronize()

    # Compare in f32.
    var out_f32 = alloc[Float32](out_size)
    var ref_f32 = alloc[Float32](out_size)
    for i in range(out_size):
        out_f32[i] = out_host[i].cast[.float32]()
        ref_f32[i] = ref_host[i].cast[.float32]()

    assert_almost_equal(
        out_f32,
        ref_f32,
        num_elements=out_size,
        atol=1e-2,
        rtol=5e-2,
    )
    print("PASSED\n")


def run_gemm_mma_cpasync_residual[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    tile_k: Int = 128,
    swapAB: Bool = False,
](gemm_m: Int, gemm_k: Int, gemm_n: Int, *, ctx: DeviceContext) raises:
    """Run GEMM TC kernel with residual epilogue: D = matmul(A,B) + residual.

    Also exercises the elementwise-lambda epilogue under `swapAB`: the lambda
    must receive the true row-major `[M, N]` index so the residual is added at
    the correct location even when the kernel tiles the transposed problem.
    """
    print(
        "== gemm_tc+residual  M=",
        gemm_m,
        " K=",
        gemm_k,
        " N=",
        gemm_n,
        " tile_k=",
        tile_k,
        " swapAB=",
        swapAB,
    )

    var act_size = gemm_m * gemm_k
    var weight_size = gemm_n * gemm_k
    var out_size = gemm_m * gemm_n

    var act_host = alloc[Scalar[a_type]](act_size)
    var weight_host = alloc[Scalar[b_type]](weight_size)
    var residual_host = alloc[Scalar[c_type]](out_size)
    var out_host = alloc[Scalar[c_type]](out_size)
    var ref_host = alloc[Scalar[c_type]](out_size)

    for i in range(act_size):
        act_host[i] = random_float64(min=-0.5, max=0.5).cast[a_type]()
    for i in range(weight_size):
        weight_host[i] = random_float64(min=-0.5, max=0.5).cast[b_type]()
    for i in range(out_size):
        residual_host[i] = random_float64(min=-0.5, max=0.5).cast[c_type]()
        out_host[i] = Scalar[c_type](0)
        ref_host[i] = Scalar[c_type](0)

    var act_dev = ctx.enqueue_create_buffer[a_type](act_size)
    var weight_dev = ctx.enqueue_create_buffer[b_type](weight_size)
    var residual_dev = ctx.enqueue_create_buffer[c_type](out_size)
    var out_dev = ctx.enqueue_create_buffer[c_type](out_size)
    var ref_dev = ctx.enqueue_create_buffer[c_type](out_size)

    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(weight_dev, weight_host)
    ctx.enqueue_copy(residual_dev, residual_host)
    ctx.enqueue_copy(out_dev, out_host)
    ctx.enqueue_copy(ref_dev, ref_host)

    var a_shape = row_major((gemm_m, gemm_k))
    var w_shape = row_major((gemm_n, gemm_k))
    var c_shape = row_major((gemm_m, gemm_n))

    var a_tensor = TileTensor(act_dev, a_shape)
    var w_tensor = TileTensor(weight_dev, w_shape)
    var c_tensor = TileTensor(out_dev, c_shape)
    var residual_tensor = TileTensor(residual_dev, c_shape)

    @__parameter
    @inline(.always)
    @__copy_capture(c_tensor, residual_tensor)
    def residual_epilogue[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        var res = residual_tensor.load[width=width](Coord(idx)).cast[dtype]()
        c_tensor.store[width=width](Coord(idx), (val + res).cast[c_type]())

    gemm_mma_cpasync[
        tile_k=tile_k,
        elementwise_lambda_fn=residual_epilogue,
        swapAB=swapAB,
    ](
        c_tensor,
        a_tensor,
        w_tensor,
        gemm_m,
        gemm_k,
        gemm_n,
        1,
        ctx,
    )
    ctx.synchronize()

    var ref_tensor = TileTensor(ref_dev, c_shape)
    vendor_blas.matmul(
        ctx,
        ref_tensor,
        a_tensor,
        w_tensor,
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var out_f32 = alloc[Float32](out_size)
    var ref_f32 = alloc[Float32](out_size)
    for i in range(out_size):
        out_f32[i] = out_host[i].cast[.float32]()
        ref_f32[i] = (
            ref_host[i].cast[.float32]() + residual_host[i].cast[.float32]()
        )

    assert_almost_equal(
        out_f32,
        ref_f32,
        num_elements=out_size,
        atol=1e-2,
        rtol=5e-2,
    )
    print("PASSED\n")


def run_gemm_mma_cpasync_epilogue_fn[
    tile_k: Int = 128,
    swapAB: Bool = False,
](gemm_m: Int, gemm_k: Int, gemm_n: Int, *, ctx: DeviceContext) raises:
    """Checks that a store closure receives each output at its row-major
    `[M, N]` index and that the kernel itself leaves `c` untouched.

    The closure writes the row-scaled result into a separate buffer; the power
    of two scale keeps the comparison against the plain kernel exact.
    """
    print(
        "== gemm_tc+epilogue_fn  M=",
        gemm_m,
        " K=",
        gemm_k,
        " N=",
        gemm_n,
        " tile_k=",
        tile_k,
        " swapAB=",
        swapAB,
    )
    comptime dtype = DType.bfloat16
    var act_size = gemm_m * gemm_k
    var weight_size = gemm_n * gemm_k
    var out_size = gemm_m * gemm_n

    var act_dev = ctx.enqueue_create_buffer[dtype](act_size)
    var weight_dev = ctx.enqueue_create_buffer[dtype](weight_size)
    var ref_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var unused_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var out_dev = ctx.enqueue_create_buffer[dtype](out_size)

    with act_dev.map_to_host() as h_act, weight_dev.map_to_host() as h_w:
        for i in range(act_size):
            h_act[i] = random_float64(min=-0.5, max=0.5).cast[dtype]()
        for i in range(weight_size):
            h_w[i] = random_float64(min=-0.5, max=0.5).cast[dtype]()
    ctx.enqueue_memset(ref_dev, 0)
    ctx.enqueue_memset(unused_dev, 0)
    ctx.enqueue_memset(out_dev, 0)

    var a_tensor = TileTensor(act_dev, row_major((gemm_m, gemm_k)))
    var w_tensor = TileTensor(weight_dev, row_major((gemm_n, gemm_k)))
    var c_shape = row_major((gemm_m, gemm_n))
    var ref_tensor = TileTensor(ref_dev, c_shape)
    var unused_tensor = TileTensor(unused_dev, c_shape)
    var out_tensor = TileTensor(out_dev, c_shape)

    def store_odd_rows_negated[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: IndexList[2], val: SIMD[dtype, width]) {var out_tensor}:
        var row_scale = -2 if idx[0] % 2 == 1 else 2
        out_tensor.store[width=width](
            Coord(idx),
            (val * SIMD[dtype, width](row_scale)).cast[out_tensor.dtype](),
        )

    gemm_mma_cpasync[tile_k=tile_k, swapAB=swapAB](
        ref_tensor, a_tensor, w_tensor, gemm_m, gemm_k, gemm_n, 1, ctx
    )
    gemm_mma_cpasync[tile_k=tile_k, swapAB=swapAB](
        unused_tensor,
        a_tensor,
        w_tensor,
        gemm_m,
        gemm_k,
        gemm_n,
        1,
        store_odd_rows_negated,
        ctx,
    )

    with ref_dev.map_to_host() as h_ref, out_dev.map_to_host() as h_out, unused_dev.map_to_host() as h_unused:
        for i in range(out_size):
            var row_scale = -2 if (i // gemm_n) % 2 == 1 else 2
            assert_equal(h_out[i], h_ref[i] * Scalar[dtype](row_scale))
            assert_equal(h_unused[i], Scalar[dtype](0))
    print("PASSED\n")


def main() raises:
    with DeviceContext() as ctx:
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=64](
            32, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=128](
            32, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=256](
            32, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=512](
            32, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=64](
            24, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=128](
            24, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=256](
            24, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=512](
            24, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=64](
            16, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=128](
            16, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=256](
            16, 7168, 384, ctx=ctx
        )
        run_gemm_mma_cpasync[.bfloat16, .bfloat16, .bfloat16, tile_k=512](
            16, 7168, 384, ctx=ctx
        )

        # swapAB tests: C[M, N] = act @ weight^T must match the non-swap path
        # exactly. Kimi decode shape (N=2112, K=7168) across small M.
        run_gemm_mma_cpasync[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            tile_k=128,
            swapAB=True,
        ](1, 7168, 2112, ctx=ctx)
        run_gemm_mma_cpasync[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            tile_k=256,
            swapAB=True,
        ](2, 7168, 2112, ctx=ctx)
        run_gemm_mma_cpasync[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            tile_k=128,
            swapAB=True,
        ](4, 7168, 2112, ctx=ctx)
        run_gemm_mma_cpasync[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            tile_k=256,
            swapAB=True,
        ](7, 7168, 2112, ctx=ctx)
        run_gemm_mma_cpasync[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            tile_k=128,
            swapAB=True,
        ](16, 7168, 2112, ctx=ctx)

        # Residual epilogue tests: D = matmul(A,B) + residual.
        run_gemm_mma_cpasync_residual[
            .bfloat16, .bfloat16, .bfloat16, tile_k=128
        ](32, 7168, 384, ctx=ctx)
        run_gemm_mma_cpasync_residual[
            .bfloat16, .bfloat16, .bfloat16, tile_k=256
        ](24, 7168, 384, ctx=ctx)
        run_gemm_mma_cpasync_residual[
            .bfloat16, .bfloat16, .bfloat16, tile_k=64
        ](16, 7168, 384, ctx=ctx)

        run_gemm_mma_cpasync_residual[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            tile_k=128,
            swapAB=True,
        ](1, 7168, 2112, ctx=ctx)
        run_gemm_mma_cpasync_residual[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            tile_k=256,
            swapAB=True,
        ](2, 7168, 2112, ctx=ctx)
        run_gemm_mma_cpasync_residual[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            tile_k=128,
            swapAB=True,
        ](4, 7168, 2112, ctx=ctx)
        run_gemm_mma_cpasync_residual[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            tile_k=256,
            swapAB=True,
        ](7, 7168, 2112, ctx=ctx)
        run_gemm_mma_cpasync_residual[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            tile_k=128,
            swapAB=True,
        ](16, 7168, 2112, ctx=ctx)

        run_gemm_mma_cpasync_epilogue_fn[tile_k=128](28, 7168, 384, ctx=ctx)
        run_gemm_mma_cpasync_epilogue_fn[tile_k=256, swapAB=True](
            7, 7168, 2112, ctx=ctx
        )
