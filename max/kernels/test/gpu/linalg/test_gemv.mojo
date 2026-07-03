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

from std.math import ceildiv
from std.random import randn, seed, random_float64

import std.gpu.primitives.warp as warp
from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext, get_gpu_target
from std.sys import simd_width_of
from linalg.gemv import gemv_kernel, gemv_split_k, gevm_kernel
from linalg.matmul.gpu import matmul_kernel
import linalg.matmul.vendor.blas as vendor_blas

from std.utils import IndexList
from internal_utils import assert_almost_equal

from layout import TileTensor, Coord, Idx, row_major
from layout.tile_layout import Layout


def run_matvec[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    accum_type: DType = DType.float32,
](M: Int, N: Int, K: Int, *, ctx: DeviceContext) raises:
    print("== run_matvec kernel")
    print("dtypes: A=", a_type, " B=", b_type, " C=", c_type)

    var iterations = 100
    var a_host = alloc[Scalar[a_type]](M * K)
    var b_host = alloc[Scalar[b_type]](K * N)
    var c_host = alloc[Scalar[c_type]](M * N)
    var c_host_blas = alloc[Scalar[accum_type]](M * N)

    # using large inputs for FP8 ctype will cause saturation and cause all output values to be FP8 max value which can cause false positives.
    for i in range(M * K):
        a_host[i] = (
            random_float64(min=-1.0, max=1.0).cast[a_type]() if c_type
            == DType.float8_e4m3fn else Float32(i).cast[a_type]()
        )

    for i in range(K * N):
        b_host[i] = (
            random_float64(min=-1.0, max=1.0).cast[b_type]() if c_type
            == DType.float8_e4m3fn else Float32(i + 1).cast[b_type]()
        )

    for i in range(M * N):
        c_host[i] = Float32(0).cast[c_type]()

    for i in range(M * N):
        c_host_blas[i] = Float32(0).cast[accum_type]()

    var a_device = ctx.enqueue_create_buffer[a_type](M * K)
    var b_device = ctx.enqueue_create_buffer[b_type](K * N)
    var c_device = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_naive = ctx.enqueue_create_buffer[accum_type](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime WARPS_PER_BLOCK = 1024 // WARP_SIZE

    @always_inline
    @parameter
    def run_func_gemv(ctx: DeviceContext) raises:
        comptime kernel = gemv_kernel[c_type, a_type, b_type]

        ctx.enqueue_function[kernel](
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=ceildiv(M, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    @always_inline
    @parameter
    def run_func_gevm(ctx: DeviceContext) raises:
        comptime kernel = gevm_kernel[
            c_type,
            a_type,
            b_type,
            tile_size=WARP_SIZE * WARPS_PER_BLOCK,
        ]

        ctx.enqueue_function[kernel](
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=ceildiv(N, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    var nstime: Float64
    var kernelType: StaticString
    if N == 1:
        run_func_gemv(ctx)
        ctx.enqueue_copy(c_host, c_device)
        nstime = Float64(ctx.execution_time[run_func_gemv](iterations))
        kernelType = "GEMV"
    elif M == 1:
        run_func_gevm(ctx)
        ctx.enqueue_copy(c_host, c_device)
        nstime = Float64(ctx.execution_time[run_func_gevm](iterations))
        kernelType = "GEVM"
    else:
        print("Incorrect input shape [MNK]")
        return
    var flops = 2 * M * N * K
    var sectime = Float64(nstime) / Float64(iterations) / 1000000000
    print(kernelType, "KERNEL:")
    print(sectime, "sec")
    print(Float64(flops) * 1e-9 / sectime, " GFLOPS")
    print()

    # running reference using vendor_blas
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    # Create tensors for vendor_blas
    # For GEMV (N=1): A is MxK, B is Kx1, C is Mx1
    # For GEVM (M=1): A is 1xK, B is KxN, C is 1xN
    var a_nd = TileTensor(a_device, row_major(M, K))
    var b_nd = TileTensor(b_device, row_major(K, N))
    var c_ref_nd = TileTensor(c_device_naive, row_major(M, N))

    vendor_blas.matmul(
        ctx,
        c_ref_nd,
        a_nd,
        b_nd,
        c_row_major=True,
        transpose_b=False,
    )

    ctx.enqueue_copy(c_host_blas, c_device_naive)
    ctx.synchronize()

    print("Reference computed using vendor_blas")
    print()

    # Compare results - convert both to float32 for comparison
    var c_host_f32 = alloc[Float32](M * N)
    var c_host_blas_f32 = alloc[Float32](M * N)

    for i in range(M * N):
        c_host_f32[i] = c_host[i].cast[DType.float32]()
        # c_host_blas is in accum_type (float32), cast to c_type to simulate quantization, then to float32
        c_host_blas_f32[i] = c_host_blas[i].cast[c_type]().cast[DType.float32]()

    # Use appropriate tolerance based on output dtype
    comptime errorTolerance = 1e-2
    assert_almost_equal(
        c_host_f32,
        c_host_blas_f32,
        num_elements=M * N,
        atol=1e-4,
        rtol=errorTolerance,
    )


def run_matvec_with_epilogue_fn(
    M: Int, N: Int, K: Int, *, ctx: DeviceContext
) raises:
    comptime c_stride = 5
    comptime seed_val = 42

    var iterations = 100
    var a_host = alloc[Float32](M * K)
    var b_host = alloc[Float32](K * N)

    seed(seed_val)

    # over-allocate C to simulate a view tensor
    var c_host = alloc[Float32](M * N * c_stride)
    var c_host_naive = alloc[Float32](M * N * c_stride)

    randn(a_host, M * K)
    randn(b_host, K * N)

    for i in range(M * N * c_stride):
        c_host[i] = 0

    for i in range(M * N * c_stride):
        c_host_naive[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N * c_stride)

    var c_device_nd = TileTensor(
        c_device, Layout((M, N), (N * c_stride, c_stride))
    )
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var const_val: Float32 = 4.0

    @parameter
    @always_inline
    @__copy_capture(c_device_nd, const_val)
    def epilogue_fn[
        dtype: DType, width: SIMDSize, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        c_device_nd.store[width=width](
            Coord(idx),
            rebind[SIMD[DType.float32, width]](
                val + SIMD[dtype, width](const_val)
            ),
        )

    comptime WARPS_PER_BLOCK = 1024 // WARP_SIZE

    @always_inline
    @parameter
    def run_func_gemv(ctx: DeviceContext) raises:
        comptime kernel = gemv_kernel[
            DType.float32,
            DType.float32,
            DType.float32,
            elementwise_lambda_fn=epilogue_fn,
        ]
        var func = ctx.compile_function[kernel]()
        ctx.enqueue_function(
            func,
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=ceildiv(M, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    @always_inline
    @parameter
    def run_func_gevm(ctx: DeviceContext) raises:
        comptime kernel = gevm_kernel[
            DType.float32,
            DType.float32,
            DType.float32,
            tile_size=WARP_SIZE * WARPS_PER_BLOCK,
            elementwise_lambda_fn=epilogue_fn,
        ]
        var func = ctx.compile_function[kernel]()
        ctx.enqueue_function(
            func,
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=ceildiv(N, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    ctx.enqueue_copy(c_device, c_host)

    var kernelType: StaticString
    if N == 1:
        run_func_gemv(ctx)
        ctx.enqueue_copy(c_host, c_device)
        nstime = Float64(ctx.execution_time[run_func_gemv](iterations))
        kernelType = "GEMV"
    elif M == 1:
        run_func_gevm(ctx)
        ctx.enqueue_copy(c_host, c_device)
        nstime = Float64(ctx.execution_time[run_func_gevm](iterations))
        kernelType = "GEVM"
    else:
        print("Incorrect input shape [MNK]")
        return

    var flops = 2 * M * N * K
    var sectime = Float64(nstime) / Float64(iterations) / 1000000000

    print(kernelType, "KERNEL:")
    print(sectime, "sec")
    print(Float64(flops) * 1e-9 / sectime, " GFLOPS")
    print()

    # running naive
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime BLOCK_DIM = 16

    @always_inline
    @parameter
    def run_func_naive(ctx: DeviceContext) raises:
        comptime kernel = matmul_kernel[
            DType.float32,
            DType.float32,
            DType.float32,
            BLOCK_DIM,
            elementwise_lambda_fn=epilogue_fn,
        ]
        var func = ctx.compile_function[kernel]()
        ctx.enqueue_function(
            func,
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    run_func_naive(ctx)
    ctx.enqueue_copy(c_host_naive, c_device)

    nstime = Float64(ctx.execution_time[run_func_naive](iterations))
    var sectime2 = Float64(nstime) / Float64(iterations) / 1000000000
    print("NAIVE MATMUL:")
    print(sectime2, "sec")
    print(Float64(flops) * 1e-9 / sectime2, " GFLOPS")
    print()

    # Due to varied pattern of FP32 arith the accumulated sum isn't exactly
    # accurate. Hence relative tolerance needs to be checked.
    comptime errorTolerance = 1e-2
    assert_almost_equal(
        c_host,
        c_host_naive,
        num_elements=M * N * c_stride,
        atol=1e-4,
        rtol=errorTolerance,
    )


def run_split_k_gemm[
    M: Int,
    N: Int,
    K: Int,
    with_epilogue: Bool,
    tile_n: Int = 2,
    tile_m: Int = 1,
    num_threads: Int = 128,
](*, ctx: DeviceContext) raises:
    comptime a_type = DType.float32
    comptime simd_width = simd_width_of[a_type, target=get_gpu_target()]()
    comptime check_bounds_n = N % tile_n != 0
    # The grid covers ceildiv(M, tile_m) * tile_m rows, so tile_m > 1 needs
    # the row guard (tile_m == 1 covers M exactly).
    comptime check_bounds_m = tile_m > 1
    comptime seed_val = 42
    comptime row_pad = 8
    comptime const_val = Float32(4.0)  # added by the epilogue

    seed(seed_val)

    var a_host = alloc[Float32](M * K)
    var w_host = alloc[Float32](N * K)
    randn(a_host, M * K)
    randn(w_host, N * K)

    var row_stride = N + row_pad
    var c_elems = M * row_stride
    var c_host = alloc[Float32](c_elems)
    var c_expected = alloc[Float32](c_elems)
    for i in range(c_elems):
        c_host[i] = 0
        c_expected[i] = 0

    var a_device = ctx.enqueue_create_buffer[a_type](M * K)
    var w_device = ctx.enqueue_create_buffer[a_type](N * K)
    var c_device = ctx.enqueue_create_buffer[a_type](c_elems)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(w_device, w_host)
    ctx.enqueue_copy(c_device, c_host)

    var a_nd = TileTensor(a_device, row_major(M, K)).as_immut()
    var w_nd = TileTensor(w_device, row_major(N, K)).as_immut()
    var c_nd = TileTensor(c_device, Layout((M, N), (row_stride, 1)))

    comptime if with_epilogue:

        @parameter
        @always_inline
        @__copy_capture(c_nd)
        def epilogue_fn[
            dtype: DType, width: SIMDSize, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[dtype, width]):
            c_nd.store[width=width](
                Coord(idx),
                rebind[SIMD[a_type, width]](
                    val + SIMD[dtype, width](const_val)
                ),
            )

        comptime kernel = gemv_split_k[
            a_type,
            a_type,
            a_type,
            type_of(c_nd).LayoutType,
            type_of(a_nd).LayoutType,
            type_of(w_nd).LayoutType,
            simd_width=simd_width,
            tile_m=tile_m,
            tile_n=tile_n,
            num_threads=num_threads,
            elementwise_lambda_fn=epilogue_fn,
            check_bounds_m=check_bounds_m,
            check_bounds_n=check_bounds_n,
        ]
        var func = ctx.compile_function[kernel]()
        ctx.enqueue_function(
            func,
            c_nd,
            a_nd,
            w_nd,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, tile_m), ceildiv(N, tile_n)),
            block_dim=num_threads,
        )
    else:
        comptime kernel = gemv_split_k[
            a_type,
            a_type,
            a_type,
            type_of(c_nd).LayoutType,
            type_of(a_nd).LayoutType,
            type_of(w_nd).LayoutType,
            simd_width=simd_width,
            tile_m=tile_m,
            tile_n=tile_n,
            num_threads=num_threads,
            check_bounds_m=check_bounds_m,
            check_bounds_n=check_bounds_n,
        ]
        ctx.enqueue_function[kernel](
            c_nd,
            a_nd,
            w_nd,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, tile_m), ceildiv(N, tile_n)),
            block_dim=num_threads,
        )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    print(
        "== run_split_k_gemm with_epilogue=",
        with_epilogue,
        " M=",
        M,
        " N=",
        N,
        " K=",
        K,
    )

    # Host reference: out[m, n] = sum_k a[m, k] * w[n, k] (+ const_val with the
    # epilogue), written into the padded layout.
    for m in range(M):
        for n in range(N):
            var acc = Float32(0)
            for kk in range(K):
                acc += a_host[m * K + kk] * w_host[n * K + kk]
            comptime if with_epilogue:
                c_expected[m * row_stride + n] = acc + const_val
            else:
                c_expected[m * row_stride + n] = acc

    comptime errorTolerance = 1e-2
    assert_almost_equal(
        c_host,
        c_expected,
        num_elements=c_elems,
        atol=1e-4,
        rtol=errorTolerance,
    )


def main() raises:
    with DeviceContext() as ctx:
        # gemv for matrix vector multiply - FP32
        run_matvec[DType.float32, DType.float32, DType.float32](
            4096, 1, 4096, ctx=ctx
        )
        run_matvec_with_epilogue_fn(4096, 1, 4096, ctx=ctx)
        # gevm for vector matrix multiply - FP32
        run_matvec[DType.float32, DType.float32, DType.float32](
            1, 4096, 4096, ctx=ctx
        )
        run_matvec_with_epilogue_fn(1, 4096, 4096, ctx=ctx)

        # gemv for matrix vector multiply - BF16 input, FP8 output
        run_matvec[DType.bfloat16, DType.bfloat16, DType.float8_e4m3fn](
            4096, 1, 4096, ctx=ctx
        )
        # gevm for vector matrix multiply - BF16 input, FP8 output
        run_matvec[DType.bfloat16, DType.bfloat16, DType.float8_e4m3fn](
            1, 4096, 4096, ctx=ctx
        )

        # gemv for matrix vector multiply - BF16 input, BF16 output
        run_matvec[DType.bfloat16, DType.bfloat16, DType.bfloat16](
            4096, 1, 4096, ctx=ctx
        )
        # gevm for vector matrix multiply - BF16 input, BF16 output
        run_matvec[DType.bfloat16, DType.bfloat16, DType.bfloat16](
            1, 4096, 4096, ctx=ctx
        )

        # gemv_split_k GEMM (M > 1, N > 1), with and without an epilogue.
        # Covers both check_bounds_n=False (N % tile_n == 0) and the
        # column-bounds-checked path (N % tile_n != 0).
        run_split_k_gemm[4, 128, 2048, with_epilogue=True, tile_n=2](ctx=ctx)
        run_split_k_gemm[33, 126, 2048, with_epilogue=True, tile_n=4](ctx=ctx)
        run_split_k_gemm[4, 128, 2048, with_epilogue=False, tile_n=2](ctx=ctx)
        run_split_k_gemm[33, 126, 2048, with_epilogue=False, tile_n=4](ctx=ctx)

        # tile_m > 1 with M % tile_m != 0 and N % tile_n != 0: both row and
        # column guards on (check_bounds_m and check_bounds_n). No tuning
        # config reaches this combination today, but it is what the kernel's
        # parameter defaults give a direct caller.
        run_split_k_gemm[5, 126, 2048, with_epilogue=True, tile_n=4, tile_m=2](
            ctx=ctx
        )
        run_split_k_gemm[5, 126, 2048, with_epilogue=False, tile_n=4, tile_m=2](
            ctx=ctx
        )

        # FP32 router-GEMM dispatch shapes (small N, large K) with an epilogue,
        # at the dispatch's tile_m buckets (tile_n=1, 256 threads). Exercises
        # the split-K GEMV epilogue path that matmul_dispatch_sm100 now routes
        # the FP32 router/gate GEMM to (the epilogue rides through as the
        # elementwise_lambda_wrapper).
        run_split_k_gemm[
            4,
            128,
            6144,
            with_epilogue=True,
            tile_n=1,
            tile_m=1,
            num_threads=256,
        ](ctx=ctx)
        run_split_k_gemm[
            8,
            128,
            6144,
            with_epilogue=True,
            tile_n=1,
            tile_m=2,
            num_threads=256,
        ](ctx=ctx)
        run_split_k_gemm[
            16,
            128,
            6144,
            with_epilogue=True,
            tile_n=1,
            tile_m=4,
            num_threads=256,
        ](ctx=ctx)
        run_split_k_gemm[
            8,
            256,
            6144,
            with_epilogue=True,
            tile_n=1,
            tile_m=2,
            num_threads=256,
        ](ctx=ctx)
