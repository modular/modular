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
from std.gpu.host import DeviceContext
from linalg.gemv import gemv_kernel, gevm_kernel
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

        ctx.enqueue_function_experimental[kernel](
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

        ctx.enqueue_function_experimental[kernel](
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
    var a_nd = TileTensor(a_device, row_major(Idx(M), Idx(K)))
    var b_nd = TileTensor(b_device, row_major(Idx(K), Idx(N)))
    var c_ref_nd = TileTensor(c_device_naive, row_major(Idx(M), Idx(N)))

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
        c_device, Layout((Idx(M), Idx(N)), (Idx(N * c_stride), Idx(c_stride)))
    )
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var const_val: Float32 = 4.0

    @parameter
    @always_inline
    @__copy_capture(c_device_nd, const_val)
    def epilogue_fn[
        dtype: DType, width: Int, *, alignment: Int = 1
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
        var func = ctx.compile_function_experimental[kernel]()
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
        var func = ctx.compile_function_experimental[kernel]()
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
        var func = ctx.compile_function_experimental[kernel]()
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
