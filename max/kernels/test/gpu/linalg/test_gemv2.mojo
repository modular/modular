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
# mojo build --debug-level=full --mcmodel=medium --large-data-threshold=1048576
# to build this file if running into linking issues with large PTX kernels.

from random import random_si64

import linalg.matmul.vendor.blas as vendor_blas
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from linalg.matmul.gpu import _matmul_gpu
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from utils import IndexList

comptime epilogue_func_type = fn[
    type: DType, width: Int, *, alignment: Int = 1
](IndexList[2], IndexList[2], SIMD[type, width]) capturing -> SIMD[type, width]

comptime to_dim[value: Optional[Int]] = value.value() if value else Dim()


@parameter
@always_inline
fn epilogue_test_fn[
    dtype: DType, width: Int, *, alignment: Int = 1
](
    idx: IndexList[2],
    dim_space: IndexList[2],
    val: SIMD[dtype, width],
) -> SIMD[
    dtype, width
]:
    var bias = SIMD[dtype, width](0)

    comptime for i in range(width):
        bias[i] = (
            0.5
            + Float64(idx[0] + idx[1] + i)
            / Float64(dim_space[0] + dim_space[1])
        ).cast[dtype]()

    return val + bias


fn test[
    in_type: DType,
    out_type: DType,
    transpose_b: Bool,
    M: Optional[Int],
    N: Optional[Int],
    K: Optional[Int],
](mut bench: Bench, ctx: DeviceContext, m: Int, n: Int, k: Int,) raises:
    comptime assert Bool(N) and Bool(
        K
    ), "This test currently requires static N and K."

    print(m, "x", n, "x", k, "transpose_b", transpose_b)

    comptime static_a_shape = DimList(to_dim[M], to_dim[K])
    comptime static_b_shape = DimList(
        to_dim[N], to_dim[K]
    ) if transpose_b else DimList(to_dim[K], to_dim[N])
    comptime static_c_shape = DimList(to_dim[M], to_dim[N])

    var dynamic_a_shape = IndexList[2](M.or_else(m), K.or_else(k))
    var dynamic_b_shape = IndexList[2](
        N.or_else(n), K.or_else(k)
    ) if transpose_b else IndexList[2](K.or_else(k), N.or_else(n))
    var dynamic_c_shape = IndexList[2](M.or_else(m), N.or_else(n))

    var a_size = m * k
    var b_size = n * k if transpose_b else k * n
    var c_size = m * n

    comptime a_layout = Layout.row_major(
        M.or_else(UNKNOWN_VALUE), K.or_else(UNKNOWN_VALUE)
    )
    comptime b_layout = Layout.row_major(
        N.or_else(UNKNOWN_VALUE), K.or_else(UNKNOWN_VALUE)
    ) if transpose_b else Layout.row_major(
        K.or_else(UNKNOWN_VALUE), N.or_else(UNKNOWN_VALUE)
    )
    comptime c_layout = Layout.row_major(
        M.or_else(UNKNOWN_VALUE), N.or_else(UNKNOWN_VALUE)
    )

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[in_type]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[in_type]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[out_type]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[out_type]].alloc(c_size)

    var a_host = LayoutTensor[in_type, a_layout](
        a_host_ptr,
        RuntimeLayout[a_layout].row_major(dynamic_a_shape),
    )
    var b_host = LayoutTensor[in_type, b_layout](
        b_host_ptr,
        RuntimeLayout[b_layout].row_major(dynamic_b_shape),
    )
    var c_host = LayoutTensor[out_type, c_layout](
        c_host_ptr,
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
    )
    var c_host_ref = LayoutTensor[out_type, c_layout](
        c_host_ref_ptr,
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
    )

    # Device allocations
    var a_device_buffer = ctx.enqueue_create_buffer[in_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[in_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[out_type](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[out_type](c_size)

    var a_device = NDBuffer[in_type, 2, _, static_a_shape](
        a_device_buffer.unsafe_ptr(),
        DimList(m, k),
    )
    var b_device = NDBuffer[in_type, 2, _, static_b_shape](
        b_device_buffer.unsafe_ptr(),
        DimList(n, k) if transpose_b else DimList(k, n),
    )
    var c_device = NDBuffer[out_type, 2, _, static_c_shape](
        c_device_buffer.unsafe_ptr(),
        DimList(m, n),
    )
    var c_device_ref = NDBuffer[out_type, 2, _, static_c_shape](
        c_device_ref_buffer.unsafe_ptr(),
        DimList(m, n),
    )

    comptime rand_min = -100
    comptime rand_max = 100

    for i in range(m * k):
        var val = random_si64(rand_min, rand_max)
        a_host_ptr[i] = val.cast[in_type]()

    for i in range(k * n):
        var val = random_si64(rand_min, rand_max)
        b_host_ptr[i] = val.cast[in_type]()

    for i in range(m * n):
        c_host_ptr[i] = 0
        c_host_ref_ptr[i] = 0

    # Move operands to the Device

    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
        c_device,
        a_device,
        b_device,
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device_buffer)

    var handle = vendor_blas.Handle()

    vendor_blas.matmul(
        ctx,
        handle,
        c_device_ref,
        a_device,
        b_device,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref_buffer)

    ctx.synchronize()
    var errors = 0
    for i in range(m * n):
        # print(i // n, i % n, c_host_ptr[i], c_host_ref_ptr[i])
        if c_host_ptr[i] != c_host_ref_ptr[i]:
            # print(i//n, i%n, c_host_ptr[i], c_host_ref_ptr[i])
            errors += 1

    print("errors", errors)

    @parameter
    fn bench_func(mut m_bench: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
                c_device,
                a_device,
                b_device,
                ctx,
            )

        m_bench.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_func](
        BenchId("mojo matmul"),
        [ThroughputMeasure(BenchMetric.elements, 2 * m * n * k)],
    )

    @parameter
    fn bench_func_vendor_blas(mut m_bench: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            vendor_blas.matmul(
                ctx,
                handle,
                c_device_ref,
                a_device,
                b_device,
                c_row_major=True,
                transpose_b=transpose_b,
            )

        m_bench.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_func_vendor_blas](
        BenchId("vendor_blas matmul"),
        [ThroughputMeasure(BenchMetric.elements, 2 * m * n * k)],
    )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^
    _ = c_device_ref_buffer^


def main():
    var bench = Bench()

    with DeviceContext() as ctx:
        # GEMV_SPLIT_K
        # M = 1, K % simd_width == 0, transpose_b = True

        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=True,
            M=None,
            N = Int(4096),
            K = Int(4096),
        ](bench, ctx, 1, 4096, 4096)

        # M = 1, N % TILE_N != 0, K % simd_width == 0, transpose_b = True
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=True,
            M=None,
            N = Int(75837),
            K = Int(5120),
        ](bench, ctx, 1, 75837, 5120)

        # GEMV_KERNEL_VECTOR

        # N = 1, K % simd_width == 0, transpose_b = False
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=False,
            M=None,
            N = Int(1),
            K = Int(4096),
        ](bench, ctx, 4096, 1, 4096)

        # N = 1, K % simd_width == 0, transpose_b = True
        test[
            in_type = DType.bfloat16,
            out_type = DType.bfloat16,
            transpose_b=True,
            M=None,
            N = Int(1),
            K = Int(13824),
        ](bench, ctx, 5120, 1, 13824)

        # GEMV_KERNEL

        # M = 1, K % simd_width !=0, transpose_b = True
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=True,
            M=None,
            N = Int(4096),
            K = Int(4095),
        ](bench, ctx, 1, 4096, 4095)

        # N = 1, K % simd_width !=0, transpose_b = False
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=False,
            M=None,
            N = Int(1),
            K = Int(4095),
        ](bench, ctx, 4096, 1, 4095)

        # matmaul_naive
        # M = 1, K % WARP_SIZE != 0, transpose_b = False
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=False,
            M=None,
            N = Int(4096),
            K = Int(4095),
        ](bench, ctx, 1, 4096, 4095)

    bench.dump_report()
