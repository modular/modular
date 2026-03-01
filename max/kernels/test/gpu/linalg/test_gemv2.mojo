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

from std.random import random_si64

import linalg.matmul.vendor.blas as vendor_blas
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from buffer import Dim, DimList, NDBuffer
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from linalg.matmul.gpu import _matmul_gpu
from std.utils import IndexList

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

    var a_managed = ManagedLayoutTensor[in_type, a_layout](
        RuntimeLayout[a_layout].row_major(dynamic_a_shape),
        ctx,
    )
    var b_managed = ManagedLayoutTensor[in_type, b_layout](ctx)
    var c_managed = ManagedLayoutTensor[out_type, c_layout](
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
        ctx,
    )
    var c_ref_managed = ManagedLayoutTensor[out_type, c_layout](
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
        ctx,
    )

    var a_host = a_managed.tensor[update=False]()
    var b_host = b_managed.tensor[update=False]()
    var c_host = c_managed.tensor[update=False]()
    var c_host_ref = c_ref_managed.tensor[update=False]()

    comptime rand_min = -100
    comptime rand_max = 100

    for i in range(m * k):
        var val = random_si64(rand_min, rand_max)
        a_host.ptr[i] = val.cast[in_type]()

    for i in range(k * n):
        var val = random_si64(rand_min, rand_max)
        b_host.ptr[i] = val.cast[in_type]()

    for i in range(m * n):
        c_host.ptr[i] = 0
        c_host_ref.ptr[i] = 0

    var a_device_tensor = a_managed.device_tensor()
    var b_device_tensor = b_managed.device_tensor()
    var c_device_tensor = c_managed.device_tensor()
    var c_device_ref_tensor = c_ref_managed.device_tensor()

    var a_device = NDBuffer[in_type, 2, _, static_a_shape](
        a_device_tensor.ptr,
        IndexList[2](m, k),
    )
    var b_device = NDBuffer[in_type, 2, _, static_b_shape](
        b_device_tensor.ptr,
        IndexList[2](n, k) if transpose_b else IndexList[2](k, n),
    )
    var c_device = NDBuffer[out_type, 2, _, static_c_shape](
        c_device_tensor.ptr,
        IndexList[2](m, n),
    )
    var c_device_ref = NDBuffer[out_type, 2, _, static_c_shape](
        c_device_ref_tensor.ptr,
        IndexList[2](m, n),
    )

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
        c_device,
        a_device,
        b_device,
        ctx,
    )

    var c_host_final = c_managed.tensor()

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

    var c_host_ref_final = c_ref_managed.tensor()

    var errors = 0
    for i in range(m * n):
        # print(i // n, i % n, c_host.ptr[i], c_host_ref.ptr[i])
        if c_host_final.ptr[i] != c_host_ref_final.ptr[i]:
            # print(i//n, i%n, c_host.ptr[i], c_host_ref.ptr[i])
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


def main() raises:
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
