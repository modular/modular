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

from sys import (
    align_of,
    bit_width_of,
    has_nvidia_gpu_accelerator,
    simd_width_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from algorithm.functional import elementwise
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext, get_gpu_target
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import arange as arange, random
from linalg.matmul.gpu import _matmul_gpu
from linalg.utils_gpu import MatmulConfig
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from test_utils import ulp_distance
from testing import assert_almost_equal

from utils import IndexList
from utils.index import Index

comptime to_dim[value: Optional[Int]] = value.value() if value else Dim()


comptime epilogue_func_type = fn[
    dtype: DType, width: Int, *, alignment: Int = 1
](IndexList[2], IndexList[2], SIMD[dtype, width]) capturing -> SIMD[
    dtype, width
]


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


fn select_max_ulp_distance[
    lambda_fn: Optional[epilogue_func_type]
](max_ulp_distance: Optional[Int]) -> Int:
    if max_ulp_distance:
        return max_ulp_distance.value()
    else:
        comptime if lambda_fn:
            return 4
        return 2


fn test[
    dtype: DType,
    /,
    *,
    transpose_b: Bool = False,
    arange_a: Bool = False,
    arange_b: Bool = False,
    lambda_fn: Optional[epilogue_func_type] = None,
    config: Optional[MatmulConfig[dtype, dtype, dtype, transpose_b]] = None,
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](
    ctx: DeviceContext,
    m: Int,
    n: Int,
    k: Int,
    rtol: Float64 = 1e-3 if dtype == DType.float32 else 1e-2,
    max_ulp_distance: Optional[Int] = None,
) raises:
    comptime assert Bool(N) and Bool(
        K
    ), "This test currently requires static N and K."

    print(m, "x", n, "x", k)

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
    var a_host_ptr = UnsafePointer[Scalar[dtype]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[dtype]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[dtype]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[dtype]].alloc(c_size)

    var a_host = LayoutTensor[dtype, a_layout](
        a_host_ptr,
        RuntimeLayout[a_layout].row_major(dynamic_a_shape),
    )
    var b_host = LayoutTensor[dtype, b_layout](
        b_host_ptr,
        RuntimeLayout[b_layout].row_major(dynamic_b_shape),
    )
    var c_host = LayoutTensor[dtype, c_layout](
        c_host_ptr,
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
    )
    var c_host_ref = LayoutTensor[dtype, c_layout](
        c_host_ref_ptr,
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
    )

    # Device allocations
    var a_device_buffer = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[dtype](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[dtype](c_size)

    var a_device = NDBuffer[dtype, 2, _, static_a_shape](
        a_device_buffer.unsafe_ptr(),
        DimList(m, k),
    )
    var b_device = NDBuffer[dtype, 2, _, static_b_shape](
        b_device_buffer.unsafe_ptr(),
        DimList(n, k) if transpose_b else DimList(k, n),
    )
    var c_device = NDBuffer[dtype, 2, _, static_c_shape](
        c_device_buffer.unsafe_ptr(),
        DimList(m, n),
    )
    var c_device_ref = NDBuffer[dtype, 2, _, static_c_shape](
        c_device_ref_buffer.unsafe_ptr(),
        DimList(m, n),
    )

    # Initialize matmul operands
    comptime if arange_a:
        arange(a_host)
    else:
        random(a_host)

    comptime if arange_b:
        arange(b_host)
    else:
        random(b_host)

    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    # Move operands to the Device
    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)
    ctx.enqueue_copy(c_device_ref_buffer, c_host_ref_ptr)

    var c_tensor = c_device

    @parameter
    @always_inline
    @__copy_capture(c_tensor, m, n)
    fn epilogue_fn[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        var update_val: SIMD[_dtype, width] = val

        comptime if lambda_fn:
            comptime func = lambda_fn.value()
            update_val = func(idx, (m, n), update_val)
        c_tensor.store[alignment=alignment](
            idx, rebind[SIMD[dtype, width]](update_val)
        )

    comptime if lambda_fn:
        _matmul_gpu[
            use_tensor_core=True,
            transpose_b=transpose_b,
            elementwise_lambda_fn=epilogue_fn,
            config=config,
        ](
            c_device,
            a_device,
            b_device,
            ctx,
        )
    else:
        _matmul_gpu[
            use_tensor_core=True,
            transpose_b=transpose_b,
            config=config,
        ](
            c_device,
            a_device,
            b_device,
            ctx,
        )

    ctx.synchronize()

    vendor_blas.matmul(
        ctx,
        c_device_ref,
        a_device,
        b_device,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    var c_ref_tensor = c_device_ref
    comptime pack_size = simd_width_of[dtype, target = get_gpu_target()]()

    @always_inline
    @__copy_capture(c_ref_tensor, m, n)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[2]](idx0)

        var val = c_ref_tensor.load[width=simd_width](idx)

        var update_val = val

        comptime if lambda_fn:
            comptime element_lambda = lambda_fn.value()
            update_val = element_lambda(idx, (m, n), val)

        c_ref_tensor.store(
            idx,
            update_val,
        )

    comptime if lambda_fn:
        elementwise[func, pack_size, target="gpu"](
            IndexList[2](m, n),
            ctx,
        )
    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device_buffer)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref_buffer)
    ctx.synchronize()

    var _max_ulp_distance = select_max_ulp_distance[lambda_fn](max_ulp_distance)
    for mi in range(m):
        for ni in range(n):
            var expect = c_host_ref[mi, ni][0]
            var actual = c_host[mi, ni][0]

            comptime if bit_width_of[dtype]() <= 16:
                var ulp_dist = ulp_distance(actual, expect)
                if ulp_dist <= _max_ulp_distance:
                    continue

            assert_almost_equal(actual, expect, rtol=rtol)

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
    with DeviceContext() as ctx:
        print("===> tfloat32-float32 mma")
        test[
            DType.float32,
            arange_a=True,
            arange_b=True,
            N = Int(12288),
            K = Int(4096),
        ](ctx, 512, 12288, 4096)
        test[DType.float32, arange_a=True, N = Int(384), K = Int(128)](
            ctx, 256, 384, 128
        )
        test[DType.float32, arange_b=True, N = Int(4096), K = Int(4096)](
            ctx, 128, 4096, 4096
        )
        test[
            DType.float32,
            arange_a=True,
            arange_b=True,
            N = Int(12288),
            K = Int(4096),
        ](ctx, 512, 12288, 4096)
        test[DType.float32, N = Int(4096), K = Int(11008)](ctx, 23, 4096, 11008)
        test[DType.float32, N = Int(4096), K = Int(12288)](ctx, 67, 4096, 12288)
        test[DType.float32, N = Int(4096), K = Int(4096)](ctx, 555, 4096, 4096)

        print("===> bfloat16-float32 mma")
        test[
            DType.bfloat16,
            arange_a=True,
            transpose_b=True,
            config = MatmulConfig[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                transpose_b=True,
            ](
                block_tile_shape=Index(64, 128, 64),
                warp_tile_shape=Index(16, 128, 64),
                num_pipeline_stages=3,
            ),
            N = Int(128),
            K = Int(128),
        ](ctx, 100, 128, 128)
        test[DType.bfloat16, arange_b=True, N = Int(12288), K = Int(3072)](
            ctx, 1024, 12288, 3072
        )
        test[
            DType.bfloat16,
            arange_a=True,
            arange_b=True,
            N = Int(5120),
            K = Int(3072),
        ](ctx, 1024, 5120, 3072)
        test[DType.bfloat16, N = Int(3072), K = Int(32768)](
            ctx, 1024, 3072, 32768
        )
        test[DType.bfloat16, N = Int(3072), K = Int(3072)](
            ctx, 1024, 3072, 3072
        )

        comptime if has_nvidia_gpu_accelerator():
            test[
                DType.bfloat16,
                transpose_b=True,
                config = MatmulConfig[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    transpose_b=True,
                ](
                    block_tile_shape=Index(16, 64, 64),
                    warp_tile_shape=Index(16, 64, 32),
                    num_pipeline_stages=3,
                    num_k_partitions=1,
                    num_warp_k_partitions=2,
                ),
                N = Int(4096),
                K = Int(4096),
            ](ctx, 32, 4096, 4096)
            test[
                DType.bfloat16,
                transpose_b=True,
                config = MatmulConfig[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    transpose_b=True,
                ](
                    block_tile_shape=Index(32, 64, 32),
                    warp_tile_shape=Index(16, 64, 32),
                    num_pipeline_stages=3,
                    num_k_partitions=1,
                    num_warp_k_partitions=4,
                ),
                N = Int(4096),
                K = Int(4096),
            ](ctx, 32, 4096, 4096)

        print("===> tfloat32-float32 mma with epilogue")
        test[
            DType.float32,
            lambda_fn=epilogue_test_fn,
            N = Int(3072),
            K = Int(3072),
        ](ctx, 999, 3072, 3072)
        test[
            DType.float32,
            lambda_fn=epilogue_test_fn,
            N = Int(12288),
            K = Int(2048),
        ](ctx, 777, 12288, 2048)

        print("===> bfloat16-float32 mma with epilogue")
        # Our default split-k reduction precision is output precision. For
        # bfloat16, we need a larger tolerance since the reference may reduce
        # in float32.
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N = Int(3072),
            K = Int(12288),
        ](ctx, 14, 3072, 12288, rtol=2e-2)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N = Int(12288),
            K = Int(3072),
        ](ctx, 33, 12288, 3072)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N = Int(5120),
            K = Int(3072),
        ](ctx, 101, 5120, 3072)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N = Int(3072),
            K = Int(32768),
        ](ctx, 400, 3072, 32768, rtol=2e-2)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N = Int(3072),
            K = Int(3072),
        ](ctx, 910, 3072, 3072)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N = Int(6144),
            K = Int(4096),
        ](ctx, 50, 6144, 4096)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N = Int(4096),
            K = Int(4096),
        ](ctx, 22, 4096, 4096)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N = Int(28672),
            K = Int(4096),
        ](ctx, 88, 28672, 4096)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N = Int(4096),
            K = Int(14336),
        ](ctx, 100, 4096, 14336)
        test[
            DType.bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N = Int(128256),
            K = Int(4096),
        ](ctx, 600, 128256, 4096)
