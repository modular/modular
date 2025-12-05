# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


from sys import has_nvidia_gpu_accelerator, simd_width_of

import linalg.matmul.vendor.blas as vendor_blas
from algorithm.functional import elementwise
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext, get_gpu_target
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from linalg.bmm import _batched_matmul_gpu
from memory import LegacyUnsafePointer as UnsafePointer
from testing import assert_almost_equal

from utils import Index, IndexList

comptime epilogue_func_type = fn[
    dtype: DType, width: Int, *, alignment: Int = 1
] (SIMD[dtype, width]) capturing -> SIMD[dtype, width]

comptime to_dim[value: Optional[Int]] = value.value() if value else Dim()


@always_inline
@parameter
fn elementwise_epilogue_fn[
    dtype: DType,
    width: Int,
    *,
    alignment: Int = 1,
](val: SIMD[dtype, width],) -> SIMD[dtype, width]:
    return val + 2


fn test[
    dtype: DType,
    /,
    *,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
    Batch: Optional[Int] = None,
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](
    ctx: DeviceContext,
    b: Int,
    m: Int,
    n: Int,
    k: Int,
    rtol: Float64 = 1e-3 if dtype is DType.float32 else 1e-2,
) raises:
    print(b, "x", m, "x", n, "x", k, "transpose_b", transpose_b)

    comptime batch_static_a_shape = DimList(to_dim[Batch], to_dim[M], to_dim[K])
    comptime batch_static_b_shape = DimList(
        to_dim[Batch], to_dim[N], to_dim[K]
    ) if transpose_b else DimList(to_dim[Batch], to_dim[K], to_dim[N])
    comptime batch_static_c_shape = DimList(to_dim[Batch], to_dim[M], to_dim[N])

    comptime static_a_shape = DimList(to_dim[M], to_dim[K])
    comptime static_b_shape = DimList(
        to_dim[N], to_dim[K]
    ) if transpose_b else DimList(to_dim[K], to_dim[N])
    comptime static_c_shape = DimList(to_dim[M], to_dim[N])

    var batch_dynamic_a_shape = IndexList[3](
        Batch.or_else(b), M.or_else(m), K.or_else(k)
    )
    var batch_dynamic_b_shape = IndexList[3](
        Batch.or_else(b), N.or_else(n), K.or_else(k)
    ) if transpose_b else IndexList[3](
        Batch.or_else(b), K.or_else(k), N.or_else(n)
    )

    var batch_dynamic_c_shape = IndexList[3](
        Batch.or_else(b), M.or_else(m), N.or_else(n)
    )

    var dynamic_a_shape = IndexList[2](M.or_else(m), K.or_else(k))
    var dynamic_b_shape = IndexList[2](
        N.or_else(n), K.or_else(k)
    ) if transpose_b else IndexList[2](K.or_else(k), N.or_else(n))

    var dynamic_c_shape = IndexList[2](M.or_else(m), N.or_else(n))

    var a_size = b * m * k
    var b_size = b * n * k if transpose_b else b * k * n
    var c_size = b * m * n

    comptime a_layout = Layout.row_major(
        Batch.or_else(UNKNOWN_VALUE),
        M.or_else(UNKNOWN_VALUE),
        K.or_else(UNKNOWN_VALUE),
    )
    comptime b_layout = Layout.row_major(
        Batch.or_else(UNKNOWN_VALUE),
        N.or_else(UNKNOWN_VALUE),
        K.or_else(UNKNOWN_VALUE),
    ) if transpose_b else Layout.row_major(
        Batch.or_else(UNKNOWN_VALUE),
        K.or_else(UNKNOWN_VALUE),
        N.or_else(UNKNOWN_VALUE),
    )
    comptime c_layout = Layout.row_major(
        Batch.or_else(UNKNOWN_VALUE),
        M.or_else(UNKNOWN_VALUE),
        N.or_else(UNKNOWN_VALUE),
    )

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[dtype]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[dtype]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[dtype]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[dtype]].alloc(c_size)

    var a_host = LayoutTensor[dtype, a_layout](
        a_host_ptr,
        RuntimeLayout[a_layout].row_major(batch_dynamic_a_shape),
    )
    var b_host = LayoutTensor[dtype, b_layout](
        b_host_ptr,
        RuntimeLayout[b_layout].row_major(batch_dynamic_b_shape),
    )
    var c_host = LayoutTensor[dtype, c_layout](
        c_host_ptr,
        RuntimeLayout[c_layout].row_major(batch_dynamic_c_shape),
    )
    var c_host_ref = LayoutTensor[dtype, c_layout](
        c_host_ref_ptr,
        RuntimeLayout[c_layout].row_major(batch_dynamic_c_shape),
    )

    var a_device_buffer = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[dtype](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[dtype](c_size)

    var a_device = NDBuffer[dtype, 3, MutAnyOrigin, batch_static_a_shape, _](
        a_device_buffer.unsafe_ptr(), batch_dynamic_a_shape
    )
    var b_device = NDBuffer[dtype, 3, MutAnyOrigin, batch_static_b_shape, _](
        b_device_buffer.unsafe_ptr(), batch_dynamic_b_shape
    )
    var c_device = NDBuffer[dtype, 3, MutAnyOrigin, batch_static_c_shape, _](
        c_device_buffer.unsafe_ptr(), batch_dynamic_c_shape
    )
    var c_device_ref = NDBuffer[
        dtype, 3, MutAnyOrigin, batch_static_c_shape, _
    ](c_device_ref_buffer.unsafe_ptr(), batch_dynamic_c_shape)

    random(a_host)
    random(b_host)
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    # Move operands to the Device

    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)
    ctx.enqueue_copy(c_device_ref_buffer, c_host_ref_ptr)

    @parameter
    @always_inline
    @__copy_capture(c_device)
    fn epilogue_fn[
        dtype: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[dtype, width],) capturing -> None:
        comptime func = lambda_fn.value()
        var update_val = func(val)
        c_device.store(
            Index(idx[0], idx[1], idx[2]), update_val.cast[c_device.type]()
        )

    @parameter
    if lambda_fn:
        _batched_matmul_gpu[
            transpose_b=transpose_b,
            elementwise_epilogue_fn=epilogue_fn,
        ](c_device, a_device, b_device, ctx)
    else:
        _batched_matmul_gpu[transpose_b=transpose_b](
            c_device, a_device, b_device, ctx
        )

    ctx.synchronize()

    # Skip equality check if N or K are 0 (causes error in vendor_blas).
    if n == 0 or k == 0:
        return
    if not has_nvidia_gpu_accelerator() and m == 0:
        # AMD doesn't support matmul with M=0
        return

    for i in range(b):
        var c_ptr = c_device_ref.data + (i * m * n)
        var a_ptr = a_device.data + (i * m * k)
        var b_ptr = b_device.data + (i * k * n)

        var c_buffer = NDBuffer[dtype, 2, _, static_c_shape](
            c_ptr, dynamic_c_shape
        )
        var a_buffer = NDBuffer[dtype, 2, _, static_a_shape](
            a_ptr, dynamic_a_shape
        )
        var b_buffer = NDBuffer[dtype, 2, _, static_b_shape](
            b_ptr, dynamic_b_shape
        )

        vendor_blas.matmul(
            ctx,
            c_buffer,
            a_buffer,
            b_buffer,
            c_row_major=True,
            transpose_b=transpose_b,
        )

    ctx.synchronize()

    comptime pack_size = simd_width_of[dtype, target = get_gpu_target()]()

    @always_inline
    @__copy_capture(c_device_ref, b, m, n)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[3]](idx0)
        var val = c_device_ref.load[width=simd_width](idx)
        comptime element_lambda = lambda_fn.value()
        var update_val = element_lambda(val)

        c_device_ref.store(
            idx,
            update_val,
        )

    @parameter
    if lambda_fn:
        elementwise[func, pack_size, target="gpu"](
            IndexList[3](b, m, Int(n)),
            ctx,
        )

    ctx.enqueue_copy(c_host_ptr, c_device_buffer)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref_buffer)
    ctx.synchronize()

    for batch_idx in range(b):
        for m_idx in range(m):
            for n_idx in range(n):
                var expect = c_host_ref[batch_idx, m_idx, n_idx][0]
                var actual = c_host[batch_idx, m_idx, n_idx][0]

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
        # Test zero-dimension edge cases
        test[
            DType.bfloat16,
            transpose_b=False,
            Batch=None,
            M=None,
            N=None,
            K=None,
        ](ctx, 0, 2, 2, 2)

        test[
            DType.bfloat16,
            transpose_b=False,
            Batch=None,
            M=None,
            N=None,
            K=None,
        ](ctx, 2, 0, 2, 2)

        test[
            DType.bfloat16,
            transpose_b=False,
            Batch=None,
            M=None,
            N=None,
            K=None,
        ](ctx, 2, 2, 0, 2)

        test[
            DType.bfloat16,
            transpose_b=False,
            Batch=None,
            M=None,
            N=None,
            K=None,
        ](ctx, 2, 2, 2, 0)

        # tests naive kernels
        test[
            DType.bfloat16,
            transpose_b=False,
            Batch=None,
            M=None,
            N=None,
            K=None,
        ](ctx, 2, 2, 2, 2)

        test[
            DType.float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
            Batch=None,
            M=None,
            N=None,
            K=None,
        ](ctx, 2, 2, 2, 2)

        test[
            DType.float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
            Batch=None,
            M=None,
            N=None,
            K=None,
        ](ctx, 64, 256, 512, 128)

        @parameter
        if has_nvidia_gpu_accelerator():
            # NOTE: these tests should be run on a100 and above

            # tests kernels.ampere_128x128_4
            test[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                Batch=None,
                M=None,
                N = Int(128256),
                K = Int(4096),
            ](ctx, 2, 600, 128256, 4096)

            # tests kernels.ampere_256x64_4
            test[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                Batch=None,
                M=None,
                N = Int(3072),
                K = Int(12288),
            ](ctx, 4, 14, 3072, 12288, rtol=2e-2)

            # tests DeepSeek Case
            test[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                Batch=None,
                M=None,
                N = Int(128),
                K = Int(512),
            ](ctx, 128, 256, 128, 512)

            test[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                Batch=None,
                M=None,
                N = Int(512),
                K = Int(128),
            ](ctx, 128, 256, 512, 128)

            test[
                DType.bfloat16,
                transpose_b=False,
                lambda_fn=elementwise_epilogue_fn,
                Batch=None,
                M=None,
                N = Int(3072),
                K = Int(12288),
            ](ctx, 4, 14, 3072, 12288, rtol=2e-2)
