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


from sys import has_nvidia_gpu_accelerator, simd_width_of

import linalg.matmul.vendor.blas as vendor_blas
from algorithm.functional import elementwise
from buffer import NDBuffer
from gpu.host import DeviceContext, get_gpu_target
from layout._coord import Coord, Idx
from layout._layout import Layout, row_major
from layout._tile_tensor import TileTensor
from linalg.bmm import _batched_matmul_gpu

from random import rand
from testing import assert_almost_equal
from utils import IndexList

comptime epilogue_func_type = fn[
    dtype: DType, width: Int, *, alignment: Int = 1
](SIMD[dtype, width]) capturing -> SIMD[dtype, width]


@always_inline
@parameter
fn elementwise_epilogue_fn[
    dtype: DType,
    width: Int,
    *,
    alignment: Int = 1,
](val: SIMD[dtype, width],) -> SIMD[dtype, width]:
    return val + 2


fn run_bmm_and_check_result[
    dtype: DType,
    //,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
    check_against_naive_kernel: Bool = False,
](
    a_host: TileTensor[mut=True, dtype, ...],
    b_host: TileTensor[mut=True, dtype, ...],
    c_host: TileTensor[mut=True, dtype, ...],
    c_host_ref: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
    rtol: Float64 = 1e-3 if dtype == DType.float32 else 1e-2,
) raises:
    comptime assert c_host.flat_rank == 3, "c_device must have rank 3"
    comptime assert c_host_ref.flat_rank == 3, "c_device_ref must have rank 3"
    var a_size = a_host.numel()
    var b_size = b_host.numel()
    var c_size = c_host.numel()

    # allocate device buffers
    var a_device_buffer = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[dtype](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[dtype](c_size)

    var a_device = TileTensor[dtype](a_device_buffer, a_host.layout)
    var b_device = TileTensor[dtype](b_device_buffer, b_host.layout)
    var c_device = TileTensor[dtype](c_device_buffer, c_host.layout)
    var c_device_ref = TileTensor[dtype](c_device_ref_buffer, c_host_ref.layout)

    rand(a_host.ptr, a_size)
    rand(b_host.ptr, b_size)
    c_device_buffer.enqueue_fill(0)
    c_device_ref_buffer.enqueue_fill(0)

    # Copy operands to the Device
    ctx.enqueue_copy(a_device_buffer, a_host.ptr)
    ctx.enqueue_copy(b_device_buffer, b_host.ptr)

    # Run BMM
    @parameter
    @always_inline
    @__copy_capture(c_device)
    fn epilogue_fn[
        dtype: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx0: IndexList[rank], val: SIMD[dtype, width],) capturing -> None:
        var idx = rebind[IndexList[3]](idx0)
        comptime func = lambda_fn.value()
        var update_val = func(val)
        var coord = Coord((Idx(idx[0]), Idx(idx[1]), Idx(idx[2])))

        c_device.store(coord, update_val.cast[c_device.dtype]())

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

    var b = Int(c_device_ref.dim(0))
    var m = Int(c_device_ref.dim(1))
    var n = Int(c_device_ref.dim(2))
    var k = Int(a_device.dim(2))

    # Skip equality check if N or K are 0 (causes error in vendor_blas).
    if n == 0 or k == 0:
        return
    if not has_nvidia_gpu_accelerator() and m == 0:
        # AMD doesn't support matmul with M=0
        return

    @parameter
    if check_against_naive_kernel:
        # erase static dimensions so that the naive kernel can be used
        _batched_matmul_gpu[transpose_b=transpose_b](
            c_device_ref.make_dynamic[DType.int64](),
            a_device.make_dynamic[DType.int64](),
            b_device.make_dynamic[DType.int64](),
            ctx,
        )
    else:
        for i in range(a_host.dim(0)):
            var c_ptr = c_device_ref.ptr + i * Scalar[a_host.linear_idx_type](
                c_device_ref.layout.stride[0]().value()
            )
            var a_ptr = a_device.ptr + i * Scalar[a_host.linear_idx_type](
                a_device.layout.stride[0]().value()
            )
            var b_ptr = b_device.ptr + i * Scalar[a_host.linear_idx_type](
                b_device.layout.stride[0]().value()
            )

            var b_shape = IndexList[2](n, k) if transpose_b else IndexList[2](
                k, n
            )
            var c_buffer = NDBuffer[dtype, 2](c_ptr, {m, n})
            var a_buffer = NDBuffer[dtype, 2](a_ptr, {m, k})
            var b_buffer = NDBuffer[dtype, 2](b_ptr, b_shape)

            vendor_blas.matmul(
                ctx,
                c_buffer,
                a_buffer,
                b_buffer,
                c_row_major=True,
                transpose_b=transpose_b,
            )

    comptime pack_size = simd_width_of[dtype, target = get_gpu_target()]()

    @always_inline
    @__copy_capture(c_device_ref)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[3]](idx0)
        var coord = Coord((Idx(idx[0]), Idx(idx[1]), Idx(idx[2])))
        var val = c_device_ref.load[width=simd_width](coord)
        comptime element_lambda = lambda_fn.value()
        var update_val = element_lambda(val)

        c_device_ref.store(
            coord,
            update_val,
        )

    @parameter
    if lambda_fn:
        elementwise[func, pack_size, target="gpu"](
            IndexList[3](b, m, n),
            ctx,
        )

    ctx.enqueue_copy(c_host.ptr, c_device_buffer)
    ctx.enqueue_copy(c_host_ref.ptr, c_device_ref_buffer)
    ctx.synchronize()

    for batch_idx in range(b):
        for m_idx in range(m):
            for n_idx in range(n):
                var expect = c_host_ref[batch_idx, m_idx, n_idx][0]
                var actual = c_host[batch_idx, m_idx, n_idx][0]

                assert_almost_equal(actual, expect, rtol=rtol)


fn test_dynamic_shapes[
    dtype: DType,
    /,
    *,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    b: Int,
    m: Int,
    n: Int,
    k: Int,
    rtol: Float64 = 1e-3 if dtype == DType.float32 else 1e-2,
) raises:
    # fmt: off
    print(
        "test_dynamic_shapes", b, "x", m, "x", n, "x", k, "transpose_b", transpose_b,
    )
    # fmt: on

    var a_size = b * m * k
    var b_size = b * n * k
    var c_size = b * m * n

    # Host allocations
    var a_host_ptr = alloc[Scalar[dtype]](a_size)
    var b_host_ptr = alloc[Scalar[dtype]](b_size)
    var c_host_ptr = alloc[Scalar[dtype]](c_size)
    var c_host_ref_ptr = alloc[Scalar[dtype]](c_size)

    var a_host = TileTensor(a_host_ptr, row_major((Idx(b), Idx(m), Idx(k))))
    var c_host = TileTensor(c_host_ptr, row_major((Idx(b), Idx(m), Idx(n))))
    var c_host_ref = TileTensor(
        c_host_ref_ptr, row_major((Idx(b), Idx(m), Idx(n)))
    )

    var b_host = TileTensor(
        b_host_ptr, row_major((Idx(b), Idx(n), Idx(k)))
    ) if transpose_b else TileTensor(
        b_host_ptr, row_major((Idx(b), Idx(k), Idx(n)))
    )
    run_bmm_and_check_result[transpose_b=transpose_b, lambda_fn=lambda_fn](
        a_host, b_host, c_host, c_host_ref, ctx, rtol
    )

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()


fn test_static_NK[
    dtype: DType,
    /,
    *,
    N: Int,
    K: Int,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    b: Int,
    m: Int,
    rtol: Float64 = 1e-3 if dtype == DType.float32 else 1e-2,
) raises:
    print(
        "test_static_NK", b, "x", m, "x", N, "x", K, "transpose_b", transpose_b
    )

    var a_size = b * m * K
    var b_size = b * N * K
    var c_size = b * m * N

    # Host allocations
    var a_host_ptr = alloc[Scalar[dtype]](a_size)
    var b_host_ptr = alloc[Scalar[dtype]](b_size)
    var c_host_ptr = alloc[Scalar[dtype]](c_size)
    var c_host_ref_ptr = alloc[Scalar[dtype]](c_size)

    var a_host = TileTensor(a_host_ptr, row_major((Idx(b), Idx(m), Idx[K]())))
    var c_host = TileTensor(c_host_ptr, row_major((Idx(b), Idx(m), Idx[N]())))
    var c_host_ref = TileTensor(
        c_host_ref_ptr, row_major((Idx(b), Idx(m), Idx[N]()))
    )

    @parameter
    if transpose_b:
        var b_host = TileTensor(
            b_host_ptr, row_major((Idx(b), Idx[N](), Idx[K]()))
        )
        run_bmm_and_check_result[transpose_b=transpose_b, lambda_fn=lambda_fn](
            a_host, b_host, c_host, c_host_ref, ctx, rtol
        )

    else:
        var b_host = TileTensor(
            b_host_ptr, row_major((Idx(b), Idx[K](), Idx[N]()))
        )
        run_bmm_and_check_result[transpose_b=transpose_b, lambda_fn=lambda_fn](
            a_host, b_host, c_host, c_host_ref, ctx, rtol
        )

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()


fn test_non_row_major_layout[
    dtype: DType,
    /,
    *,
    B: Int,
    N: Int,
    K: Int,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    m: Int,
    rtol: Float64 = 1e-3 if dtype == DType.float32 else 1e-2,
) raises:
    """
    This function tests bacthed matmul with non-row major inputs.
    For example, A's layout could be (B, m, K):(K, B*K, 1).
    """
    print("test_non_row_major_layout", B, "x", m, "x", N, "x", K)

    var a_size = B * m * K
    var b_size = B * N * K
    var c_size = B * m * N

    # Host allocations
    var a_host_ptr = alloc[Scalar[dtype]](a_size)
    var b_host_ptr = alloc[Scalar[dtype]](b_size)
    var c_host_ptr = alloc[Scalar[dtype]](c_size)
    var c_host_ref_ptr = alloc[Scalar[dtype]](c_size)

    var a_layout = Layout(
        (Idx[B](), Idx(m), Idx[K]()), (Idx[K](), Idx[B * K](), Idx(1))
    )
    var c_layout = Layout(
        (Idx[B](), Idx(m), Idx[N]()), (Idx[N](), Idx[B * N](), Idx(1))
    )

    var a_host = TileTensor(a_host_ptr, a_layout)
    var c_host = TileTensor(c_host_ptr, c_layout)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_layout)

    var b_host = TileTensor(
        b_host_ptr, row_major((Idx[B](), Idx[N](), Idx[K]()))
    )
    run_bmm_and_check_result[
        transpose_b=True, lambda_fn=lambda_fn, check_against_naive_kernel=True
    ](a_host, b_host, c_host, c_host_ref, ctx, rtol)

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()


def main():
    with DeviceContext() as ctx:
        # Test zero-dimension edge cases
        test_dynamic_shapes[
            DType.bfloat16,
            transpose_b=False,
        ](ctx, 0, 2, 2, 2)

        # Test non-batch dispatch logic
        test_dynamic_shapes[
            DType.bfloat16,
            transpose_b=False,
        ](ctx, 1, 2, 2, 2)

        test_dynamic_shapes[
            DType.bfloat16,
            transpose_b=False,
        ](ctx, 2, 0, 2, 2)

        test_dynamic_shapes[
            DType.bfloat16,
            transpose_b=False,
        ](ctx, 2, 2, 0, 2)

        test_dynamic_shapes[
            DType.bfloat16,
            transpose_b=False,
        ](ctx, 2, 2, 2, 0)

        # tests naive kernels
        test_dynamic_shapes[
            DType.bfloat16,
            transpose_b=False,
        ](ctx, 2, 2, 2, 2)

        test_dynamic_shapes[
            DType.float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
        ](ctx, 2, 2, 2, 2)

        test_dynamic_shapes[
            DType.float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
        ](ctx, 64, 256, 512, 128)

        @parameter
        if has_nvidia_gpu_accelerator():
            # NOTE: these tests should be run on a100 and above

            # tests kernels.ampere_128x128_4
            test_static_NK[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N = Int(128256),
                K = Int(4096),
            ](ctx, 2, 600)

            # tests kernels.ampere_256x64_4
            test_static_NK[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N = Int(3072),
                K = Int(12288),
            ](ctx, 4, 14, rtol=2e-2)

            # tests DeepSeek Case
            test_static_NK[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N = Int(128),
                K = Int(512),
            ](ctx, 128, 256)

            test_static_NK[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N = Int(512),
                K = Int(128),
            ](ctx, 128, 256)

            test_static_NK[
                DType.bfloat16,
                transpose_b=False,
                lambda_fn=elementwise_epilogue_fn,
                N = Int(3072),
                K = Int(12288),
            ](ctx, 4, 14, rtol=2e-2)

            # test non-row major layout
            test_non_row_major_layout[
                DType.bfloat16,
                B = Int(128),
                N = Int(128),
                K = Int(512),
            ](ctx, 22)

            test_non_row_major_layout[
                DType.bfloat16,
                B = Int(128),
                N = Int(512),
                K = Int(128),
            ](ctx, 22)
