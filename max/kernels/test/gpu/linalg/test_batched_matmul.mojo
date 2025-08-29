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


from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from linalg.bmm import _batched_matmul_gpu
from linalg import vendor_blas
from utils import Index, IndexList
from internal_utils._utils import ValOrDim, dynamic, static
from algorithm.functional import elementwise
from sys import simd_width_of
from testing import assert_almost_equal
from gpu.host import get_gpu_target
from sys import has_nvidia_gpu_accelerator


from internal_utils import (
    HostNDBuffer,
    random,
    zero,
)

alias epilogue_func_type = fn[dtype: DType, width: Int, *, alignment: Int = 1] (
    SIMD[dtype, width]
) capturing -> SIMD[dtype, width]


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
](
    ctx: DeviceContext,
    b: ValOrDim,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    rtol: Float64 = 1e-3 if dtype is DType.float32 else 1e-2,
) raises:
    var M = m.value
    var N = n.value
    var K = k.value
    var B = b.value
    print(B, "x", M, "x", N, "x", K, "transpose_b", transpose_b)

    alias batch_static_a_shape = DimList(b.dim, m.dim, k.dim)
    alias batch_static_b_shape = DimList(
        b.dim, n.dim, k.dim
    ) if transpose_b else DimList(b.dim, k.dim, n.dim)
    alias batch_static_c_shape = DimList(b.dim, m.dim, n.dim)

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)

    var batch_dynamic_a_shape = IndexList[3](b.value, m.value, k.value)
    var batch_dynamic_b_shape = IndexList[3](
        b.value, n.value, k.value
    ) if transpose_b else IndexList[3](b.value, k.value, n.value)

    var batch_dynamic_c_shape = IndexList[3](b.value, m.value, n.value)

    var dynamic_a_shape = IndexList[2](m.value, k.value)
    var dynamic_b_shape = IndexList[2](
        n.value, k.value
    ) if transpose_b else IndexList[2](k.value, n.value)

    var dynamic_c_shape = IndexList[2](m.value, n.value)

    var a_host = HostNDBuffer[dtype, 3, batch_static_a_shape](
        batch_dynamic_a_shape
    )
    var b_host = HostNDBuffer[dtype, 3, batch_static_b_shape](
        batch_dynamic_b_shape
    )
    var c_host = HostNDBuffer[dtype, 3, batch_static_c_shape](
        batch_dynamic_c_shape
    )
    var c_host_ref = HostNDBuffer[dtype, 3, batch_static_c_shape](
        batch_dynamic_c_shape
    )

    var a_device_buffer = ctx.enqueue_create_buffer[dtype](
        a_host.tensor.num_elements()
    )
    var b_device_buffer = ctx.enqueue_create_buffer[dtype](
        b_host.tensor.num_elements()
    )
    var c_device_buffer = ctx.enqueue_create_buffer[dtype](
        c_host.tensor.num_elements()
    )
    var c_device_ref_buffer = ctx.enqueue_create_buffer[dtype](
        c_host_ref.tensor.num_elements()
    )

    var a_device = NDBuffer[
        dtype, 3, MutableAnyOrigin, batch_static_a_shape, _
    ](a_device_buffer._unsafe_ptr(), batch_dynamic_a_shape)
    var b_device = NDBuffer[
        dtype, 3, MutableAnyOrigin, batch_static_b_shape, _
    ](b_device_buffer._unsafe_ptr(), batch_dynamic_b_shape)
    var c_device = NDBuffer[
        dtype, 3, MutableAnyOrigin, batch_static_c_shape, _
    ](c_device_buffer._unsafe_ptr(), batch_dynamic_c_shape)
    var c_device_ref = NDBuffer[
        dtype, 3, MutableAnyOrigin, batch_static_c_shape, _
    ](c_device_ref_buffer._unsafe_ptr(), batch_dynamic_c_shape)

    random(a_host.tensor)
    random(b_host.tensor)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Move operands to the Device

    ctx.enqueue_copy(a_device_buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device_buffer, b_host.tensor.data)
    ctx.enqueue_copy(c_device_buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref_buffer, c_host_ref.tensor.data)

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
        alias func = lambda_fn.value()
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

    for i in range(B):
        var c_ptr = c_device_ref.data + (i * M * N)
        var a_ptr = a_device.data + (i * M * K)
        var b_ptr = b_device.data + (i * K * N)

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

    alias pack_size = simd_width_of[dtype, target = get_gpu_target()]()

    @always_inline
    @__copy_capture(c_device_ref, B, M, N)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[3]](idx0)
        var val = c_device_ref.load[width=simd_width](idx)
        alias element_lambda = lambda_fn.value()
        var update_val = element_lambda(val)

        c_device_ref.store(
            idx,
            update_val,
        )

    @parameter
    if lambda_fn:
        elementwise[func, pack_size, target="gpu"](
            IndexList[3](B, M, Int(N)),
            ctx,
        )

    ctx.enqueue_copy(c_host.tensor.data, c_device_buffer)
    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref_buffer)
    ctx.synchronize()

    c_host_tensor = c_host.tensor
    c_host_ref_tensor = c_host_ref.tensor

    for b in range(B):
        for m in range(M):
            for n in range(N):
                var expect = c_host_ref_tensor[b, m, n]
                var actual = c_host_tensor[b, m, n]

                assert_almost_equal(actual, expect, rtol=rtol)

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


def main():
    with DeviceContext() as ctx:
        # tests naive kernels
        test[
            DType.bfloat16,
            transpose_b=False,
        ](ctx, dynamic(2), dynamic(2), dynamic(2), dynamic(2))

        test[
            DType.float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
        ](ctx, dynamic(2), dynamic(2), dynamic(2), dynamic(2))

        test[
            DType.float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
        ](ctx, dynamic(64), dynamic(256), dynamic(512), dynamic(128))

        @parameter
        if has_nvidia_gpu_accelerator():
            # NOTE: these tests should be run on a100 and above

            # tests kernels.ampere_128x128_4
            test[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
            ](ctx, dynamic(2), dynamic(600), static[128256](), static[4096]())

            # tests kernels.ampere_256x64_4
            test[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
            ](
                ctx,
                dynamic(4),
                dynamic(14),
                static[3072](),
                static[12288](),
                rtol=2e-2,
            )

            # tests DeepSeek Case
            test[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
            ](ctx, dynamic(128), dynamic(256), static[128](), static[512]())

            test[
                DType.bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
            ](ctx, dynamic(128), dynamic(256), static[512](), static[128]())

            test[
                DType.bfloat16,
                transpose_b=False,
                lambda_fn=elementwise_epilogue_fn,
            ](
                ctx,
                dynamic(4),
                dynamic(14),
                static[3072](),
                static[12288](),
                rtol=2e-2,
            )
