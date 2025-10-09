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

import linalg.matmul.vendor.blas as vendor_blas
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.fp8_quantization import matmul_dynamic_scaled_fp8
from linalg.matmul import matmul
from testing import assert_almost_equal
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from utils.index import Index, IndexList


fn test_matmul_dynamic_scaled_fp8[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    transpose_b: Bool,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)
    alias static_a_scales_shape = DimList(1, m.dim)
    alias static_b_scales_shape = DimList(n.dim, 1) if transpose_b else DimList(
        1, n.dim
    )

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)
    var dynamic_a_scales_shape = DimList(1, m.value)
    var dynamic_b_scales_shape = DimList(
        n.value, 1
    ) if transpose_b else DimList(1, n.value)

    var a_host = HostNDBuffer[in_dtype, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[in_dtype, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[out_dtype, 2, static_c_shape](dynamic_c_shape)
    var a_scales_host = HostNDBuffer[scales_dtype, 2, static_a_scales_shape](
        dynamic_a_scales_shape
    )
    var b_scales_host = HostNDBuffer[scales_dtype, 2, static_b_scales_shape](
        dynamic_b_scales_shape
    )
    var a_host_ref = HostNDBuffer[DType.float32, 2, static_a_shape](
        dynamic_a_shape
    )
    var b_host_ref = HostNDBuffer[DType.float32, 2, static_b_shape](
        dynamic_b_shape
    )
    var c_host_ref = HostNDBuffer[DType.float32, 2, static_c_shape](
        dynamic_c_shape
    )

    var a_device = DeviceNDBuffer[in_dtype, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[in_dtype, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[out_dtype, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var a_scales_device = DeviceNDBuffer[
        scales_dtype, 2, static_a_scales_shape
    ](dynamic_a_scales_shape, ctx=ctx)
    var b_scales_device = DeviceNDBuffer[
        scales_dtype, 2, static_b_scales_shape
    ](dynamic_b_scales_shape, ctx=ctx)
    var a_device_ref = DeviceNDBuffer[DType.float32, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device_ref = DeviceNDBuffer[DType.float32, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[DType.float32, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    var M = m.value
    var N = n.value
    alias K = k.dim.get()

    random(a_host.tensor)
    random(b_host.tensor)
    random(a_scales_host.tensor)
    random(b_scales_host.tensor)

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_scales_device.buffer, a_scales_host.tensor.data)
    ctx.enqueue_copy(b_scales_device.buffer, b_scales_host.tensor.data)
    ctx.enqueue_copy(a_device_ref.buffer, a_host_ref.tensor.data)
    ctx.enqueue_copy(b_device_ref.buffer, b_host_ref.tensor.data)

    matmul_dynamic_scaled_fp8[
        input_scale_granularity="colwise",
        weight_scale_granularity="rowwise",
        transpose_b=transpose_b,
        target="gpu",
    ](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        a_scales_device.tensor,
        b_scales_device.tensor,
        ctx,
    )
    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.synchronize()

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk = Index(1, 1, K),
    ](
        c_device_ref.tensor,
        a_device.tensor,
        b_device.tensor,
        a_scales_device.tensor,
        b_scales_device.tensor,
        ctx,
    )

    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()

    for i in range(m.value):
        for j in range(n.value):
            assert_almost_equal(
                c_host.tensor[i, j].cast[DType.float32](),
                c_host_ref.tensor[i, j],
                msg="At [" + String(i) + ", " + String(j) + "]",
                atol=1e-2,
                rtol=1e-2,
            )


def main():
    with DeviceContext() as ctx:
        test_matmul_dynamic_scaled_fp8[
            in_dtype = DType.float8_e4m3fn,
            out_dtype = DType.bfloat16,
            scales_dtype = DType.bfloat16,
            transpose_b=True,
        ](ctx, dynamic(17), static[256 + 256](), static[256]())

        test_matmul_dynamic_scaled_fp8[
            in_dtype = DType.float8_e4m3fn,
            out_dtype = DType.bfloat16,
            scales_dtype = DType.bfloat16,
            transpose_b=True,
        ](ctx, dynamic(124), static[512](), static[512]())

        # these tests are guaranteed to hit a mojo fp8 kernel in the dispatch table.
        # if the fp8 kernel is not registered, these tests will fail.
        test_matmul_dynamic_scaled_fp8[
            in_dtype = DType.float8_e4m3fn,
            out_dtype = DType.bfloat16,
            scales_dtype = DType.bfloat16,
            transpose_b=True,
        ](ctx, dynamic(3000), static[5376](), static[4096]())

        test_matmul_dynamic_scaled_fp8[
            in_dtype = DType.float8_e4m3fn,
            out_dtype = DType.bfloat16,
            scales_dtype = DType.bfloat16,
            transpose_b=True,
        ](ctx, dynamic(224), static[43008](), static[5376]())
