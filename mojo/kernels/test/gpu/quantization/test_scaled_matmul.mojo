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

# REQUIRES: H100-GPU
# RUN: %mojo-no-debug %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from collections import Optional
from gpu.host import DeviceContext
from internal_utils._utils import ValOrDim, dynamic, static
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    random,
)
from testing import assert_almost_equal
from linalg.fp8_quantization import matmul_dynamic_scaled_fp8
from linalg.matmul import matmul


fn test_matmul_dynamic_scaled_fp8[
    scales_dtype: DType,
    transpose_b: Bool,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(k.dim, n.dim) if transpose_b else DimList(
        n.dim, k.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)
    alias static_a_scales_shape = DimList(m.dim, 1)
    alias static_b_scales_shape = DimList(n.dim, 1) if transpose_b else DimList(
        1, n.dim
    )

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(k.value, n.value) if transpose_b else DimList(
        n.value, k.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)
    var dynamic_a_scales_shape = DimList(m.value, 1)
    var dynamic_b_scales_shape = DimList(
        n.value, 1
    ) if transpose_b else DimList(1, n.value)

    var a_host = HostNDBuffer[DType.float8_e4m3fn, 2, static_a_shape](
        dynamic_a_shape
    )
    var b_host = HostNDBuffer[DType.float8_e4m3fn, 2, static_b_shape](
        dynamic_b_shape
    )
    var c_host = HostNDBuffer[DType.bfloat16, 2, static_c_shape](
        dynamic_c_shape
    )
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

    var a_device = DeviceNDBuffer[DType.float8_e4m3fn, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[DType.float8_e4m3fn, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[DType.bfloat16, 2, static_c_shape](
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

    random(a_host.tensor)
    random(b_host.tensor)
    random(a_scales_host.tensor)
    random(b_scales_host.tensor)

    for i in range(m.value):
        for j in range(k.value):
            a_host_ref.tensor[i, j] = (
                a_host.tensor[i, j].cast[DType.float32]()
                * a_scales_host.tensor[i, 0].cast[DType.float32]()
            )

    for i in range(k.value):
        for j in range(n.value):

            @parameter
            if transpose_b:
                b_host_ref.tensor[j, i] = (
                    b_host.tensor[j, i].cast[DType.float32]()
                    * b_scales_host.tensor[j, 0].cast[DType.float32]()
                )
            else:
                b_host_ref.tensor[i, j] = (
                    b_host.tensor[i, j].cast[DType.float32]()
                    * b_scales_host.tensor[0, j].cast[DType.float32]()
                )

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_scales_device.buffer, a_scales_host.tensor.data)
    ctx.enqueue_copy(b_scales_device.buffer, b_scales_host.tensor.data)
    ctx.enqueue_copy(a_device_ref.buffer, a_host_ref.tensor.data)
    ctx.enqueue_copy(b_device_ref.buffer, b_host_ref.tensor.data)

    matmul_dynamic_scaled_fp8[transpose_b=transpose_b, target="gpu",](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        a_scales_device.tensor,
        b_scales_device.tensor,
        ctx,
    )
    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.synchronize()

    matmul[transpose_b=transpose_b, target="gpu",](
        c_device_ref.tensor,
        a_device_ref.tensor,
        b_device_ref.tensor,
        Optional[DeviceContext](ctx),
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
                rtol=6e-3,
            )


fn main() raises:
    with DeviceContext() as ctx:
        test_matmul_dynamic_scaled_fp8[
            scales_dtype = DType.bfloat16, transpose_b=True
        ](ctx, dynamic(123), static[512](), static[512]())

        test_matmul_dynamic_scaled_fp8[
            scales_dtype = DType.bfloat16, transpose_b=True
        ](ctx, dynamic(235), static[512](), static[512]())
