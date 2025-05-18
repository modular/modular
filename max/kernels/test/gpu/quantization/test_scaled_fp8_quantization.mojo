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

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random, zero
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.fp8_quantization import (
    quantize_dynamic_scaled_fp8,
    quantize_static_scaled_fp8,
)
from memory.unsafe import bitcast
from testing import assert_equal

from utils.numerics import FPUtils, max_finite, min_finite


fn test_scaled_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
](ctx: DeviceContext, scale: Float32, m: ValOrDim, n: ValOrDim,) raises:
    alias static_shape = DimList(m.dim, n.dim)
    var dynamic_shape = DimList(m.value, n.value)

    var in_host = HostNDBuffer[in_dtype, 2, static_shape](dynamic_shape)
    var out_host = HostNDBuffer[out_dtype, 2, static_shape](dynamic_shape)

    var in_device = DeviceNDBuffer[in_dtype, 2, static_shape](
        dynamic_shape, ctx=ctx
    )

    var out_device = DeviceNDBuffer[out_dtype, 2, static_shape](
        dynamic_shape, ctx=ctx
    )

    random(in_host.tensor)
    zero(out_host.tensor)

    ctx.enqueue_copy(in_device.buffer, in_host.tensor.data)
    ctx.enqueue_copy(out_device.buffer, out_host.tensor.data)

    quantize_static_scaled_fp8[out_dtype, in_dtype](
        out_device.tensor, in_device.tensor, scale, ctx
    )

    ctx.enqueue_copy(out_host.tensor.data, out_device.buffer)

    ctx.synchronize()

    for i in range(m.value):
        for j in range(n.value):
            var in_val_scaled_f32: Float32

            in_val_scaled_f32 = in_host.tensor[i, j].cast[DType.float32]() * (
                1.0 / scale
            )

            in_val_scaled_f32 = max(
                Float32(min_finite[out_dtype]()),
                min(Float32(max_finite[out_dtype]()), in_val_scaled_f32),
            )

            assert_equal(
                in_val_scaled_f32.cast[DType.float8_e4m3fn]().cast[
                    DType.float64
                ](),
                out_host.tensor[i, j].cast[DType.float64](),
            )


fn test_dynamic_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
    group_size_or_per_token: Int,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim,) raises:
    alias group_size = n.dim if group_size_or_per_token == -1 else group_size_or_per_token

    alias static_shape = DimList(m.dim, n.dim)
    alias static_scales_shape = DimList(m.dim, n.dim // group_size)
    var dynamic_shape = DimList(m.value, n.value)
    var dynamic_scales_shape = DimList(m.value, n.value // group_size)

    var in_host = HostNDBuffer[in_dtype, 2, static_shape](dynamic_shape)
    var out_host = HostNDBuffer[out_dtype, 2, static_shape](dynamic_shape)
    var scales_host = HostNDBuffer[in_dtype, 2, static_scales_shape](
        dynamic_scales_shape
    )

    var in_device = DeviceNDBuffer[in_dtype, 2, static_shape](
        dynamic_shape, ctx=ctx
    )
    var out_device = DeviceNDBuffer[out_dtype, 2, static_shape](
        dynamic_shape, ctx=ctx
    )
    var scales_device = DeviceNDBuffer[in_dtype, 2, static_scales_shape](
        dynamic_scales_shape, ctx=ctx
    )

    random(in_host.tensor)

    ctx.enqueue_copy(in_device.buffer, in_host.tensor.data)

    quantize_dynamic_scaled_fp8[group_size_or_per_token](
        out_device.tensor, scales_device.tensor, in_device.tensor, 1200.0, ctx
    )

    ctx.enqueue_copy(out_host.tensor.data, out_device.buffer)
    ctx.enqueue_copy(scales_host.tensor.data, scales_device.buffer)
    ctx.synchronize()

    for i in range(m.value):
        for group_idx in range(n.value // group_size):
            var group_max = Scalar[in_dtype](0)
            for j in range(group_size):
                group_max = max(
                    group_max,
                    abs(in_host.tensor[i, j + group_idx * Int(group_size)]),
                )

            var scale_factor = min(group_max, 1200.0) / Scalar[
                out_dtype
            ].MAX_FINITE.cast[in_dtype]()

            assert_equal(
                scales_host.tensor[i, group_idx].cast[DType.float64](),
                scale_factor.cast[DType.float64](),
            )

            for j in range(group_size):
                var in_val = in_host.tensor[i, j + group_idx * Int(group_size)]
                var out_val = out_host.tensor[
                    i, j + group_idx * Int(group_size)
                ]

                assert_equal(
                    out_val.cast[DType.float32](),
                    (in_val / scale_factor)
                    .cast[out_dtype]()
                    .cast[DType.float32](),
                    msg="At ["
                    + String(i)
                    + ", "
                    + String(j + group_idx * Int(group_size))
                    + "]",
                )


fn main() raises:
    with DeviceContext() as ctx:
        test_scaled_fp8_quant[DType.float8_e4m3fn, DType.bfloat16](
            ctx, 0.5, dynamic(32), static[16]()
        )
        test_scaled_fp8_quant[DType.float8_e4m3fn, DType.float16](
            ctx, 0.33, dynamic(31), static[15]()
        )
        test_scaled_fp8_quant[DType.float8_e4m3fn, DType.bfloat16](
            ctx, 0.3323, dynamic(31), static[15]()
        )

        test_dynamic_fp8_quant[DType.float8_e4m3fn, DType.bfloat16, -1](
            ctx, dynamic(1), static[256]()
        )
        test_dynamic_fp8_quant[DType.float8_e4m3fn, DType.bfloat16, -1](
            ctx, dynamic(1), static[1024]()
        )
        test_dynamic_fp8_quant[DType.float8_e4m3fn, DType.bfloat16, -1](
            ctx, dynamic(1), static[16384]()
        )
        test_dynamic_fp8_quant[DType.float8_e4m3fn, DType.bfloat16, 128](
            ctx, dynamic(1), static[16384]()
        )
