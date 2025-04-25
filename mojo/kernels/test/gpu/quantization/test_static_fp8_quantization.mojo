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

from utils.numerics import FPUtils, max_finite, min_finite
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils._utils import ValOrDim, dynamic, static
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    random,
    zero,
)
from testing import assert_equal
from linalg.fp8_quantization import static_scaled_fp8_quantization


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

    static_scaled_fp8_quantization[out_dtype, in_dtype](
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
