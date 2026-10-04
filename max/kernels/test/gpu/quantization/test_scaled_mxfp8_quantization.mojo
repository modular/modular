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

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.block_scaled_quantization import (
    quantize_dynamic_scaled_fp4fp8,
)
from std.math import ceildiv
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    get_scale_factor,
)
from std.math import isnan


def test_dynamic_mxfp8_quant[
    in_dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
    N: Int,
](ctx: DeviceContext, m: Int, n: Int, tensor_scale: Float32 = 1.0) raises:
    if N % (SF_VECTOR_SIZE) != 0:
        raise Error(
            "n must be a multiple of (SF_VECTOR_SIZE // 2) due to kernel"
            " constraints"
        )

    comptime out_dtype = DType.float8_e4m3fn

    var input_layout = row_major(m, Idx[N])
    var scales_layout = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        ceildiv(n, SF_VECTOR_SIZE * SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var in_device = ctx.enqueue_create_buffer[in_dtype](input_layout.size())
    var out_device = ctx.enqueue_create_buffer[out_dtype](input_layout.size())
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](
        scales_layout.size()
    )
    var input_tensor = TileTensor(in_device, input_layout)
    var output_tensor = TileTensor(out_device, input_layout)
    var scales_tensor = TileTensor(scales_device, scales_layout)

    # Initialize input with random data and output with zeros on host
    with in_device.map_to_host() as in_host:
        var in_host_tensor = TileTensor(in_host, input_layout)
        random(in_host_tensor, min=-1.0, max=1.0)

        for idx0 in range(m):
            for idx1 in range(N):
                in_host_tensor[idx0, idx1] = (
                    in_host_tensor[idx0, idx1] * tensor_scale.cast[in_dtype]()
                )

    with out_device.map_to_host() as out_host:
        for i in range(len(out_host)):
            out_host[i] = 0

    # Run the quantization kernel
    quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        output_tensor.as_unsafe_any_origin(),
        scales_tensor.as_unsafe_any_origin(),
        input_tensor.as_unsafe_any_origin(),
        num_cols=n,
        num_cols_padded=n,
    )

    ctx.synchronize()

    # Verify results by reading back from device
    var atol = Float32(1)
    var rtol = Float32(0)
    var mismatch_count = 0
    with in_device.map_to_host() as in_host:
        with out_device.map_to_host() as out_host:
            with scales_device.map_to_host() as scales_host:
                var input_tensor_host = TileTensor(in_host, input_layout)
                var output_tensor_host = TileTensor(out_host, input_layout)
                var scales_tensor_host = TileTensor(scales_host, scales_layout)

                for idx0 in range(m):
                    for idx1 in range(N):
                        var ref_output = input_tensor_host[idx0, idx1].cast[
                            DType.float32
                        ]()
                        var output = output_tensor_host[idx0, idx1].cast[
                            DType.float32
                        ]()

                        if isnan(ref_output):
                            raise Error("NaN value found in reference output!")
                        if isnan(output):
                            raise Error("NaN value found in quantized output!")

                        var fp8_sf = get_scale_factor[
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE
                        ](scales_tensor_host.as_unsafe_any_origin(), idx0, idx1)

                        var output_dequantized = (
                            output * fp8_sf.cast[.float32]()
                        )

                        var left = abs(output_dequantized - ref_output)
                        var right = atol + rtol * abs(ref_output)

                        if left > right:
                            mismatch_count += 1

                var mismatch_rate = Float64(mismatch_count) / Float64(m * n)
                if (1 - mismatch_rate) < 0.999:
                    raise Error("Too many mismatches!")
                print(
                    "M = ",
                    m,
                    "N = ",
                    n,
                    "SF_VECTOR_SIZE = ",
                    SF_VECTOR_SIZE,
                    "in_dtype = ",
                    in_dtype,
                    "scales_dtype = ",
                    scales_dtype,
                    "mismatch percentage = ",
                    mismatch_rate * 100.0,
                    "%",
                )


def main() raises:
    with DeviceContext() as ctx:
        test_dynamic_mxfp8_quant[
            DType.bfloat16,
            MXFP8_SF_DTYPE,
            MXFP8_SF_VECTOR_SIZE,
            N=128,
        ](ctx, 1, 128)
        test_dynamic_mxfp8_quant[
            DType.bfloat16,
            MXFP8_SF_DTYPE,
            MXFP8_SF_VECTOR_SIZE,
            N=128,
        ](ctx, 258, 128)

        comptime for N in range(576, 16384, 1024):
            test_dynamic_mxfp8_quant[
                DType.bfloat16,
                MXFP8_SF_DTYPE,
                MXFP8_SF_VECTOR_SIZE,
                N=N,
            ](ctx, 999, N, tensor_scale=32.0)
