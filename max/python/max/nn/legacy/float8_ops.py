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

from max.dtype import DType
from max.graph import TensorValue

from .float8_config import Float8Config
from .kernels import (
    block_scales_interleave,
    convert_weights_to_fp8_fnuz_if_needed,
    dynamic_block_scaled_matmul_fp4,
    dynamic_scaled_matmul,
    matmul_static_scaled_float8,
    quantize_dynamic_block_scaled_fp4,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
)


def matmul_float4(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    input_scale: TensorValue,
    weight_scale_2: TensorValue,
    float8_config: Float8Config,
) -> TensorValue:
    """Computes x @ weight.T with modelopt NVFP4 quantization.

    Args:
        x: The input tensor in bf16.
        weight: The weight tensor in uint8 (float4-e2m1x2).
        weight_scale: The weight scale tensor in f8e4m3fn.
        input_scale: The input scale factor in f32 (used with vLLM convention by kernel).
        weight_scale_2: Additional weight scale factor in f32.
        float8_config: The float8 configuration.

    Returns:
        The output tensor in bf16.
    """
    if not float8_config.is_nvfp4:
        raise ValueError(
            "matmul_float4 only supports modelopt NVFP4 quantization"
        )

    x, x_scales = quantize_dynamic_block_scaled_fp4(
        x,
        tensor_sf=1.0 / input_scale,
        scales_type=DType.float8_e4m3fn,
        out_type=DType.uint8,  # fp4-e2m1fnX2
    )

    weight_scale = weight_scale.to(x.device)
    weight_scale = block_scales_interleave(
        weight_scale,
    )

    res = dynamic_block_scaled_matmul_fp4(
        x,
        weight,
        x_scales,
        weight_scale,
        tensor_sf=weight_scale_2 * input_scale,
        out_type=DType.bfloat16,
    )
    return res


def matmul_float8(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    input_scale: TensorValue | None,
    float8_config: Float8Config,
    group_size_or_per_token: int = -1,
) -> TensorValue:
    """Computes x @ weight.T with float8 quantization.

    Args:
        x: The input tensor.
        weight: The weight tensor.
        weight_scale: The weight scale tensor.
        input_scale: The input scale tensor (only required for static
            fp8 quantization).
        float8_config: The float8 configuration.
        group_size_or_per_token: The group size for quantization. When set to -1,
            the quantization is column-wise.

    Returns:
        The output tensor.
    """
    if float8_config.is_nvfp4:
        raise ValueError(
            "matmul_float8 does not support modelopt NVFP4 quantization"
        )

    weight, weight_scale = convert_weights_to_fp8_fnuz_if_needed(
        weight, weight_scale
    )

    if input_scale is not None:
        x = quantize_static_scaled_float8(x, input_scale, out_type=weight.dtype)

        return matmul_static_scaled_float8(x, weight, input_scale, weight_scale)
    else:
        x, x_scales = quantize_dynamic_scaled_float8(
            x,
            float8_config.input_scale,
            float8_config.weight_scale,
            scales_type=weight_scale.dtype,
            group_size_or_per_token=group_size_or_per_token,
            out_type=weight.dtype,
        )
        weight_scale = weight_scale.to(x.device)

        return dynamic_scaled_matmul(
            x,
            weight,
            x_scales,
            weight_scale,
            float8_config.input_scale,
            float8_config.weight_scale,
            out_type=DType.bfloat16,
        )
