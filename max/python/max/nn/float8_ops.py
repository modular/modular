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

from max.dtype import DType
from max.graph import TensorValue

from .float8_config import Float8Config
from .kernels import (
    convert_weights_to_fp8_fnuz_if_needed,
    dynamic_scaled_matmul,
    matmul_static_scaled_float8,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
)


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
