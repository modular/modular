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

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from pydantic import Field
from typing_extensions import Self


def _make_nvfp4_config(num_layers: int, num_single_layers: int) -> QuantConfig:
    """Build a QuantConfig for NVFP4 block-scaled quantization.

    This matches the modelopt NVFP4 format: block size 16 on the K axis,
    static per-tensor input scales, FP8 weight scales.
    """
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.STATIC,
        dtype=DType.float32,
        block_size=(1, 16),
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e4m3fn,
        block_size=(1, 16 // 2),
    )
    all_layers = set(range(num_layers + num_single_layers))
    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=all_layers,
        attn_quantized_layers=all_layers,
        embedding_output_dtype=DType.bfloat16,
        quant_method="modelopt",
        quant_algo="NVFP4",
    )


class Flux2Config(MAXModelConfigBase):
    patch_size: int = 1
    in_channels: int = 128
    out_channels: int | None = None
    num_layers: int = 8
    num_single_layers: int = 48
    attention_head_dim: int = 128
    num_attention_heads: int = 48
    joint_attention_dim: int = 15360
    timestep_guidance_channels: int = 256
    mlp_ratio: float = 3.0
    axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
    rope_theta: int = 2000
    eps: float = 1e-6
    guidance_embeds: bool = True
    """If False (Klein/distilled), no guidance embedder weights are expected."""
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)
    quant_config: QuantConfig | None = None
    """NVFP4 quantization config, populated when encoding is float4_e2m1fnx2."""

    @classmethod
    def initialize_from_config(
        cls,
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> Self:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in cls.model_fields
        }
        raw_dtype = supported_encoding_dtype(encoding)
        # For NVFP4, the computation dtype is bfloat16 (FP4 is only for
        # weights); build the QuantConfig so layers know to use NVFP4Linear.
        quant_config: QuantConfig | None = None
        if raw_dtype == DType.uint8:
            quant_config = _make_nvfp4_config(
                init_dict.get("num_layers", 8),
                init_dict.get("num_single_layers", 48),
            )
            raw_dtype = DType.bfloat16  # activations stay bf16
        init_dict.update(
            {
                "dtype": raw_dtype,
                "device": DeviceRef.from_device(devices[0]),
                "quant_config": quant_config,
            }
        )
        return cls(**init_dict)
