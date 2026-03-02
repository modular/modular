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
from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from pydantic import Field


class LTX2TransformerConfigBase(MAXModelConfigBase):
    activation_fn: str = "gelu-approximate"
    attention_bias: bool = True
    attention_head_dim: int = 128
    attention_out_bias: bool = True
    audio_attention_head_dim: int = 64
    audio_cross_attention_dim: int = 2048
    audio_hop_length: int = 160
    audio_in_channels: int = 128
    audio_num_attention_heads: int = 32
    audio_out_channels: int = 128
    audio_patch_size: int = 1
    audio_patch_size_t: int = 1
    audio_pos_embed_max_pos: int = 20
    audio_sampling_rate: int = 16000
    audio_scale_factor: int = 4
    base_height: int = 2048
    base_width: int = 2048
    caption_channels: int = 3840
    causal_offset: int = 1
    cross_attention_dim: int = 4096
    cross_attn_timestep_scale_multiplier: int = 1000
    in_channels: int = 128
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-06
    num_attention_heads: int = 32
    num_layers: int = 48
    out_channels: int = 128
    patch_size: int = 1
    patch_size_t: int = 1
    pos_embed_max_pos: int = 20
    qk_norm: str = "rms_norm_across_heads"
    rope_double_precision: bool = True
    rope_theta: float = 10000.0
    rope_type: str = "split"
    timestep_scale_multiplier: int = 1000
    vae_scale_factors: tuple[int, int, int] = (8, 32, 32)
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)


class LTX2TransformerConfig(LTX2TransformerConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> LTX2TransformerConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in LTX2TransformerConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return LTX2TransformerConfigBase(**init_dict)


class LTX2VocoderConfig(MAXModelConfigBase):
    hidden_channels: int = 1024
    in_channels: int = 128
    leaky_relu_negative_slope: float = 0.1
    out_channels: int = 2
    output_sampling_rate: int = 24000
    resnet_dilations: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    resnet_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    upsample_factors: tuple[int, ...] = (6, 5, 2, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (16, 15, 8, 4, 4)
    dtype: DType = DType.float32  # Vocoders often run in float32
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> MAXModelConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in LTX2VocoderConfig.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return LTX2VocoderConfig(**init_dict)


class LTX2TextConnectorsConfig(MAXModelConfigBase):
    audio_connector_attention_head_dim: int = 128
    audio_connector_num_attention_heads: int = 30
    audio_connector_num_layers: int = 2
    audio_connector_num_learnable_registers: int = 128
    caption_channels: int = 3840
    causal_temporal_positioning: bool = False
    connector_rope_base_seq_len: int = 4096
    rope_double_precision: bool = True
    rope_theta: float = 10000.0
    rope_type: str = "split"
    text_proj_in_factor: int = 49
    video_connector_attention_head_dim: int = 128
    video_connector_num_attention_heads: int = 30
    video_connector_num_layers: int = 2
    video_connector_num_learnable_registers: int = 128
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> MAXModelConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in LTX2TextConnectorsConfig.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return LTX2TextConnectorsConfig(**init_dict)
