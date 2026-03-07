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
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from pydantic import Field


class LTX2TextConnectorsConfigBase(MAXModelConfigBase):
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
            if key in LTX2TextConnectorsConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return LTX2TextConnectorsConfigBase(**init_dict)


# TODO: Update above config with below's style
# class LTX2TextConnectorsConfigBase(MAXModelConfigBase):
#     vocab_size: int = 32128
#     d_model: int = 512
#     d_kv: int = 64
#     d_ff: int = 2048
#     num_layers: int = 6
#     num_decoder_layers: int | None = None
#     num_heads: int = 8
#     relative_attention_num_buckets: int = 32
#     relative_attention_max_distance: int = 128
#     dropout_rate: float = 0.1
#     layer_norm_epsilon: float = 1e-6
#     initializer_factor: float = 1.0
#     feed_forward_proj: str = "relu"
#     dense_act_fn: str | None = Field(default=None, exclude=True)
#     is_gated_act: bool = Field(default=False, exclude=True)
#     is_decoder: bool = Field(default=False, exclude=True)
#     is_encoder_decoder: bool = True
#     use_cache: bool = True
#     pad_token_id: int = 0
#     eos_token_id: int = 1
#     classifier_dropout: float = 0.0
#     device: DeviceRef = Field(default_factory=DeviceRef.GPU)
#     dtype: DType = DType.bfloat16


class LTX2TextConnectorsConfig(LTX2TextConnectorsConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> LTX2TextConnectorsConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in LTX2TextConnectorsConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return LTX2TextConnectorsConfigBase(**init_dict)
