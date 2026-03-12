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
from typing_extensions import Self


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
        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return cls(**init_dict)
