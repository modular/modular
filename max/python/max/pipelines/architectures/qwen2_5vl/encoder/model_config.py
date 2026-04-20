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

"""Configuration for Qwen2.5-VL text encoder used in QwenImage pipeline."""

from __future__ import annotations

import math
from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from pydantic import Field

_HF_KEY_MAP = {
    "max_position_embeddings": "max_seq_len",
}


class Qwen25VLTextEncoderConfigBase(MAXModelConfigBase):
    """Base configuration for Qwen2.5-VL text encoder.

    Key differences from Qwen3:
    - attention_bias: True (Qwen2.5 uses biased attention)
    - Different default dimensions matching Qwen2.5-VL-7B-Instruct
    """

    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    num_hidden_layers: int = 28
    head_dim: int = 128
    vocab_size: int = 152064
    intermediate_size: int = 18944
    rope_theta: float = 1000000.0
    max_seq_len: int = 128000
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)

    @property
    def attention_multiplier(self) -> float:
        return math.sqrt(1.0 / self.head_dim)


class Qwen25VLTextEncoderConfig(Qwen25VLTextEncoderConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> Qwen25VLTextEncoderConfigBase:
        text_config = config_dict.get("text_config", config_dict)

        init_dict = {}
        for key, value in text_config.items():
            mapped_key = _HF_KEY_MAP.get(key, key)
            if mapped_key in Qwen25VLTextEncoderConfigBase.__annotations__:
                init_dict[mapped_key] = value

        if "head_dim" not in init_dict:
            hidden_size = init_dict.get("hidden_size", 3584)
            num_attention_heads = init_dict.get("num_attention_heads", 28)
            init_dict["head_dim"] = hidden_size // num_attention_heads

        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )

        return Qwen25VLTextEncoderConfigBase(**init_dict)
