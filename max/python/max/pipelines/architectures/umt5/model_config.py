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
from pydantic import Field


class UMT5ConfigBase(MAXModelConfigBase):
    vocab_size: int = 256384
    d_model: int = 4096
    d_kv: int = 64
    d_ff: int = 10240
    num_layers: int = 24
    num_decoder_layers: int | None = 24
    num_heads: int = 64
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "gated-gelu"
    dense_act_fn: str | None = Field(default=None, exclude=True)
    is_gated_act: bool = Field(default=False, exclude=True)
    is_decoder: bool = Field(default=False, exclude=True)
    is_encoder_decoder: bool = True
    use_cache: bool = True
    output_past: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    classifier_dropout: float = 0.0
    scalable_attention: bool = True
    tie_word_embeddings: bool = False
    tokenizer_class: str = "T5Tokenizer"
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)
    dtype: DType = DType.bfloat16


class UMT5Config(UMT5ConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> UMT5ConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in UMT5ConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": encoding.dtype,
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return UMT5ConfigBase(**init_dict)
