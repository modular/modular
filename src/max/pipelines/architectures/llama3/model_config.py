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
"""Config for Llama3 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.pipelines.kv_cache import KVCacheParams
from max.pipelines.nn import Llama3RopeScalingParams


@dataclass
class Llama3Config:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rope_theta: float
    rope_scaling_params: Optional[Llama3RopeScalingParams]
    max_seq_len: int
    intermediate_size: int
    interleaved_rope_weights: bool
    vocab_size: int
    dtype: DType
    quantization_encoding: Optional[QuantizationEncoding]
    quantization_config: Optional[QuantizationConfig]
    kv_params: KVCacheParams
    all_logits: bool
    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    rms_norm_eps: Optional[float]
    tie_word_embeddings: bool
    stacked_mlp: bool
    stacked_qkv: bool
    logits_postprocessor: Callable[[TensorValue], TensorValue] | None
    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    devices: list[DeviceRef]
    clip_qkv: Optional[float]
