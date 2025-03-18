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
"""Config for Llama Vision models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.pipelines.config import (
    KVCacheConfig,
    MAXModelConfig,
)
from max.pipelines.kv_cache import KVCacheParams
from transformers import AutoConfig


@dataclass(kw_only=True)  # type: ignore[call-overload]
class LlamaVisionConfig(MAXModelConfig):
    @staticmethod
    def help() -> dict[str, str]:
        return {}

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.text_config.num_key_value_heads,
            head_dim=(
                huggingface_config.text_config.hidden_size
                // huggingface_config.text_config.num_attention_heads
            ),
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            n_devices=n_devices,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.vision_config.num_hidden_layers
