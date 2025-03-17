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
"""Config for MPNet models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.pipelines.config import (
    KVCacheConfig,
    MAXConfig,
)
from max.pipelines.kv_cache import KVCacheParams
from transformers import AutoConfig


# TODO(zheng): Move this under MAXModelConfig. The challenge here is that
# MAXModelConfig has optional fields, and MPNetConfig has required fields.
# We can work around this by having a superclass of MAXModelConfig that has
# the abstract methods, and then having MPNetConfig extend that.
@dataclass
class MPNetConfig(MAXConfig):
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
            n_kv_heads=huggingface_config.num_attention_heads,
            head_dim=(
                huggingface_config.hidden_size
                // huggingface_config.num_attention_heads
            ),
            cache_strategy=kv_cache_config.cache_strategy,
            n_devices=n_devices,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
        )
