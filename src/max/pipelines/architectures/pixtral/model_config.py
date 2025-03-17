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
"""Config for Pixtral models."""

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
# MAXModelConfig has optional fields, and PixtralConfig has required fields.
# We can work around this by having a superclass of MAXModelConfig that has
# the abstract methods, and then having PixtralConfig extend that.
@dataclass
class PixtralConfig(MAXConfig):
    @staticmethod
    def help() -> dict[str, str]:
        return {}

    # TODO(zheng): Figure out a scalable abstract method for all MAXModelConfigs.
    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            page_size=kv_cache_config.kv_cache_page_size,
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.text_config.num_key_value_heads,
            head_dim=huggingface_config.text_config.head_dim,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            n_devices=n_devices,
        )
