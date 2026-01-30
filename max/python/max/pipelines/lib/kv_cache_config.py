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
"""MAX KVCache configuration."""

import enum
from collections.abc import Mapping

from max.config import ConfigFileModel
from max.dtype import DType
from max.nn.legacy.kv_cache import KVCacheStrategy
from pydantic import Field, PrivateAttr


class KVCacheConfig(ConfigFileModel):
    cache_strategy: KVCacheStrategy = Field(
        default=KVCacheStrategy.MODEL_DEFAULT,
        description=(
            "The cache strategy to use. This defaults to model_default, which "
            "selects the default strategy for the requested architecture. You "
            "can also force a specific strategy: continuous or paged."
        ),
    )

    kv_cache_page_size: int = Field(
        default=128,
        description="The number of tokens in a single page in the paged KVCache.",
    )

    enable_prefix_caching: bool = Field(
        default=True,
        description="Whether to enable prefix caching for the paged KVCache.",
    )

    enable_kvcache_swapping_to_host: bool = Field(
        default=False,
        description=(
            "Whether to swap paged KVCache blocks to host memory when device "
            "blocks are evicted."
        ),
    )

    device_memory_utilization: float = Field(
        default=0.9,
        description=(
            "The fraction of available device memory that the process should "
            "consume. This informs the KVCache workspace size: "
            "kv_cache_workspace = (total_free_memory * device_memory_utilization) "
            "- model_weights_size."
        ),
    )

    host_kvcache_swap_space_gb: float = Field(
        default=50.0,
        description=(
            "The amount of host memory to use for the host KVCache in GiB. "
            "This space is only allocated when kvcache_swapping_to_host is "
            "enabled."
        ),
    )

    _cache_dtype: DType = PrivateAttr(default=DType.float32)
    "The data type of the KV cache. The cache dtype is determined by the model's quantization encoding, and can be overridden from CLI by the kv_cache_format parameter."

    kv_cache_format: str | None = Field(
        default=None,
        description=(
            "Override the default data type for the KV cache."
            "Supported values: float32, bfloat16, float8_e4m3fn."
        ),
    )

    # Need to use `Optional` here to support `click` with 3.9.
    _available_cache_memory: int | None = PrivateAttr(default=None)
    """The amount of available cache memory in bytes. This should only be set by internal code."""

    _config_file_section_name: str = PrivateAttr(default="kv_cache_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @property
    def cache_dtype(self) -> DType:
        return self._cache_dtype

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for KVCacheConfig."""
        return {
            "KVCacheStrategy": KVCacheStrategy,
        }
