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

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass

from max.nn.kv_cache import KVCacheStrategy

from .max_config import MAXConfig

logger = logging.getLogger("max.pipelines")


# frozen is False (for now) because of _available_cache_memory being set by
# internal code.
@dataclass(frozen=False)
class KVCacheConfig(MAXConfig):
    cache_strategy: KVCacheStrategy = KVCacheStrategy.MODEL_DEFAULT
    """The cache strategy to use. This defaults to :obj:`model_default`, which will set the cache
    strategy based on the default strategy for the architecture requested.

    You can also force the engine to use a specific caching strategy: :obj:`continuous` | :obj:`paged`.
    """

    kv_cache_page_size: int = 128
    """The number of tokens in a single page in the paged KVCache."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for the paged attention KVCache."""

    enable_kvcache_swapping_to_host: bool = False
    """Whether to enable swapping the paged attention KVCache blocks to host memory when device blocks are evicted."""

    device_memory_utilization: float = 0.9
    """The fraction of available device memory that the process should consume.

    This is used to inform the size of the KVCache workspace. The calculation is:

    .. math::

        kv\\_cache\\_workspace = (total\\_free\\_memory \\times device\\_memory\\_utilization) - model\\_weights\\_size
    """

    host_kvcache_swap_space_gb: float = 50.0
    """The amount of host memory to use for the host KVCache in GiB.

    This space is only allocated when kvcache_swapping_to_host is enabled.
    """

    _available_cache_memory: int | None = None
    """The amount of available cache memory in bytes. This should only be set by internal code."""

    _config_file_section_name: str = "kv_cache_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for KVCacheConfig."""
        return {
            "KVCacheStrategy": KVCacheStrategy,
        }

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "cache_strategy": "Force a specific cache strategy: 'paged' or 'continuous'. If not provided, the optimal caching strategy for the model requested will be selected.",
            "kv_cache_page_size": "The number of tokens in a single page in the paged KVCache. Default is set to 128.",
            "enable_prefix_caching": "Whether to enable prefix caching for the paged attention KVCache. This defaults to false.",
            "enable_kvcache_swapping_to_host": "Whether to enable swapping the paged attention KVCache blocks to host memory when device blocks are evicted. This defaults to false.",
            "device_memory_utilization": "The fraction of available device memory that the process should consume. This is used to inform the size of the KVCache workspace: kv_cache_workspace = (total_free_memory * device_memory_utilization) - model_weights_size. Default is set to 0.9.",
            "host_kvcache_swap_space_gb": "The amount of host memory to use for the host KVCache in GiB. This is only used when kvcache_swapping_to_host is enabled. Default is set to 50.0.",
        }
