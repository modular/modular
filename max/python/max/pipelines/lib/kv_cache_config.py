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
"""MAX KVCache configuration."""

from collections.abc import Sequence

from max.config import ConfigFileModel
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache.cache_params import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    KVCacheStrategy,
)
from pydantic import Field, PrivateAttr


class KVCacheConfig(ConfigFileModel):
    cache_strategy: KVCacheStrategy = Field(
        default="model_default",
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

    lmcache_config_file: str | None = Field(
        default=None,
        description=(
            "Path to an LMCache YAML configuration file. When set, enables "
            "LMCache-based external KV cache tiering (CPU, disk, remote)."
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
        """Returns the data type used for KV cache storage."""
        return self._cache_dtype

    def to_params(
        self,
        dtype: DType,
        n_kv_heads: int,
        head_dim: int,
        num_layers: int,
        devices: Sequence[DeviceRef],
        data_parallel_degree: int = 1,
        is_mla: bool = False,
        kvcache_quant_config: KVCacheQuantizationConfig | None = None,
    ) -> KVCacheParams:
        """Return KVCacheParams built from this config.

        Args:
            dtype: Data type for KV cache storage.
            n_kv_heads: Total number of KV heads across all devices.
            head_dim: Dimension of each attention head.
            num_layers: Number of model layers.
            devices: Devices that host the KV cache.
            data_parallel_degree: Degree of data parallelism.
            is_mla: Whether the model uses Multi-Latent Attention.
            kvcache_quant_config: KV cache quantization configuration.

        Returns:
            The constructed KV cache parameters.
        """
        return KVCacheParams(
            dtype=dtype,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            page_size=self.kv_cache_page_size,
            cache_strategy=self.cache_strategy,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_kvcache_swapping_to_host=self.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=self.host_kvcache_swap_space_gb,
            devices=devices,
            is_mla=is_mla,
            data_parallel_degree=data_parallel_degree,
            kvcache_quant_config=kvcache_quant_config,
            lmcache_config_file=self.lmcache_config_file,
        )
