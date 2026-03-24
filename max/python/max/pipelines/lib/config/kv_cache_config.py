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
from max.nn.kv_cache.cache_params import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    KVConnectorType,
)
from pydantic import ConfigDict, Field, PrivateAttr


class KVConnectorConfig(ConfigFileModel):
    """Connector-specific configuration for KV cache connectors.

    Common fields are typed. Additional connector-specific fields (e.g.
    LMCache settings like ``local_cpu``, ``max_local_cpu_size``) pass
    through via ``extra="allow"`` and are accessible via ``model_extra``.
    """

    model_config = ConfigDict(strict=False, extra="allow")

    host_kvcache_swap_space_gb: float = Field(
        default=50.0,
        description=(
            "Host memory (GiB) reserved for KV cache swapping. "
            "Used by local and tiered connectors."
        ),
    )
    """Host memory in GiB for KV cache swapping."""

    disk_offload_dir: str | None = Field(
        default=None,
        description=(
            "Directory for disk-based KV cache offloading. "
            "Required when kv_connector is 'tiered'."
        ),
    )
    """Directory for disk-based KV cache offloading."""

    disk_offload_max_gb: float = Field(
        default=50.0,
        description="Maximum disk space (GB) for KV cache offloading.",
    )
    """Maximum disk space in GB for KV cache offloading."""

    disk_offload_direct_io: bool = Field(
        default=False,
        description="Use O_DIRECT for disk I/O (bypasses OS page cache).",
    )
    """Whether to use O_DIRECT for disk I/O."""

    def as_lmcache_config(self) -> dict[str, object]:
        """Returns only the extra (LMCache-specific) fields as a dict.

        Filters out the typed fields defined on this class so the result
        can be passed directly to ``LMCacheEngineConfig.from_defaults()``.
        """
        return dict(self.model_extra) if self.model_extra else {}


class KVCacheConfig(ConfigFileModel):
    """Configuration for the paged KV cache."""

    kv_cache_page_size: int = Field(
        default=128,
        description="The number of tokens in a single page in the paged KVCache.",
    )
    """The number of tokens in a single page in the paged KV cache."""

    enable_prefix_caching: bool = Field(
        default=True,
        description="Whether to enable prefix caching for the paged KVCache.",
    )
    """Whether to enable prefix caching for the paged KV cache."""

    kv_connector: KVConnectorType | None = Field(
        default=None,
        description=(
            "Type of KV cache connector to use. Options: null, local, tiered, "
            "lmcache. When not set, defaults to null (no external caching)."
        ),
    )
    """Type of KV cache connector to use."""

    kv_connector_config: KVConnectorConfig | None = Field(
        default=None,
        description=(
            "Connector-specific configuration overrides as inline JSON or "
            "path to a YAML/JSON file. Each connector type has sensible "
            "defaults, so this is only needed for customization."
        ),
    )
    """Connector-specific configuration overrides."""

    device_memory_utilization: float = Field(
        default=0.9,
        description=(
            "The fraction of available device memory that the process should "
            "consume. This informs the KVCache workspace size: "
            "kv_cache_workspace = (total_free_memory * device_memory_utilization) "
            "- model_weights_size."
        ),
    )
    """The fraction of available device memory the process should consume."""

    _cache_dtype: DType = PrivateAttr(default=DType.float32)
    "The data type of the KV cache. The cache dtype is determined by the model's quantization encoding, and can be overridden from CLI by the kv_cache_format parameter."

    kv_cache_format: str | None = Field(
        default=None,
        description=(
            "Override the default data type for the KV cache."
            "Supported values: float32, bfloat16, float8_e4m3fn."
        ),
    )
    """An override for the default data type of the KV cache."""

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
        num_q_heads: int | None = None,
        kvcache_quant_config: KVCacheQuantizationConfig | None = None,
    ) -> KVCacheParams:
        """Returns :class:`~max.nn.kv_cache.cache_params.KVCacheParams` built from this config.

        Args:
            dtype: Data type for KV cache storage.
            n_kv_heads: Total number of KV heads across all devices.
            head_dim: Dimension of each attention head.
            num_layers: Number of model layers.
            devices: Devices that host the KV cache.
            data_parallel_degree: Degree of data parallelism.
            is_mla: Whether the model uses Multi-Latent Attention.
            num_q_heads: Number of query attention heads. Required when
                ``is_mla`` is True.
            kvcache_quant_config: KV cache quantization configuration.

        Returns:
            The constructed KV cache parameters.
        """
        cfg = self.kv_connector_config
        return KVCacheParams(
            dtype=dtype,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            page_size=self.kv_cache_page_size,
            enable_prefix_caching=self.enable_prefix_caching,
            kv_connector=self.kv_connector,
            kv_connector_config=cfg,
            host_kvcache_swap_space_gb=(
                cfg.host_kvcache_swap_space_gb if cfg else None
            ),
            devices=devices,
            is_mla=is_mla,
            num_q_heads=num_q_heads,
            data_parallel_degree=data_parallel_degree,
            kvcache_quant_config=kvcache_quant_config,
        )
