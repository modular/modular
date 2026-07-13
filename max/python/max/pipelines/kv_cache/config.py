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

from __future__ import annotations

__all__ = ["KVCacheConfig", "KVConnectorConfig"]

from collections.abc import Sequence
from typing import Any

from max.config import ConfigFileModel
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache.cache_params import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    KVConnectorType,
    KVHashAlgo,
    MHAKVCacheParams,
    MLAKVCacheParams,
    SpeculativeMethod,
)
from max.pipelines.kv_cache.paged_kv_cache._seed_helpers import (
    resolve_kv_hash_seed,
)
from pydantic import ConfigDict, Field, PrivateAttr


class KVConnectorConfig(ConfigFileModel):
    """Connector-specific configuration for KV cache connectors.

    Common fields are typed. Additional connector-specific fields pass
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

    block_store_endpoint: str | None = Field(
        default=None,
        description=(
            "Endpoint for the co-located dKV service. Supports IPC "
            "(ipc:///path) or TCP (tcp://host:port). "
            "Required when kv_connector is 'dkv'."
        ),
    )
    """Endpoint for the co-located dKV service.

    Remote dKV endpoints are discovered at runtime through the
    Orchestrator (via ``external_block_metadata`` on the request
    context), not configured statically. For multi-store reads, the
    discovered metadata must include MAX-native transfer-engine metadata so
    the connector can reuse ``KVTransferEngine.connect()``.
    """


class KVCacheConfig(ConfigFileModel):
    """Configuration for the paged KV cache."""

    kv_cache_page_size: int = Field(
        default=128,
        description=(
            "The number of tokens in a single page in the paged KVCache."
        ),
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
            "Type of KV cache connector to use. "
            "When not set, defaults to ``null`` (no external caching)."
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
            "The fraction of available device memory that the process "
            "should consume. The remaining headroom holds the KV cache: "
            "``kv_cache_workspace = (total_free_memory * "
            "device_memory_utilization) - model_weights_size``."
        ),
    )
    """The fraction of available device memory the process should consume."""

    allow_kv_head_replication: bool = Field(
        default=False,
        description=(
            "Allow TP wider than the KV head count by replicating each KV head "
            "across a group of devices. Used as the default for "
            "to_params(allow_kv_head_replication=...) so it reaches base-class "
            "paths that don't thread the flag. Only for architectures whose "
            "attention shards K/V projections to match."
        ),
    )
    """Default for :meth:`to_params`'s ``allow_kv_head_replication`` argument."""

    _cache_dtype: DType = PrivateAttr(default=DType.float32)
    "The data type of the KV cache. The cache dtype is determined by the model's quantization encoding, and can be overridden from CLI by the kv_cache_format parameter."

    kv_cache_format: str | None = Field(
        default=None,
        description=(
            "Override the default data type for the KV cache. "
            "Supported values: ``float32``, ``bfloat16``, ``float8_e4m3fn``."
        ),
    )
    """An override for the default data type of the KV cache."""
    kv_cache_hash_algo: KVHashAlgo = Field(
        default="ahash64",
        description=(
            "Hash algorithm used for KV-cache block identity. "
            "``ahash64`` is the legacy 64-bit non-cryptographic hasher. "
            "``sha256`` is a 256-bit cryptographic hasher with optional "
            "per-cluster seed and per-request salt. ``sha256_64`` "
            "truncates the SHA-256 chain to 64 bits for protocol "
            "compatibility."
        ),
    )
    """Hash algorithm used for KV-cache block identity."""

    kv_cache_hash_seed: str | None = Field(
        default=None,
        description=(
            "Optional 64-character hex string (32 bytes) used as a "
            "cluster-wide seed when ``kv_cache_hash_algo`` is "
            "``sha256``/``sha256_64``. When omitted, MAX generates a "
            "random seed at process start; the hex is logged once. "
            "Ignored for ``ahash64``."
        ),
    )
    """Optional 32-byte hex seed for sha256/sha256_64 hashing."""

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
        speculative_method: SpeculativeMethod | None = None,
        num_draft_tokens: int = 0,
        allow_kv_head_replication: bool | None = None,
    ) -> KVCacheParams:
        """Returns :class:`~max.nn.kv_cache.cache_params.KVCacheParams` built from this config.

        Selects the attention-type-specific subclass: a
        :class:`~max.nn.kv_cache.cache_params.MLAKVCacheParams` when ``is_mla``
        is set, otherwise a
        :class:`~max.nn.kv_cache.cache_params.MHAKVCacheParams`.

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
            speculative_method: Speculative decoding method propagated from
                :class:`~max.pipelines.speculative.SpeculativeConfig`.
                ``None`` when speculative decoding is disabled.
            num_draft_tokens: Total draft tokens generated per
                speculative iteration. Zero when no speculative decoding.
            allow_kv_head_replication: Replicate KV heads for TP wider than the
                KV head count. Defaults to ``None`` (falls back to the config's
                :attr:`allow_kv_head_replication`).

        Returns:
            The constructed KV cache parameters.
        """
        if allow_kv_head_replication is None:
            allow_kv_head_replication = self.allow_kv_head_replication
        cfg = self.kv_connector_config
        kv_hash_seed = resolve_kv_hash_seed(
            self.kv_cache_hash_algo, self.kv_cache_hash_seed
        )
        shared_kwargs: dict[str, Any] = dict(
            dtype=dtype,
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
            data_parallel_degree=data_parallel_degree,
            kvcache_quant_config=kvcache_quant_config,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
            kv_hash_algo=self.kv_cache_hash_algo,
            kv_hash_seed=kv_hash_seed,
        )
        if is_mla:
            if num_q_heads is None:
                raise ValueError("num_q_heads is required when is_mla=True.")
            return MLAKVCacheParams(num_q_heads=num_q_heads, **shared_kwargs)
        return MHAKVCacheParams(
            n_kv_heads=n_kv_heads,
            allow_kv_head_replication=allow_kv_head_replication,
            **shared_kwargs,
        )
