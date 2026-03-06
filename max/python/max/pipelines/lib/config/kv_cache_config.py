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
)
from pydantic import Field, PrivateAttr


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

    enable_kvcache_swapping_to_host: bool = Field(
        default=False,
        description=(
            "Whether to swap paged KVCache blocks to host memory when device "
            "blocks are evicted."
        ),
    )
    """Whether to swap paged KV cache blocks to host memory when device blocks are evicted."""

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

    host_kvcache_swap_space_gb: float = Field(
        default=50.0,
        description=(
            "The amount of host memory to use for the host KVCache in GiB. "
            "This space is only allocated when kvcache_swapping_to_host is "
            "enabled."
        ),
    )
    """The amount of host memory to use for the host KV cache, in GiB."""

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

    disk_offload_dir: str | None = Field(
        default=None,
        description=(
            "Directory for disk-based KV cache offloading. When set (together "
            "with kvcache_swapping_to_host), blocks are written through from "
            "CPU to disk for persistence across restarts."
        ),
    )
    """The directory for disk-based KV cache offloading."""

    disk_offload_max_gb: float = Field(
        default=50.0,
        description="Maximum disk space (GB) for KV cache offloading.",
    )
    """The maximum disk space in GB for KV cache offloading."""

    disk_offload_direct_io: bool = Field(
        default=False,
        description=(
            "Use O_DIRECT for disk I/O (bypasses OS page cache). "
            "Requires block sizes aligned to the filesystem block size. "
            "Falls back to buffered I/O if alignment is not met."
        ),
    )
    """Whether to use ``O_DIRECT`` for disk I/O, bypassing the OS page cache."""

    lmcache_config_file: str | None = Field(
        default=None,
        description=(
            "Path to an LMCache YAML configuration file. When set, enables "
            "LMCache-based external KV cache tiering (CPU, disk, remote)."
        ),
    )
    """The path to an LMCache YAML configuration file."""

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
        q_max_seq_len: int = 1,
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
            q_max_seq_len: Query tokens per sequence during decode (1 for
                standard decode, >1 for MTP).
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
            enable_prefix_caching=self.enable_prefix_caching,
            enable_kvcache_swapping_to_host=self.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=self.host_kvcache_swap_space_gb,
            devices=devices,
            is_mla=is_mla,
            num_q_heads=num_q_heads,
            q_max_seq_len=q_max_seq_len,
            data_parallel_degree=data_parallel_degree,
            kvcache_quant_config=kvcache_quant_config,
            disk_offload_dir=self.disk_offload_dir,
            disk_offload_max_gb=self.disk_offload_max_gb,
            disk_offload_direct_io=self.disk_offload_direct_io,
            lmcache_config_file=self.lmcache_config_file,
        )
