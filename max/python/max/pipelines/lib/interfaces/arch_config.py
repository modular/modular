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
"""Architecture-specific config interfaces.

The `ArchConfig` class is not to be confused with the following classes:
- PipelineConfig: Parameters that are relevant to the entire pipeline and are
  generally passed in from the top level (MAX Serve, entrypoints). For example,
  the max_batch_size, max_length, etc.
- MAXModelConfig: Model-related parameters that may be defined at the top level.
  This class is used as an organizational layer for the PipelineConfig that can
  be accessed using the `pipeline_config.model` attribute.
  Parameters include the model_path, device_specs, quantization_encoding, etc.

The architecture-specific config is defined during the startup phase, from a
PipelineConfig object.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from max.driver import load_devices, scan_available_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.lib.utils import upper_bounded_default
from typing_extensions import Self, override

from ..kv_cache_config import KVCacheConfig

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig
    from transformers import AutoConfig


@runtime_checkable
class ArchConfig(Protocol):
    """Config for a model architecture."""

    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initialize the config from a PipelineConfig."""

    def get_max_seq_len(self) -> int:
        """Returns the default maximum sequence length for the model.

        Subclasses should determine whether this value can be overridden by
        setting the ``--max-length`` (``pipeline_config.max_length``) flag.
        """


@runtime_checkable
class ArchConfigWithKVCache(ArchConfig, Protocol):
    """Config for a model architecture that uses a KV cache."""

    def get_kv_params(self) -> KVCacheParamInterface:
        """KV cache parameters to use when running the model."""


def _all_available_devices() -> list[DeviceRef]:
    return [
        DeviceRef.from_device(device)
        for device in load_devices(scan_available_devices())
    ]


@dataclass
class ArchConfigWithAttentionKVCache(ArchConfigWithKVCache, abc.ABC):
    """Predefined configuration for architectures that use attention KV cache blocks.

    Subclasses must define the following attributes:
    - num_key_value_heads: int
    - head_dim: int
    - num_layers: int
    - model_max_seq_len: int
    """

    dtype: DType
    """The data type to use for the model."""
    devices: list[DeviceRef] = field(default_factory=_all_available_devices)
    """The physical devices to use when running the model."""
    cache_dtype: DType | None = None
    """The data type to use for the KV cache."""
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    """The KV cache configuration to use when running the model."""
    data_parallel_degree: int = 1
    """The data parallel degree to use when running the model."""
    user_provided_max_length: int | None = None
    """Override for the maximum sequence length."""

    huggingface_config: AutoConfig | None = None

    _kv_params: KVCacheParams | None = None

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        if pipeline_config.model.quantization_encoding is None:
            raise ValueError(
                "Quantization encoding is required for ArchConfigWithAttentionKVCache"
            )
        return cls(
            dtype=pipeline_config.model.quantization_encoding.dtype,
            devices=[
                DeviceRef(device_type=d.device_type, id=d.id)
                for d in pipeline_config.model.device_specs
            ],
            cache_dtype=pipeline_config.model.kv_cache.cache_dtype,
            kv_cache=pipeline_config.model.kv_cache,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            user_provided_max_length=pipeline_config.max_length,
            huggingface_config=pipeline_config.model.huggingface_config,
        )

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length the model can process.

        Returns ``max_length`` if set, otherwise ``model_max_seq_len``.
        Raises ValueError if ``max_length`` exceeds ``model_max_seq_len``.
        """
        return upper_bounded_default(
            upper_bound=self.model_max_seq_len,
            default=self.user_provided_max_length,
        )

    def get_kv_params(self) -> KVCacheParams:
        """Returns the KV cache parameters for this architecture."""
        if self._kv_params is not None:
            return self._kv_params
        self._kv_params = self.kv_cache.to_params(
            dtype=self.cache_dtype or self.dtype,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_layers=self.num_layers,
            devices=self.devices,
            data_parallel_degree=self.data_parallel_degree,
        )
        return self._kv_params

    @property
    @abc.abstractmethod
    def num_key_value_heads(self) -> int:
        """Number of key-value heads to use for the KV cache."""
        pass

    @property
    @abc.abstractmethod
    def head_dim(self) -> int:
        """Dimensionality of each attention head."""
        pass

    @property
    @abc.abstractmethod
    def num_layers(self) -> int:
        """Number of hidden layers in the model."""
        pass

    @property
    @abc.abstractmethod
    def model_max_seq_len(self) -> int:
        """The maximum sequence length that can be processed by the model."""
        pass
