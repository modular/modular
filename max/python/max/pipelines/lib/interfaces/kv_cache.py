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
"""KV Cache related interfaces and protocols."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from max.driver import Device, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.interfaces import Pipeline
from max.kv_cache import (
    NullKVCacheManager,
    PagedKVCacheManager,
    load_kv_manager,
)
from max.nn.kv_cache import KVCacheParams
from transformers import AutoConfig

if TYPE_CHECKING:
    from ..config import PipelineConfig
    from ..config_enums import SupportedEncoding
    from ..kv_cache_config import KVCacheConfig


@runtime_checkable
class KVCacheMixin(Protocol):
    def load_kv_manager(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> PagedKVCacheManager | NullKVCacheManager:
        """Provided a PipelineConfig and InferenceSession, loads the KV manager.

        Args:
            session: Inference session to compile and init the KV cache.
            available_cache_memory: Amount of memory available to the KV cache,
                in bytes.

        Returns:
            Either a single KV cache manager or a tuple of KV cache managers:
            one per input modality.
        """
        model_config = pipeline_config.model_config

        # This is absolutely cursed.
        # We are converting from DeviceSpec -> Device -> DeviceRef.
        # What even is the difference between DeviceSpec and DeviceRef?
        device_specs = pipeline_config.model_config.device_specs
        devices = load_devices(device_specs)
        device_refs = [DeviceRef.from_device(d) for d in devices]

        return load_kv_manager(
            params=self.get_kv_params(
                huggingface_config=huggingface_config,
                devices=device_refs,
                kv_cache_config=model_config.kv_cache_config,
                cache_dtype=encoding.cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                pipeline_config, huggingface_config
            ),
            available_cache_memory=available_cache_memory,
            session=session,
        )

    @classmethod
    @abstractmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the pipeline model."""
        ...

    # TODO(AITLIB-265): Remove this altogether from all PipelineModels.
    @classmethod
    @abstractmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Returns the KV cache params for the pipeline model."""
        ...

    # TODO(AITLIB-265): Remove this altogether from all PipelineModels.
    @classmethod
    @abstractmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Returns the number of layers for the pipeline model."""
        ...

    @classmethod
    @abstractmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        ...


def get_paged_manager(
    pipeline: Pipeline[Any, Any],
) -> PagedKVCacheManager | None:
    """Get the paged KV cache manager from a pipeline, if available.

    Args:
        pipeline: The pipeline to extract the KV cache manager from.

    Returns:
        The paged KV cache manager if available, None otherwise.
    """
    if hasattr(pipeline, "_pipeline_model") and hasattr(
        pipeline._pipeline_model, "kv_manager"
    ):
        kv_manager = pipeline._pipeline_model.kv_manager
        # Accept standard PagedKVCacheManager
        if isinstance(kv_manager, PagedKVCacheManager):
            return kv_manager
        # Duck-type acceptance for multimodal managers exposing the same interface
        required_attrs = [
            "alloc",
            "fetch",
            "step",
            "release",
            "contains",
            "device_tensors",
            "total_num_pages",
        ]
        if all(hasattr(kv_manager, a) for a in required_attrs):
            return kv_manager

    return None
