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

from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession
from max.interfaces import Pipeline
from max.kv_cache import NullKVCacheManager
from max.kv_cache.paged_cache import PagedKVCacheManager
from max.nn.kv_cache import KVCacheParams
from transformers import AutoConfig

if TYPE_CHECKING:
    from ..config import PipelineConfig
    from ..kv_cache_config import KVCacheConfig


@runtime_checkable
class KVCacheMixin(Protocol):
    def load_kv_manager(
        self, session: InferenceSession, available_cache_memory: int | None
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
        ...

    # TODO(AITLIB-265): Remove this altogether from all PipelineModels.
    @classmethod
    @abstractmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
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
