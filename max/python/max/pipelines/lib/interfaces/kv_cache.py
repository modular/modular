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
"""KV Cache related interfaces and protocols."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.kv_cache import (
    PagedKVCacheManager,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.nn.legacy.kv_cache import KVCacheParams
from transformers import AutoConfig

if TYPE_CHECKING:
    from ..config import PipelineConfig
    from ..kv_cache_config import KVCacheConfig


@runtime_checkable
class KVCacheMixin(Protocol):
    def load_kv_manager(
        self,
        kv_params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> PagedKVCacheManager:
        """Provided a PipelineConfig and InferenceSession, loads the KV manager.

        Args:
            kv_params: KV cache parameters.
            max_batch_size: Maximum batch size of the model.
            max_seq_len: Maximum sequence length of the model.
            session: Inference session to compile and init the KV cache.
            available_cache_memory: Amount of memory available to the KV cache,
                in bytes.

        Returns:
            A single KV cache manager.
        """
        return load_kv_manager(
            params=kv_params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            available_cache_memory=available_cache_memory,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        huggingface_config: AutoConfig,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        available_cache_memory: int,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        return estimate_kv_cache_size(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            available_cache_memory=available_cache_memory,
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
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Returns the KV cache params for the pipeline model."""
        ...
