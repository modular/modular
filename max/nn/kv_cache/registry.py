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
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from max.driver import Device
from max.engine import InferenceSession

from .cache_params import KVCacheParams, KVCacheStrategy
from .continuous_batching_cache import ContinuousBatchingKVCacheManager
from .manager import KVCacheManager
from .paged_cache import PagedKVCacheManager, PagedKVCacheManagerFA3Fallback

CACHE_MANAGER_REGISTRY: dict[KVCacheStrategy, type[KVCacheManager]] = {
    KVCacheStrategy.CONTINUOUS: ContinuousBatchingKVCacheManager,
    KVCacheStrategy.PAGED: PagedKVCacheManager,
    KVCacheStrategy.PAGED_FA3_FALLBACK: PagedKVCacheManagerFA3Fallback,
}


def load_kv_manager(
    params: KVCacheParams,
    max_batch_size: Optional[int],
    max_seq_len: int,
    num_layers: int,
    devices: Sequence[Device],
    session: InferenceSession,
    available_cache_memory: Optional[int] = None,
    page_size: Optional[int] = 512,
) -> KVCacheManager:
    assert max_batch_size is not None, "Expected max_batch_size to be set"
    assert max_batch_size > 0, "max_batch_size must be greater than 0"
    if params.cache_strategy == KVCacheStrategy.CONTINUOUS:
        return ContinuousBatchingKVCacheManager(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
        )
    elif params.cache_strategy in {
        KVCacheStrategy.PAGED,
        KVCacheStrategy.PAGED_FA3_FALLBACK,
    }:
        manager_cls: type[
            PagedKVCacheManager | PagedKVCacheManagerFA3Fallback
        ] = PagedKVCacheManager
        if params.cache_strategy == KVCacheStrategy.PAGED_FA3_FALLBACK:
            manager_cls = PagedKVCacheManagerFA3Fallback

        if page_size is None:
            msg = (
                "Missing required argument page_size for KVCacheStrategy.PAGED"
            )
            raise ValueError(msg)

        # TODO(KERN-1308) remove this validation as we generalize page_size
        if page_size % 128 != 0 or page_size < 128:
            msg = "Page size must be a multiple of 128 and at least 128."
            raise ValueError(msg)

        if available_cache_memory is None:
            msg = "Missing required argument available_cache_memory for KVCacheStrategy.PAGED"
            raise ValueError(msg)

        return manager_cls(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
            cache_memory=available_cache_memory,
            page_size=page_size,
        )
    else:
        msg = f"cache type: {params.cache_strategy} not supported."
        raise ValueError(msg)


def estimate_kv_cache_size(
    params: KVCacheParams,
    max_batch_size: Optional[int],
    max_seq_len: int,
    num_layers: int,
    available_cache_memory: int,
    devices: Sequence[Device],
    **kwargs: Any,
) -> int:
    assert max_batch_size is not None, "Expected max_batch_size to be set"
    assert max_batch_size > 0, "max_batch_size must be greater than 0"
    if params.cache_strategy not in CACHE_MANAGER_REGISTRY:
        msg = f"cache type: {params.cache_strategy} not supported."
        raise ValueError(msg)

    return CACHE_MANAGER_REGISTRY[params.cache_strategy].estimated_memory_size(
        params=params,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        available_cache_memory=available_cache_memory,
        devices=devices,
        **kwargs,
    )


def infer_optimal_batch_size(
    params: KVCacheParams,
    max_seq_len: int,
    num_layers: int,
    available_cache_memory: int,
    devices: Sequence[Device],
    **kwargs: Any,
) -> int:
    return CACHE_MANAGER_REGISTRY[
        params.cache_strategy
    ].infer_optimal_batch_size(
        params=params,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        available_cache_memory=available_cache_memory,
        devices=devices,
        **kwargs,
    )
