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
from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from typing import Any
from unittest.mock import Mock

from max.driver import Device, is_virtual_device_mode
from max.engine import InferenceSession
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    MultiKVCacheParams,
    compute_num_device_blocks,
    estimated_memory_size,
)
from max.nn.legacy.kv_cache.cache_params import KVCacheParamInterface

from .paged_kv_cache import PagedKVCacheManager

logger = logging.getLogger("max.pipelines")

CACHE_MANAGER_REGISTRY: dict[KVCacheStrategy, type[PagedKVCacheManager]] = {
    KVCacheStrategy.PAGED: PagedKVCacheManager,
}


def _load_single_kv_manager(
    params: KVCacheParams,
    total_num_pages: int,
    session: InferenceSession,
) -> PagedKVCacheManager:
    # In compile-only mode (virtual device mode), use the null KV manager
    # to avoid GPU memory allocation
    if is_virtual_device_mode():
        logger.info(
            "Detected compile-only mode, Use fake KVCache to avoid GPU allocation"
        )
        return Mock()

    if params.cache_strategy != KVCacheStrategy.PAGED:
        raise ValueError(
            f"Found unsupported KVCache strategy: {params.cache_strategy}"
        )

    # TODO(KERN-1308) remove this validation as we generalize page_size
    if params.page_size % 128 != 0 or params.page_size < 128:
        raise ValueError(
            "Page size must be a multiple of 128 and at least 128."
        )

    return PagedKVCacheManager(
        params=params,
        total_num_pages=total_num_pages,
        total_num_host_pages=params.compute_num_host_blocks(),
        session=session,
    )


def load_kv_manager(
    params: KVCacheParamInterface,
    max_batch_size: int,
    max_seq_len: int,
    session: InferenceSession,
    available_cache_memory: int,
) -> PagedKVCacheManager:
    """Loads a single KV cache manager from the given params."""
    if not isinstance(params, KVCacheParams):
        raise TypeError("params must be a KVCacheParams, got: {type(params)}")

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be greater than 0")

    total_num_pages = compute_num_device_blocks(
        params=params,
        available_cache_memory=available_cache_memory,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )

    return _load_single_kv_manager(params, total_num_pages, session)


def _load_kv_managers(
    params: KVCacheParamInterface,
    total_num_pages: int,
    session: InferenceSession,
) -> list[PagedKVCacheManager]:
    if isinstance(params, KVCacheParams):
        return [_load_single_kv_manager(params, total_num_pages, session)]
    elif isinstance(params, MultiKVCacheParams):
        return list(
            itertools.chain.from_iterable(
                _load_kv_managers(p, total_num_pages, session)
                for p in params.params
            )
        )
    else:
        raise TypeError(
            f"params must be a KVCacheParams or MultiKVCacheParams, got: {type(params)}"
        )


def load_kv_managers(
    params: KVCacheParamInterface,
    max_batch_size: int,
    max_seq_len: int,
    session: InferenceSession,
    available_cache_memory: int,
) -> list[PagedKVCacheManager]:
    """Loads (potentially multiple) KV cache managers from the given params."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be greater than 0")
    total_num_pages = compute_num_device_blocks(
        params=params,
        available_cache_memory=available_cache_memory,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    return _load_kv_managers(params, total_num_pages, session)


def estimate_kv_cache_size(
    params: KVCacheParamInterface,
    max_batch_size: int,
    max_seq_len: int,
    available_cache_memory: int,
) -> int:
    """Estimates the KV cache size in bytes for the given params and constraints."""
    assert max_batch_size > 0, "max_batch_size must be greater than 0"

    return estimated_memory_size(
        params=params,
        available_cache_memory=available_cache_memory,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )


def infer_optimal_batch_size(
    params: KVCacheParamInterface,
    max_seq_len: int,
    available_cache_memory: int,
    devices: Sequence[Device],
    **kwargs: Any,
) -> int:
    """Infers the optimal batch size for the cache strategy and constraints."""
    return CACHE_MANAGER_REGISTRY[
        params.cache_strategy
    ].infer_optimal_batch_size(
        params=params,
        max_seq_len=max_seq_len,
        available_cache_memory=available_cache_memory,
        devices=devices,
        **kwargs,
    )
