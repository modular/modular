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

"""Defines a central registry mapping KV cache strategies to their manager implementations."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock, Mock

from max.driver import Device, is_virtual_device_mode
from max.engine import InferenceSession
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    MultiKVCacheParams,
    compute_num_device_blocks,
    estimated_memory_size,
)
from max.nn.kv_cache.cache_params import KVCacheParamInterface

from .paged_kv_cache import PagedKVCacheManager

logger = logging.getLogger("max.pipelines")

CACHE_MANAGER_REGISTRY: dict[KVCacheStrategy, type[PagedKVCacheManager]] = {
    "paged": PagedKVCacheManager,
}


def _load_single_kv_manager(
    params: KVCacheParams,
    total_num_pages: int,
    total_num_host_pages: int,
    session: InferenceSession,
    max_batch_size: int,
) -> PagedKVCacheManager:
    # In compile-only mode (virtual device mode), use the null KV manager
    # to avoid GPU memory allocation
    if is_virtual_device_mode():
        logger.info(
            "Detected compile-only mode, Use fake KVCache to avoid GPU allocation"
        )
        return Mock()

    if params.cache_strategy != "paged":
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
        total_num_host_pages=total_num_host_pages,
        session=session,
        max_batch_size=max_batch_size,
    )


def load_kv_manager(
    params: KVCacheParams,
    max_batch_size: int | None,
    max_seq_len: int,
    session: InferenceSession,
    available_cache_memory: int | None,
) -> PagedKVCacheManager:
    """Loads a single KV cache manager from the given params."""
    # FIXME: This is very very cursed. We can fix this by making `load_kv_manager`
    # a method on the KVCacheParams class.
    if isinstance(params, MagicMock):
        return MagicMock()

    if available_cache_memory is None:
        raise ValueError(
            "available_cache_memory should have been set during memory estimation"
        )

    if max_batch_size is None:
        raise ValueError(
            "max_batch_size should have been set during memory estimation"
        )

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be greater than 0")

    total_num_pages = compute_num_device_blocks(
        params=params,
        available_cache_memory=available_cache_memory,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )

    total_num_host_pages = params.compute_num_host_blocks()

    return _load_single_kv_manager(
        params=params,
        total_num_pages=total_num_pages,
        total_num_host_pages=total_num_host_pages,
        session=session,
        max_batch_size=max_batch_size,
    )


def load_multi_kv_managers(
    params: MultiKVCacheParams,
    max_batch_size: int | None,
    max_seq_len: int,
    session: InferenceSession,
    available_cache_memory: int | None,
) -> list[PagedKVCacheManager]:
    """Loads a list of KV cache managers from the given params."""
    # FIXME: This is very very cursed. We can fix this by making `load_multi_kv_managers`
    # a method on the MultiKVCacheParams class.
    if isinstance(params, MagicMock):
        return [MagicMock() for _ in range(len(params.params))]

    if available_cache_memory is None:
        raise ValueError(
            "available_cache_memory should have been set during memory estimation"
        )

    if max_batch_size is None:
        raise ValueError(
            "max_batch_size should have been set during memory estimation"
        )

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be greater than 0")

    total_num_pages = compute_num_device_blocks(
        params=params,
        available_cache_memory=available_cache_memory,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )

    # assume all params have the same number of host pages
    total_num_host_pages = params.params[0].compute_num_host_blocks()

    return [
        _load_single_kv_manager(
            params=params,
            total_num_pages=total_num_pages,
            total_num_host_pages=total_num_host_pages,
            session=session,
            max_batch_size=max_batch_size,
        )
        for params in params.params
    ]


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
