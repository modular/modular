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
from dataclasses import replace
from unittest.mock import MagicMock, Mock

from max.driver import Buffer, is_virtual_device_mode
from max.engine import InferenceSession
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVConnectorType,
    MultiKVCacheParams,
    compute_num_device_blocks,
    compute_num_host_blocks,
)

from .paged_kv_cache import PagedKVCacheManager

logger = logging.getLogger("max.pipelines")


def _load_single_kv_manager(
    params: KVCacheParamInterface,
    total_num_pages: int,
    total_num_host_pages: int,
    session: InferenceSession,
    max_batch_size: int,
    other_kv_managers_device_buffers_per_replica: list[list[Buffer]]
    | None = None,
) -> PagedKVCacheManager:
    # In compile-only mode (virtual device mode), use the null KV manager
    # to avoid GPU memory allocation
    if is_virtual_device_mode():
        logger.info(
            "Detected compile-only mode, Use fake KVCache to avoid GPU allocation"
        )
        return Mock()

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
        other_kv_managers_device_buffers_per_replica=other_kv_managers_device_buffers_per_replica,
    )


def load_kv_manager(
    params: KVCacheParamInterface,
    max_batch_size: int | None,
    max_seq_len: int,
    session: InferenceSession,
    available_cache_memory: int | None,
) -> PagedKVCacheManager:
    """Loads a KV cache manager from the given params.

    Accepts both ``KVCacheParams`` (single cache) and ``MultiKVCacheParams``
    (multiple caches).  The returned ``PagedKVCacheManager`` natively handles
    all caches with a single ``BlockManager`` and ``KVConnector``.
    """
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

    total_num_host_pages = compute_num_host_blocks(params)

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
    """Loads a list of KV cache managers from the given params.

    .. deprecated::
        Use ``load_kv_manager(params)`` instead, which returns a single
        ``PagedKVCacheManager`` that natively handles multiple caches.
        This function is retained for backward compatibility with
        speculative decoding.
    """
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

    total_num_host_pages = compute_num_host_blocks(params)

    param0 = params.params[0]
    if param0.kv_connector is None:
        return [
            _load_single_kv_manager(
                params=p,
                total_num_pages=total_num_pages,
                total_num_host_pages=total_num_host_pages,
                session=session,
                max_batch_size=max_batch_size,
            )
            for p in params.params
        ]

    # What follows is a terrible terrible hack to get KVCache offloading via
    # KVConnector to compose with speculative decoding. The right solution is to
    # have a unified KVCache that handles both the target and draft models
    # KVCaches buffers. Lets fix this ASAP.
    if param0.kv_connector not in (
        KVConnectorType.local,
        KVConnectorType.tiered,
    ):
        raise ValueError(
            f"Only local and tiered connectors are supported for multi-cache models. Found {param0.kv_connector}"
        )

    # HACK ALERT: Disable KVConnector for all but the first cache.
    kv_managers_1_to_n = [
        _load_single_kv_manager(
            params=replace(p, kv_connector=None),
            total_num_pages=total_num_pages,
            total_num_host_pages=total_num_host_pages,
            session=session,
            max_batch_size=max_batch_size,
        )
        for p in params.params[1:]
    ]

    dp = param0.data_parallel_degree
    other_kv_managers_device_buffers_per_replica: list[list[Buffer]] = [
        [] for _ in range(dp)
    ]

    for kv_manager_i in kv_managers_1_to_n:
        for replica_idx in range(dp):
            buffers = kv_manager_i.get_device_buffer(replica_idx).all_buffers
            other_kv_managers_device_buffers_per_replica[replica_idx].extend(
                buffers
            )

    kv_manager_0 = _load_single_kv_manager(
        params=param0,
        total_num_pages=total_num_pages,
        total_num_host_pages=total_num_host_pages,
        session=session,
        max_batch_size=max_batch_size,
        # Smuggle the other kv managers' device buffers to the first kv manager
        # for use in its KVConnector.
        other_kv_managers_device_buffers_per_replica=other_kv_managers_device_buffers_per_replica,
    )
    return [kv_manager_0] + kv_managers_1_to_n
