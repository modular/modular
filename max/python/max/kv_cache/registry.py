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

import logging
from collections.abc import Sequence
from typing import Any

from max.driver import Device, is_virtual_device_mode
from max.engine import InferenceSession
from max.nn.kv_cache.cache_params import KVCacheParams, KVCacheStrategy

from .null_cache_manager import NullKVCacheManager
from .paged_cache import PagedKVCacheManager

logger = logging.getLogger("max.pipelines")

CACHE_MANAGER_REGISTRY: dict[KVCacheStrategy, type[PagedKVCacheManager]] = {
    KVCacheStrategy.PAGED: PagedKVCacheManager,
}


def load_kv_manager(
    params: KVCacheParams,
    max_batch_size: int | None,
    max_seq_len: int,
    devices: Sequence[Device],
    session: InferenceSession,
    available_cache_memory: int | None = None,
) -> PagedKVCacheManager | NullKVCacheManager:
    assert max_batch_size is not None, "Expected max_batch_size to be set"
    assert max_batch_size > 0, "max_batch_size must be greater than 0"

    # In compile-only mode (virtual device mode), use the null KV manager
    # to avoid GPU memory allocation
    if is_virtual_device_mode():
        logger.info(
            "Detected compile-only mode, using NullKVCacheManager to avoid GPU allocation"
        )
        return NullKVCacheManager(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            devices=devices,
            session=session,
        )

    if params.cache_strategy == KVCacheStrategy.PAGED:
        page_size = params.page_size
        # TODO(KERN-1308) remove this validation as we generalize page_size
        if page_size % 128 != 0 or page_size < 128:
            raise ValueError(
                "Page size must be a multiple of 128 and at least 128."
            )

        if available_cache_memory is None:
            raise ValueError(
                "Missing required argument available_cache_memory for KVCacheStrategy.PAGED"
            )

        return PagedKVCacheManager(
            params=params,
            total_num_pages=params.compute_num_device_blocks(
                available_cache_memory=available_cache_memory,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
            ),
            total_num_host_pages=params.compute_num_host_blocks(),
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            devices=devices,
            session=session,
        )
    else:
        raise ValueError(f"cache type: {params.cache_strategy} not supported.")


def estimate_kv_cache_size(
    params: KVCacheParams,
    max_batch_size: int | None,
    max_seq_len: int,
    available_cache_memory: int,
) -> int:
    assert max_batch_size is not None, "Expected max_batch_size to be set"
    assert max_batch_size > 0, "max_batch_size must be greater than 0"

    return params.estimated_memory_size(
        available_cache_memory=available_cache_memory,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
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
