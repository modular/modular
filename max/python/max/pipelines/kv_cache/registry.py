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
import os
from unittest.mock import MagicMock, Mock

from max.driver import is_virtual_device_mode
from max.engine import InferenceSession
from max.nn.kv_cache import (
    KVCacheParamInterface,
    compute_max_seq_len_fitting_in_cache,
    compute_num_device_blocks,
    recurrent_leaf,
)

from .paged_kv_cache import PagedKVCacheManager
from .paged_kv_cache.cache_manager_interface import PagedKVCacheManagerInterface
from .paged_kv_cache.jenga_cache_manager import JengaKVCacheManager

logger = logging.getLogger("max.pipelines")

# Temporary allowlist for the Jenga cutover.
_JENGA_MODEL_NAME_SUBSTRINGS = (
    "llama",  # Llama 3/4: full
    "gemma",  # Gemma 3/4: full + SWA
    "gpt-oss",  # full + SWA
    "gptoss",  # same as gpt-oss
    "olmo",  # Olmo 2: full; Olmo 3: 3:1 SWA:full
    "step-3",  # Step-3.5: full + SWA
    "step3",  # same as step-3
    "inkling",  # full + SWA
    "kimi",  # Kimi K2.x / Kimi-VL: MLA full (Eagle3 draft adds an SWA group)
    "deepseek",  # DeepSeek V2/V3: MLA full; V3.2 adds a sparse-indexer group
)


def _use_jenga_kv_cache(
    params: KVCacheParamInterface, is_di_enabled: bool, model_name: str
) -> bool:
    """Whether the Jenga KV cache serves this cache.

    A recurrent state can only live on Jenga, so it overrides the allowlist.
    `MODULAR_USE_LEGACY_KV_CACHE=1` otherwise forces the legacy KV cache.

    TODO: temporary flag for the Jenga cutover. Delete once the transition is complete.
    """
    if recurrent_leaf(params) is not None:
        return True
    name = model_name.lower()
    if not any(token in name for token in _JENGA_MODEL_NAME_SUBSTRINGS):
        return False
    prefer_legacy = os.getenv("MODULAR_USE_LEGACY_KV_CACHE", "0").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    if prefer_legacy:
        return False
    # TODO(SERVOPT-1590): Jenga is incompatible with Disaggregated Inference.
    return not is_di_enabled


def max_seq_len_fitting_in_cache(
    params: KVCacheParamInterface,
    available_cache_memory: int,
    is_di_enabled: bool,
    model_name: str,
) -> int | None:
    """Returns the longest request the manager for this cache holds.

    The cost of a request depends on the manager: the legacy pool charges
    every leaf a page per slot, Jenga charges each leaf only what it retains.

    Returns:
        The longest sequence, or ``None`` when no length exhausts the cache.
    """
    if _use_jenga_kv_cache(params, is_di_enabled, model_name):
        return JengaKVCacheManager.max_seq_len_fitting_in_cache(
            params, available_cache_memory
        )
    return compute_max_seq_len_fitting_in_cache(
        params=params,
        available_cache_memory=available_cache_memory,
        include_null_block=True,
    )


def load_kv_manager(
    params: KVCacheParamInterface,
    max_batch_size: int | None,
    max_seq_len: int,
    session: InferenceSession,
    available_cache_memory: int | None,
    is_di_enabled: bool,
    model_name: str,
    max_num_input_tokens: int | None,
) -> PagedKVCacheManagerInterface:
    """Loads a KV cache manager from the given params.

    Accepts both ``KVCacheParams`` (single cache) and ``MultiKVCacheParams``
    (multiple caches).  The returned manager natively handles all caches
    with a single ``BlockManager`` and ``KVConnector``.

    TODO: remove `is_di_enabled` once Jenga supports DI.
    """
    if isinstance(params, MagicMock):
        return MagicMock()

    # In compile-only mode (virtual device mode), use the null KV manager
    # to avoid GPU memory allocation
    if is_virtual_device_mode():
        logger.info(
            "Detected compile-only mode, Use fake KVCache to avoid GPU allocation"
        )
        return Mock()

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

    # TODO(KERN-1308) remove this validation as we generalize page_size
    if params.page_size % 128 != 0 or params.page_size < 128:
        raise ValueError(
            "Page size must be a multiple of 128 and at least 128."
        )

    if _use_jenga_kv_cache(params, is_di_enabled, model_name):
        if recurrent_leaf(params) is not None:
            logger.info(
                "Using Jenga KV cache: this model keeps recurrent state, whose"
                " pages share the KV budget and eviction order with its KV."
            )
        else:
            logger.info(
                "Using Jenga KV cache. To fall back to using the legacy KV"
                " cache, set MODULAR_USE_LEGACY_KV_CACHE=1"
            )
        return JengaKVCacheManager.create(
            params=params,
            available_bytes=available_cache_memory,
            max_batch_size=max_batch_size,
            max_num_input_tokens=max_num_input_tokens,
            max_seq_len=max_seq_len,
        )

    logger.info("Using legacy KV cache")
    # A single request at max_seq_len must fit in the device block pool:
    # otherwise it cannot be preempted (there is nothing else to evict) and
    # overflows the pool at runtime, crashing the model worker with
    # InsufficientBlocksError. Fail startup instead.
    total_num_pages = compute_num_device_blocks(
        params=params,
        available_cache_memory=available_cache_memory,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        require_max_seq_len_fits=True,
    )

    return PagedKVCacheManager(
        params=params,
        total_num_pages=total_num_pages,
        session=session,
        max_batch_size=max_batch_size,
    )
