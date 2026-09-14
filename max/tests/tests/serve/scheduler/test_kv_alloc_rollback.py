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

"""A KV cache ``alloc`` that fails must leave the request as it found it.

``alloc`` splices prefix-cache hits into the request and advances its token
window past them before allocating blocks for the rest. Both cache managers
must undo that splice when the second step raises, so a caller that releases
and retries the request does not carry a window that claims blocks the
request no longer holds.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import PagedKVCacheManagerInterface
from max.pipelines.kv_cache.paged_kv_cache.block_utils import (
    InsufficientBlocksError,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_cache_manager import (
    JengaKVCacheManager,
)
from tests.serve.scheduler.common import (
    create_kv_cache,
    create_text_context,
    rand,
)

MAX_SEQ_LEN = 8192


def _warm_prefix_cache(
    kv_cache: PagedKVCacheManagerInterface, prefix: np.ndarray
) -> None:
    """Runs one request over ``prefix`` so its full blocks are cached."""
    ctx = create_text_context(
        len(prefix) + 1, MAX_SEQ_LEN, shared_prefix=prefix
    )
    kv_cache.claim(ctx)
    kv_cache.alloc(ctx)
    ctx.update(new_token=1)
    kv_cache.step(ctx)
    kv_cache.release(ctx)


def _claim_and_alloc(
    kv_cache: PagedKVCacheManagerInterface,
    num_tokens: int,
    prefix: np.ndarray | None = None,
) -> TextContext:
    ctx = create_text_context(num_tokens, MAX_SEQ_LEN, shared_prefix=prefix)
    kv_cache.claim(ctx)
    kv_cache.alloc(ctx)
    return ctx


def _assert_untouched(
    kv_cache: PagedKVCacheManagerInterface, ctx: TextContext
) -> None:
    assert ctx.tokens.processed_length == 0
    assert ctx.tokens.active_length == len(ctx.tokens)
    assert ctx.cached_prefix_length == 0
    assert ctx.cached_prefix_external_length == 0
    assert kv_cache.contains(ctx)
    assert kv_cache.get_req_blocks(ctx) == []


def test_paged_alloc_failure_rolls_back_prefix_reuse() -> None:
    page_size = 128
    kv_cache = create_kv_cache(
        num_blocks=20,
        max_batch_size=8,
        max_seq_len=MAX_SEQ_LEN,
        page_size=page_size,
        enable_prefix_caching=True,
    )
    prefix = rand(2 * page_size)
    _warm_prefix_cache(kv_cache, prefix)
    # Leave four free blocks: the two cached prefix blocks and two others.
    filler = _claim_and_alloc(kv_cache, 16 * page_size - 1)
    free_before = kv_cache.block_count().free
    assert free_before == 4

    # Reuses the two cached blocks, then needs three more with two free.
    ctx = create_text_context(
        2 * page_size + 300, MAX_SEQ_LEN, shared_prefix=prefix
    )
    kv_cache.claim(ctx)
    with pytest.raises(InsufficientBlocksError):
        kv_cache.alloc(ctx)

    _assert_untouched(kv_cache, ctx)
    assert kv_cache.block_count().free == free_before

    # The cached prefix survives the rollback and serves the retry.
    kv_cache.release(filler)
    assert kv_cache.alloc(ctx).is_complete()
    assert ctx.tokens.processed_length == 2 * page_size
    assert ctx.cached_prefix_length == 2 * page_size
    assert len(kv_cache.get_req_blocks(ctx)) == 5


def _create_jenga_kv_cache(
    page_size: int, num_huge_pages: int
) -> JengaKVCacheManager:
    params = MultiKVCacheParams.from_params(
        {
            "sliding": MHAKVCacheParams(
                dtype=DType.float32,
                num_layers=1,
                n_kv_heads=1,
                head_dim=16,
                page_size=page_size,
                window_size=25,
                enable_prefix_caching=True,
                devices=[DeviceRef.CPU()],
            ),
            "full": MHAKVCacheParams(
                dtype=DType.float32,
                num_layers=1,
                n_kv_heads=1,
                head_dim=1,
                page_size=page_size,
                enable_prefix_caching=True,
                devices=[DeviceRef.CPU()],
            ),
        }
    )
    huge_page_bytes = max(
        leaf.bytes_per_page for leaf in params.leaves().values()
    )
    # No max_seq_len: the pool must be allowed to be smaller than a request.
    return JengaKVCacheManager.create(
        params=params,
        available_bytes=num_huge_pages * huge_page_bytes,
        max_batch_size=8,
    )


def test_jenga_alloc_failure_rolls_back_prefix_reuse() -> None:
    page_size = 10
    kv_cache = _create_jenga_kv_cache(page_size, num_huge_pages=8)
    prefix = rand(2 * page_size)
    _warm_prefix_cache(kv_cache, prefix)
    free_before = kv_cache.block_count().free

    # Reuses the two cached blocks, then asks for more than the pool holds.
    ctx = create_text_context(
        2 * page_size + 5000, MAX_SEQ_LEN, shared_prefix=prefix
    )
    kv_cache.claim(ctx)
    with pytest.raises(InsufficientBlocksError):
        kv_cache.alloc(ctx)

    _assert_untouched(kv_cache, ctx)
    assert kv_cache.block_count().free == free_before
    kv_cache.release(ctx)

    # The cached prefix survives the rollback and serves the next request.
    fitting = _claim_and_alloc(kv_cache, 2 * page_size + 5, prefix=prefix)
    assert fitting.tokens.processed_length == 2 * page_size
    assert fitting.cached_prefix_length == 2 * page_size
