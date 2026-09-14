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

"""Unit test for ``DecodeScheduler.reserve_memory_and_send_to_prefill``
retrying a request whose first allocation failed.

Written against the unbound method with a mocked ``self`` and a real CPU
``PagedKVCacheManager``, so the prefix-cache reuse inside ``alloc`` is the
production code path rather than a stub.
"""

from __future__ import annotations

import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import PagedKVCacheManager
from max.serve.scheduler.decode_scheduler import DecodeScheduler
from tests.serve.scheduler.common import (
    create_kv_cache,
    create_text_context,
    rand,
)

PAGE_SIZE = 128
NUM_BLOCKS = 20
MAX_SEQ_LEN = 4096


def _make_self(
    kv_cache: PagedKVCacheManager, request_queue: queue.Queue[TextContext]
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_batch_size=8, data_parallel_degree=1
        ),
        request_queue=request_queue,
        pending_reqs=OrderedDict(),
        _admission_enqueue_time={},
        requests={},
        batch_constructor=SimpleNamespace(
            all_tg_reqs={},
            get_next_replica_idx=MagicMock(return_value=0),
            structured_output_enabled=False,
        ),
        kv_cache=kv_cache,
        prefill_reqs_per_replica=[0],
        _send_admitted_request_to_prefill=MagicMock(),
    )


def _hold_blocks(kv_cache: PagedKVCacheManager, num_tokens: int) -> TextContext:
    """Claims and allocates a request so its blocks leave the free queue."""
    ctx = create_text_context(num_tokens, MAX_SEQ_LEN)
    kv_cache.claim(ctx)
    kv_cache.alloc(ctx)
    return ctx


def _warm_prefix_cache(
    kv_cache: PagedKVCacheManager, prefix: np.ndarray
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


def test_requeued_request_is_readmitted_after_cached_prefix_eviction() -> None:
    """A request re-queued after ``InsufficientBlocksError`` must go back
    to the queue as a fresh request.

    ``alloc`` splices prefix-cache hits into the request and advances its
    token window past them before allocating the rest, so a failure after
    the splice leaves the window advanced while the release hands the
    blocks back. If the retry then finds fewer cached blocks (here: none,
    evicted by another request), ``allocate_new_blocks`` sees committed
    tokens with no blocks behind them and asserts.
    """
    kv_cache = create_kv_cache(
        num_blocks=NUM_BLOCKS,
        max_batch_size=8,
        max_seq_len=MAX_SEQ_LEN,
        page_size=PAGE_SIZE,
        enable_prefix_caching=True,
    )
    prefix = rand(2 * PAGE_SIZE)
    _warm_prefix_cache(kv_cache, prefix)
    # Leave four free blocks: the two cached prefix blocks and two others.
    filler = _hold_blocks(kv_cache, 16 * PAGE_SIZE - 1)
    assert kv_cache.block_count().free == 4

    request_queue: queue.Queue[TextContext] = queue.Queue()
    self_obj = _make_self(kv_cache, request_queue)
    # Reuses the two cached blocks, then needs three more with two free.
    ctx = create_text_context(
        2 * PAGE_SIZE + 300, MAX_SEQ_LEN, shared_prefix=prefix
    )
    request_queue.put_nowait(ctx)

    DecodeScheduler.reserve_memory_and_send_to_prefill(self_obj)  # type: ignore[arg-type]

    assert list(self_obj.pending_reqs) == [ctx.request_id]
    assert not kv_cache.contains(ctx)
    self_obj._send_admitted_request_to_prefill.assert_not_called()
    assert ctx.tokens.processed_length == 0

    # Another request takes the remaining free blocks, evicting the cached
    # prefix; the filler then releases so admission is no longer blocked.
    evictor = _hold_blocks(kv_cache, 4 * PAGE_SIZE - 1)
    kv_cache.release(filler)

    DecodeScheduler.reserve_memory_and_send_to_prefill(self_obj)  # type: ignore[arg-type]

    self_obj._send_admitted_request_to_prefill.assert_called_once()
    assert not self_obj.pending_reqs
    assert ctx.tokens.processed_length == 0
    assert ctx.tokens.active_length == len(ctx.tokens)
    assert len(kv_cache.get_req_blocks(ctx)) == 5
    kv_cache.release(evictor)
