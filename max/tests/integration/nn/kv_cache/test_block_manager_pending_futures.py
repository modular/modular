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

"""Depth-2 pending-future correctness for BlockManager.

With schedule-ahead decoding, a request may hold TWO unrealized future-token
placeholders (one per in-flight forward). These tests drive a real
``TextContext`` through that lifecycle against a real ``BlockManager`` and pin
the KV bookkeeping invariants:

- ``_compute_seq_len`` subtracts ``max(1, pending_future_count)`` (each
  placeholder is a not-yet-run forward's input with no KV entry yet).
- Block hashing excludes every trailing placeholder; the hash chain matches a
  placeholder-free request over the same realized prefix.
- ``commit_to_prefix_cache`` never commits a block containing an unrealized
  placeholder, even though the older placeholder counts as a *processed*
  position once the second forward is enqueued behind it. (Pre-fix, crossing
  a block boundary with two placeholders raised IndexError: one more computed
  block than computed hashes.)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.pipelines.kv_cache.paged_kv_cache.block_manager import (
    BlockManager,
    _compute_seq_len,
)
from max.pipelines.modeling.types import RequestID

BLOCK_SIZE = 4


def _make_block_manager() -> BlockManager:
    return BlockManager(
        device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
        total_num_blocks=64,
        block_size=BLOCK_SIZE,
        connector=cast(object, NullConnector()),  # type: ignore[arg-type]
        enable_prefix_caching=True,
    )


def _make_context(prompt: list[int], request_id: str) -> TextContext:
    return TextContext(
        request_id=RequestID(request_id),
        max_length=64,
        tokens=TokenBuffer(np.array(prompt, dtype=np.int64)),
    )


def test_compute_seq_len_subtracts_pending_future_count() -> None:
    """Each pending placeholder is a trailing token with no KV entry."""
    ctx = _make_context(list(range(7)), "req-seq-len")

    # No placeholder: classic "last generated token has no KV entry" minus 1.
    base = _compute_seq_len(ctx, num_draft_tokens=0)
    assert base == len(ctx.tokens) + 1 - 1

    # One placeholder: identical accounting (max(1, 1) == 1), one more token.
    ctx.update_with_future_token(max_pending_futures=2)
    assert _compute_seq_len(ctx, num_draft_tokens=0) == base + 1

    # Two placeholders: the buffer grew by one token AND the exclusion grew by
    # one, so the seq_len is unchanged -- both placeholders are inputs of
    # forwards that have not written KV yet.
    ctx.update_with_future_token(max_pending_futures=2)
    assert _compute_seq_len(ctx, num_draft_tokens=0) == base + 1


def test_hashes_exclude_both_pending_placeholders() -> None:
    """The hash chain with two live placeholders equals the chain of a
    placeholder-free request over the same realized prefix (no -999 value is
    ever hashed into a block key)."""
    prompt = list(range(10, 10 + 2 * BLOCK_SIZE + 1))  # 9 tokens, 2 full blocks

    bm = _make_block_manager()
    ctx = _make_context(prompt, "req-with-futures")
    ctx.update_with_future_token(max_pending_futures=2)
    ctx.update_with_future_token(max_pending_futures=2)
    assert ctx.pending_future_count == 2
    bm.compute_hashes_for_request(ctx)

    # Reference: a plain request over the realized prefix plus enough real
    # trailing tokens to make the same number of positions hashable.
    # With 2 placeholders on 11 tokens, 9 are hashable => 2 full blocks.
    bm_ref = _make_block_manager()
    ref_ctx = cast(
        TextContext,
        SimpleNamespace(
            request_id=RequestID("req-reference"),
            tokens=np.array(prompt + [99], dtype=np.int64),
            cache_salt=None,
            pending_future_count=0,
        ),
    )
    bm_ref.compute_hashes_for_request(ref_ctx)

    hashes = bm.req_to_hashes[ctx.request_id]
    ref_hashes = bm_ref.req_to_hashes[RequestID("req-reference")]
    assert len(hashes) == 2
    assert hashes == ref_hashes


def test_commit_never_commits_placeholder_block() -> None:
    """Depth-2 commit-on-block-boundary: the block containing the older
    placeholder must not commit while the placeholder is unrealized.

    The older placeholder becomes a *processed* position as soon as the second
    forward is enqueued behind it, so the classic processed_length-based
    commit would try to commit its block: with block_size=4 and 8 processed
    positions [t0..t6, PH1], block 1 = [t4, t5, t6, PH1] -- an unrealized
    -999 in committed prefix-cache content, and no hash exists for that block
    (pre-fix: IndexError in commit_to_prefix_cache).
    """
    bm = _make_block_manager()
    ctx = _make_context(list(range(7)), "req-commit")  # 7 prompt tokens
    bm.req_to_blocks[ctx.request_id] = [
        bm.allocate_device_block() for _ in range(3)
    ]

    # Step A: forward 1 enqueued; PH1 appended. processed=7, len=8, count=1.
    ctx.update_with_future_token(max_pending_futures=2)
    bm.step(ctx)
    assert bm.req_to_committed_idx[ctx.request_id] == BLOCK_SIZE

    # Step B: schedule-ahead enqueues forward 2 before step A's token is
    # realized; PH2 appended. processed=8 (includes PH1), len=9, count=2.
    ctx.update_with_future_token(max_pending_futures=2)
    assert ctx.pending_future_count == 2
    assert ctx.tokens.processed_length == 2 * BLOCK_SIZE
    bm.step(ctx)  # pre-fix: IndexError (req_hashes[1] does not exist)

    # Block 1 spans [t4, t5, t6, PH1]; PH1 is unrealized, so nothing new
    # may commit and no hash may exist for that block yet.
    assert bm.req_to_committed_idx[ctx.request_id] == BLOCK_SIZE
    assert len(bm.req_to_hashes[ctx.request_id]) == 1

    # Sync realizes both steps oldest-first; the next scheduled step appends
    # a fresh placeholder, then the block manager steps again.
    ctx.realize_future_token(7)
    ctx.realize_future_token(8)
    ctx.update_with_future_token(max_pending_futures=2)
    bm.step(ctx)

    # Block 1 = [t4, t5, t6, 7] is now fully realized: committed, with its
    # hash computed over real token values only.
    assert bm.req_to_committed_idx[ctx.request_id] == 2 * BLOCK_SIZE
    assert len(bm.req_to_hashes[ctx.request_id]) == 2

    # Prefix-cache key/content consistency: a second request over the same
    # realized tokens computes the same chain and finds both blocks.
    ctx_b = cast(
        TextContext,
        SimpleNamespace(
            request_id=RequestID("req-reuser"),
            tokens=np.array(list(range(9)) + [99], dtype=np.int64),
            cache_salt=None,
            pending_future_count=0,
        ),
    )
    bm.compute_hashes_for_request(ctx_b)
    assert (
        bm.req_to_hashes[RequestID("req-reuser")][:2]
        == bm.req_to_hashes[ctx.request_id]
    )
    reused = bm.get_full_blocks_from_prefix_cache(ctx_b)
    assert len(reused) == 2
