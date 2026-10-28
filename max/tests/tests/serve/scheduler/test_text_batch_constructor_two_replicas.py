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
"""Batch construction across two data-parallel replicas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from max.nn.legacy.kv_cache import KVCacheGroupId, PagedKVLeafRegion
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.kv_connector import (
    BlockCount,
    CompletedTransfer,
    KVLoadFailed,
    KVLookup,
    KVTransfer,
)
from max.pipelines.kv_cache.paged_kv_cache import (
    PrefixCacheHits,
    jenga_block_manager,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_manager import (
    JengaBlockManager,
    KVLeafInfo,
    create_groups,
    create_pools,
)
from max.pipelines.modeling.types import RequestID
from max.serve.scheduler.batch_constructor import text_batch_constructor
from max.serve.scheduler.batch_constructor.text_batch_constructor import (
    LOOKUP_WAIT_ENV_VAR,
    TextBatchConstructor,
)
from max.serve.scheduler.config import TokenGenerationSchedulerConfig


class _IncompleteOnload:
    def __init__(self) -> None:
        self.complete = False
        self.fails = False

    @property
    def g0_blocks_per_leaf(self) -> Mapping[str, Sequence[int]]:
        return {}

    def is_complete(self) -> bool:
        if self.fails:
            raise KVLoadFailed("memory transfer failed")
        return self.complete

    def synchronize(self) -> None:
        self.complete = True


class _AnsweredLookup:
    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks

    def is_complete(self) -> bool:
        return True

    def masks(self) -> Mapping[str, Sequence[bool]]:
        return {"full": [True] * self._num_blocks}


class _NeverLookup:
    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks

    def is_complete(self) -> bool:
        return False

    def masks(self) -> Mapping[str, Sequence[bool]]:
        raise RuntimeError("never answers")


class _Connector(NullConnector):
    """Answers a hit on every block while ``answer_hits``; else never."""

    def __init__(self) -> None:
        self.answer_hits = True
        self.onloads: list[_IncompleteOnload] = []
        self.started: list[KVLookup] = []

    @property
    def name(self) -> str:
        return "HitOrNeverConnector"

    def start_lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVLookup:
        lookup: KVLookup = (
            _AnsweredLookup(len(block_hashes))
            if self.answer_hits
            else _NeverLookup(len(block_hashes))
        )
        self.started.append(lookup)
        return lookup

    def defers_lookup(self, replica_idx: int = 0) -> bool:
        return True

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        onload = _IncompleteOnload()
        self.onloads.append(onload)
        return onload

    def abandon_lookup(self, lookup: KVLookup) -> None:
        return


class _FakeBuf:
    def __init__(self, replica: int) -> None:
        self.replica = replica


class _FakeMem:
    def __init__(self, replica: int) -> None:
        self.buffers = [_FakeBuf(replica)]
        self.bytes_per_page = 1


def _ctx(tokens: list[int], max_length: int = 100) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(np.array(tokens, dtype=np.int64)),
    )


def _manager(
    connector: NullConnector, num_blocks: int, num_replicas: int
) -> JengaBlockManager:
    leaf_infos = {"full": KVLeafInfo(1, KVCacheGroupId.full())}
    pools = create_pools(leaf_infos, num_blocks, num_replicas=num_replicas)
    return JengaBlockManager(
        pools=pools,
        groups=create_groups(leaf_infos, pools, 1, enable_prefix_caching=True),
        leaves={
            "full": PagedKVLeafRegion(
                leaf_id="full",
                group_id=KVCacheGroupId.full(),
                bytes_per_page=1,
                page_size=1,
            )
        },
        block_size=1,
        enable_prefix_caching=True,
        max_num_input_tokens=None,
        num_draft_tokens=0,
        num_draft_tokens_per_step=0,
        connector=connector,
        replica_kv_memory=cast(
            Any, [{"full": _FakeMem(i)} for i in range(num_replicas)]
        ),
        enable_dp_cross_replica_prefix_copy=True,
    )


def _kv_cache_over(manager: JengaBlockManager, log: list[object]) -> Mock:
    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.params = Mock()
    kv_cache.params.page_size = 1

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        log.append(("alloc", ctx.request_id))
        return manager.alloc(ctx, defer_lookup=defer_lookup)

    kv_cache.alloc = Mock(side_effect=alloc)
    kv_cache.claim = Mock(side_effect=manager.claim)
    kv_cache.contains = Mock(side_effect=manager.contains)
    kv_cache.release = Mock(side_effect=manager.release)
    kv_cache.starts_lookup = Mock(side_effect=manager.starts_lookup)
    kv_cache.start_lookup = Mock(side_effect=manager.start_lookup)
    kv_cache.awaits_lookup = Mock(side_effect=manager.awaits_lookup)
    kv_cache.hand_back_lookup = Mock(side_effect=manager.hand_back_lookup)
    kv_cache.active_length_after_alloc = Mock(
        side_effect=manager.active_length_after_alloc
    )
    kv_cache.num_req_blocks = Mock(side_effect=manager.num_req_blocks)
    kv_cache.poll_transfers = Mock(side_effect=manager.poll_transfers)
    kv_cache.pending_transfers_exist = Mock(
        side_effect=manager.pending_transfers_exist
    )
    kv_cache.block_count = Mock(return_value=BlockCount(free=100, total=100))
    kv_cache.pressure_pct = Mock(return_value=0.0)
    kv_cache.get_prefix_cache_hit_counts = Mock(
        return_value=[PrefixCacheHits(), PrefixCacheHits()]
    )
    return kv_cache


def test_text_batch_constructor__decode_fallback_runs_before_a_later_replica_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the wait a replica's decode fallback allocs before the next admits.

    Replica 0's committed prefix pages are the only free pages on its pool.
    Replica 0 cordons its prompt on a lookup, so its prefill step falls back
    to decode, and replica 1 holds the same prefix as a cross-replica hit.
    A cross-replica copy does not pin the pages it reads, so the decode row
    must take its pages before replica 1 looks at replica 0's prefix cache,
    and no copy may read a page the decode row now holds.
    """
    monkeypatch.delenv(LOOKUP_WAIT_ENV_VAR, raising=False)
    copied_from: list[tuple[int, int]] = []
    log: list[object] = []

    def record_copy(
        dsts: Sequence[object],
        srcs: Sequence[object],
        dst_rows: Sequence[int],
        src_rows: Sequence[int],
    ) -> None:
        pages = []
        for src, row in zip(srcs, src_rows, strict=True):
            assert isinstance(src, _FakeBuf)
            pages.append((src.replica, row))
        log.append(("copy", pages))
        copied_from.extend(pages)

    monkeypatch.setattr(jenga_block_manager, "_batch_copy_rows", record_copy)
    connector = _Connector()
    manager = _manager(connector, num_blocks=12, num_replicas=2)
    kv_cache = _kv_cache_over(manager, log)
    batch_constructor = TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=5,
            max_batch_total_tokens=None,
            enable_in_flight_batching=False,
            enable_chunked_prefill=False,
            target_tokens_per_batch_ce=30,
            data_parallel_degree=2,
        ),
        pipeline=Mock(spec=["release"]),
        kv_cache=kv_cache,
    )
    r0 = _ctx([1] * 9)
    batch_constructor.enqueue_new_request(r0, replica_idx=0)
    batch_constructor.construct_batch()
    connector.onloads[0].complete = True
    batch_constructor.construct_batch()
    batch_constructor.release_request(r0.request_id)
    connector.answer_hits = False
    c0 = _ctx([2] * 5)
    t0 = _ctx([3] * 10, max_length=20)
    t0.update(999)
    manager.claim(t0, replica_idx=0)
    r1 = _ctx([1] * 9)
    batch_constructor.enqueue_new_request(c0, replica_idx=0)
    batch_constructor.enqueue_new_request(t0, replica_idx=0)
    batch_constructor.enqueue_new_request(r1, replica_idx=1)

    inputs = batch_constructor.construct_batch()

    assert any(c.request_id == t0.request_id for c in inputs.batches[0])
    t0_blocks = set(manager.get_req_blocks_per_leaf(t0)["full"])
    src_bids = {bid for rep, bid in copied_from if rep == 0}
    assert not src_bids & t0_blocks
    t0_alloc_pos = log.index(("alloc", t0.request_id))
    copy_positions = [
        i for i, e in enumerate(log) if isinstance(e, tuple) and e[0] == "copy"
    ]
    assert all(t0_alloc_pos < i for i in copy_positions)


def test_text_batch_constructor__rerun_leaves_a_pressure_blocked_head_as_the_first_left_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rerun does not reset a head KV pressure keeps out a second time.

    Under in-flight batching with decode rows and KV pressure over the
    threshold, only requests that hold pages are admitted. The first
    admission resets and rebinds the fresh head, which under DP CE balancing
    prices it again. The rerun, admitting an onload that landed in the wait,
    reaches the same head and leaves it as the first admission left it.
    """
    monkeypatch.setenv(LOOKUP_WAIT_ENV_VAR, "10")

    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    clock = _Clock()
    monkeypatch.setattr(text_batch_constructor, "time", clock)

    class _ClockedEvent:
        def __init__(self, settles_at: float) -> None:
            self.settles_at = settles_at

        @property
        def g0_blocks_per_leaf(self) -> Mapping[str, Sequence[int]]:
            return {}

        def is_complete(self) -> bool:
            return clock.now >= self.settles_at

        def synchronize(self) -> None:
            self.settles_at = clock.now

    claimed: set[RequestID] = set()
    blocks: dict[RequestID, int] = {}
    lookups_out: set[RequestID] = set()
    events: dict[RequestID, _ClockedEvent] = {}

    onloaded = _ctx([1] * 9)
    looked_up = _ctx([2] * 9)
    fresh = _ctx([3] * 9)
    decode = _ctx([4] * 9)
    decode.update(999)

    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.params = Mock()
    kv_cache.params.page_size = 1
    kv_cache.params.enable_dp_cross_replica_prefix_copy = False
    kv_cache.claim = Mock(
        side_effect=lambda ctx, replica_idx=0: claimed.add(ctx.request_id)
    )
    kv_cache.contains = Mock(side_effect=lambda ctx: ctx.request_id in claimed)

    def release(ctx: TextContext) -> None:
        claimed.discard(ctx.request_id)
        blocks.pop(ctx.request_id, None)

    kv_cache.release = Mock(side_effect=release)
    kv_cache.num_req_blocks = Mock(
        side_effect=lambda ctx: blocks.get(ctx.request_id, 0)
    )
    kv_cache.starts_lookup = Mock(
        side_effect=lambda ctx: (
            ctx is looked_up and ctx.request_id not in events
        )
    )
    kv_cache.awaits_lookup = Mock(
        side_effect=lambda ctx: ctx.request_id in lookups_out
    )
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    kv_cache.hand_back_lookup = Mock(return_value=True)
    kv_cache.active_length_after_alloc = Mock(
        side_effect=lambda ctx: ctx.tokens.active_length
    )
    kv_cache.block_count = Mock(return_value=BlockCount(free=100, total=100))
    kv_cache.pressure_pct = Mock(return_value=0.0)
    probes: list[RequestID] = []

    def hit_counts(ctx: TextContext) -> list[PrefixCacheHits]:
        probes.append(ctx.request_id)
        return [PrefixCacheHits(), PrefixCacheHits()]

    kv_cache.get_prefix_cache_hit_counts = Mock(side_effect=hit_counts)

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is onloaded:
            blocks[ctx.request_id] = 4
            event = _ClockedEvent(settles_at=0.002)
            events[ctx.request_id] = event
            return event
        if ctx is looked_up:
            if ctx.request_id not in events:
                event = _ClockedEvent(settles_at=0.003)
                events[ctx.request_id] = event
                lookups_out.add(ctx.request_id)
                return event
            lookups_out.discard(ctx.request_id)
            blocks[ctx.request_id] = 1
            return CompletedTransfer()
        blocks[ctx.request_id] = 7
        return CompletedTransfer()

    kv_cache.alloc = Mock(side_effect=alloc)
    batch_constructor = TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=10,
            target_tokens_per_batch_ce=100,
            data_parallel_degree=2,
            dp_ce_balance_timeout_ms=10_000.0,
            enable_in_flight_batching=True,
        ),
        pipeline=Mock(spec=["release"]),
        kv_cache=kv_cache,
    )

    # Pass 1: replica 1 cordons the onload.
    batch_constructor.enqueue_new_request(onloaded, replica_idx=1)
    batch_constructor.construct_batch()
    assert onloaded.request_id in batch_constructor._onloading_reqs

    # Pass 2: replica 1 has a decode row and KV pressure past the threshold;
    # replica 0 starts a lookup that answers in the wait.
    claimed.add(decode.request_id)
    blocks[decode.request_id] = 9
    batch_constructor.enqueue_new_request(decode, replica_idx=1)
    batch_constructor.enqueue_new_request(fresh, replica_idx=1)
    batch_constructor.enqueue_new_request(looked_up, replica_idx=0)
    kv_cache.pressure_pct = Mock(return_value=95.0)
    probes.clear()

    inputs = batch_constructor.construct_batch()

    assert any(c.request_id == decode.request_id for c in inputs.batches[1])
    assert any(c.request_id == onloaded.request_id for c in inputs.batches[1])
    assert any(c.request_id == looked_up.request_id for c in inputs.batches[0])
    assert not any(c.request_id == fresh.request_id for c in inputs.batches[1])
    assert probes.count(fresh.request_id) == 1
