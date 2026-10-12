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

"""A connector load whose copy fails, through ``PagedKVCacheManager.alloc``.

CPU only: a real cache manager whose connector is swapped for a fake that
serves whatever hashes it is told it holds and fails on command. The failure
has to cost the request its onload and nothing else: not the worker, not the
device hit, and no page published with garbage in it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheGroupId, MHAKVCacheParams
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.kv_cache.kv_connector import (
    ByteCount,
    CompletedTransfer,
    KVConnectorTransfer,
    KVLoadFailed,
    KVTransfer,
)
from test_common.context_utils import create_text_context

LEAF = "full"


class _Transfer:
    """A load's copy, landing or failing when the test says so.

    A failure is sticky, as the dKV connector's is, so every poll after the
    first reports it too.
    """

    def __init__(self, fails: bool = False) -> None:
        self.complete = False
        self.fails = fails

    def is_complete(self) -> bool:
        if self.fails:
            raise KVLoadFailed("memory transfer failed")
        return self.complete

    def synchronize(self) -> None:
        if self.fails:
            raise KVLoadFailed("memory transfer failed")
        self.complete = True


class _Connector:
    """Holds whatever hashes the test puts in ``held``, loading asynchronously."""

    name = "fake"

    def __init__(self) -> None:
        self.held: set[bytes] = set()
        self.lookups = 0
        self.loads: list[tuple[list[int], _Transfer]] = []
        # Whether the next load's transfer fails on its very first poll.
        self.fail_on_first_poll = False

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return {LEAF: KVCacheGroupId.full()}

    def lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> Mapping[str, Sequence[bool]]:
        self.lookups += 1
        return {LEAF: [h in self.held for h in block_hashes]}

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        transfer = _Transfer(fails=self.fail_on_first_poll)
        self.loads.append((list(block_ids[LEAF]), transfer))
        return transfer

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        return CompletedTransfer()

    def touch(
        self, block_hashes: Sequence[bytes], replica_idx: int = 0
    ) -> None: ...

    def poll_transfers(self) -> None: ...

    def shutdown(self) -> None: ...

    def reset_prefix_cache(self) -> None: ...

    @property
    def host_byte_count(self) -> ByteCount:
        return ByteCount(free=0, total=0)

    @property
    def disk_byte_count(self) -> ByteCount:
        return ByteCount(free=0, total=0)

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics()

    def take_metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics()


def _make_kv_manager(
    enable_runtime_checks: bool = True,
) -> tuple[PagedKVCacheManager, _Connector]:
    params = MHAKVCacheParams(
        dtype=DType.float32,
        num_layers=1,
        n_kv_heads=1,
        head_dim=1,
        enable_prefix_caching=True,
        page_size=1,
        devices=[DeviceRef.CPU()],
        data_parallel_degree=1,
    )
    kv_manager = PagedKVCacheManager(
        params=params,
        total_num_pages=64,
        session=InferenceSession(devices=[CPU()]),
        enable_runtime_checks=enable_runtime_checks,
        max_batch_size=8,
    )
    connector = _Connector()
    kv_manager._block_manager.connector = connector
    return kv_manager, connector


def _ctx(tokens: Sequence[int]) -> TextContext:
    return create_text_context(np.array(tokens, dtype=np.int64))


def _run_forward(kv_manager: PagedKVCacheManager, ctx: TextContext) -> None:
    """Commits a forward over ``ctx``'s active tokens, as a step does."""
    ctx.update(99)
    kv_manager.step(ctx)


def _hold_in_connector_with_a_device_hit(
    kv_manager: PagedKVCacheManager,
    connector: _Connector,
    tokens: Sequence[int],
    num_device_blocks: int,
) -> list[bytes]:
    """Puts ``tokens``' KV in the connector and its first blocks on device.

    Returns the hashes the connector holds, in prefix order.
    """
    full = _ctx(tokens)
    kv_manager.claim(full)
    kv_manager.alloc(full)
    _run_forward(kv_manager, full)
    hashes = list(kv_manager._block_manager.req_to_hashes[full.request_id])
    kv_manager.release(full)
    kv_manager.reset_prefix_cache()

    device = _ctx(tokens[:num_device_blocks])
    kv_manager.claim(device)
    kv_manager.alloc(device)
    _run_forward(kv_manager, device)
    kv_manager.release(device)

    connector.held = set(hashes)
    return hashes


@pytest.mark.parametrize("first_poller", ["manager", "scheduler"])
def test_a_failed_onload_recomputes_from_the_device_hit(
    first_poller: str,
) -> None:
    """The failed copy costs the onload and nothing else.

    Whichever poller sees the failure first, the request's next ``alloc``
    undoes the splice and takes the device hit back without the connector.
    The onloaded pages go back unpublished, and the request recomputes from
    where its device hit ends, with no external tokens credited to it.
    """
    kv_manager, connector = _make_kv_manager()
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    hashes = _hold_in_connector_with_a_device_hit(
        kv_manager, connector, tokens, num_device_blocks=3
    )
    pool = kv_manager._block_manager.device_block_pool
    device_bids = [pool.prefix_cache[h].bid for h in hashes[:3]]

    ctx = _ctx(tokens)
    kv_manager.claim(ctx)
    transfer = kv_manager.alloc(ctx)
    assert not transfer.is_complete()
    assert ctx.tokens.processed_length == 7
    assert ctx.cached_prefix_external_length == 4
    loaded_bids, posted = connector.loads[-1]
    posted.fails = True

    if first_poller == "manager":
        kv_manager.poll_transfers()
    else:
        with pytest.raises(KVLoadFailed):
            transfer.is_complete()
    lookups = connector.lookups
    resumed = kv_manager.alloc(ctx)

    assert resumed.is_complete()
    assert connector.lookups == lookups, "the recompute skips the connector"
    assert ctx.tokens.processed_length == 3
    assert ctx.cached_prefix_length == 3
    assert ctx.cached_prefix_external_length == 0
    (blocks,) = kv_manager.get_req_blocks_per_leaf(ctx).values()
    assert blocks[:3] == device_bids
    assert not kv_manager.pending_transfers_exist()
    assert all(h not in pool.prefix_cache for h in hashes[3:7]), (
        "a failed copy's pages must not be published"
    )
    assert all(pool.pool[bid].ref_cnt == 0 for bid in loaded_bids), (
        "the failed pages went back exactly once"
    )
    assert kv_manager.take_metrics_aggregated().connector_load_failures == 1

    _run_forward(kv_manager, ctx)
    kv_manager.release(ctx)
    assert pool.num_free_blocks == pool.total_num_blocks


def test_a_request_released_before_its_onload_fails_leaves_nothing() -> None:
    """A request cancelled mid-onload has nothing left to roll back."""
    kv_manager, connector = _make_kv_manager()
    tokens = [1, 2, 3, 4, 5, 6]
    _hold_in_connector_with_a_device_hit(
        kv_manager, connector, tokens, num_device_blocks=2
    )
    pool = kv_manager._block_manager.device_block_pool

    ctx = _ctx(tokens)
    kv_manager.claim(ctx)
    kv_manager.alloc(ctx)
    kv_manager.release(ctx)
    connector.loads[-1][1].fails = True
    kv_manager.poll_transfers()

    assert not kv_manager.pending_transfers_exist()
    assert pool.num_free_blocks == pool.total_num_blocks
    assert kv_manager.take_metrics_aggregated().connector_load_failures == 1


def test_a_transfer_that_fails_on_its_first_poll_is_a_miss() -> None:
    """The manager's own poll right after ``load`` can see the failure.

    Nothing is spliced at that point, so it is served like a post that failed.
    """
    kv_manager, connector = _make_kv_manager()
    tokens = [1, 2, 3, 4, 5, 6]
    _hold_in_connector_with_a_device_hit(
        kv_manager, connector, tokens, num_device_blocks=2
    )
    connector.fail_on_first_poll = True

    ctx = _ctx(tokens)
    kv_manager.claim(ctx)
    transfer = kv_manager.alloc(ctx)

    assert transfer.is_complete()
    assert ctx.tokens.processed_length == 2
    assert ctx.cached_prefix_external_length == 0
    assert not kv_manager.pending_transfers_exist()
    assert kv_manager.take_metrics_aggregated().connector_load_failures == 1
    lookups = connector.lookups
    kv_manager.alloc(ctx)
    assert connector.lookups == lookups


def test_a_request_that_ran_past_a_failed_onload_is_not_recomputed() -> None:
    """A caller that never held the request back has already read the pages.

    There is no clean prefix to go back to by then, so the next ``alloc``
    fails loudly instead of recomputing over KV it cannot trust. Runtime
    checks are off because stepping a request whose onload is still in flight
    trips them first, which is a different loud failure for the same misuse.
    """
    kv_manager, connector = _make_kv_manager(enable_runtime_checks=False)
    tokens = [1, 2, 3, 4, 5, 6]
    _hold_in_connector_with_a_device_hit(
        kv_manager, connector, tokens, num_device_blocks=2
    )

    ctx = _ctx(tokens)
    kv_manager.claim(ctx)
    kv_manager.alloc(ctx)
    _run_forward(kv_manager, ctx)
    connector.loads[-1][1].fails = True

    with pytest.raises(RuntimeError, match="ran past") as excinfo:
        kv_manager.alloc(ctx)
    assert isinstance(excinfo.value.__cause__, KVLoadFailed)
