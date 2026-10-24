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
"""What a KV connector sees of ``JengaKVCacheManager``, through its public API.

Each test drives the manager the way a pipeline does (``claim``, ``alloc``,
``runtime_inputs``, ``step``, ``release``) against a scripted connector, and
asserts only on what the connector was asked, what the request was handed back
and what the manager reports. Nothing reads the manager's internals, so the
tests do not care how the block manager is built.

The scripted connector mirrors ``max._kv_core.testing.FakeConnector``: it is
told what it holds, completes transfers as scripted, and records every call as
a tuple led by its name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheGroupId, MHAKVCacheParams
from max.nn.kv_cache.cache_params import KVConnectorType, KVHashAlgo
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.config import KVConnectorConfig
from max.pipelines.kv_cache.connectors import tier_connector
from max.pipelines.kv_cache.kv_connector import (
    ByteCount,
    CompletedLookup,
    KVConnectorTransfer,
    KVLoadFailed,
    KVLoadRefused,
    KVLookup,
    KVTransfer,
)
from max.pipelines.kv_cache.paged_kv_cache import (
    jenga_cache_manager as jenga_mod,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_cache_manager import (
    JengaKVCacheManager,
)
from max.pipelines.request.base import RequestID

PAGE = 4
"""Tokens per page."""

PROMPT = 17
"""Tokens that fill four pages and spill one onto a fifth."""

PROMPT_PAGES = 4
"""Pages ``PROMPT`` publishes: the last token is never hashed."""


class ScriptedTransfer:
    """A transfer that completes, or fails, only when told to."""

    def __init__(self, g0: Mapping[str, Sequence[int]] | None = None) -> None:
        self.g0_blocks_per_leaf = {} if g0 is None else g0
        self.done = False
        self.error: Exception | None = None

    def is_complete(self) -> bool:
        if self.error is not None:
            raise self.error
        return self.done

    def synchronize(self) -> None:
        self.done = True


class ScriptedConnector:
    """Holds the hashes it is told to and records every call."""

    name = "ScriptedConnector"

    def __init__(self, leaves: Mapping[str, KVCacheGroupId]) -> None:
        self.leaves = dict(leaves)
        self.held: dict[str, set[bytes]] = {leaf: set() for leaf in leaves}
        self._sync = False
        self._loads: list[str] = []
        self._transfers: list[ScriptedTransfer] = []
        self._calls: list[tuple[Any, ...]] = []

    # -- What the test scripts ----------------------------------------------

    def hold(self, hashes: Sequence[bytes]) -> None:
        """Marks the hashes held by every leaf."""
        for held in self.held.values():
            held.update(hashes)

    def set_default_sync(self, sync: bool) -> None:
        """Makes a transfer complete at once unless an outcome is queued."""
        self._sync = sync

    def push_load(self, kind: str) -> None:
        """Queues the outcome of the next load."""
        self._loads.append(kind)

    def complete(self) -> None:
        for transfer in self._transfers:
            transfer.done = True

    def fail(self, message: str = "transfer failed") -> None:
        for transfer in self._transfers:
            if not transfer.done:
                transfer.error = KVLoadFailed(message)

    def calls(self) -> list[tuple[Any, ...]]:
        return list(self._calls)

    def take_calls(self) -> list[tuple[Any, ...]]:
        calls, self._calls = self._calls, []
        return calls

    # -- The KVConnector surface --------------------------------------------

    def lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> Mapping[str, Sequence[bool]]:
        self._calls.append(("lookup", list(block_hashes), replica_idx, hint))
        return {
            leaf: [h in held for h in block_hashes]
            for leaf, held in self.held.items()
        }

    # The asynchronous surface answers from the synchronous calls, the way
    # the protocol defaults do, so the manager runs the same paths as before.
    def start_lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVLookup:
        return CompletedLookup(
            self.lookup(block_hashes, replica_idx=replica_idx, hint=hint)
        )

    def defers_lookup(self, replica_idx: int = 0) -> bool:
        return False

    def load_lookup(
        self,
        lookup: KVLookup,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        return self.load(
            block_ids, block_hashes, replica_idx=replica_idx, hint=hint
        )

    def abandon_lookup(self, lookup: KVLookup) -> None:
        return

    def probe(
        self,
        block_hashes: Sequence[bytes],
        replica_idxs: Sequence[int],
        hint: bytes | None = None,
    ) -> Sequence[Mapping[str, Sequence[bool]]] | None:
        return None

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        self._calls.append(
            (
                "load",
                [list(block_ids[leaf]) for leaf in self.leaves],
                [list(block_hashes[leaf]) for leaf in self.leaves],
                replica_idx,
                hint,
            )
        )
        kind = self._loads.pop(0) if self._loads else None
        if kind == "refused":
            raise KVLoadRefused("evicted between lookup and load")
        if kind == "fail_at_post":
            raise KVLoadFailed("memory transfer failed")
        transfer = ScriptedTransfer()
        if kind == "fail_at_first_poll":
            transfer.error = KVLoadFailed("transfer failed")
        elif kind == "sync" or (kind is None and self._sync):
            transfer.done = True
        self._transfers.append(transfer)
        return transfer

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        self._calls.append(
            (
                "offload",
                [list(block_ids[leaf]) for leaf in self.leaves],
                [list(block_hashes[leaf]) for leaf in self.leaves],
                replica_idx,
            )
        )
        transfer = ScriptedTransfer({k: list(v) for k, v in block_ids.items()})
        transfer.done = self._sync
        self._transfers.append(transfer)
        return transfer

    def touch(
        self, block_hashes: Sequence[bytes], replica_idx: int = 0
    ) -> None:
        self._calls.append(("touch", list(block_hashes), replica_idx))

    def poll_transfers(self) -> None:
        self._calls.append(("poll_transfers",))

    def reset_prefix_cache(self) -> None:
        self._calls.append(("reset_prefix_cache",))
        for held in self.held.values():
            held.clear()

    def shutdown(self) -> None:
        return None

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics()

    def take_metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics()

    @property
    def host_byte_count(self) -> ByteCount:
        return ByteCount(free=1, total=2)

    @property
    def disk_byte_count(self) -> ByteCount:
        return ByteCount(free=3, total=4)


def make_leaf(
    *,
    dp: int = 1,
    cross_replica_copy: bool = True,
    kv_hash_algo: KVHashAlgo = "ahash64",
    kv_hash_seed: bytes | None = None,
) -> MHAKVCacheParams:
    """Returns a one-leaf cache that names the tiered connector."""
    return MHAKVCacheParams(
        dtype=DType.float32,
        num_layers=1,
        n_kv_heads=1,
        head_dim=1,
        page_size=PAGE,
        devices=[DeviceRef.CPU()] * dp,
        data_parallel_degree=dp,
        enable_prefix_caching=True,
        enable_dp_cross_replica_prefix_copy=cross_replica_copy,
        kv_hash_algo=kv_hash_algo,
        kv_hash_seed=kv_hash_seed,
        kv_connector_config=KVConnectorConfig(
            type=KVConnectorType.tiered, host_offload_max_gb=1.0
        ),
    )


def create_manager(
    leaf: MHAKVCacheParams, num_huge_blocks: int
) -> JengaKVCacheManager:
    (page_bytes,) = {p.bytes_per_page for p in leaf.leaves().values()}
    return JengaKVCacheManager.create(
        params=leaf,
        available_bytes=num_huge_blocks * page_bytes * len(leaf.devices),
        max_batch_size=4,
        max_seq_len=5 * PAGE,
    )


def make_manager(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dp: int = 1,
    num_huge_blocks: int = 16,
    cross_replica_copy: bool = True,
    kv_hash_algo: KVHashAlgo = "ahash64",
    kv_hash_seed: bytes | None = None,
) -> tuple[JengaKVCacheManager, ScriptedConnector]:
    """Builds a one-leaf manager whose connector is a ``ScriptedConnector``."""
    created: list[ScriptedConnector] = []

    def create_connector(**kwargs: Any) -> ScriptedConnector:
        created.append(ScriptedConnector(kwargs["leaves"]))
        return created[0]

    monkeypatch.setattr(jenga_mod, "create_connector", create_connector)
    manager = create_manager(
        make_leaf(
            dp=dp,
            cross_replica_copy=cross_replica_copy,
            kv_hash_algo=kv_hash_algo,
            kv_hash_seed=kv_hash_seed,
        ),
        num_huge_blocks,
    )
    (connector,) = created
    connector.set_default_sync(True)
    return manager, connector


def make_ctx(
    num_tokens: int, *, offset: int = 0, cache_salt: str | None = None
) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        cache_salt=cache_salt,
        tokens=TokenBuffer(np.arange(offset, offset + num_tokens)),
    )


def forward(
    mgr: JengaKVCacheManager, ctx: TextContext, replica_idx: int = 0
) -> None:
    """Runs one forward the way the pipeline does: alloc, inputs, step."""
    mgr.claim(ctx, replica_idx)
    mgr.alloc(ctx)
    run(mgr, ctx, replica_idx)


def run(
    mgr: JengaKVCacheManager, ctx: TextContext, replica_idx: int = 0
) -> None:
    """Runs the forward of a request that is already allocated."""
    batches: list[list[TextContext]] = [
        [] for _ in range(mgr.params.data_parallel_degree)
    ]
    batches[replica_idx] = [ctx]
    mgr.runtime_inputs(batches)
    ctx.update(42)
    mgr.step(ctx)


def settle(mgr: JengaKVCacheManager) -> None:
    """Sends the connector what the last steps committed, and lands it."""
    mgr.runtime_inputs([[] for _ in range(mgr.params.data_parallel_degree)])
    mgr.poll_transfers()


def calls_named(
    connector: ScriptedConnector, name: str
) -> list[tuple[Any, ...]]:
    return [call for call in connector.calls() if call[0] == name]


def call_names(connector: ScriptedConnector) -> list[str]:
    """Names every call but the per-step drain, in order."""
    return [c[0] for c in connector.calls() if c[0] != "poll_transfers"]


def offloaded_hashes(connector: ScriptedConnector) -> list[bytes]:
    """Returns every hash the connector was offered, in order."""
    return [
        block_hash
        for call in calls_named(connector, "offload")
        for block_hash in call[2][0]
    ]


def leave_prompt_in_the_connector(
    mgr: JengaKVCacheManager,
    connector: ScriptedConnector,
    *,
    num_tokens: int = PROMPT,
    replica_idx: int = 0,
    held_pages: int | None = None,
) -> list[bytes]:
    """Runs a prompt to completion and leaves its KV only in the connector.

    Returns the hashes of the pages it published, root first, of which the
    connector holds the first ``held_pages`` (all, by default). The device tier
    is emptied, so the prompt can only come back over an onload.
    """
    ctx = make_ctx(num_tokens)
    forward(mgr, ctx, replica_idx)
    settle(mgr)
    mgr.release(ctx)
    hashes = offloaded_hashes(connector)
    mgr.reset_prefix_cache()
    connector.hold(hashes[:held_pages])
    connector.take_calls()
    return hashes


# ===--------------------------------------------------------------------=== #
# Recency: what `touch` is asked to rank
# ===--------------------------------------------------------------------=== #


def test_release_touches_the_whole_committed_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release re-ranks the request's committed sequence in the external tier.

    The tier orders a group by the keys of one call, and a sequence reaches it
    over several (one per step), so each later commit outranks everything
    committed before it (CLIN-1893). Release is the one point that fires for
    every request, not only the ones read back, so it re-ranks the whole chain.
    A cold request touches nothing on admission: nothing was reused.
    """
    mgr, connector = make_manager(monkeypatch)
    ctx = make_ctx(2 * PAGE + 1)
    forward(mgr, ctx)
    settle(mgr)
    assert calls_named(connector, "touch") == []

    mgr.release(ctx)

    committed = offloaded_hashes(connector)
    assert len(committed) == 2
    assert calls_named(connector, "touch") == [("touch", committed, 0)]


def test_release_touches_nothing_when_no_page_was_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request too short to fill one page has nothing in the tier to rank."""
    mgr, connector = make_manager(monkeypatch)
    ctx = make_ctx(PAGE - 1)
    forward(mgr, ctx)

    mgr.release(ctx)

    assert calls_named(connector, "touch") == []


def test_a_host_only_hit_touches_once_after_the_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anchor is the load path's one recency signal, so it comes last.

    Nothing is on the device, so the prefix is pulled from the connector. The
    touch carries the root-anchored run the request reused, not the uncached
    tail past it, and fires after the load so that no touch the load makes
    itself can invert it (CLIN-1533).
    """
    mgr, connector = make_manager(monkeypatch)
    hashes = leave_prompt_in_the_connector(mgr, connector)
    assert len(hashes) == PROMPT_PAGES

    ctx = make_ctx(PROMPT)
    mgr.claim(ctx)
    mgr.alloc(ctx)

    assert ctx.cached_prefix_external_length == PROMPT_PAGES * PAGE
    assert call_names(connector) == ["lookup", "load", "touch"]
    assert calls_named(connector, "touch") == [("touch", hashes, 0)]


# ===--------------------------------------------------------------------=== #
# Replicas
# ===--------------------------------------------------------------------=== #


def test_a_peer_copy_touches_the_full_sequence_for_the_serving_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page held on another replica's device is copied over, then touched.

    The hit is served by a device-to-device copy onto replica 1, so the touch
    is keyed to the serving replica rather than the one the pages came from,
    and no load is posted.
    """
    mgr, connector = make_manager(monkeypatch, dp=2)
    first = make_ctx(PROMPT)
    forward(mgr, first, replica_idx=0)
    settle(mgr)
    mgr.release(first)
    hashes = offloaded_hashes(connector)
    assert len(hashes) == PROMPT_PAGES
    connector.take_calls()

    second = make_ctx(PROMPT)
    mgr.claim(second, replica_idx=1)
    mgr.alloc(second)

    assert second.cached_prefix_length == PROMPT_PAGES * PAGE
    assert second.cached_prefix_external_length == 0
    assert calls_named(connector, "touch") == [("touch", hashes, 1)]
    assert calls_named(connector, "load") == []
    metrics = mgr.take_metrics_aggregated()
    assert metrics.cross_replica_blocks_copied == PROMPT_PAGES
    assert metrics.cross_replica_bytes_copied > 0


def test_a_peer_page_is_served_by_the_connector_when_copies_are_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With cross-replica copies off, a peer's device pages are out of reach.

    The lookup stops at the local miss, and the connector serves the whole
    prefix, so it is all external rather than the device hit it would have
    been with the copy on.
    """
    mgr, connector = make_manager(monkeypatch, dp=2, cross_replica_copy=False)
    first = make_ctx(PROMPT)
    forward(mgr, first, replica_idx=0)
    settle(mgr)
    mgr.release(first)
    hashes = offloaded_hashes(connector)
    connector.hold(hashes)
    connector.take_calls()

    second = make_ctx(PROMPT)
    mgr.claim(second, replica_idx=1)
    mgr.alloc(second)

    assert second.cached_prefix_length == PROMPT_PAGES * PAGE
    assert second.cached_prefix_external_length == PROMPT_PAGES * PAGE
    assert call_names(connector) == ["lookup", "load", "touch"]
    assert mgr.take_metrics_aggregated().cross_replica_blocks_copied == 0


def test_hit_counts_split_each_replicas_run_between_device_and_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pricing an arrival is read-only, and asks the connector only for a gap.

    Replica 0 holds the whole prompt on its device, so it needs no lookup.
    Replica 1 holds none of it (copies are off), so its whole run is the
    connector's. Nothing is loaded and nothing is touched.
    """
    mgr, connector = make_manager(monkeypatch, dp=2, cross_replica_copy=False)
    first = make_ctx(PROMPT)
    forward(mgr, first, replica_idx=0)
    settle(mgr)
    mgr.release(first)
    connector.hold(offloaded_hashes(connector))
    connector.take_calls()

    hits = mgr.get_prefix_cache_hit_counts(make_ctx(PROMPT))

    assert [(h.device_blocks, h.external_blocks) for h in hits] == [
        (PROMPT_PAGES, 0),
        (0, PROMPT_PAGES),
    ]
    assert [call[:1] + call[2:3] for call in connector.calls()] == [
        ("lookup", 1)
    ]


# ===--------------------------------------------------------------------=== #
# Onloads that do not go through
# ===--------------------------------------------------------------------=== #


def test_a_refused_load_keeps_using_the_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal is a race the next lookup may win, not a broken transport.

    Unlike a failure to post, it does not make the request skip the connector,
    and it is not counted as a load failure.
    """
    mgr, connector = make_manager(monkeypatch)
    leave_prompt_in_the_connector(mgr, connector)
    connector.push_load("refused")

    ctx = make_ctx(PROMPT)
    mgr.claim(ctx)
    mgr.alloc(ctx)

    assert ctx.cached_prefix_length == 0
    assert mgr.take_metrics_aggregated().connector_load_failures == 0
    lookups = len(calls_named(connector, "lookup"))

    again = make_ctx(PROMPT)
    mgr.claim(again)
    mgr.alloc(again)

    assert len(calls_named(connector, "lookup")) == lookups + 1
    assert again.cached_prefix_external_length == PROMPT_PAGES * PAGE


def test_a_partial_hit_allocates_no_surplus_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manager draws one page per hash the lookup reported, and no more."""
    mgr, connector = make_manager(monkeypatch)
    leave_prompt_in_the_connector(mgr, connector, held_pages=2)
    free_before = mgr.block_count().free

    ctx = make_ctx(PROMPT)
    mgr.claim(ctx)
    mgr.alloc(ctx)

    assert ctx.cached_prefix_external_length == 2 * PAGE
    assert mgr.block_count().free == free_before - mgr.num_req_blocks(ctx)


def test_a_failure_the_scheduler_already_saw_still_blocks_the_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manager polls a transfer the scheduler already saw fail.

    The scheduler sweeps its own cordon over the same transfer objects, and it
    runs before the manager's drain, so it sees the failure first. The failure
    being sticky is what keeps the manager's poll from reading complete and
    publishing pages the copy never filled.
    """
    mgr, connector = make_manager(monkeypatch)
    leave_prompt_in_the_connector(mgr, connector)
    connector.push_load("manual")
    ctx = make_ctx(PROMPT)
    mgr.claim(ctx)
    transfer = mgr.alloc(ctx)
    assert not transfer.is_complete()
    assert mgr.pending_transfers_exist()

    connector.fail()
    with pytest.raises(KVLoadFailed):
        transfer.is_complete()
    mgr.poll_transfers()

    assert not mgr.pending_transfers_exist()
    (hits,) = mgr.get_prefix_cache_hit_counts(make_ctx(PROMPT))
    assert hits.device_blocks == 0, "garbage must not be published"


# ===--------------------------------------------------------------------=== #
# Offload
# ===--------------------------------------------------------------------=== #


def test_an_offload_stops_at_the_first_evicted_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pages evicted before the offload ran are left out, with their tail.

    The first request's four pages are committed but not yet sent when it
    releases them and a second request takes two of them back. What reaches
    the connector is the run up to the first page that is gone.
    """
    # Five allocable pages: the first request holds all five.
    reference, reference_connector = make_manager(
        monkeypatch, num_huge_blocks=6
    )
    full = make_ctx(PROMPT)
    forward(reference, full)
    settle(reference)
    hashes = offloaded_hashes(reference_connector)
    assert len(hashes) == PROMPT_PAGES

    mgr, connector = make_manager(monkeypatch, num_huge_blocks=6)
    first = make_ctx(PROMPT)
    forward(mgr, first)
    mgr.release(first)
    # One free page, then the tails of the released prefix: two are evicted.
    squatter = make_ctx(3 * PAGE, offset=1000)
    mgr.claim(squatter)
    mgr.alloc(squatter)
    run(mgr, squatter)

    assert len(calls_named(connector, "offload")) == 1
    assert offloaded_hashes(connector) == hashes[:2]


def test_an_offload_of_pages_that_are_all_evicted_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr, connector = make_manager(monkeypatch, num_huge_blocks=6)
    first = make_ctx(PROMPT)
    forward(mgr, first)
    mgr.release(first)
    squatter = make_ctx(5 * PAGE, offset=1000)
    mgr.claim(squatter)
    mgr.alloc(squatter)
    mgr.runtime_inputs([[squatter]])

    assert calls_named(connector, "offload") == []


def test_runs_reach_the_connector_in_commit_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two requests that step together are offloaded one run each, in order."""
    mgr, connector = make_manager(monkeypatch)
    first = make_ctx(2 * PAGE + 1)
    second = make_ctx(2 * PAGE + 1, offset=1000)
    for ctx in (first, second):
        mgr.claim(ctx)
        mgr.alloc(ctx)
    mgr.runtime_inputs([[first, second]])
    for ctx in (first, second):
        ctx.update(42)
        mgr.step(ctx)

    settle(mgr)

    runs = [call[2][0] for call in calls_named(connector, "offload")]
    assert [len(run) for run in runs] == [2, 2]
    # Both prefixes are published, so each comes back as its own device hit.
    for ctx in (first, second):
        probe = make_ctx(2 * PAGE + 1, offset=0 if ctx is first else 1000)
        (hits,) = mgr.get_prefix_cache_hit_counts(probe)
        assert hits.device_blocks == 2


def test_paging_to_host_on_cpu_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tiered connector has nothing to offload from a CPU cache."""
    # The class lookup runs before the device check; this target does not
    # depend on the extension, so stub it.
    monkeypatch.setattr(
        tier_connector, "_native_connector_class", lambda: object
    )
    with pytest.raises(ValueError, match="cannot offload"):
        create_manager(make_leaf(), num_huge_blocks=16)


# ===--------------------------------------------------------------------=== #
# Hashing: what reaches the connector, and what a salt isolates
# ===--------------------------------------------------------------------=== #


def hashes_of_a_prompt(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kv_hash_algo: KVHashAlgo = "ahash64",
    kv_hash_seed: bytes | None = None,
    cache_salt: str | None = None,
) -> list[bytes]:
    """Returns the hashes a fresh manager offers for the same prompt."""
    mgr, connector = make_manager(
        monkeypatch, kv_hash_algo=kv_hash_algo, kv_hash_seed=kv_hash_seed
    )
    ctx = make_ctx(PROMPT, cache_salt=cache_salt)
    forward(mgr, ctx)
    settle(mgr)
    return offloaded_hashes(connector)


def test_the_hash_algorithm_sets_the_width_of_the_hashes_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = hashes_of_a_prompt(monkeypatch)
    sha256 = hashes_of_a_prompt(monkeypatch, kv_hash_algo="sha256")

    assert len(default) == len(sha256) == PROMPT_PAGES
    assert {len(h) for h in default} == {8}
    assert {len(h) for h in sha256} == {32}


@pytest.mark.parametrize("kv_hash_algo", ["ahash64", "sha256"])
def test_the_seed_and_the_salt_each_isolate_a_prompt(
    monkeypatch: pytest.MonkeyPatch, kv_hash_algo: KVHashAlgo
) -> None:
    seed = b"\xab" * 32
    base = hashes_of_a_prompt(
        monkeypatch, kv_hash_algo=kv_hash_algo, kv_hash_seed=seed
    )

    assert base == hashes_of_a_prompt(
        monkeypatch, kv_hash_algo=kv_hash_algo, kv_hash_seed=seed
    )
    other_seed = hashes_of_a_prompt(
        monkeypatch, kv_hash_algo=kv_hash_algo, kv_hash_seed=b"\xcd" * 32
    )
    salted = hashes_of_a_prompt(
        monkeypatch,
        kv_hash_algo=kv_hash_algo,
        kv_hash_seed=seed,
        cache_salt="tenant-a",
    )
    assert not set(base) & set(other_seed)
    assert not set(base) & set(salted)


def test_a_cache_salt_keeps_tenants_from_sharing_a_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr, _ = make_manager(monkeypatch)
    first = make_ctx(PROMPT, cache_salt="tenant-a")
    forward(mgr, first)
    mgr.release(first)

    (same,) = mgr.get_prefix_cache_hit_counts(
        make_ctx(PROMPT, cache_salt="tenant-a")
    )
    (other,) = mgr.get_prefix_cache_hit_counts(
        make_ctx(PROMPT, cache_salt="tenant-b")
    )
    (unsalted,) = mgr.get_prefix_cache_hit_counts(make_ctx(PROMPT))

    assert same.device_blocks == PROMPT_PAGES
    assert other.device_blocks == 0
    assert unsalted.device_blocks == 0
