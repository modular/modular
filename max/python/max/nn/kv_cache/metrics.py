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
"""KV cache metrics and the containers that carry them."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from typing_extensions import Self

_K = TypeVar("_K", bound=Hashable)
_N = TypeVar("_N", int, float)

ConnectorLookupDrop = Literal[
    "poll_failed",
    "timeout",
    "release",
    "hand_back",
    "stale_reask",
    "sync_alloc",
    "shrink_reask",
    "shrink_cap",
    "shrink_at_load",
    "covered",
    "skip",
    "nothing_to_load",
    "no_room",
    "load_refused",
    "load_failed",
    "exception",
]
"""Why the Jenga block manager dropped a connector lookup's answer.

One value per place an answer goes back to the connector, or is discarded,
without a load:

- ``poll_failed``: polling the handle raised, so it was given up on.
- ``timeout``: no answer within the lookup deadline.
- ``release``: the request was released with the lookup still held.
- ``hand_back``: the scheduler handed back a lookup past the hand back age,
  answered or not.
- ``stale_reask``: an answer was found past the hand back age before it was
  loaded, and the lookup was asked again.
- ``sync_alloc``: an alloc that cannot wait found the lookup still out and
  asked in place instead, so the answer never came back.
- ``shrink_reask``: the device hit shrank below where the lookup was asked,
  so it was asked again from the shorter hit.
- ``shrink_cap``: the same, past the request's re-asks. The rest is
  computed, unless the connector answers in place and is asked about the gap
  at once.
- ``shrink_at_load``: the device hit came up short of where the lookup was
  asked only when it was taken. The gap is then asked in place if the
  connector answers in place, and computed otherwise.
- ``covered``: the device hit grew over the run the answer offered, so the
  device served it.
- ``skip``: the request skips the connector after a failed load.
- ``nothing_to_load``: the answer offered no block. Counted only for a
  started lookup, since one asked in place that finds nothing leaves no
  answer to drop.
- ``no_room``: the pool cannot hold the run the answer serves.
- ``load_refused``: the connector refused the load, typically evicted since.
- ``load_failed``: the load's copy failed to post or on its first poll.
- ``exception``: the load path raised. That includes a raise after a load
  was posted, whose blocks may still land although the failed alloc rolls
  the request back. Counted only for a started lookup, since a raise past a
  lookup asked in place leaves no handle to count.
"""

ConnectorLookupAnswer = Literal["offered", "empty", "unseen"]
"""What a dropped connector lookup's answer said.

- ``offered``: it offered at least one block.
- ``empty``: it answered with no block. A store read that failed, was shed
  or found the store degraded answers this way too, until the connector
  reports read failures apart from misses.
- ``unseen``: it was dropped before an answer was seen, including one that
  came back but was never polled, given up on, or unreadable.
"""


def _add_by_key(a: Mapping[_K, _N], b: Mapping[_K, _N]) -> dict[_K, _N]:
    """Sums two per-label counter maps, keeping labels either one carries."""
    total = dict(a)
    for key, value in b.items():
        total[key] = total.get(key, 0) + value
    return total


def dkv_tier_degraded(connected_clients: int, total_clients: int) -> bool:
    """Returns whether a dKV tier is present but not fully connected.

    True when at least one dKV client exists and fewer than all of them are
    connected to the external tier, which is the state an operator alerts on for
    a dead or degraded dKV deployment. Shared by KVCacheMetrics.dkv_degraded and
    the scheduler's per-batch log so the two cannot fall out of sync.
    """
    return total_clients > 0 and connected_clients < total_clients


@dataclass
class KVCacheMetrics:
    """Metrics for the KV cache.

    Tracks token usage and transfer statistics for KV cache operations.

    Device (G0) figures are in blocks, the unit the manager allocates in. The
    connector's external host and disk tiers are in bytes: those are byte
    budgets the operator sizes in bytes, their block width need not match the
    device's, and bytes rate directly against PCIe and disk bandwidth.
    """

    input_tokens: int = 0
    """Number of tokens processed as new input (cache misses)."""
    cache_tokens: int = 0
    """Number of tokens retrieved from cache (cache hits)."""
    device_blocks_served: int = 0
    """Number of cache blocks served from the device tier, with no host or disk
    promotion. Jenga counts a block copied in from a peer replica; the legacy
    cache counts only blocks the replica already held."""
    h2d_bytes_copied: int = 0
    """Bytes of KV copied from the connector's host tier to device."""
    d2h_bytes_copied: int = 0
    """Bytes of KV copied from device to the connector's host tier."""
    cross_replica_blocks_copied: int = 0
    """Number of cache blocks copied device-to-device across DP replicas."""
    cross_replica_bytes_copied: int = 0
    """Bytes moved by device-to-device copies across DP replicas."""
    disk_bytes_written: int = 0
    """Bytes of KV written to disk."""
    disk_bytes_read: int = 0
    """Bytes of KV read from disk."""
    inflight_disk_ops: int = 0
    """Number of in-flight disk operations."""
    connector_loads_refused: int = 0
    """Number of loads the connector refused: its host and disk tiers, or dKV.

    A refused load is served by recomputing its blocks, so a steady rate of
    these is hit rate the tiers reported and then could not deliver.
    """
    connector_load_failures: int = 0
    """Number of connector loads whose copy failed.

    Counted by the cache manager, which serves each one by recomputing the
    request's prefix. A transport fault rather than a miss, so any of these is
    worth chasing.
    """
    connector_offload_blocks_dropped: int = 0
    """Number of offloaded blocks the connector's host pool had no room for.

    Counted per leaf block. A pool starved by blocks pinned for in-flight
    transfers drops offloads before its hit rate falls.
    """

    # Connector lookup answers the Jenga block manager dropped without a
    # load. Per-window deltas, counted once per lookup rather than per leaf
    # client, and in prompt blocks, the unit one block of every leaf shares.
    connector_lookup_drops: dict[
        tuple[ConnectorLookupDrop, ConnectorLookupAnswer, bool], int
    ] = field(default_factory=dict)
    """Dropped lookup answers, keyed by reason, by what the answer said, and
    by whether the connector routed the lookup by a cache hint."""
    connector_lookup_dropped_blocks: dict[
        tuple[ConnectorLookupDrop, bool], int
    ] = field(default_factory=dict)
    """Blocks those drops cost, keyed by reason and by hint. For a drop at
    the load (``no_room``, ``load_refused``, ``load_failed``) it is the run
    past the device hit then, and ``covered`` costs none. Otherwise it is the
    run the answer offered from where the device hit ended when it was
    asked, which ``shrink_reask``, ``shrink_at_load``, ``hand_back`` and
    ``stale_reask`` may get back by asking again."""
    connector_lookup_shrink_depths: dict[int, int] = field(default_factory=dict)
    """Answers dropped because the device hit shrank under them, by how many
    blocks it shrank between the ask and the check."""
    connector_lookup_shrink_cap_blocks: int = 0
    """An estimate of the blocks requests computed because a shrink came
    after their last re-ask: those the device hit lost plus the run the
    dropped answer offered past them. Counted only for a connector whose
    lookup is deferred, since one that answers in place asks about the gap
    at once."""
    connector_lookup_reasks_per_request: dict[int, int] = field(
        default_factory=dict
    )
    """Released claims by how often their lookup was asked again after a
    shrink, for claims that re-asked at least once."""
    connector_lookup_hand_backs_per_request: dict[int, int] = field(
        default_factory=dict
    )
    """Released claims by how many answers they handed back for age, for
    claims that handed back at least one."""
    nixl_read_blocks: int = 0
    """Number of cache blocks read via NIXL (dKV GET)."""
    nixl_write_blocks: int = 0
    """Number of cache blocks written via NIXL (dKV PUT)."""

    # dKV latency pairs: total_ms + count sum correctly across DP replicas.
    nixl_read_latency_total_ms: float = 0.0
    """Cumulative NIXL READ transfer latency in milliseconds."""
    nixl_read_latency_count: int = 0
    """Number of NIXL READ transfer completions."""
    nixl_read_latency_max_ms: float = 0.0
    """Longest single dKV READ transfer in milliseconds.

    The tail the total/count pair above cannot show, since that pair only ever
    yields a mean. Combined across replicas by taking the max, not by adding.
    The connector's ``ConnectorMetrics::read_transfer_latency_max_ms`` is the
    canonical account of what the figure does and does not mean.
    """
    nixl_write_latency_total_ms: float = 0.0
    """Cumulative NIXL WRITE transfer latency in milliseconds."""
    nixl_write_latency_count: int = 0
    """Number of NIXL WRITE transfer completions."""
    rpc_acquire_latency_total_ms: float = 0.0
    """Cumulative dKV acquire_blocks RPC latency in milliseconds."""
    rpc_acquire_latency_count: int = 0
    """Number of acquire_blocks RPC calls."""
    rpc_read_latency_total_ms: float = 0.0
    """Cumulative dKV read_blocks RPC latency in milliseconds."""
    rpc_read_latency_count: int = 0
    """Number of read_blocks RPC calls."""
    nixl_read_bytes: int = 0
    """Total bytes transferred via NIXL READ."""
    nixl_write_bytes: int = 0
    """Total bytes transferred via NIXL WRITE."""

    # dKV external-tier health. These are a level and a lifetime-cumulative
    # counter read live from the connector rather than per-batch transfer
    # deltas, so they export as gauges and take_metrics does not clear them.
    dkv_connected_clients: int = 0
    """Number of dKV data-parallel replicas fully connected to the external tier.

    Per REPLICA, not per client. A replica owns one connector client per KV
    cache leaf, and counts as connected only when EVERY one of them is up,
    because one dead leaf cannot serve a block. A four-leaf replica with one
    leaf down reports 0 here against 1 in ``dkv_total_clients``.
    """
    dkv_total_clients: int = 0
    """Number of dKV data-parallel replicas, regardless of connection state.

    The denominator for ``dkv_connected_clients``. Named "clients" from when a
    replica owned exactly one; it counts replicas, not the per-leaf clients a
    replica now holds.
    """
    dkv_reconnect_attempts: int = 0
    """Cumulative dKV reconnect attempts per replica, summed over replicas.

    Per REPLICA, not per client: a replica owns one client per KV cache leaf and
    they share a server and a network path, so they lose and recover it
    together. The connector reports the max over a replica's leaf clients, which
    keeps this on the same denominator as ``dkv_total_clients`` -- summing over
    leaves would multiply a four-leaf deployment's reading by four.
    """

    # The dKV settle worker and the owner release halt. Every leaf client runs
    # its own worker, so these fold over leaf clients as well as replicas. The
    # depth, the age and the halt are levels the connector reads fresh on every
    # snapshot, so take_metrics leaves them in place like the health keys
    # above. settle_jobs_finished is a per-window delta that take_metrics
    # clears.
    dkv_settle_queue_depth: int = 0
    """Settle jobs queued and not yet finished, summed over every leaf client."""
    dkv_settle_queue_oldest_age_ms: float = 0.0
    """How long the oldest unfinished settle job has waited, in milliseconds.

    The max over every leaf client, and zero when none is queued. Read it
    against ``dkv_settle_jobs_finished``. A head stuck on an RPC or a reconnect
    ages at wall-clock rate while that count stays flat, and a worker that is
    only behind under load ages more slowly while the count keeps rising.
    """
    dkv_settle_jobs_finished: int = 0
    """Settle jobs the workers finished or dropped this window, summed."""
    dkv_owner_releases_halted: int = 0
    """1 while any leaf client's owner release halt is up, else 0.

    While it is up, the owner's failing cleanups wait on the orphan queue and
    its drain backs off.
    """

    # Cross-node pull. Per-window deltas like nixl_read_blocks and the latency
    # pairs, so they sum and take_metrics clears them, unlike the three dKV
    # health keys above. The connector's ConnectorMetrics is the canonical
    # account of each one's unit.
    dkv_peer_attaches: int = 0
    """Remote peers attached from a cache hint.

    Cached per peer, so a stable peer set leaves this at zero while pulls keep
    succeeding. ``dkv_peer_loads`` is the steady-state signal.
    """
    dkv_peer_attach_failures: int = 0
    """Peer attaches that failed at dial, probe, handshake, or timeout."""
    dkv_peers_dropped: int = 0
    """Remote peers dropped, whether replaced after a restart, evicted over the
    peer table's cap, or torn down by the caller. Read against
    ``dkv_peer_attaches`` to tell a churning peer table from a stable one."""
    dkv_peer_loads: int = 0
    """Loads served from a hint-routed peer."""
    dkv_peer_load_failures: int = 0
    """Hinted peer loads that fell through to another source or to the
    co-located tier.

    Memo-capped rather than per-request: a failed source suppresses further
    attempts against the same instance and epoch for a fixed window, so one
    dead peer charges roughly one failure per window however many requests it
    affects.
    """
    dkv_peer_transfer_failures: int = 0
    """Hint-routed peer leases whose NIXL read failed to build, post or
    complete.

    Charged to the peer, which is skipped for the failure memo window and
    marked stale for a metadata refresh, rather than degrading the connector.
    """
    dkv_peer_metadata_refreshes: int = 0
    """Stale peers whose NIXL metadata the connector reloaded in place."""
    dkv_peer_metadata_refresh_failures: int = 0
    """In-place peer metadata reloads that NIXL refused.

    Each strands its peer: the connector never routes to or dials it again for
    the life of the process.
    """
    dkv_hints_rejected: int = 0
    """Cache hints the connector rejected, never one counted in
    ``dkv_hints_unused``."""
    dkv_hints_rejected_by_reason: dict[str, int] = field(default_factory=dict)
    """``dkv_hints_rejected`` split by why each hint was rejected."""
    dkv_hints_unused: int = 0
    """Cache hints that agree with the request but cannot serve it, such as
    one whose chain ends before the lookup starts. Normal traffic, and never
    counted in ``dkv_hints_rejected``."""
    dkv_hints_unused_by_reason: dict[str, int] = field(default_factory=dict)
    """``dkv_hints_unused`` split by why each hint could not serve."""
    dkv_touches_deduplicated: int = 0
    """Recency touches the client dropped because a touch it sent within the
    last second already covered their keys, each an RPC the store was spared."""

    # What lost cache hints and short peer pulls cost, in blocks (CLIN-2219).
    # Per-window deltas like the hint counters above, so the blocks lost to
    # each cause can be ranked rather than guessed from request counts.
    dkv_hints_received: int = 0
    """Cache hints that reached a lookup which counts them, aligned or not.
    The denominator for the ``unparseable``, ``no_group_entry`` and
    ``chain_mismatch`` rejections and for ``dkv_hints_unused``, per entry per
    leaf client like them. An ``invalid_source`` rejection is counted per
    source, not per hint, so it is not a share of this."""
    dkv_hint_blocks_forgone_by_reason: dict[str, int] = field(
        default_factory=dict
    )
    """Blocks the peers a rejected or unused hint named had offered from the
    request's first block, by the rejection or unused reason, per entry per
    leaf client like ``dkv_hints_received``. For ``chain_mismatch`` it counts
    only the blocks before the chains part, since the rest were never the
    request's. It is zero for a request sent without ``chain_start`` and for a
    sliding window hint, neither of which can be placed on the hinted
    chain."""
    dkv_own_run_blocks_short_of_hint: int = 0
    """Blocks the co-located run fell short of the run the hint credited it
    with, over the entries ``dkv_own_runs_short_of_hint`` counts."""
    dkv_peer_blocks_short_of_hint: int = 0
    """Blocks a hinted peer was asked for that neither its answer nor the
    co-located run covered, a failed read counting from where the co-located
    run ends. A read sent back to the peer counts on the read that asks again,
    or at the end of the round when none does. A later source can still serve
    some of these blocks."""
    dkv_hint_blocks_unserved: int = 0
    """Blocks each hinted entry's hint promised that the finished lookup's
    plan does not serve, per entry per leaf client like
    ``dkv_hints_received``. The two ``*_short_of_hint`` counts are per hop and
    can count one block more than once, so this is the count to rank blocks
    lost by. Counted on the leased path MAX drives only, since Mach's
    presence path does not count it yet."""

    # dKV fail-opens: loads answered as misses without a transfer, and the
    # lookups hinted peers still served through the same outage. Per-window
    # deltas like the peer counters, so a tier serving every load as a miss
    # reads as one rather than as a cold cache.
    dkv_prepare_load_degraded: int = 0
    """Lookups that found the co-located dKV down and left blocks to it, which
    missed. Blocks a hinted peer served load as usual."""
    dkv_load_with_plan_degraded: int = 0
    """Loads whose co-located dKV went down between the lookup and the post,
    releasing what it leased. Blocks leased from a hinted peer still load."""
    dkv_load_with_plan_expired: int = 0
    """Leases that aged past their deadline before the load posted them."""
    dkv_peer_rounds_while_degraded: int = 0
    """Lookups that found the co-located dKV down and still leased a loadable
    run from a hinted peer, so they served hits the fail-opens above would
    have missed."""

    # The overlapped hinted lookup (CLIN-2163). Per-window deltas like the
    # fail-opens above. The hinted lookup pair sums across replicas the way
    # the rpc latency pairs do, so the average stays correct.
    dkv_own_runs_short_of_hint: int = 0
    """Hinted entries whose co-located run ended before the run the hint
    credited the co-located dKV with, so their peer was asked from where that
    run ends. A rising rate means hints overstate what the co-located dKV
    holds."""
    dkv_split_loads: int = 0
    """Entries whose tail loaded from a hinted peer behind a head loaded from
    the co-located dKV."""
    dkv_overlapped_peer_lookups: int = 0
    """Peer lookups begun alongside the co-located lookup rather than after
    it, under ``MODULAR_DKV_OVERLAP_PEER_LOOKUPS``."""
    dkv_peer_writeback_blocks: int = 0
    """Peer-filled blocks offloaded to write back to the co-located dKV once
    their onload landed, under ``MODULAR_DKV_OFFLOAD_PEER_LOADS``, counted per
    leaf for the hashes the device prefix cache still held, up to the first
    one it lacks on a full-attention leaf, where the offload stops."""
    dkv_peer_writeback_blocks_evicted: int = 0
    """Peer-filled hashes a write-back's offload found missing from the
    device prefix cache while its request still held its own pages, per leaf,
    with those behind the first on a full-attention leaf. Another request
    published these as twins and they were evicted after it let them go, and
    that request normally wrote them to dKV already, so this is not a loss."""
    dkv_peer_writeback_blocks_dropped: int = 0
    """Peer-filled hashes lost to the co-located dKV, per leaf: those a
    write-back found missing from the device prefix cache when its request
    had let the pages go, by a rollback or a release before the copy landed,
    with those behind the first on a full-attention leaf. Nobody wrote these
    back."""
    dkv_peer_writeback_hold_expired: int = 0
    """Peer write-backs not seen to land within their deadline, which then
    stop holding back the offloads that follow them in their chains. One with
    an offload held behind it lets dKV briefly register that offload's run
    ahead of the pulled blocks."""
    dkv_peer_writeback_held_offloads: int = 0
    """Offloads released from a hold behind a peer write-back, directly or
    through another held run."""
    dkv_peer_writeback_hold_ms_total: float = 0.0
    """Cumulative time the offloads ``dkv_peer_writeback_held_offloads``
    counts were held, in milliseconds."""
    dkv_hinted_lookups: int = 0
    """Lookup rounds in which some entry routed to an attached peer."""
    dkv_hinted_lookup_ms_total: float = 0.0
    """Cumulative wall time of the rounds ``dkv_hinted_lookups`` counts, in
    milliseconds."""

    # The dKV orphan queue, where a failed reader-pin release or Filling abort
    # waits to be sent again. The outcome counts are per-window deltas like the
    # fail-opens above. The depth is a level the client reads fresh on every
    # snapshot, so take_metrics leaves it, and it sums over every leaf client
    # like dkv_settle_queue_depth.
    dkv_orphan_cleanups_reissued: int = 0
    """Deferred cleanups a drain sent again and dKV took."""
    dkv_orphan_cleanups_abandoned: int = 0
    """Deferred cleanups left to dKV's sweep because their instance was gone."""
    dkv_orphan_cleanups_dropped: int = 0
    """Deferred cleanups dropped to keep the queue under its cap."""
    dkv_orphan_cleanups_refused: int = 0
    """Deferred cleanups dKV refused, dropped rather than sent again."""
    dkv_orphan_cleanups_expired: int = 0
    """Deferred cleanups dropped once dKV's sweep had freed what they named."""
    dkv_orphan_queue_depth: int = 0
    """Deferred cleanups queued or being reissued now, summed over every leaf
    client."""

    # Per-call dKV client counters, per-window deltas keyed by RPC (or, for the
    # lock waits, by connector entry point). Time over calls is the per-call
    # mean, which the per-batch latency means above cannot give.
    dkv_rpc_calls: dict[str, int] = field(default_factory=dict)
    """dKV RPCs issued, by operation."""
    dkv_rpc_time_ms: dict[str, float] = field(default_factory=dict)
    """Wall time of those RPCs in milliseconds, socket wait included."""
    dkv_rpc_socket_wait_ms: dict[str, float] = field(default_factory=dict)
    """Milliseconds those RPCs queued for their client's one REQ socket."""
    dkv_lock_waits: dict[str, int] = field(default_factory=dict)
    """Times each connector entry point took the connector lock."""
    dkv_lock_wait_ms: dict[str, float] = field(default_factory=dict)
    """Milliseconds each entry point waited for the connector lock."""
    dkv_blocked_calls: dict[str, int] = field(default_factory=dict)
    """Calls into each connector entry point."""
    dkv_blocked_ms: dict[str, float] = field(default_factory=dict)
    """Milliseconds the calling thread spent inside each entry point, lock
    wait and any RPC included. Near zero for an entry that returns at once,
    which is how the scheduler thread is shown to have stopped waiting."""

    @property
    def prompt_tokens(self) -> int:
        """Total number of prompt tokens (input + cached).

        Returns:
            Sum of input_tokens and cache_tokens.
        """
        return self.input_tokens + self.cache_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Proportion of prompt tokens that were retrieved from cache.

        Returns:
            Ratio of cache_tokens to total prompt_tokens, or 0.0 if no tokens
            were processed.
        """
        if self.prompt_tokens == 0:
            return 0.0
        return self.cache_tokens / self.prompt_tokens

    @property
    def nixl_read_latency_avg_ms(self) -> float:
        """Average NIXL READ transfer latency in milliseconds."""
        if self.nixl_read_latency_count == 0:
            return 0.0
        return self.nixl_read_latency_total_ms / self.nixl_read_latency_count

    @property
    def nixl_write_latency_avg_ms(self) -> float:
        """Average NIXL WRITE transfer latency in milliseconds."""
        if self.nixl_write_latency_count == 0:
            return 0.0
        return self.nixl_write_latency_total_ms / self.nixl_write_latency_count

    @property
    def rpc_acquire_latency_avg_ms(self) -> float:
        """Average dKV acquire_blocks RPC latency in milliseconds."""
        if self.rpc_acquire_latency_count == 0:
            return 0.0
        return (
            self.rpc_acquire_latency_total_ms / self.rpc_acquire_latency_count
        )

    @property
    def rpc_read_latency_avg_ms(self) -> float:
        """Average dKV read_blocks RPC latency in milliseconds."""
        if self.rpc_read_latency_count == 0:
            return 0.0
        return self.rpc_read_latency_total_ms / self.rpc_read_latency_count

    @property
    def dkv_hinted_lookup_avg_ms(self) -> float:
        """Average hinted dKV lookup round in milliseconds."""
        if self.dkv_hinted_lookups == 0:
            return 0.0
        return self.dkv_hinted_lookup_ms_total / self.dkv_hinted_lookups

    @property
    def dkv_peer_writeback_hold_avg_ms(self) -> float:
        """Average hold behind a peer write-back in milliseconds."""
        if self.dkv_peer_writeback_held_offloads == 0:
            return 0.0
        return (
            self.dkv_peer_writeback_hold_ms_total
            / self.dkv_peer_writeback_held_offloads
        )

    @property
    def nixl_read_gib_per_s(self) -> float:
        """NIXL READ throughput in GiB/s."""
        if self.nixl_read_latency_total_ms <= 0:
            return 0.0
        return (self.nixl_read_bytes / (1 << 30)) / (
            self.nixl_read_latency_total_ms / 1000
        )

    @property
    def nixl_write_gib_per_s(self) -> float:
        """NIXL WRITE throughput in GiB/s."""
        if self.nixl_write_latency_total_ms <= 0:
            return 0.0
        return (self.nixl_write_bytes / (1 << 30)) / (
            self.nixl_write_latency_total_ms / 1000
        )

    @property
    def dkv_degraded(self) -> bool:
        """Whether a dKV tier is present but not every client is connected.

        Delegates to the module-level dkv_tier_degraded so this predicate has a
        single definition shared with the scheduler's per-batch log.
        """
        return dkv_tier_degraded(
            self.dkv_connected_clients, self.dkv_total_clients
        )

    def __add__(self, other: Self) -> Self:
        """Combines two KVCacheMetrics field by field.

        Counters sum; ``nixl_read_latency_max_ms`` takes the max, since a peak
        latency is a property of one transfer rather than a quantity to pool.

        Args:
            other: Another KVCacheMetrics instance to add.

        Returns:
            A new KVCacheMetrics instance with the combined values.
        """
        return type(self)(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_tokens=self.cache_tokens + other.cache_tokens,
            device_blocks_served=self.device_blocks_served
            + other.device_blocks_served,
            h2d_bytes_copied=self.h2d_bytes_copied + other.h2d_bytes_copied,
            d2h_bytes_copied=self.d2h_bytes_copied + other.d2h_bytes_copied,
            cross_replica_blocks_copied=self.cross_replica_blocks_copied
            + other.cross_replica_blocks_copied,
            cross_replica_bytes_copied=self.cross_replica_bytes_copied
            + other.cross_replica_bytes_copied,
            disk_bytes_written=self.disk_bytes_written
            + other.disk_bytes_written,
            disk_bytes_read=self.disk_bytes_read + other.disk_bytes_read,
            connector_loads_refused=self.connector_loads_refused
            + other.connector_loads_refused,
            connector_load_failures=self.connector_load_failures
            + other.connector_load_failures,
            connector_offload_blocks_dropped=self.connector_offload_blocks_dropped
            + other.connector_offload_blocks_dropped,
            connector_lookup_drops=_add_by_key(
                self.connector_lookup_drops, other.connector_lookup_drops
            ),
            connector_lookup_dropped_blocks=_add_by_key(
                self.connector_lookup_dropped_blocks,
                other.connector_lookup_dropped_blocks,
            ),
            connector_lookup_shrink_depths=_add_by_key(
                self.connector_lookup_shrink_depths,
                other.connector_lookup_shrink_depths,
            ),
            connector_lookup_shrink_cap_blocks=self.connector_lookup_shrink_cap_blocks
            + other.connector_lookup_shrink_cap_blocks,
            connector_lookup_reasks_per_request=_add_by_key(
                self.connector_lookup_reasks_per_request,
                other.connector_lookup_reasks_per_request,
            ),
            connector_lookup_hand_backs_per_request=_add_by_key(
                self.connector_lookup_hand_backs_per_request,
                other.connector_lookup_hand_backs_per_request,
            ),
            inflight_disk_ops=self.inflight_disk_ops + other.inflight_disk_ops,
            nixl_read_blocks=self.nixl_read_blocks + other.nixl_read_blocks,
            nixl_write_blocks=self.nixl_write_blocks + other.nixl_write_blocks,
            nixl_read_latency_total_ms=self.nixl_read_latency_total_ms
            + other.nixl_read_latency_total_ms,
            nixl_read_latency_count=self.nixl_read_latency_count
            + other.nixl_read_latency_count,
            # A max, not a sum: the slowest read across the replicas is still
            # one read, and adding two replicas' peaks would invent a latency
            # neither of them saw.
            nixl_read_latency_max_ms=max(
                self.nixl_read_latency_max_ms, other.nixl_read_latency_max_ms
            ),
            nixl_write_latency_total_ms=self.nixl_write_latency_total_ms
            + other.nixl_write_latency_total_ms,
            nixl_write_latency_count=self.nixl_write_latency_count
            + other.nixl_write_latency_count,
            rpc_acquire_latency_total_ms=self.rpc_acquire_latency_total_ms
            + other.rpc_acquire_latency_total_ms,
            rpc_acquire_latency_count=self.rpc_acquire_latency_count
            + other.rpc_acquire_latency_count,
            rpc_read_latency_total_ms=self.rpc_read_latency_total_ms
            + other.rpc_read_latency_total_ms,
            rpc_read_latency_count=self.rpc_read_latency_count
            + other.rpc_read_latency_count,
            nixl_read_bytes=self.nixl_read_bytes + other.nixl_read_bytes,
            nixl_write_bytes=self.nixl_write_bytes + other.nixl_write_bytes,
            dkv_connected_clients=self.dkv_connected_clients
            + other.dkv_connected_clients,
            dkv_total_clients=self.dkv_total_clients + other.dkv_total_clients,
            dkv_reconnect_attempts=self.dkv_reconnect_attempts
            + other.dkv_reconnect_attempts,
            dkv_settle_queue_depth=self.dkv_settle_queue_depth
            + other.dkv_settle_queue_depth,
            # A max, not a sum, for the same reason as the read latency peak:
            # the oldest job across two workers is still one job's wait.
            dkv_settle_queue_oldest_age_ms=max(
                self.dkv_settle_queue_oldest_age_ms,
                other.dkv_settle_queue_oldest_age_ms,
            ),
            dkv_settle_jobs_finished=self.dkv_settle_jobs_finished
            + other.dkv_settle_jobs_finished,
            # Up when either side's is, and still 1 rather than a count.
            dkv_owner_releases_halted=max(
                self.dkv_owner_releases_halted, other.dkv_owner_releases_halted
            ),
            dkv_peer_attaches=self.dkv_peer_attaches + other.dkv_peer_attaches,
            dkv_peer_attach_failures=self.dkv_peer_attach_failures
            + other.dkv_peer_attach_failures,
            dkv_peers_dropped=self.dkv_peers_dropped + other.dkv_peers_dropped,
            dkv_peer_loads=self.dkv_peer_loads + other.dkv_peer_loads,
            dkv_peer_load_failures=self.dkv_peer_load_failures
            + other.dkv_peer_load_failures,
            dkv_peer_transfer_failures=self.dkv_peer_transfer_failures
            + other.dkv_peer_transfer_failures,
            dkv_peer_metadata_refreshes=self.dkv_peer_metadata_refreshes
            + other.dkv_peer_metadata_refreshes,
            dkv_peer_metadata_refresh_failures=self.dkv_peer_metadata_refresh_failures
            + other.dkv_peer_metadata_refresh_failures,
            dkv_hints_rejected=self.dkv_hints_rejected
            + other.dkv_hints_rejected,
            dkv_hints_rejected_by_reason=_add_by_key(
                self.dkv_hints_rejected_by_reason,
                other.dkv_hints_rejected_by_reason,
            ),
            dkv_hints_unused=self.dkv_hints_unused + other.dkv_hints_unused,
            dkv_hints_unused_by_reason=_add_by_key(
                self.dkv_hints_unused_by_reason,
                other.dkv_hints_unused_by_reason,
            ),
            dkv_hints_received=self.dkv_hints_received
            + other.dkv_hints_received,
            dkv_hint_blocks_forgone_by_reason=_add_by_key(
                self.dkv_hint_blocks_forgone_by_reason,
                other.dkv_hint_blocks_forgone_by_reason,
            ),
            dkv_touches_deduplicated=self.dkv_touches_deduplicated
            + other.dkv_touches_deduplicated,
            dkv_prepare_load_degraded=self.dkv_prepare_load_degraded
            + other.dkv_prepare_load_degraded,
            dkv_load_with_plan_degraded=self.dkv_load_with_plan_degraded
            + other.dkv_load_with_plan_degraded,
            dkv_load_with_plan_expired=self.dkv_load_with_plan_expired
            + other.dkv_load_with_plan_expired,
            dkv_peer_rounds_while_degraded=self.dkv_peer_rounds_while_degraded
            + other.dkv_peer_rounds_while_degraded,
            dkv_own_runs_short_of_hint=self.dkv_own_runs_short_of_hint
            + other.dkv_own_runs_short_of_hint,
            dkv_own_run_blocks_short_of_hint=self.dkv_own_run_blocks_short_of_hint
            + other.dkv_own_run_blocks_short_of_hint,
            dkv_peer_blocks_short_of_hint=self.dkv_peer_blocks_short_of_hint
            + other.dkv_peer_blocks_short_of_hint,
            dkv_hint_blocks_unserved=self.dkv_hint_blocks_unserved
            + other.dkv_hint_blocks_unserved,
            dkv_split_loads=self.dkv_split_loads + other.dkv_split_loads,
            dkv_overlapped_peer_lookups=self.dkv_overlapped_peer_lookups
            + other.dkv_overlapped_peer_lookups,
            dkv_peer_writeback_blocks=self.dkv_peer_writeback_blocks
            + other.dkv_peer_writeback_blocks,
            dkv_peer_writeback_blocks_evicted=(
                self.dkv_peer_writeback_blocks_evicted
                + other.dkv_peer_writeback_blocks_evicted
            ),
            dkv_peer_writeback_blocks_dropped=(
                self.dkv_peer_writeback_blocks_dropped
                + other.dkv_peer_writeback_blocks_dropped
            ),
            dkv_peer_writeback_hold_expired=(
                self.dkv_peer_writeback_hold_expired
                + other.dkv_peer_writeback_hold_expired
            ),
            dkv_peer_writeback_held_offloads=(
                self.dkv_peer_writeback_held_offloads
                + other.dkv_peer_writeback_held_offloads
            ),
            dkv_peer_writeback_hold_ms_total=(
                self.dkv_peer_writeback_hold_ms_total
                + other.dkv_peer_writeback_hold_ms_total
            ),
            dkv_hinted_lookups=self.dkv_hinted_lookups
            + other.dkv_hinted_lookups,
            dkv_hinted_lookup_ms_total=self.dkv_hinted_lookup_ms_total
            + other.dkv_hinted_lookup_ms_total,
            dkv_orphan_cleanups_reissued=self.dkv_orphan_cleanups_reissued
            + other.dkv_orphan_cleanups_reissued,
            dkv_orphan_cleanups_abandoned=self.dkv_orphan_cleanups_abandoned
            + other.dkv_orphan_cleanups_abandoned,
            dkv_orphan_cleanups_dropped=self.dkv_orphan_cleanups_dropped
            + other.dkv_orphan_cleanups_dropped,
            dkv_orphan_cleanups_refused=self.dkv_orphan_cleanups_refused
            + other.dkv_orphan_cleanups_refused,
            dkv_orphan_cleanups_expired=self.dkv_orphan_cleanups_expired
            + other.dkv_orphan_cleanups_expired,
            dkv_orphan_queue_depth=self.dkv_orphan_queue_depth
            + other.dkv_orphan_queue_depth,
            dkv_rpc_calls=_add_by_key(self.dkv_rpc_calls, other.dkv_rpc_calls),
            dkv_rpc_time_ms=_add_by_key(
                self.dkv_rpc_time_ms, other.dkv_rpc_time_ms
            ),
            dkv_rpc_socket_wait_ms=_add_by_key(
                self.dkv_rpc_socket_wait_ms, other.dkv_rpc_socket_wait_ms
            ),
            dkv_lock_waits=_add_by_key(
                self.dkv_lock_waits, other.dkv_lock_waits
            ),
            dkv_lock_wait_ms=_add_by_key(
                self.dkv_lock_wait_ms, other.dkv_lock_wait_ms
            ),
            dkv_blocked_calls=_add_by_key(
                self.dkv_blocked_calls, other.dkv_blocked_calls
            ),
            dkv_blocked_ms=_add_by_key(
                self.dkv_blocked_ms, other.dkv_blocked_ms
            ),
        )
