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

"""Distributed KV cache connector via the dKV service.

A thin :class:`~max.pipelines.kv_cache.kv_connector.KVConnector` shim over the
``dkv_connector`` Rust client (``dkv_connector.DkvConnector``). The Rust client
owns the NIXL agent, all block transfers, the control-plane RPCs, inline
reconnection, and metrics; this shim only adapts the MAX-side types (device
``KVCacheMemory``, ``KVCacheMetrics``) to the client's API.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from max.driver import Buffer, Device
from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.cache_params import (
    KVCacheMemory,
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.kv_cache.data_parallelism_utils import split_into_groups
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache._nixl_backend import (
    NIXL_BACKEND_ENV_VAR,
    SUPPORTED_NIXL_BACKENDS,
    NixlBackendType,
    validate_nixl_backend,
)
from max.pipelines.kv_cache._nixl_plugin_deps import preload_nixl_plugin_deps
from max.pipelines.kv_cache.kv_connector import (
    CompletedTransfer,
    KVConnector,
    KVConnectorTransfer,
    TransferDirection,
)
from max.profiler import traced

_logger = logging.getLogger("max.pipelines")


class _DkvClient(Protocol):
    """The ``dkv_connector.DkvConnector`` surface this shim drives.

    The concrete class is a pyo3 extension type imported lazily inside
    :meth:`DKVConnector.__init__`, so this is the only place the FFI surface is
    written down. One client owns one leaf's buffers and that leaf's store
    namespace; ``group_id`` selects the leaf on every per-leaf call.
    """

    def load(
        self,
        *,
        group_id: int,
        block_ids: Sequence[int],
        block_hashes: Sequence[int],
        hint: bytes | None = ...,
    ) -> int: ...

    def offload(
        self,
        *,
        group_id: int,
        block_ids: Sequence[int],
        block_hashes: Sequence[int],
    ) -> None: ...

    def touch(self, *, group_id: int, block_hashes: Sequence[int]) -> None: ...

    def wait_for_loads(self) -> None: ...

    def wait_for_offloads(self) -> None: ...

    def metrics(self) -> Mapping[str, Any]: ...

    def take_metrics(self) -> Mapping[str, Any]: ...

    def broadcast_peer_count(self) -> int: ...


_DKV_RECENCY_FULL_SEQUENCE = 1
_DKV_RECENCY_SLIDING_WINDOW = 2
# Jenga's per-leaf null block is bid 0 (``JengaBlockPool.null_little_blocks``),
# which is what a sliding leaf's slots below the window hold.
_NULL_BLOCK_ID = 0


def _validate_dkv_leaves(leaves: Mapping[str, KVCacheGroupId]) -> None:
    """Rejects leaf trees the dKV connector cannot serve.

    Called once, from ``DKVConnector.__init__``. Every caller reaches the
    connector through that constructor, so neither ``create_connector`` nor a
    cache manager repeats the check -- a cache manager especially should not,
    since it would have to know which connector is configured.

    Args:
        leaves: Leaf id to attention group, as ``params.leaves()`` names them.

    Raises:
        ValueError: If the tree is empty, holds a group that is neither full
            attention nor sliding window, or mixes sliding-window leaves with
            different ``window_size`` values -- ``load`` derives ONE window
            length for every sliding leaf, so two windows cannot be honored.
    """
    if not leaves:
        raise ValueError("DKVConnector requires at least one KV cache leaf")
    if not all(
        group_id.is_full() or group_id.is_sliding_window()
        for group_id in leaves.values()
    ):
        raise ValueError(
            "DKVConnector supports full-attention and sliding-window leaves "
            f"only. Found: {leaves}"
        )
    windows = {
        group_id.window_size
        for group_id in leaves.values()
        if group_id.is_sliding_window()
    }
    if len(windows) > 1:
        raise ValueError(
            "DKVConnector requires all sliding-window leaves to share one "
            f"window_size; found {windows} in {leaves}"
        )


def _leaf_wire_ids(leaves: Mapping[str, KVCacheGroupId]) -> dict[str, int]:
    """Dense unique store ids in ``leaves()`` order, starting at 1.

    Positional, and safe to be positional: the id only has to mean the same
    leaf to everyone sharing a store, and a store is keyed by
    ``(tenant_id, kv_config_hash, kv_shard_id)``. :func:`_layout_fields` folds
    every child's INDEX AND NAME into ``kv_config_hash``, so adding, removing
    or reordering a leaf flips the fingerprint and dKV purges and rebuilds the
    share instead of reattaching. A reader therefore can never resolve id 2 to
    a different leaf than the writer meant -- across a restart, across DP
    replicas, or across instances a cache hint names -- because a leaf set that
    would renumber cannot present the same hash.

    Ids must fit ``u16``: ``StoreFacade::negotiate_groups`` narrows to one and
    rejects anything wider, because a truncated id silently aliases two leaves
    onto one store namespace. Numbering from 1 also keeps 0 --
    ``GROUP_ID_UNSPECIFIED``, which names no leaf -- off the wire, and makes a
    single-leaf tree present the same id (1 == ``GroupId::FullAttention``) that
    every pre-per-leaf client sent. Recency is independent of the id, so extra
    SWA leaves (values and scales) can all slide without the id encoding the
    kind.
    """
    return {leaf_id: i for i, leaf_id in enumerate(leaves, start=1)}


def _group_recency(group_id: KVCacheGroupId) -> int:
    if group_id.is_sliding_window():
        return _DKV_RECENCY_SLIDING_WINDOW
    return _DKV_RECENCY_FULL_SEQUENCE


def _sliding_row(
    leaf_block_ids: Sequence[int], window_length: int, num_loaded: int
) -> list[int]:
    """One sliding leaf's loaded row: nulls below the window, blocks inside it.

    Every leaf's row must be exactly ``num_loaded`` long -- Jenga rejects ragged
    rows -- so the real window is capped at ``num_loaded`` and the remaining
    leading slots are the null block. This mirrors what
    ``SlidingWindowKVGroupCoordinator.claim_hit_blocks`` builds for the device
    tier: ``low = max(0, N - blocks_in_window)`` nulls, then the window.

    ``window_length == 0`` (``window_size == 1``, so the window spans no whole
    block) is an all-null row of the right length, not an empty one -- the same
    thing the device coordinator produces, which counts it as a full hit.
    """
    if num_loaded <= 0:
        return []
    real = min(num_loaded, max(window_length, 0))
    window = list(leaf_block_ids)[-window_length:][:real] if real else []
    return [_NULL_BLOCK_ID] * (num_loaded - real) + window


def _to_dkv_u64(h: bytes) -> int:
    """Packs a connector-level block hash into the 64-bit dkv wire key.

    The dkv proto stays ``uint64 seq_hash``. Accepts the canonical bytes
    forms the block hasher produces:

    * 8 bytes (``ahash64`` / ``sha256_64``): used as-is, big-endian unsigned.
    * 32 bytes (full ``sha256``): truncated to the first 8 bytes (big-endian
      unsigned). Byte-identical to ``sha256_64`` of the same digest, so the
      same logical block collapses to the same dkv key under either algo.

    Args:
        h: Canonical block-hash bytes, length 8 or 32.

    Returns:
        Unsigned 64-bit integer the Rust client carries on the wire.

    Raises:
        ValueError: If ``h`` is not exactly 8 or 32 bytes long.
    """
    if len(h) not in (8, 32):
        raise ValueError(
            f"DKVConnector block hash must be 8 or 32 bytes, got {len(h)}"
        )
    return int.from_bytes(h[:8], "big", signed=False)


def _buffer_nbytes(buffer: Buffer) -> int:
    """Returns the byte length of a device buffer.

    Computed as the element count times the element width, matching what the
    Rust client divides by ``total_num_pages`` to derive the per-page stride.
    """
    return buffer.num_elements * buffer.dtype.size_in_bytes


def _group_units_by_shard(
    kv_memory: Sequence[KVCacheMemory],
) -> tuple[list[tuple[int, list[tuple[int, int]]]], bool]:
    """Groups one replica's KV memory units into per-shard unit lists.

    Each unit carries every TP shard, so shard ``s`` is ``mem.buffers[s]``
    whether the unit is replicated or sharded; the two layouts need no separate
    handling. The Rust client concatenates each shard's units in this order
    into one dKV block. That block matches the CPU block the tiered connector
    builds only when every unit is sharded, because the tiered connector
    stores a replicated unit once while dKV stores it once per shard.

    Args:
        kv_memory: One replica's offload-ready KV memory units.

    Returns:
        A ``(shards, shards_are_identical)`` pair, where ``shards`` has one
        ``(device_id, [(ptr, nbytes), ...])`` entry per TP shard in canonical
        device order, and ``shards_are_identical`` reports whether every unit
        is replicated.

    Raises:
        ValueError: If unit page counts disagree, or if units span different
            device topologies.
    """
    if not kv_memory:
        raise ValueError("kv_memory must contain at least one unit")

    # every unit must agree on the page count because the Rust client derives
    # each unit's per-page stride by dividing its length by one shared count
    unique_total_num_pages = {mem.total_num_pages for mem in kv_memory}
    if len(unique_total_num_pages) > 1:
        raise ValueError(
            "all kv_memory units must have the same total_num_pages; got "
            f"{unique_total_num_pages}"
        )

    # every unit replicated makes each shard's block byte-identical, so the
    # client may store shard 0 alone; one sharded unit makes them differ.
    shards_are_identical = all(mem.replicated for mem in kv_memory)

    # every unit must span the same device topology so shard s names the same
    # device in every unit (mirrors BlockOffloadEngine)
    topologies = {
        tuple(buffer.device.id for buffer in mem.buffers) for mem in kv_memory
    }
    if len(topologies) > 1:
        raise ValueError(
            "all KVCacheMemory units must share the same TP device topology; "
            f"got {sorted(topologies)}"
        )

    topology = next(iter(topologies))
    shards = [
        (
            device_id,
            [
                (
                    mem.buffers[rank]._data_ptr(),
                    _buffer_nbytes(mem.buffers[rank]),
                )
                for mem in kv_memory
            ],
        )
        for rank, device_id in enumerate(topology)
    ]
    return shards, shards_are_identical


def _shard_unit_strides(kv_memory: Sequence[KVCacheMemory]) -> list[int]:
    """Derives one shard's per-unit page strides in canonical order.

    A dKV block holds one shard's buffer units concatenated, so the layout
    fingerprint folds the strides of a single shard's unit list rather than
    the flat per-physical-buffer list. This keeps the folded shape identical
    between replicated and sharded layouts, where the flat list would
    otherwise repeat each unit once per shard. Shard 0 stands in for every
    shard because each shard carries one entry per unit by construction and
    the Rust config validates that the stride vectors match.

    Args:
        kv_memory: One replica's offload-ready KV memory units.

    Returns:
        The per-page byte stride of each of shard 0's units in canonical
        order.
    """
    shards, _ = _group_units_by_shard(kv_memory)

    # every unit length is the page count times the stride because the
    # grouping validated a uniform page count over 2-D [pages, stride] views
    total_num_pages = kv_memory[0].total_num_pages
    _, units = shards[0]

    return [nbytes // total_num_pages for _, nbytes in units]


# Default wall-clock budget for admitting (connect + handshake) one per-replica
# dKV client. dKV is co-located and usually up within seconds, but a still-
# starting server (connection refused), a cold slab warm-up (deferred region
# carve), or a node with no room until a departed tenant's pages drain can all
# take much longer; admission retries transient failures until this budget is
# spent, then fails model load. Override via MODULAR_DKV_ADMISSION_TIMEOUT_S.
#
# Sized above a worst-case cold carve, which budgets to roughly 600s for a
# 1.6 TiB slab. A single attempt does not have to cover that carve, because the
# server builds a share single-flight and a retry blocks on the in-flight build
# rather than starting a second one, so what matters is that the budget spans
# enough attempts to outlast it.
_DEFAULT_ADMISSION_TIMEOUT_S = 600.0
_ADMISSION_INITIAL_BACKOFF_S = 1.0
_ADMISSION_MAX_BACKOFF_S = 10.0

# The Rust client's default per-attempt handshake bound, mirrored from
# DEFAULT_HANDSHAKE_REQUEST_TIMEOUT in dkv-connector/src/transport.rs, purely to
# size the admission floor below.
#
# Deliberately the constant and not MODULAR_DKV_HANDSHAKE_TIMEOUT_S. That
# variable belongs to the transport, which reads it permissively (anything
# unparsable, non-positive, or above its 3600s cap silently falls back), and a
# second parser here would disagree with it in both directions: rejecting values
# the transport accepts, and sizing the floor off values the transport ignores.
_DEFAULT_HANDSHAKE_TIMEOUT_S = 60.0

# An admission budget has to cover several whole attempts. At or just above the
# per-attempt timeout it is spent inside the first attempt and retries nothing,
# so a refusal that would clear in seconds fails model load instead. That is the
# shape of CLIN-1842.
#
# The floor is computed against the DEFAULT per-attempt timeout, not the
# configured one. An operator raising the handshake timeout is covering one long
# cold carve, not asking for a proportionally longer budget: a retry blocks on
# the in-flight single-flight build rather than starting a second one, so the
# per-attempt bound does not multiply the work. Scaling the floor by it would
# reject configurations that raise both together, which is exactly what the
# in-tree kimi26 deployment and transport.rs both instruct.
_MIN_ADMISSION_ATTEMPTS = 4


# Env-var overrides for the Rust client's background heartbeat poller, mapped to
# the constructor keywords they feed. Every one is optional: an unset variable is
# omitted from the call so the Rust default applies, which keeps the poller on the
# timings it has always used. A shorter interval detects a dKV restart sooner at
# the cost of one more probe per interval from every replica.
_HEARTBEAT_ENV_KWARGS = {
    "MODULAR_DKV_HEARTBEAT_INTERVAL_MS": "heartbeat_interval_ms",
    "MODULAR_DKV_HEARTBEAT_REQUEST_TIMEOUT_MS": "heartbeat_request_timeout_ms",
    "MODULAR_DKV_HEARTBEAT_RECONNECT_TIMEOUT_MS": "heartbeat_reconnect_timeout_ms",
    "MODULAR_DKV_HEARTBEAT_RECONNECT_COOLDOWN_MS": (
        "heartbeat_reconnect_cooldown_ms"
    ),
    "MODULAR_DKV_HEARTBEAT_MAX_FAILURES": "heartbeat_max_failures",
    "MODULAR_DKV_HEARTBEAT_DEGRADED_WARN_EVERY": (
        "heartbeat_degraded_warn_every"
    ),
}


def _heartbeat_overrides() -> dict[str, int]:
    """Collects the heartbeat-poller overrides set in the environment.

    Returns:
        The Rust client constructor keywords for the variables in
        :data:`_HEARTBEAT_ENV_KWARGS` that are set, keyed by keyword name. An
        unset variable is absent, leaving the Rust default in place.

    Raises:
        ValueError: If a variable is set to something other than a non-negative
            integer. Failing model load beats silently polling on an unintended
            cadence, and a negative value is worth catching here rather than at
            the extension, whose keywords are unsigned and so reject it with an
            ``OverflowError`` about converting a negative int that names no
            variable. Which non-negative values are meaningful is deliberately
            not checked here, because that is per-field and belongs to
            ``HeartbeatConfig::validate`` on the Rust side, where the reasons
            live: a zero cooldown means no spacing between reconnects and is
            legitimate, while a zero interval would spin the poll loop.
    """
    overrides: dict[str, int] = {}
    for env_var, keyword in _HEARTBEAT_ENV_KWARGS.items():
        raw = os.getenv(env_var, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{env_var} must be an integer number, got {raw!r}"
            ) from exc
        if value < 0:
            raise ValueError(f"{env_var} must not be negative, got {value}")

        overrides[keyword] = value

    return overrides


def _required_nixl_backend() -> NixlBackendType:
    """The validated NIXL transfer backend this connector must use.

    Reads ``MODULAR_NIXL_TRANSFER_BACKEND``, which is mandatory: dKV has no
    auto-select mode, so there is nothing for an unset variable to mean. An
    unset, empty, or unknown value fails model load naming the variable and
    the accepted set, rather than surfacing later as a handshake mismatch or a
    transfer against a transport that cannot do the job. ``auto`` reaches the
    validator like any other unknown name — it was the dKV server's own
    spelling for the mode that was removed, so an operator mirroring
    ``DKV_MEMXFER_BACKEND`` gets a clear error instead of a silent fallback.

    This deliberately differs from the KV transfer engine, which assumes
    ``"ucx"`` when the variable is unset. The two consumers share the
    validator, not the default.

    Returns:
        The transport to hand the Rust client, normalized to lowercase.

    Raises:
        ValueError: If the variable is unset, empty, or not a supported
            backend.
    """
    raw = os.getenv(NIXL_BACKEND_ENV_VAR, "").strip()
    if not raw:
        raise ValueError(
            f"{NIXL_BACKEND_ENV_VAR} must be set to the NIXL transport this "
            "host's fabric needs (libfabric on EFA, ucx on InfiniBand). The "
            "dKV connector has no auto-select mode, so there is no default. "
            f"Supported backends: {sorted(SUPPORTED_NIXL_BACKENDS)}"
        )
    return validate_nixl_backend(raw)


def _required_operator_env() -> tuple[NixlBackendType, str]:
    """The operator-injected settings dKV refuses to start without.

    Both the NIXL transport and the tenant identity are the deployment
    operator's to set, and a pod missing one is usually missing both, so they
    are validated in one pass: an operator learns every missing variable from
    a single model load instead of discovering the next one after fixing the
    first. A lone problem is reported on its own, so the common
    one-variable-missing message stays as direct as it was.

    Returns:
        The transport to hand the Rust client and the tenant identity.

    Raises:
        ValueError: If either variable is unset or empty, or the transport is
            not a supported backend, naming every problem found.
    """
    problems: list[str] = []

    backend: NixlBackendType | None = None
    try:
        backend = _required_nixl_backend()
    except ValueError as exc:
        problems.append(str(exc))

    # MODULAR_DKV_TENANT_ID is injected by the operator (the trust boundary —
    # not a user-facing override flag, which would be forgeable). It is
    # REQUIRED: dKV has no default/legacy single-tenant path, so an unset or
    # empty value fails model load rather than silently keying an unfenced
    # shared store. Every DP replica handshakes the same per-tenant identity
    # (kv_shard_id/replica_id zeroed), so the server keys ONE region-sharded
    # store per tenant_id (backend dedup).
    tenant_id = os.getenv("MODULAR_DKV_TENANT_ID", "")
    if not tenant_id:
        problems.append(
            "dKV requires MODULAR_DKV_TENANT_ID to be set to a non-empty "
            "tenant identity (the operator injects it); the legacy "
            "empty-tenant default path has been removed."
        )

    if problems:
        raise ValueError(" ".join(problems))

    assert backend is not None  # no problems recorded means it parsed
    return backend, tenant_id


def _dtype_tag(dtype: object) -> str:
    """Returns a stable, restart-invariant text tag for a ``DType``.

    Uses the enum member ``name`` (e.g. ``"bfloat16"``) when present, else
    ``str(dtype)``. Never uses Python's per-process-randomized ``hash``, so the
    fingerprint it feeds is identical across process restarts.
    """
    return getattr(dtype, "name", None) or str(dtype)


def _layout_fields(
    params: KVCacheParams | MultiKVCacheParams,
) -> list[tuple[str, str]]:
    """Folds one cache's byte-layout into ordered ``key=value`` fields.

    A leaf :class:`KVCacheParams` contributes its per-shard geometry; a
    :class:`MultiKVCacheParams` tree contributes a ``multi`` marker, its child
    count, and each child's fields recursively, prefixed by the child's index
    and name in the tree's insertion order. That order is deterministic for a
    fixed model config and matches the declared leaf order the concatenated
    block (and thus ``unit_strides``) follows, so folding the index and name
    makes any child add/remove/reorder flip the fingerprint.

    Excludes the contract version and the concatenated ``unit_strides``, which
    :func:`_kv_config_hash` owns at the top level so a multi-cache tree folds
    one stride list spanning all its leaves.
    """
    if isinstance(params, MultiKVCacheParams):
        fields: list[tuple[str, str]] = [
            ("multi", "1"),
            ("child_count", str(len(params.children))),
        ]
        for i, (name, child) in enumerate(params.children.items()):
            # children are leaf KVCacheParams or nested MultiKVCacheParams
            assert isinstance(child, (KVCacheParams, MultiKVCacheParams))
            fields.append((f"c{i}.name", name))
            fields += [(f"c{i}.{k}", v) for k, v in _layout_fields(child)]
        return fields

    quant = params.kvcache_quant_config
    if params.quantized_kv_cache and quant is not None:
        quant_desc = (
            f"{_dtype_tag(quant.scale_dtype)}:{quant.quantization_granularity}"
        )
    else:
        quant_desc = "none"
    block_value_bytes = (
        math.prod(params.shape_per_block) * params.dtype.size_in_bytes
    )
    return [
        ("dtype", _dtype_tag(params.dtype)),
        ("dtype_bytes", str(params.dtype.size_in_bytes)),
        ("kv_dim", str(params.kv_dim)),
        ("head_dim", str(params.head_dim)),
        ("num_layers", str(params.num_layers)),
        ("page_size", str(params.page_size)),
        ("n_kv_heads_per_device", str(params.n_kv_heads_per_device)),
        ("tensor_parallel_degree", str(params.tensor_parallel_degree)),
        ("quant", quant_desc),
        ("block_value_bytes", str(block_value_bytes)),
        # Folded because it is what makes this leaf windowed, and so what
        # decides the GroupRecency it advertises (`_group_recency` reads the
        # same windowing off the leaf's `KVCacheGroupId`). Recency is NOT part
        # of the negotiated geometry the reattach path compares -- so without
        # it two configs differing only in whether a leaf is windowed hash
        # identically while advertising a different eviction policy for the
        # same id. A share built under the other one keeps its GroupKind forever
        # (`build_loaded` never re-runs), and a FullAttention share handed a
        # tail-anchored window reverse-walks it, making the NEWEST in-window
        # block the coldest. That is an inversion, not a degradation. Free to
        # add here because v=3 already purges every share. A real
        # `KVCacheParams` always carries the field -- `None` on a full leaf, an
        # int on a sliding one -- so the `getattr` default is reached only by a
        # params stand-in that predates it, which is why it is not `None`: the
        # fold has to stay a stable string for whatever a caller hands over.
        ("window_size", str(getattr(params, "window_size", -1))),
    ]


def _kv_config_hash(
    params: KVCacheParams | MultiKVCacheParams, unit_strides: Sequence[int]
) -> int:
    """Computes the stable 64-bit KV-cache layout fingerprint.

    This is the producer of ``ExchangeMetadataRequest.kv_config_hash``. Two KV
    shares hold byte-identical KV only when their ``(tenant_id, kv_config_hash,
    kv_shard_id)`` all match, so this value must capture everything that makes
    two KV blocks byte-incompatible and must be byte-stable across restarts (the
    dKV reattach / compatibility consumer, CLIN-1474, compares it).

    Contract (bump the ``v`` field on any change). The fields below are folded,
    in this order, into a canonical newline-joined ``key=value`` UTF-8 string;
    the hash is the first 8 bytes of that string's SHA-256 as a big-endian
    unsigned integer — the same 64-bit convention as the dkv ``seq_hash`` and
    :func:`_to_dkv_u64`:

    * ``v`` — contract version (``3``; see the bump note below).
    * the cache geometry from :func:`_layout_fields`: for a single-group leaf,
      its dtype/kv_dim/head_dim/num_layers/page_size/head-count/TP/quant/value
      bytes (unchanged from the ``v=2`` leaf encoding, so single-group
      fingerprints stay byte-identical); for a :class:`MultiKVCacheParams` tree
      (speculative draft+target, quantized values+scales), a ``multi`` marker
      plus each child's fields folded recursively under its index and name.
    * ``unit_strides`` — comma-joined per-page byte stride of one shard's
      buffer units in declared leaf order (values, quant scales, indexer,
      draft, and so on), derived by :func:`_shard_unit_strides`. A
      shard's dKV block is these strides concatenated across the WHOLE cache
      tree, so any change to the unit set or its ordering makes stored blocks
      byte-incompatible and must flip the hash. Folding one shard's subsequence
      rather than the flat physical buffer list keeps the folded shape identical
      between replicated and sharded layouts; the shard count is already
      pinned by ``tensor_parallel_degree``.

    Model/weights identity is deliberately NOT folded here: a different model is
    a different Mammoth deployment, hence a different ``tenant_id`` (already part
    of the dedup key). Reattaching persisted shares across a same-``tenant_id``
    weights swap (CLIN-1474) must additionally fold a weights/version fingerprint
    threaded from the pipeline config — a documented follow-up, out of scope for
    this handshake.

    ``v=3`` marks the per-leaf group geometry. Before it, a shard advertised no
    groups and the server fell back to ``ShardBuffers::bytes_per_page``, so a
    values+scales tree stored ONE group of ``V+S`` bytes; now it advertises
    ``[(1, V), (2, S)]`` and stores one group per leaf. ``_layout_fields`` and
    ``unit_strides`` do not move under that change -- both encode the same
    per-unit strides either way -- so without the bump a live share would take
    the matching-hash reattach branch in ``prepare_share_slot``, find a
    different geometry, and return ``StorageError::Config``. That surfaces as a
    ``ValueError``, which ``_is_permanent_admission_error`` treats as permanent,
    so the pod would crashloop rather than rebuild the share. (``v=2`` was
    itself the bump for the block layout becoming the concatenation of every
    buffer unit rather than the value buffer alone.)

    The bump invalidates EVERY share, not only the multi-unit trees whose
    geometry actually moved: ``v`` is folded for every config, so a
    single-full-leaf llama share is purged and rebuilt too even though it
    advertises byte-identically before and after. That is the intended trade --
    purge and cold-start beats crashloop, and a per-field bump would have to be
    read by a server that does not have one -- but it means the rollout cost is
    one cold start per TENANT, not per hybrid model.

    Rollout note for a deployment with G2 enabled: ``build_g2`` handles a stamp
    mismatch with an inline ``remove_dir_all(disk_dir)`` ON THE HANDSHAKE PATH,
    which is exactly the work ``unlink_condemned_g2`` exists to keep off it
    ("releasing the extents of a preallocated pool is far too slow to hold up a
    restart"). Every tenant's first post-upgrade handshake pays that delete
    synchronously, against an admission budget sized for a 1.6 TiB carve rather
    than for deleting a multi-TiB pool. Clearing the G2 subtree before rolling
    this out removes the delete entirely, since the ``fallocate`` was going to
    happen either way.

    Args:
        params: The KV-cache parameters for this deployment — a single-group
            leaf or a multi-cache tree.
        unit_strides: Per-page byte stride of one shard's buffer units in
            canonical order, from :func:`_shard_unit_strides`.

    Returns:
        The 64-bit layout fingerprint.
    """
    fields = [
        ("v", "3"),
        *_layout_fields(params),
        ("unit_strides", ",".join(str(s) for s in unit_strides)),
    ]
    canonical = "\n".join(f"{k}={v}" for k, v in fields).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(canonical).digest()[:8], "big", signed=False
    )


def _resolve_replica_identities(
    num_replicas: int,
    params: KVCacheParamInterface,
    unit_strides: Sequence[int],
) -> tuple[int, list[tuple[int, int]]]:
    """Resolves the per-DP-replica dKV handshake identity.

    Returns the shared ``kv_config_hash`` and, per DP replica in order, its
    ``(kv_shard_id, replica_id)``. Under backend dedup one store is keyed per
    tenant: every DP replica handshakes the same zeroed store-key identity
    ``(kv_shard_id, replica_id) == (0, 0)`` and registers its full TP GPU set in
    one client, so the dKV server keys a single (region-sharded) store per
    ``tenant_id``. Every topology is admitted — single-group and shallow
    multi-cache (speculative / quantized) alike; the layout hash folds the whole
    cache tree (:func:`_kv_config_hash`).

    The MHA/GQA-vs-MLA distinction is deliberately NOT in the store key — it
    lives in the per-block ``BlockKey`` ``tp_shard_id`` the Rust client derives
    from ``num_participating_shards`` — so identical-KV shards (DP replicas, or
    MLA's replicated latent) dedup while distinct head shards co-reside in the
    one store, never deduped against each other.

    Args:
        num_replicas: Number of DP replicas (one dKV client each, registering
            that replica's full TP GPU set).
        params: KV-cache parameters, folded into the shared layout hash.
        unit_strides: Per-page byte stride of one shard's buffer units in
            canonical order, folded into the layout hash.

    Returns:
        ``(kv_config_hash, [(kv_shard_id, replica_id), ...])`` — one zeroed
        identity per DP replica.

    Raises:
        ValueError: If ``num_replicas`` disagrees with ``data_parallel_degree``.
    """
    # Every KV topology resolves here: single-group and shallow multi-cache
    # (speculative draft+target, quantized values+scales) alike. A multi-cache
    # block rides as the concatenated-unit block _group_units_by_shard builds,
    # and the layout hash folds the whole cache tree (_kv_config_hash). True
    # per-group tagging for independent hybrid/SWA groups is a separate
    # block-manager effort (the connector keys every op under the full-attention
    # group today), out of scope here.
    assert isinstance(params, (KVCacheParams, MultiKVCacheParams))
    if num_replicas != params.data_parallel_degree:
        raise ValueError(
            f"replica count {num_replicas} does not match data_parallel_degree "
            f"{params.data_parallel_degree}; the per-replica client mapping "
            "would be wrong"
        )
    # One store per tenant: every DP replica handshakes the same zeroed store-key
    # identity ((kv_shard_id, replica_id) == (0, 0)); the shard/replica
    # distinctions are carried in the per-block BlockKey, not here.
    return _kv_config_hash(params, unit_strides), [(0, 0)] * num_replicas


# Exception types that always signal a permanent config or programming bug in
# the admission path, never a transient/connection failure. Retrying these just
# burns the whole admission budget before a real bug surfaces, so they
# short-circuit the retry loop. ``ValueError`` also covers the pyo3
# ``ConnectorError::Config`` mapping and this module's own argument validation;
# the rest are the shapes a bug inside ``_make_client`` raises (a bad attribute,
# wrong call signature, undefined name, missing key, or a failed import).
_PERMANENT_ADMISSION_EXC_TYPES: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    AttributeError,
    NameError,
    KeyError,
    ImportError,
)


def _is_permanent_admission_error(exc: Exception) -> bool:
    """Returns whether an admission failure will not recover on retry.

    Retrying is worthwhile for a still-starting dKV (connection refused),
    ``NotReady`` timeouts, and transient transport errors; it is pointless for a
    caller/config bug or a programming bug. A permanent failure is one of
    :data:`_PERMANENT_ADMISSION_EXC_TYPES` — a config error (the pyo3
    ``ConnectorError::Config`` maps to :class:`ValueError`) or a programming bug
    such as :class:`AttributeError` / :class:`TypeError` raised inside
    ``_make_client`` — or a runtime error the Rust layer tagged
    ``[retriable=false]``. Everything else (including an untagged "failed to
    connect to dKV" error) is treated as transient and retried.
    """
    if isinstance(exc, _PERMANENT_ADMISSION_EXC_TYPES):
        return True
    return "[retriable=false]" in str(exc)


def _resolve_admission_timeout_s(env: Mapping[str, str] | None = None) -> float:
    """Resolves the admission retry budget, raising it to cover several attempts.

    A budget too small to retry is corrected with a warning rather than
    rejected. The point of the floor is to guarantee that a transient refusal is
    retried; failing model load at construction would trade one broken outcome
    for another, and would do it to deployments that merely pinned the old
    default.

    Args:
        env: Environment to read; defaults to :data:`os.environ`. Injectable for
            tests.

    Returns:
        The admission budget in seconds, never below the floor.

    Raises:
        ValueError: If ``MODULAR_DKV_ADMISSION_TIMEOUT_S`` is set but not a
            positive, finite number. This shim is its only reader, so there is
            no permissive parser to mirror and a typo is worth surfacing.
    """
    env = os.environ if env is None else env

    raw = env.get("MODULAR_DKV_ADMISSION_TIMEOUT_S")
    if raw is None:
        admission_s = _DEFAULT_ADMISSION_TIMEOUT_S
    else:
        try:
            admission_s = float(raw)
        except ValueError:
            raise ValueError(
                f"MODULAR_DKV_ADMISSION_TIMEOUT_S={raw!r} is not a number"
            ) from None
        if not (0 < admission_s < float("inf")):
            raise ValueError(
                f"MODULAR_DKV_ADMISSION_TIMEOUT_S={raw!r} must be a positive, "
                f"finite number"
            )

    minimum = _MIN_ADMISSION_ATTEMPTS * _DEFAULT_HANDSHAKE_TIMEOUT_S
    if admission_s < minimum:
        _logger.warning(
            "dKV admission budget %gs is below %.0fs, the time %d handshake "
            "attempts can take, so a transient refusal would fail model load "
            "instead of being retried; using %.0fs. Set "
            "MODULAR_DKV_ADMISSION_TIMEOUT_S at or above %.0fs to silence this.",
            admission_s,
            minimum,
            _MIN_ADMISSION_ATTEMPTS,
            minimum,
            minimum,
        )
        return minimum
    return admission_s


def _admit_with_retry(
    factory: Callable[[], object],
    *,
    timeout_s: float,
    label: str = "",
    initial_backoff_s: float = _ADMISSION_INITIAL_BACKOFF_S,
    max_backoff_s: float = _ADMISSION_MAX_BACKOFF_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> object:
    """Calls ``factory`` until it succeeds, retrying transient failures.

    Retries with exponential backoff (capped at ``max_backoff_s``) until
    ``factory`` returns, a permanent error surfaces
    (:func:`_is_permanent_admission_error`), or ``timeout_s`` is exhausted (the
    last exception is then re-raised — the readiness gate: model load fails if a
    replica never admits). ``monotonic`` and ``sleep`` are injectable for tests.

    Args:
        factory: Zero-arg callable performing one admission attempt.
        timeout_s: Total wall-clock retry budget.
        label: Short identifier for the retry log line (e.g. ``"replica 3"``).
        initial_backoff_s: First backoff, doubled each retry.
        max_backoff_s: Backoff ceiling.
        monotonic: Monotonic clock source (injectable).
        sleep: Sleep function (injectable).

    Returns:
        Whatever ``factory`` returns on success.
    """
    deadline = monotonic() + timeout_s
    backoff = initial_backoff_s
    attempt = 0
    while True:
        attempt += 1
        try:
            return factory()
        except Exception as exc:
            if _is_permanent_admission_error(exc):
                raise
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise
            _logger.warning(
                "dKV admission%s attempt %d failed (%s); retrying",
                f" ({label})" if label else "",
                attempt,
                exc,
            )
            sleep(min(backoff, max_backoff_s, remaining))
            backoff *= 2


class _DKVCompletedTransfer(CompletedTransfer):
    """Completed dKV transfer with distinct per-leaf device block rows.

    Writes the base class's ``_g0_blocks_per_leaf`` from outside, as
    ``rust_tier_connector`` also does. A per-leaf mapping is the general case
    now, so the base constructor should own the shape and this subclass should
    not exist -- SERVOPT-1614.
    """

    def __init__(
        self,
        direction: TransferDirection,
        blocks_per_leaf: Mapping[str, Sequence[int]],
    ) -> None:
        super().__init__(direction)
        self._g0_blocks_per_leaf = {
            leaf_id: list(block_ids)
            for leaf_id, block_ids in blocks_per_leaf.items()
        }


class DKVConnector(KVConnector):
    """``KVConnector`` backed by the ``dkv_connector`` Rust client.

    A single instance serves every DP replica. The underlying Rust client is
    inherently per-endpoint (its ``load``/``offload`` reference block ids into
    one registered device-buffer set and carry no replica/group key), so this
    shim owns one client per leaf per DP replica in ``self._clients`` and routes
    each call to the matching leaf client for the request's processing replica.

    Under backend dedup there is one client per DP replica. Each client
    registers its replica's FULL TP GPU set (so MLA keeps
    ``device_buffers.len() == tp`` and NVLink broadcast stays engaged), and every
    DP replica of a tenant handshakes the same per-tenant store identity
    (``kv_shard_id`` / ``replica_id`` zeroed), so the server keys ONE
    region-sharded store per ``tenant_id``. :meth:`load` / :meth:`offload`
    therefore issue exactly one call, to the processing replica's client. The
    MHA/GQA-vs-MLA distinction is carried by the per-block ``BlockKey``
    ``tp_shard_id`` the Rust client builds, not by the store key: identical-KV
    shards dedup, distinct head shards co-reside.
    """

    @traced
    def __init__(
        self,
        leaves: Mapping[str, KVCacheGroupId],
        replica_kv_memory: Sequence[Mapping[str, KVCacheMemory]],
        local_block_store_endpoint: str,
        devices: Sequence[Device],
        params: KVCacheParamInterface,
    ) -> None:
        """Constructs and admits one dKV Rust client per DP replica.

        Args:
            leaves: Mapping from leaf identity to attention group.
            replica_kv_memory: Per-replica offload-ready KV memory by leaf.
            local_block_store_endpoint: Co-located dKV control-plane endpoint.
            devices: The pipeline's flat, ordered device list across replicas.
            params: KV-cache parameters, folded into the tenant store's layout
                ``kv_config_hash``.

        Raises:
            ValueError: If either operator-injected variable is missing —
                ``MODULAR_NIXL_TRANSFER_BACKEND`` (dKV auto-selects no
                transport) or ``MODULAR_DKV_TENANT_ID`` (dKV has no
                default/legacy single-tenant path, so it fails model load
                rather than silently keying an unfenced shared store). Both are
                checked in one pass, so a pod missing both is told about both.
        """
        # First, and before the deferred import below: this is a pure check on
        # the leaf tree, it is the cheapest thing here, and an unsupported tree
        # is a config error nothing further can make serviceable. Running it
        # ahead of the import also means a bad tree reports itself as such on a
        # host where the optional extension is missing, rather than as a
        # confusing ModuleNotFoundError. The ONLY place the tree is validated:
        # every caller builds through this constructor, so neither
        # create_connector nor a cache manager repeats it.
        _validate_dkv_leaves(leaves)

        # Deferred so importing this module (e.g. by a non-dKV pipeline) does
        # not require the optional, runtime-provided dkv_connector extension to
        # be installed.
        from dkv_connector import DkvConnector as _DkvConnectorClient

        # The Rust client creates a NIXL agent, which dlopens the transport
        # plugin RTLD_LOCAL; the UCX flavors carry unresolved CUDA/NVML (and,
        # for the verbs flavors, rdma-core) symbols that must already be
        # RTLD_GLOBAL in this process or the plugin faults as it loads. The
        # libfabric flavor links its stack and needs none of this, but the
        # preload skips absent libraries, so it stays backend-agnostic here
        # rather than duplicating the backend dispatch.
        preload_nixl_plugin_deps()

        if not replica_kv_memory or not all(replica_kv_memory):
            raise ValueError(
                "DKVConnector requires at least one KV cache buffer per replica"
            )
        if any(set(memory) != set(leaves) for memory in replica_kv_memory):
            raise ValueError(
                "DKVConnector leaf mapping must match every replica's KV memory"
            )
        self._leaves = dict(leaves)
        self._wire_ids = _leaf_wire_ids(self._leaves)
        self._page_size = params.page_size
        self._derive_leaf_shape()

        listen_port = int(os.getenv("MODULAR_DKV_NIXL_LISTEN_PORT", "0"))
        # Both operator-injected variables at once, so a pod missing both is
        # told about both (CLIN-1730 made the transport required alongside the
        # tenant identity, and reading them separately reported only the first).
        backend, tenant_id = _required_operator_env()

        # Kill-switch (CLIN-1534): a G0 prefix-cache hit refreshes dKV recency
        # via touch(). Set MODULAR_DKV_DISABLE_G0_TOUCH to make touch() a no-op
        # so the behavior can be backed out with an env var + restart, no code
        # revert. Read once here (not per call); BlockManager always calls
        # touch, the connector decides. Same truthy convention as block_manager's
        # MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE flag.
        self._g0_touch_disabled = os.getenv(
            "MODULAR_DKV_DISABLE_G0_TOUCH", "0"
        ).lower() in (
            "1",
            "true",
            "yes",
            "y",
        )

        num_replicas = len(replica_kv_memory)
        # one shard's per-unit page strides in canonical order, from replica 0
        # because every DP replica runs the same model and config and so the
        # same layout; folded into the layout hash because a shard's dKV block
        # is these strides concatenated
        unit_strides = [
            stride
            for leaf_id in leaves
            for stride in _shard_unit_strides([replica_kv_memory[0][leaf_id]])
        ]
        # No mixed replicated/sharded warning here any more: with one client
        # per leaf, ``_group_units_by_shard`` computes ``shards_are_identical``
        # over a single leaf, so a replicated leaf gets its own ``is_mla=True``
        # client and is stored once rather than once per TP shard. A mixed tree
        # no longer rides the per-shard path, so the footprint multiplier the
        # warning described is gone.
        #
        # That is a pure win only where the NVLink broadcast arms. ``is_mla``
        # makes the client mint one key per hash and fill device buffer 0; the
        # other shards are filled by the broadcast, which needs a same-host
        # endpoint, peer access and tp >= 2, and which the Rust side arms
        # best-effort. Where it does not arm, shards 1..tp-1 keep stale KV --
        # the loop after admission warns on exactly that combination.
        kv_config_hash, replica_identities = _resolve_replica_identities(
            num_replicas, params, unit_strides
        )
        # Total GPUs this tenant occupies across the node (dp * tp), sent in every
        # replica's handshake so the server sizes the ONE per-tenant store to
        # per_gpu_slice * tenant_gpu_count and region-shards it into that many
        # per-GPU NUMA-local regions.
        tenant_gpu_count = num_replicas * params.tensor_parallel_degree
        # The full per-GPU device ordinals this tenant occupies (all dp * tp
        # GPUs, in region order), threaded to every replica's client so each
        # handshake conveys the tenant's WHOLE per-socket NUMA layout. A DP
        # replica's own client registers only its 1/tp of the GPUs, so without
        # this the server would region-shard the store from one replica's
        # single-socket view and bind every region to that socket, breaking
        # NUMA-awareness on the DP path. dKV resolves each ordinal's NUMA node
        # the same way the connector resolves its own.
        tenant_gpu_device_ids = [device.id for device in devices]

        # ``devices`` is the pipeline's flat, ordered device list across every
        # replica; split it into each replica's canonical device order so a
        # client can bind its shard ids to that order. This is the same split the
        # cache manager applies, and it is sourced independently of
        # ``to_memory``, so it is a real cross-check on the buffer ordering rather
        # than a restatement of it.
        devices_per_replica = split_into_groups(list(devices), num_replicas)

        admission_timeout_s = _resolve_admission_timeout_s()

        heartbeat_overrides = _heartbeat_overrides()
        if heartbeat_overrides:
            _logger.info(
                "dKV heartbeat poller overridden from the environment: %s",
                heartbeat_overrides,
            )

        # Each client's connect + handshake ("admission") is retried on transient
        # failures (dKV still starting); model readiness is gated on ALL clients
        # admitting, so a client whose retry budget is exhausted raises here and
        # fails model load rather than serving with a partial dKV.
        # ``self._clients[replica_idx][leaf_id]`` is one client per leaf per DP
        # replica: ``load`` / ``offload`` / ``touch`` index both, and the
        # client-wide fan-outs (wait_for_*, metrics, take_metrics) iterate
        # every client of every replica.
        self._clients: list[dict[str, _DkvClient]] = []
        # Each client registers its replica's FULL TP GPU set (that leaf's units
        # concatenated per shard by _make_client). For MLA this restores
        # device_buffers.len() == tp, so the Rust client's NVLink broadcast +
        # NUMA-local first hop re-engage (the CLIN-1512 per-GPU split had made
        # them inert). The store key stays per-tenant — replica_identities zeros
        # kv_shard_id/replica_id, so every DP replica of a tenant resolves to
        # ONE store — and the per-block BlockKey tp_shard_id carries the
        # MHA/GQA-vs-MLA distinction.
        for idx, (
            replica_memory,
            replica_devices,
            (kv_shard_id, replica_id),
        ) in enumerate(
            zip(
                replica_kv_memory,
                devices_per_replica,
                replica_identities,
                strict=True,
            )
        ):
            clients_for_replica: dict[str, _DkvClient] = {}
            # One geometry per leaf so dKV pages are lcm(each leaf's
            # block_bytes), not lcm of concatenated values+scales. Each entry is
            # the whole triple (id, block bytes, recency) -- the shape
            # ``dkv_connector::GroupSpec`` carries end to end, so nothing
            # downstream re-joins a recency to an id by index.
            group_specs = [
                (
                    self._wire_ids[leaf_id],
                    sum(_shard_unit_strides([replica_memory[leaf_id]])),
                    recency,
                )
                for leaf_id, recency in zip(
                    self._leaves, self._group_recencies, strict=True
                )
            ]
            for leaf_id in self._leaves:
                factory = functools.partial(
                    self._make_client,
                    _DkvConnectorClient,
                    [replica_memory[leaf_id]],
                    local_block_store_endpoint,
                    listen_port,
                    backend,
                    replica_devices,
                    groups=group_specs if self._advertise_groups else [],
                    owner_group_id=self._wire_ids[leaf_id],
                    tenant_id=tenant_id,
                    kv_config_hash=kv_config_hash,
                    kv_shard_id=kv_shard_id,
                    replica_id=replica_id,
                    tenant_gpu_count=tenant_gpu_count,
                    tenant_gpu_device_ids=tenant_gpu_device_ids,
                    heartbeat_overrides=heartbeat_overrides,
                )
                clients_for_replica[leaf_id] = _admit_with_retry(
                    factory,
                    timeout_s=admission_timeout_s,
                    label=f"replica {idx}, leaf {leaf_id}",
                )
            self._clients.append(clients_for_replica)
        # One client per leaf per DP replica. load/offload index
        # self._clients[replica_idx][leaf_id]; each Rust client owns one leaf's
        # buffers and that leaf's store namespace. Guard the mapping fail-loud.
        if len(self._clients) != num_replicas or any(
            set(clients) != set(self._leaves) for clients in self._clients
        ):
            raise RuntimeError(
                "dKV connector client topology does not match its leaf mapping"
            )

        # Surface the Rust connector's MLA NVLink-broadcast status to the serve
        # process: the Rust side logs it through ``tracing`` at ``info``, and the
        # subscriber the connector now installs defaults to ``warn``, so it stays
        # quiet unless ``RUST_LOG`` raises it. ``broadcast_peer_count`` is
        # ``tp - 1`` once the broadcast armed at handshake, and 0 for a non-MLA
        # model, a single device, or a topology without peer access, so log only
        # when it engaged.
        for idx, clients in enumerate(self._clients):
            for leaf_id, client in clients.items():
                peers = client.broadcast_peer_count()
                if peers:
                    _logger.info(
                        "dKV MLA NVLink broadcast enabled: replica %d, leaf %s "
                        "broadcast_peer_count=%d",
                        idx,
                        leaf_id,
                        peers,
                    )
                elif (
                    params.tensor_parallel_degree > 1
                    and replica_kv_memory[idx][leaf_id].replicated
                ):
                    # A replicated leaf takes the is_mla path, which mints ONE
                    # key per hash and fills device buffer 0. Shards 1..tp-1
                    # are filled by the NVLink broadcast -- and only by it. The
                    # Rust side arms that best-effort: a peer-access or stream
                    # failure warns and leaves the connector on
                    # source-device-only behavior without failing the
                    # handshake. So this combination (replicated leaf, TP>1, no
                    # broadcast) is the one where shard 0 is the only shard
                    # that gets real KV, and every other shard reads whatever
                    # was in its buffer. Loud, because nothing downstream can
                    # tell.
                    _logger.warning(
                        "dKV replicated leaf %s on replica %d has "
                        "broadcast_peer_count=0 at tensor_parallel_degree=%d: "
                        "loads fill device buffer 0 only and shards 1..%d keep "
                        "stale KV. Requires a same-host (ipc://) endpoint and "
                        "a full peer-access mesh; check the connector's "
                        "broadcast-setup warnings.",
                        leaf_id,
                        idx,
                        params.tensor_parallel_degree,
                        params.tensor_parallel_degree - 1,
                    )

        _logger.info(
            "dKV admitted all %d handshake(s) across %d replica(s) for "
            "tenant %r",
            sum(len(clients) for clients in self._clients),
            num_replicas,
            tenant_id,
        )

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return self._leaves

    @staticmethod
    def _make_client(
        client_cls: type,
        kv_memory: Sequence[KVCacheMemory],
        local_block_store_endpoint: str,
        listen_port: int,
        backend: str | None,
        expected_devices: Sequence[Device],
        *,
        groups: Sequence[tuple[int, int, int]],
        owner_group_id: int,
        tenant_id: str,
        kv_config_hash: int,
        kv_shard_id: int,
        replica_id: int,
        tenant_gpu_count: int,
        tenant_gpu_device_ids: Sequence[int],
        heartbeat_overrides: Mapping[str, int],
    ) -> _DkvClient:
        # Group the per-leaf units into one (device_id, units) entry
        # per TP shard. The Rust client concatenates each shard's units, in
        # this order, into one dKV block, so a quantized cache's scale buffers
        # and a multi-cache buffer's extra caches (speculative draft and
        # target) all land inside the block rather than being dropped. The
        # stored bytes are the shard's portion of the CPU block the local and
        # tiered connectors build (CLIN-1460) only for an all-sharded tree;
        # those connectors keep one copy of a replicated unit, dKV one per
        # shard.
        shards, shards_are_identical = _group_units_by_shard(kv_memory)

        # MAX's compute stream per device ordinal, so the same-host offload can
        # order each device's D2H after the forward pass that wrote its blocks
        # via a CUDA event in that device's own context. Events and streams are
        # per context, so multi-device TP needs a handle per device rather than
        # one shared handle. A device whose stream has no native handle (e.g. a
        # CPU stream) maps to 0, which routes that device's transfers over NIXL.
        compute_streams: dict[int, int] = {}
        for mem in kv_memory:
            for buffer in mem.buffers:
                compute_streams[buffer.device.id] = (
                    buffer.device.default_queue.native_stream_handle
                )

        # Bind ``tp_shard_id`` to device identity rather than to registration
        # luck. A remote peer fetches a block by the ``(tp_shard_id, group,
        # seq_hash)`` key, so its shard ids must line up with ours by device
        # rank. ``expected_devices`` is the replica's device order sourced from
        # the pipeline config, so comparing it against the shard order the
        # grouping derived catches a future change that reorders buffers
        # before it silently shifts every key.
        registered_order = [device_id for device_id, _ in shards]
        expected_order = [device.id for device in expected_devices]
        if registered_order != expected_order:
            raise ValueError(
                "dKV grouped KV buffers into shard device order "
                f"{registered_order}, which does not match the replica's "
                f"canonical device order {expected_order}. tp_shard_id is bound "
                "to that order, so a mismatch would mis-key blocks across peers."
            )

        # ``total_num_pages`` is the buffer's physical page count
        # (``buffer.shape[0]``), which already includes MAX's trailing "null"
        # page beyond the logical block count. The Rust client divides each
        # registered unit's length by this to derive that unit's per-page byte
        # stride, so it must be the physical count; valid-block offsets are
        # unaffected since the null page is last and is never transferred.
        total_num_pages = kv_memory[0].total_num_pages

        return client_cls(
            local_block_store_endpoint,
            shards,
            0,  # page_size (tokens): unused by the Rust client
            total_num_pages,
            len(shards),
            # the client's parameter is still named ``is_mla``, but
            # replication is not MLA-specific: M3's index-K cache replicates.
            shards_are_identical,
            listen_port=listen_port,
            backend=backend,
            compute_streams=compute_streams,
            groups=list(groups),
            owner_group_id=owner_group_id,
            tenant_id=tenant_id,
            kv_config_hash=kv_config_hash,
            kv_shard_id=kv_shard_id,
            replica_id=replica_id,
            tenant_gpu_count=tenant_gpu_count,
            tenant_gpu_device_ids=list(tenant_gpu_device_ids),
            **heartbeat_overrides,
        )

    @property
    def name(self) -> str:
        return "dkv"

    def _derive_leaf_shape(self) -> None:
        """Caches what ``load`` and ``touch`` need from the leaf tree.

        Everything here depends only on ``_leaves``, ``_wire_ids`` and
        ``_page_size``, all fixed for the life of the connector, so deriving it
        per admission would put two comprehensions and a ``ceildiv`` on the
        scheduler's hot path for values that cannot change (SERVOPT-1526).

        Called from ``__init__`` once those three are bound. Tests that
        construct through ``__new__`` bind them by hand and call this, so the
        derived shape cannot drift from the real one.
        """
        self._full_leaf_ids = [
            leaf_id
            for leaf_id, group_id in self._leaves.items()
            if group_id.is_full()
        ]
        self._sliding_leaf_ids = [
            leaf_id
            for leaf_id, group_id in self._leaves.items()
            if group_id.is_sliding_window()
        ]
        # One window for every sliding leaf (``_validate_dkv_leaves`` rejects a
        # tree with two), so resolve it once. 0 when there is no sliding leaf,
        # and also when ``window_size == 1`` -- see :func:`_sliding_row`.
        self._window_blocks = (
            self._leaves[self._sliding_leaf_ids[0]].blocks_in_window(
                self._page_size
            )
            if self._sliding_leaf_ids
            else 0
        )
        # Recency is a property of the leaf's attention group, so it is fixed
        # here rather than rebuilt inside the per-replica admission loop.
        self._group_recencies = [
            _group_recency(group_id) for group_id in self._leaves.values()
        ]
        # A single sliding leaf must still advertise, or the server sees no
        # recency, infers from wire id 1 (== GroupId::FullAttention) and gives a
        # pure-SWA model full-sequence LRU. A lone full leaf keeps the
        # pre-existing empty-groups wire shape.
        self._advertise_groups = len(self._leaves) > 1 or bool(
            self._sliding_leaf_ids
        )

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVConnectorTransfer:
        """Loads external blocks into ``replica_idx``'s device memory by hash.

        Each ``block_hashes`` element must be canonical bytes: 8 bytes for
        ``ahash64`` / ``sha256_64`` or 32 bytes for full ``sha256``. 32-byte
        digests are truncated to their first 8 bytes at the dkv boundary (see
        :func:`_to_dkv_u64`).

        ``hint`` is the request's ``dkv_cache_hint`` JSON bytes, forwarded
        unparsed: the Rust client reads it to route each block to the peer that
        holds it, and treats anything unusable as no hint, which costs a miss
        rather than a failed load.

        Routes to the processing replica's per-leaf clients, one call per leaf.
        Each client returns its own loaded-block count; the block manager frees
        ``blocks[num_loaded:]`` past it (in
        ``_get_full_blocks_from_host_prefix_cache``). The Rust client owns the
        freed-page ordering across its own GPUs, so there is no shard-client
        fan-out or cross-client drain at this layer.
        """
        if set(block_ids) != set(self._leaves):
            raise ValueError(
                "DKVConnector.load block IDs must match its leaf mapping. "
                f"Expected {self._leaves}, got {block_ids}"
            )
        dkv_hashes = [_to_dkv_u64(h) for h in block_hashes]
        clients = self._clients[replica_idx]
        full_leaf_ids = self._full_leaf_ids
        sliding_leaf_ids = self._sliding_leaf_ids

        # A hint routes the leaves that carry it to a PEER. The sliding leaves
        # cannot carry one (a v2 hint is group-major and names the producer's
        # ids, and a peer predating per-leaf ids has no id for a sliding leaf),
        # so on a hybrid tree a hinted full-leaf hit is looked up remotely
        # while the sliding leaf is looked up locally and misses. The hit-shape
        # test below then yields num_loaded = 0 -- after the remote blocks have
        # already been pulled across the network. That is strictly worse than
        # not hinting: it spends the bandwidth and the load barrier to throw
        # the result away, and shows up only as dkv_peer_loads rising with no
        # matching cached_tokens. Until a hint can name per-leaf ids
        # (SERVOPT-1617), a hybrid tree takes the local path on every leaf.
        if sliding_leaf_ids:
            hint = None

        n_fulls: list[int] = []
        for leaf_id in full_leaf_ids:
            n_fulls.append(
                clients[leaf_id].load(
                    group_id=self._wire_ids[leaf_id],
                    block_ids=list(block_ids[leaf_id]),
                    block_hashes=dkv_hashes,
                    hint=hint,
                )
            )
        # Divergent depths discard the whole hit today. SERVOPT-1613 replaces
        # this with min(n_fulls): dKV reports a leading prefix count, so the
        # shorter run is genuinely present in every full leaf, and values vs
        # scales diverging is the expected case for a quantized model rather
        # than an anomaly.
        if n_fulls and len(set(n_fulls)) > 1:
            for leaf_id in full_leaf_ids:
                clients[leaf_id].wait_for_loads()
            return _DKVCompletedTransfer(
                TransferDirection.LOAD,
                {leaf_id: [] for leaf_id in self._leaves},
            )
        # With no full leaf there is nothing to bound the prefix, so the whole
        # request is the candidate: a pure-SWA model needs only its window, and
        # every slot below it is null. This is the same accounting
        # SlidingWindowKVGroupCoordinator.longest_cache_hit does when it returns
        # ``idx + run`` -- positions under the window count as hit.
        n_full = n_fulls[0] if n_fulls else len(dkv_hashes)
        if not sliding_leaf_ids:
            return _DKVCompletedTransfer(
                TransferDirection.LOAD,
                {
                    leaf_id: list(leaf_ids)[:n_full]
                    for leaf_id, leaf_ids in block_ids.items()
                },
            )

        window_blocks = self._window_blocks
        if window_blocks == 0:
            # window_size == 1: the window spans no whole block, so a sliding
            # leaf needs no real block and there is nothing to load for it. The
            # full leaves' prefix is served in full against all-null sliding
            # rows, which is what SlidingWindowKVGroupCoordinator builds for the
            # device tier. Not reachable for a real model; the paths agreeing
            # keeps it from becoming a silent divergence if one ever is.
            return _DKVCompletedTransfer(
                TransferDirection.LOAD,
                {
                    **{
                        leaf_id: list(block_ids[leaf_id])[:n_full]
                        for leaf_id in full_leaf_ids
                    },
                    **{
                        leaf_id: [_NULL_BLOCK_ID] * n_full
                        for leaf_id in sliding_leaf_ids
                    },
                },
            )
        window_length = min(
            n_full,
            window_blocks,
            *(len(block_ids[leaf_id]) for leaf_id in sliding_leaf_ids),
        )
        if window_length == 0:
            if n_full:
                for leaf_id in full_leaf_ids:
                    clients[leaf_id].wait_for_loads()
            return _DKVCompletedTransfer(
                TransferDirection.LOAD,
                {leaf_id: [] for leaf_id in self._leaves},
            )

        # No hint on the sliding path. A v2 hint is group-major and names the
        # PRODUCER's group ids; those line up for the full leaves a hint has
        # always covered, but a peer that predates per-leaf ids has no id for a
        # sliding leaf to match. An absent hint reads as no hint, which routes
        # to the co-located dKV -- a miss at worst, never a wrong block.
        n_slidings: list[int] = []
        for leaf_id in sliding_leaf_ids:
            sliding_block_ids = list(block_ids[leaf_id])
            n_slidings.append(
                clients[leaf_id].load(
                    group_id=self._wire_ids[leaf_id],
                    block_ids=sliding_block_ids[-window_length:],
                    block_hashes=dkv_hashes[n_full - window_length : n_full],
                )
            )

        # The two hit shapes SlidingWindowKVGroupCoordinator recognizes:
        #
        #   * a COMPLETE window ending at n_full serves the whole n_full prefix,
        #     because the slots below the window are null either way, and
        #   * a partial run anchored at the SEQUENCE START is a hit of just that
        #     run -- the window_length == n_full case, where the requested range
        #     is the prefix itself so there is nothing below it.
        #
        # dKV reports a LEADING prefix count, so a short n_sliding is always a
        # run from the start of the requested range. That is only usable when
        # the range starts at the root; otherwise the run floats in the middle
        # of the window and no prefix length is serviceable, so the whole load
        # misses rather than reporting a hit the KV cannot back.
        if all(n == window_length for n in n_slidings):
            num_loaded = n_full
        elif window_length == n_full:
            num_loaded = min(n_slidings)
        else:
            num_loaded = 0

        # Blocks we do not claim are freed by the caller, so in-flight reads
        # into them must land first.
        if num_loaded < n_full:
            for leaf_id in (*full_leaf_ids, *sliding_leaf_ids):
                clients[leaf_id].wait_for_loads()

        loaded: dict[str, list[int]] = {}
        for leaf_id in full_leaf_ids:
            loaded[leaf_id] = list(block_ids[leaf_id])[:num_loaded]
        for leaf_id in sliding_leaf_ids:
            loaded[leaf_id] = _sliding_row(
                block_ids[leaf_id], window_length, num_loaded
            )
        return _DKVCompletedTransfer(TransferDirection.LOAD, loaded)

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        """Offloads ``replica_idx``'s device blocks to the dkv service by hash.

        Each ``block_hashes`` element follows the same 8-or-32 byte
        contract as :meth:`load` (truncated to its first 8 bytes at the
        dkv boundary; see :func:`_to_dkv_u64`).

        The dKV store dedups by composite key ``(tp_shard_id, group,
        seq_hash)`` and does not chain blocks under a parent, so the Rust
        client builds the keys (and the NUMA striping plan) from the hashes
        alone.

        Routes to the processing replica's per-leaf clients, one call per leaf.
        """
        if set(block_ids) != set(self._leaves):
            raise ValueError(
                "DKVConnector.offload block IDs must match its leaf mapping. "
                f"Expected {self._leaves}, got {block_ids}"
            )
        dkv_hashes = [_to_dkv_u64(h) for h in block_hashes]
        # Every leaf commits the same run in lockstep, so a leaf whose row is
        # not one block per hash would pair ids with the wrong hashes.
        ragged = {
            leaf_id: len(ids)
            for leaf_id, ids in block_ids.items()
            if len(ids) != len(dkv_hashes)
        }
        if ragged:
            raise ValueError(
                "DKVConnector.offload needs one block per hash on every leaf; "
                f"got {ragged} for {len(dkv_hashes)} hashes"
            )
        clients = self._clients[replica_idx]
        # Offload every block in the run for every leaf, sliding ones included.
        # The caller hands us one newly committed run (BlockManager slices
        # ``req_hashes[first:last]``), NOT the sequence from its root, so a
        # run-local ``[-blocks_in_window:]`` tail is not the sequence's window:
        # it drops blocks a later request with a shorter shared prefix needs,
        # and that request then misses its window and discards the full leaf's
        # hit with it. Capacity is the server's job -- GROUP_RECENCY_SLIDING_WINDOW
        # is what makes dKV evict slid-out blocks first.
        for leaf_id in self._leaves:
            clients[leaf_id].offload(
                group_id=self._wire_ids[leaf_id],
                block_ids=list(block_ids[leaf_id]),
                block_hashes=dkv_hashes,
            )
        # dKV registers its posted WRITEs in the deprecated ``wait_for_offloads``
        # barrier, so the manager keeps no pin on the source blocks.
        return _DKVCompletedTransfer(
            TransferDirection.OFFLOAD,
            block_ids,
        )

    def touch(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> None:
        """Refreshes ``replica_idx``'s dkv recency for blocks it served.

        A block served from MAX's on-device (G0) prefix cache issues no other
        dkv traffic, so its dkv LRU recency can freeze and dkv can evict it
        while it is still hot on device. This forwards the served blocks'
        hashes to the Rust client's ``touch``, which the server treats as an
        access that bumps recency.

        Touch contract: the CALLER passes the full root-anchored sequence, and
        this narrows it per leaf before it goes on the wire.
        ``RegionLru::touch`` forward-touches exactly the keys it is handed and
        states that obligation (``region_lru.rs``, ``GroupKind::Sliding``), so:

        * a full-attention leaf takes the whole sequence, which the server
          walks in REVERSE so the shared prefix ends most-recently-used, and
        * a sliding leaf takes only its active window, which the server walks
          FORWARD so the window ends most-recently-used and slid-out blocks
          age toward eviction.

        Handing a sliding leaf the whole chain would bump every slid-out block
        back to MRU on each admission and erase the pressure
        ``GROUP_RECENCY_SLIDING_WINDOW`` exists to create -- roughly 780 blocks
        touched to protect 8 for gemma-4 at 100k tokens with a 1k window. The
        offload path leaves capacity to the server precisely because that
        pressure exists, so the two have to agree.

        Each ``block_hashes`` element follows the same 8-or-32 byte contract as
        :meth:`load` (truncated to its first 8 bytes at the dkv boundary; see
        :func:`_to_dkv_u64`). Best-effort and fire-and-forget: the Rust client
        spawns the touch RPC and returns immediately, so this never blocks the
        caller and a missed touch costs at most a later refetch, never
        correctness. A no-op when ``MODULAR_DKV_DISABLE_G0_TOUCH`` is set (the
        kill-switch, read once at construction).

        Not chunked. ``shard_keys`` mints one key per TP shard, so a long
        sequence on a full leaf is a large request (TP=8 over 780 blocks is
        ~6k keys), but splitting it here would break the walk: the pyo3 shim
        spawns each ``touch`` as its own task that then contends for the
        connector mutex, so two calls have no ordering, and which key ends up
        MRU depends on which chunk lands last. Chunking belongs inside
        ``Connector::touch``, where the splits can be awaited in order
        (SERVOPT-1615). Windowing the sliding leaves removes the pathological
        case (a 1k window is 8 blocks however long the sequence is).

        Routes to the processing replica's per-leaf clients, one call per leaf.
        """
        if self._g0_touch_disabled:
            return
        # Honor the KVConnector.touch contract ("never raises into the caller").
        # This runs on the scheduler thread BEFORE the Rust client's fire-and-
        # forget spawn, so a bad hash length (_to_dkv_u64 -> ValueError) or an
        # out-of-range replica_idx (IndexError) would otherwise propagate here.
        # A missed recency touch is never a correctness issue, so swallow and
        # log at debug (matches offload's swallow posture; design section 4).
        try:
            dkv_hashes = [_to_dkv_u64(h) for h in block_hashes]
            clients = self._clients[replica_idx]
            # The whole root-anchored chain: the reverse walk over all of it is
            # what protects a shared prefix, so this one cannot be truncated.
            for leaf_id in self._full_leaf_ids:
                clients[leaf_id].touch(
                    group_id=self._wire_ids[leaf_id],
                    block_hashes=dkv_hashes,
                )
            # The active window only, per the contract above. The window is the
            # sequence TAIL, and dkv_hashes is root-anchored, so the last
            # ``_window_blocks`` entries are it. Guarded because ``[-0:]`` is
            # the whole list, which is the bug this fixes.
            window = (
                dkv_hashes[-self._window_blocks :]
                if self._window_blocks
                else []
            )
            for leaf_id in self._sliding_leaf_ids:
                clients[leaf_id].touch(
                    group_id=self._wire_ids[leaf_id],
                    block_hashes=window,
                )
        except Exception as exc:
            _logger.debug("dKV touch skipped: %s", exc)

    def wait_for_loads(self) -> None:
        for clients in self._clients:
            for client in clients.values():
                client.wait_for_loads()

    def wait_for_offloads(self) -> None:
        for clients in self._clients:
            for client in clients.values():
                client.wait_for_offloads()

    def shutdown(self) -> None:
        # No-op: the Rust client releases its NIXL agent, heartbeat poller, and
        # RPC connection when the object is dropped (at process teardown).
        # Per-batch transfer throughput is surfaced by the scheduler from
        # ``metrics`` below, so no background logger is needed here.
        pass

    def reset_prefix_cache(self) -> None:
        # No-op: dKV manages its own external block lifecycle server-side.
        pass

    @property
    def metrics(self) -> KVCacheMetrics:
        return self._aggregate_metrics(lambda client: client.metrics())

    def take_metrics(self) -> KVCacheMetrics:
        return self._aggregate_metrics(lambda client: client.take_metrics())

    def _aggregate_metrics(
        self, snapshot: Callable[[_DkvClient], Mapping[str, Any]]
    ) -> KVCacheMetrics:
        total = KVCacheMetrics()
        for clients in self._clients:
            leaf_clients = list(clients.values())
            # One snapshot per client. It crosses into Rust and the scheduler
            # calls this every batch, so reading twice would double the FFI cost
            # and could straddle a reconnect, pairing a connected flag with
            # transfer counts from a different moment.
            snapshots = [snapshot(client) for client in leaf_clients]
            # dkv_total_clients counts DP REPLICAS, not leaf clients. It
            # predates the per-leaf split and the scheduler reports it as "how
            # many replicas are up"; counting leaves instead would multiply
            # every deployment's reading by its leaf count and silently move
            # anything alerting on the ratio. A replica is up only when every
            # leaf it owns is up, because one dead leaf cannot serve a block.
            replica_connected = all(m["connected"] for m in snapshots)
            # dkv_reconnect_attempts is per REPLICA for the same reason
            # dkv_total_clients is: summing it over leaves would multiply every
            # deployment's reading by its leaf count and move anything alerting
            # on the current shape. A replica's leaves share one server and one
            # network path, so they lose and recover it together; the max is
            # "how many times this replica had to reconnect".
            total = total + KVCacheMetrics(
                dkv_connected_clients=1 if replica_connected else 0,
                dkv_total_clients=1,
                dkv_reconnect_attempts=max(
                    m["reconnect_attempts"] for m in snapshots
                ),
            )
            for m in snapshots:
                # Transfer volume is a per-client sum: every leaf moves its own
                # bytes.
                total = total + KVCacheMetrics(
                    nixl_read_blocks=m["read_blocks"],
                    nixl_write_blocks=m["write_blocks"],
                    nixl_read_bytes=m["read_bytes"],
                    nixl_write_bytes=m["write_bytes"],
                    nixl_read_latency_total_ms=m[
                        "read_transfer_latency_total_ms"
                    ],
                    nixl_read_latency_count=m["read_transfer_latency_count"],
                    nixl_read_latency_max_ms=m["read_transfer_latency_max_ms"],
                    # The lookup RPCs, which the transfer pairs above exclude by
                    # construction: they bracket the copies, these bracket the
                    # round trip that finds and pins the blocks. Unfed until
                    # CLIN-1844, which is why the scheduler log used to print
                    # "acquire 0.0ms, pin 0.0ms" on every line.
                    rpc_read_latency_total_ms=m["rpc_read_latency_total_ms"],
                    rpc_read_latency_count=m["rpc_read_latency_count"],
                    rpc_acquire_latency_total_ms=m[
                        "rpc_acquire_latency_total_ms"
                    ],
                    rpc_acquire_latency_count=m["rpc_acquire_latency_count"],
                    nixl_write_latency_total_ms=m[
                        "write_transfer_latency_total_ms"
                    ],
                    nixl_write_latency_count=m["write_transfer_latency_count"],
                    # Cross-node pull. Ordinary per-window deltas that
                    # take_metrics clears, unlike the health keys above, so
                    # they fold in the same way as the transfer keys: per leaf
                    # client, because each leaf pulls its own bytes from its
                    # own peers. The dict also carries attached_peers, a level
                    # neither engine exports yet.
                    dkv_peer_attaches=m["peer_attaches"],
                    dkv_peer_attach_failures=m["peer_attach_failures"],
                    dkv_peers_dropped=m["peers_dropped"],
                    dkv_peer_loads=m["peer_loads"],
                    dkv_peer_load_failures=m["peer_load_failures"],
                    dkv_hints_rejected=m["hints_rejected"],
                )
        return total
