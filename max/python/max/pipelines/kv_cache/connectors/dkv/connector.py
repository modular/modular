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

Block-hash contract: the dkv wire format carries a ``uint64 seq_hash`` and is
unchanged by this shim. Callers may pass either the 8-byte canonical encoding
used by ``ahash64`` / ``sha256_64`` or the 32-byte canonical encoding used by
full ``sha256``; in the 32-byte case the shim truncates to the first 8 bytes
at the boundary. Truncation is byte-identical to the existing ``sha256_64``
algorithm, so configuring MAX with ``sha256`` or ``sha256_64`` yields the
same dkv key for the same logical digest.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Sequence

import msgspec
from max.driver import Buffer, Device
from max.nn.kv_cache.cache_params import (
    KVCacheMemory,
    KVCacheParamInterface,
    KVCacheParams,
    KVHashAlgo,
    ReplicatedKVCacheMemory,
)
from max.nn.kv_cache.data_parallelism_utils import split_into_groups
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced

_logger = logging.getLogger("max.pipelines")

# dKV keys every block under a composite (tp_shard_id, group_id, seq_hash). MAX's
# paged KV cache is single-group (full attention) today; SWA/hybrid groups are not
# yet wired through this connector (block_manager has no group dimension), so we key
# all load/offload under the full-attention group. REVISIT when windowed-KV groups
# land. Mirrors GroupId::FullAttention (== 1) in the dkv proto
# (dkv/dkv-proto/src/gen/modular.dkv.v1.rs). GroupId::Unspecified (== 0) is rejected
# server-side (no geometry), so 0 is never a valid substitute.
_DKV_GROUP_FULL_ATTENTION = 1


def _to_dkv_u64(h: bytes) -> int:
    """Packs a connector-level block hash into the 64-bit dkv wire key.

    The dkv proto stays ``uint64 seq_hash``. Accepts the canonical bytes
    forms produced by :func:`max.pipelines.kv_cache.kv_connector.to_block_hash_bytes`:

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
    """Groups one replica's flat KV memory units into per-shard unit lists.

    ``kv_memory`` is the flat ``to_memory()`` output: one unit per logical
    buffer (target values, FP8 scales, indexer, draft, and so on), where a
    quantized or multi-cache model contributes several units. The Rust client
    concatenates each shard's units, in this order, into one dKV block, which
    makes the stored bytes match the CPU block the local and tiered connectors
    build from the same units.

    Two layouts exist:

    * Replicated (MLA with TP > 1): every unit is a
      :class:`ReplicatedKVCacheMemory` whose rank-0 buffer plus peers span the
      same device topology. Shard ``s`` takes each unit's buffer for rank
      ``s``, so every shard carries the full unit list and holds identical
      bytes.
    * Sharded (everything else): each unit is a plain buffer on one device,
      and a shard is a device. Units group under their device in flat order,
      which is the shard-restricted subsequence of the canonical unit order.

    Args:
        kv_memory: One replica's offload-ready KV memory units.

    Returns:
        A ``(shards, is_mla)`` pair, where ``shards`` has one
        ``(device_id, [(ptr, nbytes), ...])`` entry per TP shard in canonical
        device order.

    Raises:
        NotImplementedError: If replicated and non-replicated units are mixed.
        ValueError: If unit page counts or per-shard unit counts disagree, or
            if replicated units span different device topologies.
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

    replicated = [
        mem for mem in kv_memory if isinstance(mem, ReplicatedKVCacheMemory)
    ]
    is_mla = bool(replicated)

    if is_mla:
        if len(replicated) != len(kv_memory):
            raise NotImplementedError(
                "the dKV connector cannot mix replicated (MLA) and "
                "non-replicated KV memory units in one replica; every unit "
                "must be replicated across the same TP shards"
            )

        # every replicated unit must span the same device topology so shard s
        # names the same device in every unit (mirrors BlockOffloadEngine)
        topologies = {
            tuple(buffer.device.id for buffer in mem.all_buffers)
            for mem in kv_memory
        }
        if len(topologies) > 1:
            raise ValueError(
                "all replicated KVCacheMemory units must share the same TP "
                f"device topology; got {sorted(topologies)}"
            )

        topology = next(iter(topologies))
        shards = [
            (
                device_id,
                [
                    (
                        mem.all_buffers[rank]._data_ptr(),
                        _buffer_nbytes(mem.all_buffers[rank]),
                    )
                    for mem in kv_memory
                ],
            )
            for rank, device_id in enumerate(topology)
        ]
        return shards, True

    # sharded layout: group the flat units under their device, preserving the
    # canonical to_memory() order within each shard
    units_by_device: dict[int, list[tuple[int, int]]] = {}
    for mem in kv_memory:
        buffer = mem.buffer
        units_by_device.setdefault(buffer.device.id, []).append(
            (buffer._data_ptr(), _buffer_nbytes(buffer))
        )

    unit_counts = {len(units) for units in units_by_device.values()}
    if len(unit_counts) > 1:
        raise ValueError(
            "every dKV shard must carry the same number of KV memory units; "
            f"got counts {sorted(unit_counts)} across devices "
            f"{sorted(units_by_device)}"
        )

    return list(units_by_device.items()), False


def _shard_unit_strides(kv_memory: Sequence[KVCacheMemory]) -> list[int]:
    """Derives one shard's per-unit page strides in canonical order.

    A dKV block holds one shard's buffer units concatenated, so the layout
    fingerprint folds the strides of a single shard's unit list rather than
    the flat per-physical-buffer list. This keeps the folded shape identical
    between replicated (MLA) and sharded layouts, where the flat list would
    otherwise repeat each unit once per shard. Shard 0 stands in for every
    shard because :func:`_group_units_by_shard` validates a uniform unit
    count per shard and the Rust config validates that the stride vectors
    match.

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
# starting server (connection refused) or a cold slab warm-up (deferred region
# carve) can take longer; admission retries transient failures until this budget
# is spent, then fails model load. Override via MODULAR_DKV_ADMISSION_TIMEOUT_S.
_DEFAULT_ADMISSION_TIMEOUT_S = 120.0
_ADMISSION_INITIAL_BACKOFF_S = 1.0
_ADMISSION_MAX_BACKOFF_S = 10.0


def _dtype_tag(dtype: object) -> str:
    """Returns a stable, restart-invariant text tag for a ``DType``.

    Uses the enum member ``name`` (e.g. ``"bfloat16"``) when present, else
    ``str(dtype)``. Never uses Python's per-process-randomized ``hash``, so the
    fingerprint it feeds is identical across process restarts.
    """
    return getattr(dtype, "name", None) or str(dtype)


def _kv_config_hash(params: KVCacheParams, unit_strides: Sequence[int]) -> int:
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

    * ``v`` — contract version (``2``; bumped when the block layout became the
      concatenation of every buffer unit rather than the value buffer alone).
    * ``dtype`` / ``dtype_bytes`` — KV storage dtype name and byte width.
    * ``kv_dim`` — 2 for MHA/GQA, 1 for MLA (the attention family).
    * ``head_dim`` — per-head dimension.
    * ``num_layers`` — transformer layer count.
    * ``page_size`` — tokens per block/page.
    * ``n_kv_heads_per_device`` — per-shard KV-head count (TP-sensitive).
    * ``tensor_parallel_degree`` — with the per-shard head count pins the total.
    * ``quant`` — ``"<scale_dtype>:<granularity>"`` when FP8-quantized, else
      ``"none"``.
    * ``block_value_bytes`` — per-shard value-block bytes
      (``prod(shape_per_block) * dtype.size``).
    * ``unit_strides`` — comma-joined per-page byte stride of one shard's
      buffer units in canonical ``to_memory()`` order (values, quant scales,
      indexer, draft, and so on), derived by :func:`_shard_unit_strides`. A
      shard's dKV block is these strides concatenated, so any change to the
      unit set or its ordering makes stored blocks byte-incompatible and must
      flip the hash. Folding one shard's subsequence rather than the flat
      physical buffer list keeps the folded shape identical between
      replicated (MLA) and sharded layouts; the shard count is already
      pinned by ``tensor_parallel_degree``.

    Model/weights identity is deliberately NOT folded here: a different model is
    a different Mammoth deployment, hence a different ``tenant_id`` (already part
    of the dedup key). Reattaching persisted shares across a same-``tenant_id``
    weights swap (CLIN-1474) must additionally fold a weights/version fingerprint
    threaded from the pipeline config — a documented follow-up, out of scope for
    this handshake.

    The ``v`` bump to ``2`` invalidates every share persisted under the ``v=1``
    contract once on upgrade; that is the intended behavior for a cache whose
    on-disk layout changed.

    Args:
        params: The single-group KV-cache parameters for this deployment.
        unit_strides: Per-page byte stride of one shard's buffer units in
            canonical order, from :func:`_shard_unit_strides`.

    Returns:
        The 64-bit layout fingerprint.
    """
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
    fields = [
        ("v", "2"),
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
        ("unit_strides", ",".join(str(s) for s in unit_strides)),
    ]
    canonical = "\n".join(f"{k}={v}" for k, v in fields).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(canonical).digest()[:8], "big", signed=False
    )


def _resolve_kv_share_identity(
    global_device_index: int,
    tensor_parallel_degree: int,
    replicates_kv_across_tp: bool,
) -> tuple[int, int]:
    """Resolves ``(kv_shard_id, replica_id)`` for one GPU's KV share.

    ``global_device_index`` is the GPU's replica-major flat index in the
    ``[dp][tp]`` device grid (``replica * tp + tp_rank``). The result names which
    KV slice the GPU holds and which data-parallel copy of that slice it is,
    mirroring
    :func:`max.pipelines.kv_cache.paged_kv_cache.transfer_engine.resolve_peer_view`:

    * ``replicates_kv_across_tp`` (MLA with TP>1): every TP rank holds a full,
      byte-identical copy of the latent KV, so the grid flattens to
      ``[dp*tp][1]`` — one shard (``kv_shard_id == 0``), each GPU its own replica
      (``replica_id == global_device_index``); dedup group size ``dp*tp``.
    * otherwise (MHA/GQA head-sharded, or TP==1): the TP rank is the shard and
      the DP index is the replica — ``kv_shard_id == global_device_index % tp``,
      ``replica_id == global_device_index // tp``; dedup group size ``dp``.

    Invariant (the acceptance criterion): two GPUs get identical ``kv_shard_id``
    exactly when they hold byte-identical KV. Replicated: all share
    ``kv_shard_id == 0`` and all are byte-identical. Head-sharded: equal TP rank
    holds the same head slice (byte-identical across DP replicas), a different
    rank a different slice.

    Args:
        global_device_index: Replica-major flat GPU index in the ``[dp][tp]``
            grid.
        tensor_parallel_degree: TP degree (``n_devices // dp``).
        replicates_kv_across_tp: Whether every TP rank holds identical KV.

    Returns:
        ``(kv_shard_id, replica_id)`` for the ``ExchangeMetadata`` handshake.
    """
    if tensor_parallel_degree < 1:
        raise ValueError(
            f"tensor_parallel_degree must be >= 1, got {tensor_parallel_degree}"
        )
    if global_device_index < 0:
        raise ValueError(
            f"global_device_index must be >= 0, got {global_device_index}"
        )
    if replicates_kv_across_tp:
        return 0, global_device_index
    replica_id, tp_rank = divmod(global_device_index, tensor_parallel_degree)
    return tp_rank, replica_id


def _validate_tenant_topology(*, is_single_group: bool) -> None:
    """Guards the model topology a multi-tenant dKV handshake can represent.

    The ``ExchangeMetadata`` identity fields (``kv_shard_id``, ``replica_id``,
    ``device_id``, ``numa_node``) are scalars that name one GPU's share. TP>1 is
    carried by splitting the handshake per GPU (one client / connection per GPU,
    each sending its own ``kv_shard_id``); see :func:`_resolve_replica_identities`
    and :meth:`DKVConnector.__init__`. The remaining TP>1 case still deferred is
    MLA replicated KV (``replicates_kv_across_tp``), which is guarded in
    :func:`_resolve_replica_identities` (CLIN-1512 TP-2). Multi-group caches
    (hybrid / SWA / speculative) are likewise not yet wired through this
    connector.

    Args:
        is_single_group: Whether the cache is single-group (full attention).

    Raises:
        NotImplementedError: If the cache is multi-group.
    """
    if not is_single_group:
        raise NotImplementedError(
            "Multi-tenant dKV (MODULAR_DKV_TENANT_ID set) requires a single-group "
            "KV cache; hybrid / sliding-window / speculative multi-cache models "
            "are not wired through the dKV connector yet."
        )


# The default (non-multi-tenant) DP identity every replica handshakes: one
# shared, replica-agnostic store. Both fields are 0, so the dKV server keys a
# SINGLE store for ``(tenant_id="", kv_shard_id=0, replica_id=0)`` and every DP
# replica resolves to it — intentional cross-replica prefix sharing (the analog
# of the local/tiered connectors' single shared host pool), exercised by the
# dkv-e2e ``clin1452_dp.rs`` cross-replica roundtrip. See
# :func:`_resolve_replica_identities` for the full contract.
_DEFAULT_SHARED_STORE_REPLICA_IDENTITY = (0, 0)


def _resolve_replica_identities(
    tenant_id: str,
    num_replicas: int,
    params: KVCacheParamInterface,
    unit_strides: Sequence[int],
) -> tuple[int, list[tuple[int, int]]]:
    """Resolves the per-DP-replica dKV handshake identity.

    Returns the shared ``kv_config_hash`` and, per DP replica in order, its
    ``(kv_shard_id, replica_id)``:

    * Default path (empty ``tenant_id``): every DP replica resolves to the same
      all-zero identity (:data:`_DEFAULT_SHARED_STORE_REPLICA_IDENTITY`), so the
      dKV server routes them ALL to ONE shared, replica-agnostic store —
      intentional DP prefix sharing (the analog of the ``local``/``tiered``
      connectors' single shared host pool), not an accidental collapse. This is
      sound because (a) the block key is replica-agnostic (the ``BlockKey``
      proto's ``(tp_shard_id, group_id, seq_hash)``, built in ``connector.rs``,
      no replica component), so a block any replica offloads is visible to all,
      and (b)
      CLIN-1343 gives each per-replica client a UNIQUE NIXL agent name +
      port, so the N clients coexist in one process. Per-replica isolation
      (peer reads, content-hash routing) is out of scope — CLIN-1478 slice-3.
    * Multi-tenant path: guards the topology (:func:`_validate_tenant_topology`)
      and maps each GPU (replica-major flat index ``i * tp + j``) to its share
      via :func:`_resolve_kv_share_identity`, sharing the layout
      ``kv_config_hash``. The list has one entry per GPU (``dp * tp`` entries);
      TP==1 collapses to one per replica. MLA replicated KV with TP>1 is deferred
      (CLIN-1512 TP-2) because its per-GPU buffer split is not yet wired.

    Args:
        tenant_id: The resolved tenant identity (empty is the default
            single-tenant path, resolving to one shared, replica-agnostic
            store).
        num_replicas: Number of DP replicas (one dKV client each on the default
            path; ``tp`` per-GPU clients each on the multi-tenant path).
        params: KV-cache parameters (used only on the multi-tenant path).
        unit_strides: Per-page byte stride of one shard's buffer units in
            canonical order, folded into the layout hash (used only on the
            multi-tenant path).

    Returns:
        ``(kv_config_hash, [(kv_shard_id, replica_id), ...])`` — one identity per
        DP replica on the default path, one per GPU (``dp * tp``) on the
        multi-tenant path.

    Raises:
        NotImplementedError: If the multi-tenant topology is unsupported
            (multi-group cache, or MLA replicated KV with ``TP > 1``).
        ValueError: If ``num_replicas`` disagrees with ``data_parallel_degree``.
    """
    if not tenant_id:
        # Default path: all DP replicas resolve to ONE shared, replica-agnostic
        # store (see _DEFAULT_SHARED_STORE_REPLICA_IDENTITY). The paired
        # kv_config_hash is 0; the server ignores it on the empty-tenant path.
        return 0, [_DEFAULT_SHARED_STORE_REPLICA_IDENTITY] * num_replicas
    _validate_tenant_topology(is_single_group=isinstance(params, KVCacheParams))
    assert isinstance(params, KVCacheParams)  # narrowed by the guard above
    if num_replicas != params.data_parallel_degree:
        raise ValueError(
            f"replica count {num_replicas} does not match data_parallel_degree "
            f"{params.data_parallel_degree}; the replica-major device mapping "
            "the identity rule relies on would be wrong"
        )
    tp = params.tensor_parallel_degree
    replicates = params.replicates_kv_across_tp
    if replicates and tp > 1:
        raise NotImplementedError(
            "MLA multi-tenant TP>1 (replicated KV, per-GPU buffer split) is a "
            "follow-up (CLIN-1512 TP-2)"
        )
    # One identity per GPU in replica-major order: replica ``i``'s TP rank ``j``
    # has flat index ``i * tp + j``. TP==1 collapses to one identity per replica.
    identities = [
        _resolve_kv_share_identity(i * tp + j, tp, replicates)
        for i in range(num_replicas)
        for j in range(tp)
    ]
    return _kv_config_hash(params, unit_strides), identities


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


class DKVExternalBlockMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Marker that a block hash is referenced by the orchestrator hint.

    The slim hint only carries ``seq_hash``; the dKV server resolves slab
    location and length when the connector reads the block. We still wrap the
    hash in a typed struct so the context payload survives the
    API-server -> model-worker process boundary via msgspec's tagged-struct
    serialization.

    The struct is intentionally retained even though it degenerates to a single
    ``seq_hash`` field today. The orchestrator's hint shape is expected to evolve
    to mix blocks from multiple source dKV instances in a single hint (per-block
    ``instance_name`` for routing); keeping the per-block container in place now
    lets that land without re-introducing a context-side data structure.
    """

    seq_hash: int


class DKVConnector:
    """``KVConnector`` backed by the ``dkv_connector`` Rust client.

    A single instance serves every DP replica. The underlying Rust client is
    inherently per-endpoint (its ``load``/``offload`` reference block ids into
    one registered device-buffer set and carry no replica/group key), so this
    shim owns a group of clients per DP replica (``self._replica_client_groups``)
    and fans each call out to the group for the request's ``replica_idx``.

    * Default / pure-DP path: one client per replica registering that replica's
      full TP GPU set (TP rank kept in the block key), so each group holds a
      single client — :meth:`load` / :meth:`offload` issue exactly one call.
    * Multi-tenant TP>1 (non-MLA): one client per GPU, so replica ``i``'s group
      holds its ``tp`` shard-clients, each addressing a distinct per-GPU store
      (``kv_shard_id = tp_rank``); :meth:`load` / :meth:`offload` fan out to all
      shard-clients of the processing replica.

    On the default path all replicas share ONE replica-agnostic store (see
    :func:`_resolve_replica_identities`), so the group for ``replica_idx``
    selects only WHICH endpoint moves the blocks, not which store is addressed —
    routing by the PROCESSING replica. That selection in :meth:`load` /
    :meth:`offload` is the seam CLIN-1478 slice-3 later swaps to route by the
    content-hash OWNER replica (``hash(seq_hash) % dp_size``) for multi-tenant
    per-replica isolation.
    """

    @traced
    def __init__(
        self,
        replica_kv_memory: Sequence[Sequence[KVCacheMemory]],
        local_block_store_endpoint: str,
        devices: Sequence[Device],
        params: KVCacheParamInterface,
    ) -> None:
        """Constructs and admits one dKV Rust client per DP replica.

        Args:
            replica_kv_memory: Per-replica offload-ready KV memory.
            local_block_store_endpoint: Co-located dKV control-plane endpoint.
            devices: The pipeline's flat, ordered device list across replicas.
            params: KV-cache parameters, used to derive the multi-tenant
                per-GPU identity (``kv_shard_id`` / ``replica_id`` /
                ``kv_config_hash``) when ``MODULAR_DKV_TENANT_ID`` is set.
        """
        # Deferred so importing this module (e.g. for DKVExternalBlockMetadata,
        # or by non-dKV pipelines) does not require the optional, runtime-
        # provided dkv_connector extension to be installed.
        from dkv_connector import DkvConnector as _DkvConnectorClient

        if not replica_kv_memory or not all(replica_kv_memory):
            raise ValueError(
                "DKVConnector requires at least one KV cache buffer per replica"
            )

        listen_port = int(os.getenv("MODULAR_DKV_NIXL_LISTEN_PORT", "0"))
        backend = os.getenv("MODULAR_NIXL_TRANSFER_BACKEND") or None

        # Multi-tenant deployment identity (CLIN-1477). MODULAR_DKV_TENANT_ID is
        # injected by the operator (the trust boundary — not a user-facing
        # override flag, which would be forgeable). Unset/empty is the default
        # single-tenant path: every handshake identity field stays default and
        # every DP replica resolves to one shared, replica-agnostic store
        # (intentional cross-replica prefix sharing, not an accidental
        # collapse). When set, each GPU sends a distinct identity so the server
        # keys a distinct store per (tenant_id, kv_shard_id, replica_id).
        tenant_id = os.getenv("MODULAR_DKV_TENANT_ID", "")
        num_replicas = len(replica_kv_memory)
        # one shard's per-unit page strides in canonical order, from replica 0
        # because every DP replica runs the same model and config and so the
        # same layout; folded into the layout hash because a shard's dKV block
        # is these strides concatenated
        unit_strides = _shard_unit_strides(replica_kv_memory[0])
        kv_config_hash, replica_identities = _resolve_replica_identities(
            tenant_id, num_replicas, params, unit_strides
        )

        # ``devices`` is the pipeline's flat, ordered device list across every
        # replica; split it into each replica's canonical device order so a
        # client can bind its shard ids to that order. This is the same split the
        # cache manager applies, and it is sourced independently of
        # ``to_memory``, so it is a real cross-check on the buffer ordering rather
        # than a restatement of it.
        devices_per_replica = split_into_groups(list(devices), num_replicas)

        admission_timeout_s = float(
            os.getenv(
                "MODULAR_DKV_ADMISSION_TIMEOUT_S",
                str(_DEFAULT_ADMISSION_TIMEOUT_S),
            )
        )

        # Each client's connect + handshake ("admission") is retried on transient
        # failures (dKV still starting); model readiness is gated on ALL clients
        # admitting, so a client whose retry budget is exhausted raises here and
        # fails model load rather than serving with a partial dKV. ``load`` /
        # ``offload`` fan out per DP replica over ``self._replica_client_groups``;
        # ``self._clients`` is the flat list every client-wide fan-out
        # (wait_for_*, metrics, reset_metrics) iterates.
        self._clients = []
        self._replica_client_groups = []
        # tensor_parallel_degree is a KVCacheParamInterface member, valid on
        # every path; _resolve_replica_identities above validated params is a
        # KVCacheParams whenever tenant_id is set.
        tp = params.tensor_parallel_degree
        if not tenant_id or tp == 1:
            # Default / pure-DP path AND multi-tenant TP==1: one client per DP
            # replica, registering that replica's full TP GPU set (TP rank kept
            # in the block key) with units concatenated per shard by
            # _make_client. Byte-identical to the pre-CLIN-1512 path. MT TP==1
            # keeps its tenant identity here: replica_identities carries
            # (kv_shard_id=0, replica_id=dp_i) and tenant_id is threaded into
            # every client, so a multi-unit (FP8 / multi-cache) MT TP==1 cache
            # builds one concatenated client per replica instead of hitting the
            # per-GPU split's one-buffer-per-GPU guard. Each group holds one
            # client so load/offload issue one call.
            for idx, (
                kv_memory,
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
                factory = functools.partial(
                    self._make_client,
                    _DkvConnectorClient,
                    kv_memory,
                    local_block_store_endpoint,
                    listen_port,
                    backend,
                    replica_devices,
                    tenant_id=tenant_id,
                    kv_config_hash=kv_config_hash,
                    kv_shard_id=kv_shard_id,
                    replica_id=replica_id,
                )
                self._clients.append(
                    _admit_with_retry(
                        factory,
                        timeout_s=admission_timeout_s,
                        label=f"replica {idx}",
                    )
                )
            self._replica_client_groups = [[c] for c in self._clients]
        else:
            # Multi-tenant TP>1 (non-MLA): promote the TP rank to the STORE key
            # (kv_shard_id), so each GPU gets its own store / client / connection
            # / handshake, registering exactly one buffer (tp_shard_id == 0).
            # replica_identities has one entry per GPU (dp * tp), replica-major.
            for i in range(num_replicas):
                units = replica_kv_memory[i]
                # The per-GPU split indexes one KV buffer per TP rank. A
                # quantized (FP8 scale) or multi-cache buffer yields more units
                # than TP shards, which this split would silently drop, so fail
                # loudly rather than register a partial cache. Concatenating
                # those units per shard (via _group_units_by_shard, as the
                # default / TP==1 path already does) on the MT TP>1 split is a
                # follow-up, CLIN-1518. (MLA replicated KV is one unit for all
                # shards, so it is rejected above, not split here.)
                if len(units) != tp:
                    raise NotImplementedError(
                        "Multi-tenant dKV TP>1 requires exactly one KV buffer "
                        f"per GPU, but replica {i} has {len(units)} units for "
                        f"{tp} TP shards. Quantized (FP8 scale) and multi-cache "
                        "buffers are not supported on the dKV connector yet "
                        "(CLIN-1460)."
                    )
                group: list[object] = []
                for j in range(tp):
                    kv_shard_id, replica_id = replica_identities[i * tp + j]
                    factory = functools.partial(
                        self._make_client,
                        _DkvConnectorClient,
                        [units[j]],
                        local_block_store_endpoint,
                        listen_port,
                        backend,
                        [devices_per_replica[i][j]],
                        tenant_id=tenant_id,
                        kv_config_hash=kv_config_hash,
                        kv_shard_id=kv_shard_id,
                        replica_id=replica_id,
                    )
                    client = _admit_with_retry(
                        factory,
                        timeout_s=admission_timeout_s,
                        label=f"replica {i} shard {j}",
                    )
                    self._clients.append(client)
                    group.append(client)
                self._replica_client_groups.append(group)

        if tenant_id:
            _logger.info(
                "dKV admitted all %d handshake(s) across %d replica(s) for "
                "tenant %r",
                len(self._clients),
                num_replicas,
                tenant_id,
            )

    @staticmethod
    def _make_client(
        client_cls: type,
        kv_memory: Sequence[KVCacheMemory],
        local_block_store_endpoint: str,
        listen_port: int,
        backend: str | None,
        expected_devices: Sequence[Device],
        *,
        tenant_id: str,
        kv_config_hash: int,
        kv_shard_id: int,
        replica_id: int,
    ) -> object:
        # Group the flat to_memory() units into one (device_id, units) entry
        # per TP shard. The Rust client concatenates each shard's units, in
        # this order, into one dKV block, so a quantized cache's scale buffers
        # and a multi-cache buffer's extra caches (speculative draft and
        # target) all land inside the block rather than being dropped, and the
        # stored bytes are identical to the shard's portion of the CPU block
        # the local and tiered connectors build (CLIN-1460).
        shards, is_mla = _group_units_by_shard(kv_memory)

        # MAX's compute stream per device ordinal, so the same-host offload can
        # order each device's D2H after the forward pass that wrote its blocks
        # via a CUDA event in that device's own context. Events and streams are
        # per context, so multi-device TP needs a handle per device rather than
        # one shared handle. A device whose stream has no native handle (e.g. a
        # CPU stream) maps to 0, which routes that device's transfers over NIXL.
        compute_streams: dict[int, int] = {}
        for mem in kv_memory:
            for buffer in mem.all_buffers:
                compute_streams[buffer.device.id] = (
                    buffer.device.default_stream.native_stream_handle
                )

        # Bind ``tp_shard_id`` to device identity rather than to registration
        # luck. A remote peer fetches a block by the ``(tp_shard_id, group,
        # seq_hash)`` key, so its shard ids must line up with ours by device
        # rank. ``expected_devices`` is the replica's device order sourced from
        # the pipeline config, so comparing it against the shard order the
        # grouping derived catches a future ``to_memory`` change that reorders
        # buffers before it silently shifts every key.
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
            is_mla,
            listen_port=listen_port,
            backend=backend,
            compute_streams=compute_streams,
            tenant_id=tenant_id,
            kv_config_hash=kv_config_hash,
            kv_shard_id=kv_shard_id,
            replica_id=replica_id,
        )

    @property
    def name(self) -> str:
        return "dkv"

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> int:
        """Loads external blocks into ``replica_idx``'s device memory by hash.

        Each ``block_hashes`` element must be canonical bytes from
        :func:`to_block_hash_bytes`: 8 bytes for ``ahash64`` / ``sha256_64``
        or 32 bytes for full ``sha256``. 32-byte digests are truncated to
        their first 8 bytes at the dkv boundary (see :func:`_to_dkv_u64`).

        Fans out to every shard-client of the processing replica (one client on
        the default path, ``tp`` on the multi-tenant TP>1 path) with identical
        logical block ids and hashes. Returns the MIN loaded count across shards:
        a block is usable only if it landed on EVERY shard, and the block manager
        frees blocks past the returned count.
        """
        dkv_hashes = [_to_dkv_u64(h) for h in block_hashes]
        counts = [
            client.load(
                group_id=_DKV_GROUP_FULL_ATTENTION,
                device_block_ids=device_block_ids,
                block_hashes=dkv_hashes,
            )
            for client in self._replica_client_groups[replica_idx]
        ]
        if len(set(counts)) > 1:
            _logger.debug(
                "dKV load: shard-clients for replica %d returned differing "
                "loaded-block counts %s; using min",
                replica_idx,
                counts,
            )
        # TODO(CLIN-1512 TP-3): returning min is not enough once real per-GPU
        # transfers land. A shard-count mismatch (partial offload failure, or
        # independent per-store eviction) means an OVER-loading shard has already
        # enqueued async H2D reads into device pages beyond `min`. The block
        # manager (``_get_full_blocks_from_host_prefix_cache``) frees
        # ``blocks[num_loaded:]`` immediately, so a later same-batch same-replica
        # reallocation of a freed page can be clobbered by that stray in-flight
        # read -> silent wrong KV. This is inert TODAY (cross-shard transfer is
        # TP-3, not wired), but the fan-out + free path IS live. TP-3 must align
        # shard loads (all shards agree before returning) or drain/cancel the
        # stray reads before the free -- or fail loud on a count mismatch.
        return min(counts)

    def offload(
        self,
        block_ids: list[int],
        block_hashes: Sequence[bytes],
        parent_seq_hash: bytes | None = None,
        replica_idx: int = 0,
    ) -> None:
        """Offloads ``replica_idx``'s device blocks to the dkv service by hash.

        Each ``block_hashes`` element follows the same 8-or-32 byte
        contract as :meth:`load` (truncated to its first 8 bytes at the
        dkv boundary; see :func:`_to_dkv_u64`).

        ``parent_seq_hash`` is accepted for ``KVConnector`` protocol
        compatibility but no longer forwarded: the dKV store now dedups
        by composite key ``(tp_shard_id, group, seq_hash)`` and does not
        chain blocks under a parent, so the Rust client builds the keys
        (and the NUMA striping plan) from the hashes alone.

        Fans out to every shard-client of the processing replica (one client on
        the default path, ``tp`` on the multi-tenant TP>1 path) with identical
        logical block ids and hashes.
        """
        dkv_hashes = [_to_dkv_u64(h) for h in block_hashes]
        for client in self._replica_client_groups[replica_idx]:
            client.offload(
                group_id=_DKV_GROUP_FULL_ATTENTION,
                block_ids=block_ids,
                block_hashes=dkv_hashes,
            )

    def wait_for_loads(self) -> None:
        for client in self._clients:
            client.wait_for_loads()

    def wait_for_offloads(self) -> None:
        for client in self._clients:
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
    def num_host_blocks(self) -> int:
        # BlockManager gates the load path on num_host_blocks > 0. dKV capacity
        # is managed externally by the dKV service.
        return sys.maxsize

    @property
    def num_used_host_blocks(self) -> int:
        return 0

    @property
    def num_disk_blocks(self) -> int:
        return 0

    @property
    def num_used_disk_blocks(self) -> int:
        return 0

    def reset_metrics(self) -> None:
        """Clear Rust-side transfer counters after the scheduler samples a batch."""
        for client in self._clients:
            client.reset_metrics()

    @property
    def metrics(self) -> KVCacheMetrics:
        total = KVCacheMetrics()
        for client in self._clients:
            m = client.metrics()
            total = total + KVCacheMetrics(
                nixl_read_blocks=m["read_blocks"],
                nixl_write_blocks=m["write_blocks"],
                nixl_read_bytes=m["read_bytes"],
                nixl_write_bytes=m["write_bytes"],
                nixl_read_latency_total_ms=m["read_transfer_latency_total_ms"],
                nixl_read_latency_count=m["read_transfer_latency_count"],
                nixl_write_latency_total_ms=m[
                    "write_transfer_latency_total_ms"
                ],
                nixl_write_latency_count=m["write_transfer_latency_count"],
            )
        return total

    @property
    def supported_hash_algos(self) -> frozenset[KVHashAlgo]:
        """Algos this connector accepts in :meth:`load` / :meth:`offload`.

        Accepts the full ahash64-family set plus 32-byte ``sha256``: 32-byte
        digests are truncated to their first 8 bytes at the boundary, which
        is byte-identical to the ``sha256_64`` algo (see :func:`_to_dkv_u64`
        and the module docstring). The dkv wire format stays ``uint64
        seq_hash``.
        """
        return frozenset({"ahash64", "sha256", "sha256_64"})
