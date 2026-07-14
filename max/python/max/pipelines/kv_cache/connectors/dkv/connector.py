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
    MultiKVCacheParams,
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


def _layout_fields(
    params: KVCacheParams | MultiKVCacheParams,
) -> list[tuple[str, str]]:
    """Folds one cache's byte-layout into ordered ``key=value`` fields.

    A leaf :class:`KVCacheParams` contributes its per-shard geometry; a
    :class:`MultiKVCacheParams` tree contributes a ``multi`` marker, its child
    count, and each child's fields recursively, prefixed by the child's index
    and name in the tree's insertion order. That order is deterministic for a
    fixed model config and matches the ``to_memory()`` unit order the
    concatenated block (and thus ``unit_strides``) follows, so folding the index
    and name makes any child add/remove/reorder flip the fingerprint.

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

    * ``v`` — contract version (``2``; bumped when the block layout became the
      concatenation of every buffer unit rather than the value buffer alone).
    * the cache geometry from :func:`_layout_fields`: for a single-group leaf,
      its dtype/kv_dim/head_dim/num_layers/page_size/head-count/TP/quant/value
      bytes (unchanged from the ``v=2`` leaf encoding, so single-group
      fingerprints stay byte-identical); for a :class:`MultiKVCacheParams` tree
      (speculative draft+target, quantized values+scales), a ``multi`` marker
      plus each child's fields folded recursively under its index and name.
    * ``unit_strides`` — comma-joined per-page byte stride of one shard's
      buffer units in canonical ``to_memory()`` order (values, quant scales,
      indexer, draft, and so on), derived by :func:`_shard_unit_strides`. A
      shard's dKV block is these strides concatenated across the WHOLE cache
      tree, so any change to the unit set or its ordering makes stored blocks
      byte-incompatible and must flip the hash. Folding one shard's subsequence
      rather than the flat physical buffer list keeps the folded shape identical
      between replicated (MLA) and sharded layouts; the shard count is already
      pinned by ``tensor_parallel_degree``.

    Model/weights identity is deliberately NOT folded here: a different model is
    a different Mammoth deployment, hence a different ``tenant_id`` (already part
    of the dedup key). Reattaching persisted shares across a same-``tenant_id``
    weights swap (CLIN-1474) must additionally fold a weights/version fingerprint
    threaded from the pipeline config — a documented follow-up, out of scope for
    this handshake.

    Multi-group support was added without a ``v`` bump: the leaf encoding is
    byte-identical to ``v=2`` (no single-group share is invalidated), and the
    multi-group path previously raised, so no multi-group share could have been
    persisted under an earlier contract.

    Args:
        params: The KV-cache parameters for this deployment — a single-group
            leaf or a multi-cache tree.
        unit_strides: Per-page byte stride of one shard's buffer units in
            canonical order, from :func:`_shard_unit_strides`.

    Returns:
        The 64-bit layout fingerprint.
    """
    fields = [
        ("v", "2"),
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
    shim owns ONE client per DP replica in ``self._clients`` and routes each call
    to ``self._clients[replica_idx]`` for the request's processing replica.

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
            params: KV-cache parameters, folded into the tenant store's layout
                ``kv_config_hash``.

        Raises:
            ValueError: If ``MODULAR_DKV_TENANT_ID`` is unset or empty — dKV has
                no default/legacy single-tenant path, so it fails model load
                rather than silently keying an unfenced shared store.
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

        # Tenant deployment identity (CLIN-1477). MODULAR_DKV_TENANT_ID is
        # injected by the operator (the trust boundary — not a user-facing
        # override flag, which would be forgeable). It is REQUIRED: dKV has no
        # default/legacy single-tenant path, so an unset or empty value fails
        # model load rather than silently keying an unfenced shared store. Every
        # DP replica handshakes the same per-tenant identity (kv_shard_id/
        # replica_id zeroed), so the server keys ONE region-sharded store per
        # tenant_id (backend dedup).
        tenant_id = os.getenv("MODULAR_DKV_TENANT_ID", "")
        if not tenant_id:
            raise ValueError(
                "dKV requires MODULAR_DKV_TENANT_ID to be set to a non-empty "
                "tenant identity (the operator injects it); the legacy "
                "empty-tenant default path has been removed."
            )
        num_replicas = len(replica_kv_memory)
        # one shard's per-unit page strides in canonical order, from replica 0
        # because every DP replica runs the same model and config and so the
        # same layout; folded into the layout hash because a shard's dKV block
        # is these strides concatenated
        unit_strides = _shard_unit_strides(replica_kv_memory[0])
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

        admission_timeout_s = float(
            os.getenv(
                "MODULAR_DKV_ADMISSION_TIMEOUT_S",
                str(_DEFAULT_ADMISSION_TIMEOUT_S),
            )
        )

        # Each client's connect + handshake ("admission") is retried on transient
        # failures (dKV still starting); model readiness is gated on ALL clients
        # admitting, so a client whose retry budget is exhausted raises here and
        # fails model load rather than serving with a partial dKV. ``self._clients``
        # holds one client per DP replica: ``load`` / ``offload`` route by
        # ``replica_idx`` to ``self._clients[replica_idx]``, and the client-wide
        # fan-outs (wait_for_*, metrics, reset_metrics) iterate the whole list.
        self._clients = []
        # Backend dedup: one client per DP replica, each registering that
        # replica's FULL TP GPU set (its flat units concatenated per shard by
        # _make_client). For MLA this restores device_buffers.len() == tp, so the
        # Rust client's NVLink broadcast + NUMA-local first hop re-engage
        # (the CLIN-1512 per-GPU split had made them inert). The store key stays
        # per-tenant — replica_identities zeros kv_shard_id/replica_id, so every
        # DP replica of a tenant resolves to ONE store — and the per-block
        # BlockKey tp_shard_id carries the MHA/GQA-vs-MLA distinction.
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
                tenant_gpu_count=tenant_gpu_count,
                tenant_gpu_device_ids=tenant_gpu_device_ids,
            )
            self._clients.append(
                _admit_with_retry(
                    factory,
                    timeout_s=admission_timeout_s,
                    label=f"replica {idx}",
                )
            )
        # One client per DP replica is the backend-dedup invariant that lets
        # load/offload index self._clients[replica_idx] directly, with no
        # per-replica shard-client fan-out. #91376's divergent-load drain was a
        # cross-CLIENT concern; one client per replica cannot produce cross-client
        # divergence (the single Rust client owns its own multi-GPU ordering), so
        # that drain is gone. Guard the invariant fail-loud: a future change that
        # rebuilds multiple clients per replica trips here rather than silently
        # reindexing the wrong client or skipping the removed drain.
        if len(self._clients) != num_replicas:
            raise RuntimeError(
                f"dKV backend dedup expects one client per DP replica; built "
                f"{len(self._clients)} clients for {num_replicas} replica(s)"
            )

        # Surface the Rust connector's MLA NVLink-broadcast status to the serve
        # process: the Rust side logs it via ``tracing``, which has no subscriber
        # under MAX. ``broadcast_peer_count`` is ``tp - 1`` once the broadcast
        # armed at handshake, and 0 for a non-MLA model, a single device, or a
        # topology without peer access, so log only when it engaged.
        for idx, client in enumerate(self._clients):
            peers = client.broadcast_peer_count()
            if peers:
                _logger.info(
                    "dKV MLA NVLink broadcast enabled: replica %d "
                    "broadcast_peer_count=%d",
                    idx,
                    peers,
                )

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
        tenant_gpu_count: int,
        tenant_gpu_device_ids: Sequence[int],
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
            tenant_gpu_count=tenant_gpu_count,
            tenant_gpu_device_ids=list(tenant_gpu_device_ids),
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

        Routes to the processing replica's single client (backend dedup: one
        client per DP replica, registering that replica's full TP GPU set). The
        client returns the loaded-block count; the block manager frees
        ``blocks[num_loaded:]`` past it (in
        ``_get_full_blocks_from_host_prefix_cache``). The Rust client owns the
        freed-page ordering across its own GPUs, so there is no shard-client
        fan-out or cross-client drain at this layer.
        """
        dkv_hashes = [_to_dkv_u64(h) for h in block_hashes]
        return self._clients[replica_idx].load(
            group_id=_DKV_GROUP_FULL_ATTENTION,
            device_block_ids=device_block_ids,
            block_hashes=dkv_hashes,
        )

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

        Routes to the processing replica's single client (backend dedup: one
        client per DP replica, registering that replica's full TP GPU set).
        """
        dkv_hashes = [_to_dkv_u64(h) for h in block_hashes]
        self._clients[replica_idx].offload(
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
