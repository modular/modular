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
from max.driver import Device
from max.nn.kv_cache.cache_params import (
    KVCacheMemory,
    KVCacheParamInterface,
    KVCacheParams,
    KVHashAlgo,
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


def _kv_config_hash(params: KVCacheParams) -> int:
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

    * ``v`` — contract version (``1``).
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
      (``prod(shape_per_block) * dtype.size``); the geometry source, equal to the
      connector's negotiated full-attention ``block_bytes`` (quant scales are
      captured separately by ``quant``).

    Model/weights identity is deliberately NOT folded here: a different model is
    a different Mammoth deployment, hence a different ``tenant_id`` (already part
    of the dedup key). Reattaching persisted shares across a same-``tenant_id``
    weights swap (CLIN-1474) must additionally fold a weights/version fingerprint
    threaded from the pipeline config — a documented follow-up, out of scope for
    this handshake.

    Args:
        params: The single-group KV-cache parameters for this deployment.

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
        ("v", "1"),
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


def _validate_tenant_topology(
    *, is_single_group: bool, tensor_parallel_degree: int
) -> None:
    """Guards the model topology a multi-tenant dKV handshake can represent.

    The ``ExchangeMetadata`` identity fields (``kv_shard_id``, ``replica_id``,
    ``device_id``, ``numa_node``) are scalars that name one GPU's share, and MAX
    issues one handshake per DP replica. That faithfully carries per-GPU identity
    only when each replica maps to exactly one GPU, i.e. ``TP == 1`` (which
    covers the v1 target Kimi K2.5 at DP=8/TP=1, and any TP=1 deployment). For
    ``TP > 1`` a replica spans several GPUs whose distinct per-GPU identities a
    single request cannot carry, so a per-GPU handshake split is required — a
    documented follow-up. (The general :func:`_resolve_kv_share_identity` rule
    would also need refining for ``allow_kv_head_replication``, where distinct TP
    ranks hold byte-identical GQA heads.) Multi-group caches (hybrid / SWA /
    speculative) are likewise not yet wired through this connector.

    Raises:
        NotImplementedError: If the cache is multi-group, or ``TP != 1``.
    """
    if not is_single_group:
        raise NotImplementedError(
            "Multi-tenant dKV (MODULAR_DKV_TENANT_ID set) requires a single-group "
            "KV cache; hybrid / sliding-window / speculative multi-cache models "
            "are not wired through the dKV connector yet."
        )
    if tensor_parallel_degree != 1:
        raise NotImplementedError(
            "Multi-tenant dKV (MODULAR_DKV_TENANT_ID set) currently supports "
            f"tensor_parallel_degree == 1 only, got {tensor_parallel_degree}. "
            "Each DP replica must map to exactly one GPU so the per-replica "
            "ExchangeMetadata handshake carries that GPU's scalar identity; a "
            "per-GPU handshake split for TP>1 is a follow-up."
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
      and maps each replica's one GPU (TP==1) to its share via
      :func:`_resolve_kv_share_identity`, sharing the layout ``kv_config_hash``.

    Args:
        tenant_id: The resolved tenant identity (empty is the default
            single-tenant path, resolving to one shared, replica-agnostic
            store).
        num_replicas: Number of DP replicas (one dKV client each).
        params: KV-cache parameters (used only on the multi-tenant path).

    Returns:
        ``(kv_config_hash, [(kv_shard_id, replica_id), ...])``.

    Raises:
        NotImplementedError: If the multi-tenant topology is unsupported
            (multi-group cache, or ``TP != 1``).
        ValueError: If ``num_replicas`` disagrees with ``data_parallel_degree``.
    """
    if not tenant_id:
        # Default path: all DP replicas resolve to ONE shared, replica-agnostic
        # store (see _DEFAULT_SHARED_STORE_REPLICA_IDENTITY). The paired
        # kv_config_hash is 0; the server ignores it on the empty-tenant path.
        return 0, [_DEFAULT_SHARED_STORE_REPLICA_IDENTITY] * num_replicas
    _validate_tenant_topology(
        is_single_group=isinstance(params, KVCacheParams),
        tensor_parallel_degree=params.tensor_parallel_degree,
    )
    assert isinstance(params, KVCacheParams)  # narrowed by the guard above
    if num_replicas != params.data_parallel_degree:
        raise ValueError(
            f"replica count {num_replicas} does not match data_parallel_degree "
            f"{params.data_parallel_degree}; the replica-major device mapping "
            "the identity rule relies on would be wrong"
        )
    tp = params.tensor_parallel_degree
    replicates = params.replicates_kv_across_tp
    # TP==1 (guarded), so the replica's one GPU has replica-major flat index
    # ``idx * tp + 0 == idx``.
    identities = [
        _resolve_kv_share_identity(idx * tp, tp, replicates)
        for idx in range(num_replicas)
    ]
    return _kv_config_hash(params), identities


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
    inherently per-replica (its ``load``/``offload`` reference block ids into
    one replica's registered device buffers and carry no replica/group key), so
    this shim owns one Rust client per replica and routes each call to the
    client for the request's ``replica_idx``.

    On the default path all replicas share ONE replica-agnostic store (see
    :func:`_resolve_replica_identities`), so ``self._clients[replica_idx]``
    selects only WHICH per-replica client (device endpoint) moves the
    blocks, not which store is addressed — routing by the PROCESSING replica.
    That selection in :meth:`load` / :meth:`offload` is the seam CLIN-1478
    slice-3 later swaps to route by the content-hash OWNER replica
    (``hash(seq_hash) % dp_size``) for multi-tenant per-replica isolation.
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
        kv_config_hash, replica_identities = _resolve_replica_identities(
            tenant_id, num_replicas, params
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

        # One Rust client per replica, each registering only its replica's
        # device buffers with NIXL, in that replica's canonical device order.
        # Each client's connect + handshake ("admission") is retried on transient
        # failures (dKV still starting); model readiness is gated on ALL replicas
        # admitting, so a client whose retry budget is exhausted raises here and
        # fails model load rather than serving with a partial dKV.
        self._clients = []
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

        if tenant_id:
            _logger.info(
                "dKV admitted all %d replica handshake(s) for tenant %r",
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
        # Per-buffer (ptr, byte length, device ordinal) for NIXL registration.
        # Each KVCacheMemory wraps a ``[num_pages, bytes_per_page]`` uint8 device
        # buffer; MLA caches (ReplicatedKVCacheMemory) also carry TP-peer replicas
        # that hold identical data and must be registered too.
        device_buffer_meta: list[tuple[int, int, int]] = []
        device_ids: set[int] = set()
        # MAX's compute stream per device ordinal, so the same-host offload can
        # order each device's D2H after the forward pass that wrote its blocks
        # via a CUDA event in that device's own context. Events and streams are
        # per context, so multi-device TP needs a handle per device rather than
        # one shared handle. A device whose stream has no native handle (e.g. a
        # CPU stream) maps to 0, which routes that device's transfers over NIXL.
        compute_streams: dict[int, int] = {}
        is_mla = False
        for mem in kv_memory:
            buffers = mem.all_buffers
            if len(buffers) > 1:
                is_mla = True
            for buffer in buffers:
                device_buffer_meta.append(
                    (
                        buffer._data_ptr(),
                        buffer.num_elements * buffer.dtype.size_in_bytes,
                        buffer.device.id,
                    )
                )
                device_ids.add(buffer.device.id)
                compute_streams[buffer.device.id] = (
                    buffer.device.default_stream.native_stream_handle
                )

        # The Rust client keys each block by its shard's position in
        # ``device_buffer_meta`` (``tp_shard_id``), so that position must equal
        # the shard's rank in the replica's canonical device order. The per-shard
        # keying assumes exactly one buffer per device, so two layouts would
        # break it and are rejected here rather than silently mis-keyed. A
        # quantized cache appends its scale buffers after the value buffers, so
        # each device appears twice and the scales would never be keyed or
        # transferred, giving a wrong dequant. A ``MultiKVCacheBuffer`` (hybrid
        # sliding and global, or speculative draft and target) concatenates
        # several caches, so a device appears once per cache and ``tp_shard_id``
        # stops equalling a device rank. Both show up as more registered buffers
        # than distinct devices. dKV does not carry the extra buffers yet,
        # tracked by CLIN-1460, so fail loudly.
        if not is_mla and len(device_buffer_meta) != len(device_ids):
            raise NotImplementedError(
                "The dKV connector requires exactly one KV buffer per device for "
                f"non-MLA TP, but got {len(device_buffer_meta)} buffers across "
                f"{len(device_ids)} devices. Quantized (FP8 scale) caches and "
                "multi-cache buffers (hybrid or speculative decoding) are not "
                "supported on the dKV connector yet (CLIN-1460)."
            )

        # Bind ``tp_shard_id`` to device identity rather than to registration
        # luck. A remote peer fetches a block by the ``(tp_shard_id, group,
        # seq_hash)`` key, so its shard ids must line up with ours by device
        # rank. ``expected_devices`` is the replica's device order sourced from
        # the pipeline config, so comparing it against the order the buffers
        # actually registered in catches a future ``to_memory`` change that
        # reorders buffers before it silently shifts every key. First-occurrence
        # order collapses an MLA replica's peer buffers and a quantized cache's
        # scale buffers down to one entry per device.
        registered_order = list(
            dict.fromkeys(device_id for _, _, device_id in device_buffer_meta)
        )
        expected_order = [device.id for device in expected_devices]
        if registered_order != expected_order:
            raise ValueError(
                "dKV registered KV buffers in device order "
                f"{registered_order}, which does not match the replica's "
                f"canonical device order {expected_order}. tp_shard_id is bound "
                "to that order, so a mismatch would mis-key blocks across peers."
            )

        # ``total_num_pages`` is the buffer's physical page count
        # (``buffer.shape[0]``), which already includes MAX's trailing "null"
        # page beyond the logical block count. The Rust client divides the
        # registered buffer length by this to derive the per-page byte stride,
        # so it must be the physical count; valid-block offsets are unaffected
        # since the null page is last and is never transferred.
        total_num_pages = kv_memory[0].total_num_pages

        return client_cls(
            local_block_store_endpoint,
            device_buffer_meta,
            0,  # page_size (tokens): unused by the Rust client
            total_num_pages,
            len(device_ids),
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
        """
        return self._clients[replica_idx].load(
            group_id=_DKV_GROUP_FULL_ATTENTION,
            device_block_ids=device_block_ids,
            block_hashes=[_to_dkv_u64(h) for h in block_hashes],
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
        """
        self._clients[replica_idx].offload(
            group_id=_DKV_GROUP_FULL_ATTENTION,
            block_ids=block_ids,
            block_hashes=[_to_dkv_u64(h) for h in block_hashes],
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
