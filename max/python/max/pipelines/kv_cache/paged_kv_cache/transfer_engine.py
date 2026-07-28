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

"""KVCache transfer engine."""

from __future__ import annotations

import ctypes
import logging
import os
import random
import socket
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

import msgspec
from max._core import nixl
from max.driver import Buffer, Device

from .cache_manager import PagedKVCacheManager

logger = logging.getLogger("max.pipelines")

NixlBackendType = Literal["ucx", "libfabric"]

_NIXL_BACKEND_ENV_VAR = "MODULAR_NIXL_TRANSFER_BACKEND"
_SUPPORTED_BACKENDS: set[NixlBackendType] = {"ucx", "libfabric"}

# GPU runtime libraries that the upstream UCX plugin (libplugin_UCX.so, in
# its per-vendor flavors) references but does not itself dlopen. The upstream
# plugin manager loads plugins with ``dlopen(..., RTLD_NOW | RTLD_LOCAL)``;
# ``RTLD_NOW`` requires every undefined symbol (CUDA driver, NVML, HSA,
# optionally RDMA verbs) to be resolvable at load time, and ``RTLD_LOCAL``
# means the plugin cannot see symbols unless they were already loaded
# ``RTLD_GLOBAL`` into the process.
_NIXL_PLUGIN_DEP_LIBS: tuple[str, ...] = (
    # RDMA verbs (only needed by the *-verbs UCX flavors); harmless if absent.
    "libibverbs.so.1",
    "libmlx5.so.1",
    # CUDA driver + NVML: required by the CUDA-flavor UCX plugin.
    "libcuda.so.1",
    "libnvidia-ml.so.1",
    # HSA runtime: required by the ROCm-flavor UCX plugins.
    "libhsa-runtime64.so.1",
)

_nixl_plugin_deps_preloaded = False


def _preload_nixl_plugin_deps() -> None:
    """Pre-loads the UCX plugin's runtime dependencies with ``RTLD_GLOBAL``.

    The upstream NIXL plugin manager ``dlopen``s ``libplugin_UCX.so`` with
    ``RTLD_NOW | RTLD_LOCAL``. The vendored plugin flavor (selected per host
    GPU vendor via ``NIXL_PLUGIN_DIR``) references CUDA/NVML, HSA, or RDMA
    verbs symbols; ``RTLD_LOCAL`` prevents the plugin from resolving them
    against the process unless they were previously loaded with
    ``RTLD_GLOBAL``. Without this, ``get_plugin_params("UCX")`` returns
    ``NIXL_ERR_NOT_FOUND`` because the plugin fails to load.

    The Modular NIXL fork performed this preload inside its plugin-manager
    constructor; upstream does not, so we restore it here. It runs in every
    process that constructs a transfer engine — including ``spawn``-ed
    multiprocessing children, which do NOT inherit the parent's ``RTLD_GLOBAL``
    handles. Libraries that are absent on the host (e.g. CUDA on an AMD/CPU
    box) are skipped; the plugin simply cannot load there, which is reported by
    the existing availability checks rather than masked.
    """
    global _nixl_plugin_deps_preloaded
    if _nixl_plugin_deps_preloaded:
        return
    for lib_name in _NIXL_PLUGIN_DEP_LIBS:
        try:
            ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Not present on this host; the corresponding UCX flavor cannot be
            # used here. This is not an error to swallow — it is a genuine
            # "this transport is unavailable on this machine" signal that
            # surfaces downstream via get_available_plugins / get_plugin_params.
            logger.debug(
                "NIXL plugin dependency %s not found; skipping preload",
                lib_name,
            )
    _nixl_plugin_deps_preloaded = True


def _get_nixl_backend_type() -> NixlBackendType:
    """Returns the NIXL backend type from the environment.

    Reads ``MODULAR_NIXL_TRANSFER_BACKEND`` (default ``"ucx"``).
    """
    raw = os.environ.get(_NIXL_BACKEND_ENV_VAR, "ucx").strip().lower()
    if raw not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported NIXL transfer backend {raw!r} "
            f"(set via {_NIXL_BACKEND_ENV_VAR}). "
            f"Supported backends: {sorted(_SUPPORTED_BACKENDS)}"
        )
    return raw  # type: ignore[return-value]


def available_port(
    start_port: int = 8000, end_port: int = 9000, max_attempts: int = 100
) -> int:
    """Finds an available TCP port in the given range.

    Args:
        start_port: The lower bound of the port range (inclusive).
        end_port: The upper bound of the port range (inclusive).
        max_attempts: Maximum number of attempts to find a free port.

    Returns:
        int: An available port number.

    Raises:
        RuntimeError: If no available port is found after max_attempts.
    """
    for _ in range(max_attempts):
        port = random.randint(start_port, end_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Set SO_REUSEADDR to avoid TIME_WAIT issues
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found in the specified range.")


def _validate_device_type(devices: Sequence[Device]) -> None:
    is_gpu = False
    is_cpu = False
    for d in devices:
        if d.is_host:
            is_cpu = True
        else:
            is_gpu = True

    if is_cpu and is_gpu:
        raise ValueError(
            "Mixed device tensors detected. All tensors must be either on CPU or GPU, not both."
        )

    first_device = devices[0]
    if not first_device.is_host and (
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT" not in os.environ
        and "BAZEL_TEST" not in os.environ
    ):
        # See GEX-2445 for more details.
        # We intentionally make falling back to the slower CUDA_COPY transport
        # a hard error. This check is best effort. Just because it is not
        # tripped does not guarantee that the we will end up using CUDA_IPC.
        # Note that we will use MemoryManager regardless when running under
        # bazel test.
        raise ValueError(
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT must be set when using TransferEngine with GPU memory. "
            "This flag enables the MemoryManager which is required for the fast CUDA_IPC transport. "
            "Try rerunning your command with MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=99"
        )


def _validate_tensor_shape(
    tensors: Sequence[Buffer], total_num_pages: int
) -> tuple[int, int]:
    # Validate all tensors have the same shape
    first_tensor = tensors[0]
    if len(tensors) > 1:
        first_shape = first_tensor.num_elements
        first_dtype = first_tensor.dtype

        for i, tensor in enumerate(tensors[1:], 1):
            if tensor.num_elements != first_shape:
                raise ValueError(
                    f"All tensors must have the same shape. Tensor 0 has {first_shape} elements, but Tensor {i} has {tensor.num_elements} elements"
                )
            if tensor.dtype != first_dtype:
                raise ValueError(
                    f"All tensors must have the same dtype. Tensor 0 has {first_dtype}, but Tensor {i} has {tensor.dtype}"
                )

    for i, tensor in enumerate(tensors):
        if tensor.num_elements % total_num_pages != 0:
            raise ValueError(
                f"Tensor {i} num elements {tensor.num_elements} must be divisible by total number of pages {total_num_pages}"
            )

    # Calculate bytes per page
    bytes_per_page = (
        first_tensor.num_elements
        * first_tensor.dtype.size_in_bytes
        // total_num_pages
    )
    elts_per_page = first_tensor.num_elements // total_num_pages
    return bytes_per_page, elts_per_page


def _build_group_descriptors(
    base_addrs: Sequence[int],
    bytes_per_group: Sequence[int],
    page_idxs: Sequence[int],
    device_id: int,
) -> list[tuple[int, int, int]]:
    """Build NIXL ``(addr, size, device)`` descriptors for all groups.

    For each group ``g`` and page index ``i``, emits
    ``(base_addrs[g] + i * bytes_per_group[g], bytes_per_group[g], device_id)``,
    iterating group-major then page-index (the order the paired src/dst
    descriptor lists rely on).

    Each group uses its OWN base address and per-page stride, so groups with
    different ``bytes_per_page`` never share addressing -- the invariant that
    guards against the draft-KV stride-mismatch class (SERVOPT-1456).
    """
    descs: list[tuple[int, int, int]] = []
    for group_idx, bpp in enumerate(bytes_per_group):
        base = base_addrs[group_idx]
        for idx in page_idxs:
            descs.append((base + idx * bpp, bpp, device_id))
    return descs


class TensorAgentMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Metadata for a single tensor/agent in the transfer engine.

    This is used for serialization and communication between engines.
    """

    agent_name: str
    """Name of this agent."""

    metadata: bytes
    """Metadata for this agent."""

    base_addrs: list[int]
    """Base memory address per NIXL group for this shard, indexed by group.
    ``base_addrs[g]`` is the base of group ``g`` (e.g. values, scales, or a
    per-child cache). Parallel to the engine's ``bytes_per_group``; there is
    no special "main" group."""

    device_id: int
    """Device ID for this tensor."""


@dataclass
class TensorAgent:
    """Manages a single tensor and its associated NIXL agent for transfers.

    This class holds both the runtime state (live objects) and can generate
    the serializable metadata for communication between engines.
    """

    agent: nixl.Agent
    """NIXL agent for this tensor."""

    agent_name: str
    """Name of this agent."""

    base_addrs: list[int]
    """Base memory address per NIXL group for this shard, indexed by group.
    Parallel to ``reg_dlists`` and to the engine's ``bytes_per_group``; there
    is no special "main" group."""

    backend: int
    """NIXL backend handle (UCX or libfabric)."""

    device_id: int
    """Device ID for this tensor."""

    agent_metadata: bytes
    """Metadata for this agent."""

    reg_dlists: list[nixl.RegistrationDescriptorList]
    """Registration descriptor list per NIXL group, parallel to ``base_addrs``."""

    @classmethod
    def create_agent(
        cls,
        agent_name: str,
        listen_port: int,
        tensors: Sequence[Buffer],
        memory_type: nixl.MemoryType,
        backend_type: NixlBackendType = "ucx",
    ) -> TensorAgent:
        """Creates and registers a NIXL agent for a shard's per-group buffers.

        Args:
            agent_name: Unique name for this agent.
            listen_port: TCP port for the NIXL listener.
            tensors: This shard's buffers, one per NIXL group, group-major
                (e.g. ``[main_values, main_scales, draft_values,
                draft_scales]``). All must share the same device. Must be
                non-empty.
            memory_type: NIXL memory segment type (DRAM or VRAM).
            backend_type: NIXL transport backend (``"ucx"`` or ``"libfabric"``).
        """
        # Pre-load the UCX plugin's GPU runtime dependencies with RTLD_GLOBAL
        # before the NIXL plugin manager dlopens the plugin. Must run in this
        # process (e.g. spawn-ed children do not inherit RTLD_GLOBAL handles).
        _preload_nixl_plugin_deps()

        # Create NIXL agent
        agent = nixl.Agent(
            agent_name,
            nixl.AgentConfig(
                # Always use progress thread.
                # - It helps with async notification delivery.
                # - It enables overlapping transfers from multiple agents.
                use_prog_thread=True,
                use_listen_thread=True,
                listen_port=listen_port,
            ),
        )

        # Check backend availability.
        # Upstream NIXL plugin names are uppercase (UCX, LIBFABRIC); the
        # Modular-facing API (MODULAR_NIXL_TRANSFER_BACKEND) keeps lowercase
        # values for backwards compatibility. Map to upstream internally.
        upstream_backend_type = backend_type.upper()
        available = agent.get_available_plugins()
        if upstream_backend_type not in available:
            raise RuntimeError(
                f"NIXL backend {backend_type!r} not available for agent "
                f"{agent_name}. Available plugins: {available}"
            )

        # All groups for one shard live on the same device.
        device = tensors[0].device
        backend_params = agent.get_plugin_params(upstream_backend_type)[0]
        if not device.is_host:
            backend_params["gpu_device_id"] = str(device.id)

        backend = agent.create_backend(
            type=upstream_backend_type,
            init_params=backend_params,
        )

        # Register one memory region per group, uniformly.
        base_addrs: list[int] = []
        reg_dlists: list[nixl.RegistrationDescriptorList] = []
        for tensor in tensors:
            base_addr = tensor._data_ptr()
            num_bytes = tensor.num_elements * tensor.dtype.size_in_bytes
            reg_dlist = nixl.RegistrationDescriptorList(
                type=memory_type,
                descs=[(base_addr, num_bytes, device.id, "")],
            )
            status = agent.register_memory(reg_dlist, [backend])
            if status != nixl.Status.SUCCESS:
                raise ValueError(
                    f"Failed to register memory for {agent_name}: {status}"
                )
            base_addrs.append(base_addr)
            reg_dlists.append(reg_dlist)

        # Get metadata after registration
        agent_metadata = agent.get_local_metadata()

        # Create TensorAgent and add to list
        return TensorAgent(
            agent=agent,
            agent_name=agent_name,
            base_addrs=base_addrs,
            backend=backend,
            device_id=device.id,
            agent_metadata=agent_metadata,
            reg_dlists=reg_dlists,
        )

    def to_metadata(self) -> TensorAgentMetadata:
        """Convert to serializable metadata for communication."""
        return TensorAgentMetadata(
            agent_name=self.agent_name,
            metadata=self.agent_metadata,
            base_addrs=self.base_addrs,
            device_id=self.device_id,
        )


@dataclass
class _PeerView:
    """Per-peer routing view computed at connect() time.

    Captures whether either side's ``[dp][tp]`` must be reinterpreted as
    ``[dp*tp][1]`` for this peer, and the resulting effective DP.
    """

    flatten_local: bool
    flatten_remote: bool
    effective_dp: int


def resolve_peer_view(
    local_dp: int,
    local_tp: int,
    local_replicate: bool,
    remote_dp: int,
    remote_tp: int,
    remote_replicate: bool,
) -> _PeerView:
    """Decide how to view the local and remote ``[dp][tp]`` for this peer.

    Homogeneous shapes match as-is. Heterogeneous shapes are accepted only
    when exactly one side has ``replicate=True``, ``tp > 1``, and its
    ``dp * tp`` matches the other side's ``dp``. Anything else raises.
    """
    if (local_dp, local_tp) == (remote_dp, remote_tp):
        return _PeerView(
            flatten_local=False, flatten_remote=False, effective_dp=local_dp
        )

    if (
        local_replicate
        and local_tp > 1
        and remote_tp == 1
        and local_dp * local_tp == remote_dp
    ):
        return _PeerView(
            flatten_local=True, flatten_remote=False, effective_dp=remote_dp
        )

    if (
        remote_replicate
        and remote_tp > 1
        and local_tp == 1
        and remote_dp * remote_tp == local_dp
    ):
        return _PeerView(
            flatten_local=False, flatten_remote=True, effective_dp=local_dp
        )

    raise ValueError(
        f"Incompatible transfer engine shapes: "
        f"local=(dp={local_dp},tp={local_tp},replicate={local_replicate}) "
        f"remote=(dp={remote_dp},tp={remote_tp},replicate={remote_replicate}). "
        f"Heterogeneous DP/TP is only supported when exactly one side "
        f"has replicate_kv_across_tp=True (e.g. MLA) with TP>1 and its "
        f"DP*TP matches the other side's DP. Both-TP>1 and sharded TP "
        f"reshards are not supported."
    )


# ---------------------------------------------------------------------------
# Topology resolver (pure, NIXL-free)
#
# These functions turn a resolved ``_PeerView`` plus the two engines' ``[dp][tp]``
# shapes into *index plans*: which (local, remote) agent pairs to wire at
# connect time, and which (source, destination) shards to pair for a single
# transfer. They hold no ``self`` and touch no NIXL objects, so they are
# directly CPU-unit-testable. :class:`TransferEngine` maps the returned indices
# onto its live ``TensorAgent`` grid and makes the NIXL calls.
# ---------------------------------------------------------------------------


def _effective_grid(
    dp: int, tp: int, flatten: bool
) -> list[list[tuple[int, int]]]:
    """View a ``[dp][tp]`` grid as ``[effective_dp][effective_tp]`` (physical) indices.

    When ``flatten`` is True the natural ``[dp][tp]`` is reinterpreted as
    ``[dp*tp][1]`` -- each TP shard becomes its own single-shard replica.
    Every entry is the physical ``(replica, shard)`` coordinate.
    """
    if flatten:
        return [[(r, s)] for r in range(dp) for s in range(tp)]
    return [[(r, s) for s in range(tp)] for r in range(dp)]


def connect_pairing(
    view: _PeerView,
    local_dp: int,
    local_tp: int,
    remote_dp: int,
    remote_tp: int,
) -> list[tuple[int, int, int, int]]:
    """Plan the connect/disconnect/cleanup wiring for a peer.

    Returns physical index quads ``(local_replica, local_shard,
    remote_replica, remote_shard)`` in the order the NIXL metadata
    load/invalidate must iterate: the full cartesian product of local x remote
    effective replicas, zipping shards within each replica pair. Applies the
    peer view's flatten flags so heterogeneous shapes line up under ``zip``.
    """
    local_grid = _effective_grid(local_dp, local_tp, view.flatten_local)
    remote_grid = _effective_grid(remote_dp, remote_tp, view.flatten_remote)
    assert len(local_grid) == len(remote_grid) == view.effective_dp
    pairs: list[tuple[int, int, int, int]] = []
    for local_replica in local_grid:
        for remote_replica in remote_grid:
            for (lr, ls), (rr, rs) in zip(
                local_replica, remote_replica, strict=True
            ):
                pairs.append((lr, ls, rr, rs))
    return pairs


def transfer_shard_pairing(
    flatten_source: bool,
    source_tp: int,
    dest_tp: int,
) -> list[tuple[int, int]]:
    """Plan the (source_shard, dest_shard) pairs for one transfer.

    The source side may be collapsed to a single shard (``flatten_source`` --
    an MLA-replicated source, where any shard's copy suffices and shard 0 saves
    bandwidth). The destination always spans all its shards (each owns distinct
    GPU memory). When the source is a single shard but the destination has many
    (DP-source -> TP-dest), the source is fanned out so every destination shard
    is paired.

    The caller reads ``local_shards_used`` off whichever side is local: the
    source shards for a send, the destination shards for a read.
    """
    if flatten_source:
        # TODO(SERVOPT-1337): always picking shard 0 hotspots one NIC/PCIe
        # path; rotate (round-robin or hashed) to spread load across shards.
        source_shards = [0]
    else:
        source_shards = list(range(source_tp))

    dest_shards = list(range(dest_tp))

    if len(source_shards) == 1 and len(dest_shards) > 1:
        source_shards = source_shards * len(dest_shards)

    return list(zip(source_shards, dest_shards, strict=True))


class TransferEngineMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Transport-only metadata for a :class:`TransferEngine`.

    Carries just the fields a generic NIXL transport needs to connect to a
    peer: the engine name, memory type, hostname, and per-shard agent
    metadata. KV/topology-specific fields live on
    :class:`KVTransferEngineMetadata`.

    This is safe to send between threads/processes.
    """

    name: str
    """Base name of the transfer engine."""

    memory_type: nixl.MemoryType
    """Memory type of the transfer engine."""

    hostname: str
    """Hostname of the machine that the transfer engine is running on."""

    agents_meta: list[list[TensorAgentMetadata]]
    """Metadata for each replica's agents: [replica][tp_shard]."""


class KVTransferEngineMetadata(TransferEngineMetadata):
    """Metadata associated with a KV cache transfer engine.

    Extends the transport-only :class:`TransferEngineMetadata` with the
    KV-cache/topology fields (page geometry and TP replication).

    This is safe to send between threads/processes.
    """

    total_num_pages: int
    """Total number of pages in each tensor."""

    bytes_per_page: int
    """Bytes per page for each tensor."""

    # Wire key is the field name; renaming it breaks decode against peers on an
    # older build, so add a `msgspec.field(name=...)` alias if DI ever does rolling deploys.
    replicate_kv_across_tp: bool = False
    """True iff buffers are identical across TP ranks (e.g. MLA KV with
    num_kv_heads=1). When both sides declare different (dp, tp) but one
    replicates, the engine can reinterpret the replicating side as
    ``[dp*tp][1]`` to let a prefill worker at (DP=m, TP=n) connect to a
    decode worker at (DP=m*n, TP=1)."""

    bytes_per_group: list[int] = []
    """Bytes per page for each tensor group. The first entry is the main
    group; subsequent entries correspond to extra groups (e.g., draft KV in
    speculative decoding). When non-empty, ``bytes_per_page`` equals
    ``sum(bytes_per_group)``; when empty, the engine is single-group and
    ``bytes_per_page`` holds the only group's value."""


class TransferReqData(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Metadata associated with a transfer request.

    This is safe to send between threads/processes.
    """

    dst_name: str
    """Base name of destination engine."""

    src_name: str
    """Base name of source engine."""

    transfer_name: str
    """Transfer name."""

    transfer_ids: list[int]
    """Transfer IDs (one per TP shard in the replica)."""

    src_idxs: list[int]
    """Length of source indices can differ from len(transfer_ids)."""

    dst_idxs: list[int]
    """Length of destination indices can differ from len(transfer_ids)."""

    src_replica_idx: int
    """Index of the source replica this transfer is from."""

    dst_replica_idx: int
    """Index of the destination replica this transfer is to."""

    is_read: bool = False
    """True if this is a READ (pull) transfer initiated by the destination."""

    tp_shard_count: int = 0
    """Number of TP shards participating. 0 = all shards (backwards compat)."""

    local_shards_used: list[int] = []
    """Physical TP shard indices on the initiator that own this transfer's
    handles. Empty means "all shards in the recorded replica" (pre-flatten
    behavior). Required to release/status-check transfers when flatten_local
    has picked a subset of shards."""


class TransferEngine:
    """NIXL transfer engine that owns the NIXL plumbing.

    - Agent lifecycle (create, connect, disconnect, cleanup)
    - Memory registration / deregistration
    - Descriptor list construction for (buffer, offset, size) ranges
    - Send / read transfer initiation and completion tracking
      (``initiate_send_transfer``, ``initiate_read_transfer``,
      ``is_complete``, ``cleanup_transfer``, ``sync_and_release``)

    This base still carries KV-cache topology today -- the ``[dp][tp]``
    ``tensor_agents`` grid, page geometry, and ``.metadata`` returns
    :class:`KVTransferEngineMetadata`; :class:`KVTransferEngine` is a thin
    construction subclass on top. Making the transport KV-agnostic (so it is
    testable without KV scaffolding) is tracked in MXSERV-313.

    ``TransferEngine`` is not thread-safe and is intended to be driven by
    MAX's single-threaded scheduler.
    """

    name: str
    """Name of this engine / NIXL agent group."""

    tensor_agents: list[list[TensorAgent]]
    """2D list of TensorAgent objects: [replica][tp_shard]."""

    total_num_pages: int
    """Total number of pages in each tensor."""

    bytes_per_page: int
    """Total bytes per page across all groups. For single-group engines this
    equals the main group's bytes per page; for multi-group engines it is
    ``sum(bytes_per_group)``."""

    bytes_per_group: list[int]
    """Bytes per page for each group. ``bytes_per_group[0]`` is the main
    group; subsequent entries are extra groups (e.g., draft KV in
    speculative decoding)."""

    memory_type: nixl.MemoryType
    """Type of memory being managed."""

    remote_connections: dict[str, KVTransferEngineMetadata]
    """Map of remote engine names to their metadata."""

    remote_agent_to_engine: dict[str, str]
    """Map of remote agent names to their engine names."""

    completed_recv_transfers: dict[str, dict[str, int]]
    """Map of agent names to completed recv transfers."""

    inflight_send_transfers: dict[str, TransferReqData]
    """Map of transfer names to send transfer request data."""

    dp: int
    """Number of DP replicas."""

    tp: int
    """Number of TP shards per replica."""

    def __init__(
        self,
        name: str,
        tensor_agents: list[list[TensorAgent]],
        *,
        total_num_pages: int,
        bytes_per_page: int,
        bytes_per_group: list[int],
        memory_type: nixl.MemoryType,
        dp: int,
        tp: int,
        backend_type: NixlBackendType,
        replicate_kv_across_tp: bool = False,
    ) -> None:
        self.name = name
        self.tensor_agents = tensor_agents
        self.total_num_pages = total_num_pages
        self.bytes_per_page = bytes_per_page
        self.bytes_per_group = bytes_per_group
        self.memory_type = memory_type
        self.dp = dp
        self.tp = tp
        self._backend_type = backend_type
        self.replicate_kv_across_tp = replicate_kv_across_tp

        # Remote connections
        self.remote_connections: dict[str, KVTransferEngineMetadata] = {}

        # Per-peer routing view populated at connect().
        self._peer_views: dict[str, _PeerView] = {}

        # Map of agents to completed transfers
        self.completed_recv_transfers: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Map of remote agent names to their engine names
        self.remote_agent_to_engine: dict[str, str] = {}

        # All send transfers - maps transfer_name to list of (tensor_idx, transfer_id) tuples
        self.inflight_send_transfers: dict[str, TransferReqData] = {}

        # All read transfers - maps transfer_name to TransferReqData
        self.inflight_read_transfers: dict[str, TransferReqData] = {}

    @property
    def metadata(self) -> KVTransferEngineMetadata:
        """Get metadata for all replicas.

        Returns:
            Metadata for the entire engine (all replicas).
        """
        agents_meta = [
            [ta.to_metadata() for ta in replica_agents]
            for replica_agents in self.tensor_agents
        ]

        return KVTransferEngineMetadata(
            name=self.name,
            total_num_pages=self.total_num_pages,
            bytes_per_page=self.bytes_per_page,
            memory_type=self.memory_type,
            agents_meta=agents_meta,
            hostname=socket.gethostname(),
            replicate_kv_across_tp=self.replicate_kv_across_tp,
            bytes_per_group=self.bytes_per_group,
        )

    def _resolve_local_agents_for_transfer(
        self, replica_idx: int, transfer_req: TransferReqData
    ) -> list[TensorAgent]:
        """Return the local ``TensorAgent``s that own a transfer's handles.

        Consults ``transfer_req.local_shards_used`` when populated; falls
        back to every shard in the replica when empty (pre-flatten behavior).
        """
        if not transfer_req.local_shards_used:
            return list(self.tensor_agents[replica_idx])
        return [
            self.tensor_agents[replica_idx][s]
            for s in transfer_req.local_shards_used
        ]

    def _compute_peer_view(self, remote: KVTransferEngineMetadata) -> _PeerView:
        """Decide how the local and remote shapes should be viewed for this peer.

        Thin wrapper around :func:`resolve_peer_view`.
        """
        rdp = len(remote.agents_meta)
        rtp = len(remote.agents_meta[0]) if remote.agents_meta else 0
        return resolve_peer_view(
            local_dp=self.dp,
            local_tp=self.tp,
            local_replicate=self.replicate_kv_across_tp,
            remote_dp=rdp,
            remote_tp=rtp,
            remote_replicate=remote.replicate_kv_across_tp,
        )

    def _iter_peer_agents(
        self, remote: KVTransferEngineMetadata, view: _PeerView
    ) -> Iterator[tuple[TensorAgent, TensorAgentMetadata]]:
        """Yield ``(local agent, remote agent-meta)`` pairs for a peer.

        Maps the resolver's physical index quads onto the live agent grids, in
        the order connect / disconnect / cleanup must iterate. Sharing this one
        iterator is what makes teardown mirror ``connect()``.
        """
        rdp = len(remote.agents_meta)
        rtp = len(remote.agents_meta[0]) if remote.agents_meta else 0
        for lr, ls, rr, rs in connect_pairing(view, self.dp, self.tp, rdp, rtp):
            yield self.tensor_agents[lr][ls], remote.agents_meta[rr][rs]

    def connect(self, remote: KVTransferEngineMetadata) -> None:
        """Connect to a remote engine (all replicas).

        Args:
            remote: Metadata for the remote engine (all replicas).
        """
        if remote.name in self.remote_connections:
            raise ValueError(f"Agent {remote.name} already connected")

        view = self._compute_peer_view(remote)

        if self.bytes_per_page != remote.bytes_per_page:
            raise ValueError(
                f"Bytes per page mismatch: {self.bytes_per_page} != {remote.bytes_per_page}"
            )

        # Validate per-group breakdown when both sides advertise it
        remote_bpg = remote.bytes_per_group
        if remote_bpg and self.bytes_per_group != remote_bpg:
            raise ValueError(
                f"Per-group bytes-per-page mismatch: "
                f"local={self.bytes_per_group} remote={remote_bpg}"
            )

        # Check if the relevant transport env vars are set. You can get away
        # with eliding these for intra-node DI. However, for inter-node DI,
        # loading metadata appears to hang (UCX) or performance degrades
        # severely (libfabric without GPU-direct RDMA) if they are not set.
        hostname = socket.gethostname()
        is_internode = hostname != remote.hostname
        if is_internode:
            backend_type = _get_nixl_backend_type()
            if backend_type == "ucx" and not (
                "UCX_NET_DEVICES" in os.environ and "UCX_TLS" in os.environ
            ):
                raise ValueError(
                    f"Attempted to connect to a TransferEngine on a different node but UCX transports are not configured ({hostname} <-> {remote.hostname}). "
                    "Please re-run and specify both the UCX_TLS and UCX_NET_DEVICES env vars."
                )
            if backend_type == "libfabric" and not os.environ.get(
                "FI_EFA_USE_DEVICE_RDMA"
            ):
                logger.warning(
                    "Inter-node libfabric connection (%s <-> %s) without "
                    "FI_EFA_USE_DEVICE_RDMA set. EFA GPU-direct RDMA will "
                    "be disabled, which may severely impact KV transfer "
                    "throughput. Set FI_EFA_USE_DEVICE_RDMA=1.",
                    hostname,
                    remote.hostname,
                )

        # Load remote metadata for every wired (local, remote) agent pair.
        for local_ta, remote_agent_meta in self._iter_peer_agents(remote, view):
            loaded_bytes = local_ta.agent.load_remote_metadata(
                remote_agent_meta.metadata
            )
            try:
                loaded_remote_name = loaded_bytes.decode()
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"Metadata loading failed. "
                    f"Expected string, found {loaded_bytes!r}"
                ) from e
            if loaded_remote_name != remote_agent_meta.agent_name:
                raise ValueError(
                    f"Metadata loading failed. "
                    f"Expected {remote_agent_meta.agent_name}, got {loaded_remote_name}"
                )

        self.remote_connections[remote.name] = remote
        self._peer_views[remote.name] = view

        # Update the remote agent to engine mapping
        for replica_agents_meta in remote.agents_meta:
            for agent_meta in replica_agents_meta:
                self.remote_agent_to_engine[agent_meta.agent_name] = remote.name

    def disconnect(self, name: str) -> None:
        """Tear down a single remote connection.

        Releases inflight transfer handles referencing this remote,
        invalidates NIXL metadata, and removes bookkeeping entries.
        After disconnect, ``connect()`` will accept the same name again.

        Args:
            name: The name of the remote engine to disconnect.

        Raises:
            ValueError: If the named remote is not currently connected.
        """
        remote = self.remote_connections.pop(name, None)
        if remote is None:
            raise ValueError(
                f"Remote connection '{name}' not found; cannot disconnect"
            )
        view = self._peer_views.pop(name, None)
        # Defensive: connect() populates _peer_views and remote_connections
        # together, so this is unreachable today. If they ever desync,
        # recompute (rather than assuming non-flattened) so a flattened peer's
        # teardown still mirrors connect()'s pairing.
        if view is None:
            view = self._compute_peer_view(remote)

        # Release inflight send transfers targeting this remote.
        stale_sends = [
            tname
            for tname, req in self.inflight_send_transfers.items()
            if req.dst_name == name
        ]
        for tname in stale_sends:
            req = self.inflight_send_transfers.pop(tname)
            src_agents = self._resolve_local_agents_for_transfer(
                req.src_replica_idx, req
            )
            for tp_idx, tid in enumerate(req.transfer_ids):
                try:
                    src_agents[tp_idx].agent.release_transfer_request(tid)
                except Exception:
                    logger.warning(
                        "Failed to release send transfer %s tp=%d"
                        " during disconnect of '%s'",
                        tname,
                        tp_idx,
                        name,
                        exc_info=True,
                    )

        # Release inflight read transfers sourced from this remote.
        stale_reads = [
            tname
            for tname, req in self.inflight_read_transfers.items()
            if req.src_name == name
        ]
        for tname in stale_reads:
            req = self.inflight_read_transfers.pop(tname)
            dst_agents = self._resolve_local_agents_for_transfer(
                req.dst_replica_idx, req
            )
            for tp_idx, tid in enumerate(req.transfer_ids):
                try:
                    dst_agents[tp_idx].agent.release_transfer_request(tid)
                except Exception:
                    logger.warning(
                        "Failed to release read transfer %s tp=%d"
                        " during disconnect of '%s'",
                        tname,
                        tp_idx,
                        name,
                        exc_info=True,
                    )

        # Teardown iterates the same pairs as connect() (shared iterator).
        for local_ta, remote_agent_meta in self._iter_peer_agents(remote, view):
            try:
                status = local_ta.agent.invalidate_remote_metadata(
                    remote_agent_meta.agent_name
                )
                if status != nixl.Status.SUCCESS:
                    logger.warning(
                        "invalidate_remote_metadata returned %s for"
                        " agent '%s' during disconnect of '%s'",
                        status,
                        remote_agent_meta.agent_name,
                        name,
                    )
            except Exception:
                logger.warning(
                    "Failed to invalidate metadata for agent '%s'"
                    " during disconnect of '%s'",
                    remote_agent_meta.agent_name,
                    name,
                    exc_info=True,
                )

        # Clean up agent-to-engine mapping entries for this remote.
        stale_agent_names = [
            agent_name
            for agent_name, engine_name in self.remote_agent_to_engine.items()
            if engine_name == name
        ]
        for agent_name in stale_agent_names:
            del self.remote_agent_to_engine[agent_name]

        # Drop completed recv transfer tracking for this remote.
        self.completed_recv_transfers.pop(name, None)

        logger.info("Disconnected remote '%s'", name)

    def initiate_send_transfer(
        self,
        remote_metadata: KVTransferEngineMetadata,
        src_idxs: list[int],
        dst_idxs: list[int],
        src_replica_idx: int,
        dst_replica_idx: int,
    ) -> TransferReqData:
        """Initiate a transfer from current engine to remote engine.

        The same page indices are broadcast to all TP shards within the source and destination replicas.

        Args:
            remote_metadata: Metadata for the remote engine.
            src_idxs: List of indices of the source pages in the current engine.
            dst_idxs: List of indices of the destination pages in the remote engine.
            src_replica_idx: Index of the source replica to transfer from.
            dst_replica_idx: Index of the destination replica to transfer to.
        """
        if not (0 <= src_replica_idx < self.dp):
            raise ValueError(
                f"src_replica_idx {src_replica_idx} must be between 0 and {self.dp - 1}"
            )

        if not (0 <= dst_replica_idx < len(remote_metadata.agents_meta)):
            raise ValueError(
                f"dst_replica_idx {dst_replica_idx} must be between 0 and {len(remote_metadata.agents_meta) - 1}"
            )

        if remote_metadata.name not in self.remote_connections:
            raise ValueError(
                f"Remote connection {remote_metadata.name} not found"
            )

        remote = self.remote_connections[remote_metadata.name]
        view = self._peer_views[remote_metadata.name]

        if len(src_idxs) != len(dst_idxs):
            raise ValueError(
                f"Source and destination indices must have the same length. Got {len(src_idxs)} and {len(dst_idxs)}"
            )

        # Each dst idx must be unique so that we don't write to the same page
        if len(set(dst_idxs)) != len(dst_idxs):
            raise ValueError(
                f"Destination indices must be unique. Found duplicate index: {dst_idxs}"
            )

        for src_idx in src_idxs:
            if not (0 <= src_idx < self.total_num_pages):
                raise ValueError(
                    f"Source index {src_idx} must be between 0 and {self.total_num_pages - 1}"
                )

        for dst_idx in dst_idxs:
            if not (0 <= dst_idx < remote.total_num_pages):
                raise ValueError(
                    f"Destination index {dst_idx} must be between 0 and {remote.total_num_pages - 1}"
                )

        transfer_name = str(uuid4())
        transfer_ids = []

        # Plan (source_shard, dest_shard) pairs. flatten_local collapses the
        # MLA-replicated source to shard 0 (any shard's copy suffices, saves
        # bandwidth); the destination always spans all its TP shards since each
        # owns distinct GPU memory. The source is the local side here.
        local_replica_agents = self.tensor_agents[src_replica_idx]
        remote_replica_agents_meta = remote.agents_meta[dst_replica_idx]
        shard_pairs = transfer_shard_pairing(
            flatten_source=view.flatten_local,
            source_tp=len(local_replica_agents),
            dest_tp=len(remote_replica_agents_meta),
        )
        local_shards_used = [src_shard for src_shard, _ in shard_pairs]

        for src_shard, dst_shard in shard_pairs:
            ta = local_replica_agents[src_shard]
            remote_agent_meta = remote_replica_agents_meta[dst_shard]

            # Build descriptors for each group.
            # Each group uses its own base address and bytes_per_page; all
            # groups share the same logical page indices.
            src_base_addrs = ta.base_addrs
            dst_base_addrs = remote_agent_meta.base_addrs

            descs_src = _build_group_descriptors(
                src_base_addrs, self.bytes_per_group, src_idxs, ta.device_id
            )
            descs_dst = _build_group_descriptors(
                dst_base_addrs,
                self.bytes_per_group,
                dst_idxs,
                remote_agent_meta.device_id,
            )

            transfer_dlist_src = nixl.TransferDescriptorList(
                type=self.memory_type, descs=descs_src
            )
            transfer_dlist_dst = nixl.TransferDescriptorList(
                type=remote.memory_type, descs=descs_dst
            )

            # Use the appropriate agent for this tensor
            remote_agent_name = remote_agent_meta.agent_name

            transfer_id = ta.agent.create_transfer_request(
                operation=nixl.TransferOpType.WRITE,
                local_descs=transfer_dlist_src,
                remote_descs=transfer_dlist_dst,
                remote_agent=remote_agent_name,
                notif_msg=transfer_name,
            )
            status = ta.agent.post_transfer_request(transfer_id)

            if status not in [nixl.Status.SUCCESS, nixl.Status.IN_PROG]:
                raise ValueError(
                    f"Transfer request failed with status {status} for TP shard {src_shard}"
                )

            transfer_ids.append(transfer_id)

        transfer_req = TransferReqData(
            dst_name=remote_metadata.name,
            src_name=self.name,
            transfer_name=transfer_name,
            transfer_ids=transfer_ids,
            src_idxs=src_idxs,
            dst_idxs=dst_idxs,
            src_replica_idx=src_replica_idx,
            dst_replica_idx=dst_replica_idx,
            tp_shard_count=len(transfer_ids),
            local_shards_used=local_shards_used,
        )
        self.inflight_send_transfers[transfer_name] = transfer_req
        return transfer_req

    def initiate_read_transfer(
        self,
        remote_metadata: KVTransferEngineMetadata,
        src_idxs: list[int],
        dst_idxs: list[int],
        src_replica_idx: int,
        dst_replica_idx: int,
    ) -> TransferReqData:
        """Initiate a READ transfer from remote engine to current engine.

        The current engine pulls data from the remote. Used by DKVConnector
        to read KV blocks from BlockStore DRAM into GPU VRAM.

        Args:
            remote_metadata: Metadata for the remote engine (source).
            src_idxs: Page indices in the remote engine (source).
            dst_idxs: Page indices in the current engine (destination).
            src_replica_idx: Replica index in the remote engine.
            dst_replica_idx: Replica index in the current engine.
        """
        if not (0 <= dst_replica_idx < self.dp):
            raise ValueError(
                f"dst_replica_idx {dst_replica_idx} must be between 0 and {self.dp - 1}"
            )

        if not (0 <= src_replica_idx < len(remote_metadata.agents_meta)):
            raise ValueError(
                f"src_replica_idx {src_replica_idx} must be between 0 and {len(remote_metadata.agents_meta) - 1}"
            )

        if remote_metadata.name not in self.remote_connections:
            raise ValueError(
                f"Remote connection {remote_metadata.name} not found"
            )

        remote = self.remote_connections[remote_metadata.name]
        view = self._peer_views[remote_metadata.name]

        if len(src_idxs) != len(dst_idxs):
            raise ValueError(
                f"Source and destination indices must have the same length. Got {len(src_idxs)} and {len(dst_idxs)}"
            )

        for dst_idx in dst_idxs:
            if not (0 <= dst_idx < self.total_num_pages):
                raise ValueError(
                    f"Destination index {dst_idx} must be between 0 and {self.total_num_pages - 1}"
                )

        for src_idx in src_idxs:
            if not (0 <= src_idx < remote.total_num_pages):
                raise ValueError(
                    f"Source index {src_idx} must be between 0 and {remote.total_num_pages - 1}"
                )

        transfer_name = str(uuid4())
        transfer_ids = []

        # Plan (source_shard, dest_shard) pairs. Here the remote is the source
        # (flatten_remote collapses an MLA-replicated remote to shard 0) and the
        # local engine is the destination, always spanning all its TP shards.
        local_replica_agents = self.tensor_agents[dst_replica_idx]
        remote_replica_agents_meta = remote.agents_meta[src_replica_idx]
        shard_pairs = transfer_shard_pairing(
            flatten_source=view.flatten_remote,
            source_tp=len(remote_replica_agents_meta),
            dest_tp=len(local_replica_agents),
        )
        # Local is the destination for a read.
        local_shards_used = [dst_shard for _, dst_shard in shard_pairs]

        # Determine per-group bytes_per_page for the remote (source) engine.
        # If the remote advertises bytes_per_group, use it; otherwise fall back
        # to treating bytes_per_page as a single group.
        remote_bpg = (
            remote.bytes_per_group
            if remote.bytes_per_group
            else [remote.bytes_per_page]
        )

        for remote_shard, dst_shard in shard_pairs:
            ta = local_replica_agents[dst_shard]
            remote_agent_meta = remote_replica_agents_meta[remote_shard]

            # Build descriptors for each group. Local uses this engine's
            # bytes_per_group; remote uses the peer's advertised strides,
            # falling back to local for any group the peer doesn't advertise.
            effective_remote_bpg = [
                remote_bpg[g] if g < len(remote_bpg) else bpp
                for g, bpp in enumerate(self.bytes_per_group)
            ]
            descs_local = _build_group_descriptors(
                ta.base_addrs, self.bytes_per_group, dst_idxs, ta.device_id
            )
            descs_remote = _build_group_descriptors(
                remote_agent_meta.base_addrs,
                effective_remote_bpg,
                src_idxs,
                remote_agent_meta.device_id,
            )

            local_dlist = nixl.TransferDescriptorList(
                type=self.memory_type, descs=descs_local
            )
            remote_dlist = nixl.TransferDescriptorList(
                type=remote.memory_type, descs=descs_remote
            )

            transfer_id = ta.agent.create_transfer_request(
                operation=nixl.TransferOpType.READ,
                local_descs=local_dlist,
                remote_descs=remote_dlist,
                remote_agent=remote_agent_meta.agent_name,
                notif_msg=transfer_name,
            )
            status = ta.agent.post_transfer_request(transfer_id)

            if status not in [nixl.Status.SUCCESS, nixl.Status.IN_PROG]:
                raise ValueError(
                    f"Read transfer request failed with status {status} for TP shard {dst_shard}"
                )

            transfer_ids.append(transfer_id)

        transfer_req = TransferReqData(
            dst_name=self.name,
            src_name=remote_metadata.name,
            transfer_name=transfer_name,
            transfer_ids=transfer_ids,
            src_idxs=src_idxs,
            dst_idxs=dst_idxs,
            src_replica_idx=src_replica_idx,
            dst_replica_idx=dst_replica_idx,
            is_read=True,
            tp_shard_count=len(transfer_ids),
            local_shards_used=local_shards_used,
        )
        self.inflight_read_transfers[transfer_name] = transfer_req
        return transfer_req

    def _is_sender_of(self, transfer_req: TransferReqData) -> bool:
        """Check if the current engine is the sender of a transfer."""
        return transfer_req.src_name == self.name

    def _owns_transfer_request(self, transfer_req: TransferReqData) -> bool:
        """Check if the current engine owns the transfer request handles."""
        if transfer_req.is_read:
            return transfer_req.dst_name == self.name
        return self._is_sender_of(transfer_req)

    def _notification_remote_name(self, transfer_req: TransferReqData) -> str:
        """Return the remote engine name associated with completion notifications."""
        if transfer_req.is_read:
            return transfer_req.dst_name
        return transfer_req.src_name

    def _is_send_complete(self, transfer_req: TransferReqData) -> bool:
        """Check if a send transfer is complete.

        Args:
            transfer_req: The transfer request data containing transfer metadata.

        Returns:
            True if the send transfer is complete, False otherwise.
        """
        assert self._is_sender_of(transfer_req)

        is_complete = True
        src_replica_idx = transfer_req.src_replica_idx
        tp_agents = self._resolve_local_agents_for_transfer(
            src_replica_idx, transfer_req
        )
        for tp_idx, transfer_id in enumerate(transfer_req.transfer_ids):
            agent = tp_agents[tp_idx].agent
            status = agent.get_transfer_status(transfer_id)

            if status == nixl.Status.SUCCESS:
                continue
            elif status == nixl.Status.IN_PROG:
                is_complete = False
                break
            else:
                raise ValueError(
                    f"Transfer request failed with status {status} in source replica {src_replica_idx}"
                )

        return is_complete

    def _is_recv_complete(self, transfer_req: TransferReqData) -> bool:
        """Check if a recv transfer is complete."""
        assert not self._owns_transfer_request(transfer_req)

        # Check what recv completion notifications have been received
        # We only check agents in the replica local to the current engine.
        local_replica_idx = (
            transfer_req.src_replica_idx
            if transfer_req.is_read
            else transfer_req.dst_replica_idx
        )
        tp_agents = self.tensor_agents[local_replica_idx]
        for ta in tp_agents:
            notifs = ta.agent.get_notifs()
            for remote_agent_name, notifications in notifs.items():
                engine_name = self.remote_agent_to_engine[remote_agent_name]
                for notif in notifications:
                    notif_decoded = notif.decode()
                    self.completed_recv_transfers[engine_name][
                        notif_decoded
                    ] += 1

        # A recv is complete when we get expected number of notifications
        transfer_name = transfer_req.transfer_name
        expected = (
            transfer_req.tp_shard_count
            if transfer_req.tp_shard_count > 0
            else self.tp
        )
        remote_name = self._notification_remote_name(transfer_req)
        return (
            self.completed_recv_transfers[remote_name][transfer_name]
            == expected
        )

    def _is_read_complete(self, transfer_req: TransferReqData) -> bool:
        """Check if a read transfer is complete.

        For READ ops the local agent initiates the transfer, so we poll
        get_transfer_status on our own agents (same pattern as send).
        """
        assert transfer_req.is_read
        assert self._owns_transfer_request(transfer_req)

        dst_replica_idx = transfer_req.dst_replica_idx
        tp_agents = self._resolve_local_agents_for_transfer(
            dst_replica_idx, transfer_req
        )

        for tp_idx, transfer_id in enumerate(transfer_req.transfer_ids):
            agent = tp_agents[tp_idx].agent
            status = agent.get_transfer_status(transfer_id)

            if status == nixl.Status.SUCCESS:
                continue
            elif status == nixl.Status.IN_PROG:
                return False
            else:
                raise ValueError(
                    f"Read transfer failed with status {status} in replica {dst_replica_idx}"
                )

        return True

    def is_complete(self, transfer_req: TransferReqData) -> bool:
        """Checks if a given send, recv, or read transfer is completed.

        .. caution::
           This method is prone to infinite loops. For the transfer to progress,
           the remote engine MUST call wait_recv_complete. As such, the following
           code will hang:

           .. code-block:: python

              transfer_req = engine_1.write_to(...)
              while not engine_1.is_complete(transfer_req):
                  pass
              while not engine_2.is_complete(transfer_req):
                  pass

           Instead do:

           .. code-block:: python

              transfer_req = engine_1.write_to(...)
              while not engine_1.is_complete(transfer_req) or not engine_2.is_complete(transfer_req):
                  pass

        Args:
            transfer_req: The transfer request.

        Returns:
            bool: True if all transfers have completed; false otherwise.
        """
        if transfer_req.is_read:
            if self._owns_transfer_request(transfer_req):
                return self._is_read_complete(transfer_req)
            return self._is_recv_complete(transfer_req)
        elif self._is_sender_of(transfer_req):
            return self._is_send_complete(transfer_req)
        else:
            return self._is_recv_complete(transfer_req)

    def _cleanup_recv_transfer(self, transfer_req: TransferReqData) -> None:
        """Cleanup a transfer."""
        assert not self._owns_transfer_request(transfer_req)
        assert transfer_req.transfer_name not in self.inflight_send_transfers

        remote_name = self._notification_remote_name(transfer_req)
        del self.completed_recv_transfers[remote_name][
            transfer_req.transfer_name
        ]

    def _cleanup_send_transfer(self, transfer_req: TransferReqData) -> None:
        """Cleanup a send transfer."""
        assert self._is_sender_of(transfer_req)
        transfer_name = transfer_req.transfer_name
        assert transfer_name in self.inflight_send_transfers

        del self.inflight_send_transfers[transfer_name]

        src_replica_idx = transfer_req.src_replica_idx
        tp_agents = self._resolve_local_agents_for_transfer(
            src_replica_idx, transfer_req
        )
        for tp_idx, transfer_id in enumerate(transfer_req.transfer_ids):
            agent = tp_agents[tp_idx].agent
            status = agent.release_transfer_request(transfer_id)
            if status != nixl.Status.SUCCESS:
                raise ValueError(
                    f"Failed to release transfer request: {status}"
                )

    def _cleanup_read_transfer(self, transfer_req: TransferReqData) -> None:
        """Cleanup a read transfer by releasing transfer requests."""
        assert transfer_req.is_read
        transfer_name = transfer_req.transfer_name
        assert transfer_name in self.inflight_read_transfers

        del self.inflight_read_transfers[transfer_name]

        dst_replica_idx = transfer_req.dst_replica_idx
        tp_agents = self._resolve_local_agents_for_transfer(
            dst_replica_idx, transfer_req
        )
        for tp_idx, transfer_id in enumerate(transfer_req.transfer_ids):
            agent = tp_agents[tp_idx].agent
            status = agent.release_transfer_request(transfer_id)
            if status != nixl.Status.SUCCESS:
                raise ValueError(
                    f"Failed to release read transfer request: {status}"
                )

    def cleanup_transfer(self, transfer_req: TransferReqData) -> None:
        """Cleanup a transfer. This should be called after a transfer is complete.

        Args:
            transfer_req: The transfer request to cleanup.
        """
        if not self.is_complete(transfer_req):
            raise ValueError(
                f"Transfer {transfer_req.transfer_name} is not complete"
            )

        if transfer_req.is_read:
            if self._owns_transfer_request(transfer_req):
                self._cleanup_read_transfer(transfer_req)
            else:
                self._cleanup_recv_transfer(transfer_req)
        elif self._is_sender_of(transfer_req):
            self._cleanup_send_transfer(transfer_req)
        else:
            self._cleanup_recv_transfer(transfer_req)

    def sync_and_release(
        self,
        transfer_req: TransferReqData,
        timeout_s: float = 30.0,
    ) -> None:
        """Waits for a transfer to complete and releases it.

        Args:
            transfer_req: The transfer request to wait on.
            timeout_s: Maximum seconds to wait before raising TimeoutError.

        Raises:
            TimeoutError: If the transfer does not complete within timeout_s.
        """
        deadline = time.monotonic() + timeout_s
        while not self.is_complete(transfer_req):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"NIXL transfer did not complete within {timeout_s}s"
                )
            time.sleep(0.001)
        self.cleanup_transfer(transfer_req)

    def cleanup(self) -> None:
        """Release all resources associated with the transfer engine.

        Should be called before the transfer engine is garbage collected.
        Moving this logic into the __del__ destructor does causes a UCX error for
        unknown reasons.
        """
        # Release all send transfers
        for send_transfer_req in list(self.inflight_send_transfers.values()):
            self._cleanup_send_transfer(send_transfer_req)

        # Release all read transfers
        for read_transfer_req in list(self.inflight_read_transfers.values()):
            self._cleanup_read_transfer(read_transfer_req)

        # Invalidate metadata of other agents. Iterate via the recorded
        # peer view so heterogeneous flatten shapes line up under zip.
        for remote_name in self.remote_connections:
            remote = self.remote_connections[remote_name]
            view = self._peer_views.get(remote_name)
            # Defensive: connect() populates _peer_views and remote_connections
            # together, so this is unreachable today. If they ever desync,
            # recompute (rather than assuming non-flattened) so a flattened
            # peer's teardown still mirrors connect()'s pairing.
            if view is None:
                view = self._compute_peer_view(remote)
            for local_ta, remote_agent_meta in self._iter_peer_agents(
                remote, view
            ):
                status = local_ta.agent.invalidate_remote_metadata(
                    remote_agent_meta.agent_name
                )
                if status != nixl.Status.SUCCESS:
                    raise ValueError(f"Failed to invalidate metadata: {status}")

        # Deregister NIXL memory for all tensors (all replicas, all groups)
        for replica_agents in self.tensor_agents:
            for ta in replica_agents:
                for reg_dlist in ta.reg_dlists:
                    status = ta.agent.deregister_memory(reg_dlist, [ta.backend])
                    if status != nixl.Status.SUCCESS:
                        raise ValueError(
                            f"Failed to deregister memory: {status}"
                        )


class KVTransferEngine(TransferEngine):
    """KVCache Transfer Engine with support for Data Parallelism (DP) and Tensor Parallelism (TP).

    The engine accepts a 2D list of tensors: list[list[Buffer]] where the outer list
    represents DP replicas and the inner list represents TP shards within each replica.

    ``KVTransferEngine`` is a thin layer on top of :class:`TransferEngine`: it
    validates the KV buffer grid, builds the per-shard NIXL groups, and derives
    ``replicate_kv_across_tp`` before delegating all NIXL transport to the base.

    The TransferEngine communicates with other TransferEngines in other threads
    or processes. However, individual TransferEngines themselves are not
    thread-safe. It is intended to be used by MAX's single-threaded scheduler.
    """

    def __init__(
        self,
        name: str,
        tensors: Sequence[Sequence[Buffer]],
        *,
        total_num_pages: int,
        replicate_kv_across_tp: bool = False,
        extra_tensor_groups: Sequence[Sequence[Sequence[Buffer]]] | None = None,
    ) -> None:
        """Initialize the transfer engine.

        Args:
            name: Unique name for this engine.
            tensors: Main group tensors as ``[replica][tp_shard]``.
            total_num_pages: Total KV cache pages per tensor.
            replicate_kv_across_tp: Whether KV is replicated across TP ranks.
            extra_tensor_groups: Additional tensor groups (e.g., draft KV for
                speculative decoding). Each entry has the same ``[replica][tp_shard]``
                structure as ``tensors``. All tensors in each group must have
                the same shape within that group, but groups may differ in shape.
        """
        if total_num_pages <= 0:
            raise ValueError(
                f"Total number of pages {total_num_pages} must be greater than 0"
            )

        # Validate 2D structure
        if not tensors:
            raise ValueError("tensors must contain at least one replica")

        if not all(replica_tensors for replica_tensors in tensors):
            raise ValueError("Each replica must contain at least one tensor")

        # Validate all replicas have same number of TP shards
        dp = len(tensors)
        tp = len(tensors[0])
        for replica_idx, replica_tensors in enumerate(tensors):
            if len(replica_tensors) != tp:
                raise ValueError(
                    f"All replicas must have the same number of tensors. "
                    f"Replica 0 has {tp} tensors, "
                    f"but replica {replica_idx} has {len(replica_tensors)} tensors"
                )

        # Assemble the uniform group grid: all_groups[group_idx][replica_idx]
        # = [shard0, shard1, ...]. The main group is group 0; each extra tensor
        # group follows. From here on every NIXL group is treated uniformly.
        extra_groups: list[Sequence[Sequence[Buffer]]] = (
            list(extra_tensor_groups) if extra_tensor_groups else []
        )
        all_groups: list[list[list[Buffer]]] = [
            [list(replica_tensors) for replica_tensors in tensors]
        ]
        for group_idx, group_tensors in enumerate(extra_groups):
            if len(group_tensors) != dp:
                raise ValueError(
                    f"Extra group {group_idx} must have {dp} replicas, "
                    f"but has {len(group_tensors)}"
                )
            all_groups.append(
                [list(replica_tensors) for replica_tensors in group_tensors]
            )

        num_groups = len(all_groups)
        effective_replicate = replicate_kv_across_tp and tp > 1

        backend_type = _get_nixl_backend_type()

        # Validate every group across replicas and compute per-group bytes/page.
        bytes_per_group: list[int] = []  # [group_idx] → bytes_per_page
        memory_types: list[nixl.MemoryType] = []
        for group_idx, group_replicas in enumerate(all_groups):
            group_bpp_list: list[int] = []
            for replica_idx, replica_shards in enumerate(group_replicas):
                if len(replica_shards) != tp:
                    raise ValueError(
                        f"Group {group_idx} replica {replica_idx} has "
                        f"{len(replica_shards)} TP shards, but expected {tp}. "
                        "All groups and replicas must share the same TP degree."
                    )
                _validate_device_type([t.device for t in replica_shards])
                gbpp, _ = _validate_tensor_shape(
                    replica_shards, total_num_pages
                )
                group_bpp_list.append(gbpp)

                is_cpu = replica_shards[0].device.is_host
                memory_types.append(
                    nixl.MemoryType.DRAM if is_cpu else nixl.MemoryType.VRAM
                )
            if len(set(group_bpp_list)) != 1:
                raise ValueError(
                    f"All replicas must have the same bytes_per_page. "
                    f"Group {group_idx} found: {group_bpp_list}"
                )
            bytes_per_group.append(group_bpp_list[0])

        if len(set(memory_types)) != 1:
            raise ValueError(
                f"All groups/replicas must have the same memory type. "
                f"Found: {set(memory_types)}"
            )

        bytes_per_page = sum(bytes_per_group)
        memory_type = memory_types[0]

        # Create one agent per (replica, shard), registering every group's
        # buffer for that shard uniformly (group-major).
        tensor_agents: list[list[TensorAgent]] = []
        for replica_idx in range(dp):
            replica_agents = []
            for tp_idx in range(tp):
                shard_tensors = [
                    all_groups[g][replica_idx][tp_idx]
                    for g in range(num_groups)
                ]
                tensor_agent = TensorAgent.create_agent(
                    agent_name=f"{name}_{replica_idx}_{tp_idx}",
                    listen_port=available_port(),
                    tensors=shard_tensors,
                    memory_type=memory_type,
                    backend_type=backend_type,
                )
                replica_agents.append(tensor_agent)
            tensor_agents.append(replica_agents)

        super().__init__(
            name=name,
            tensor_agents=tensor_agents,
            total_num_pages=total_num_pages,
            bytes_per_page=bytes_per_page,
            bytes_per_group=bytes_per_group,
            memory_type=memory_type,
            dp=dp,
            tp=tp,
            backend_type=backend_type,
            replicate_kv_across_tp=effective_replicate,
        )

        logger.info(
            "NIXL memory registration complete for %s (%s backend): "
            "%d agent(s) (dp=%d, tp=%d), %d bytes per agent (%d group(s)).",
            self.name,
            backend_type,
            self.dp * self.tp,
            self.dp,
            self.tp,
            self.bytes_per_page * total_num_pages,
            len(self.bytes_per_group),
        )

    @classmethod
    def from_paged_kv_cache(
        cls, name: str, kv_cache: PagedKVCacheManager
    ) -> KVTransferEngine:
        """Construct an engine wired to a ``PagedKVCacheManager``.

        Pulls the per-replica device buffers, sets ``total_num_pages``, and
        derives ``replicate_kv_across_tp`` from the cache params. Equivalent to
        constructing the engine manually but consolidates the boilerplate that
        prefill/decode schedulers share.

        For models with multiple KV caches (e.g., speculative decoding with a
        separate target and draft KV), each child cache is registered as its
        own NIXL group so that heterogeneous buffer shapes (e.g., 61-layer MLA
        target vs. 1-layer Eagle draft) are validated and transferred
        independently.
        """
        from max.nn.kv_cache.cache_params import MultiKVCacheBuffer

        cache_params = kv_cache.params
        dp = cache_params.data_parallel_degree
        total_num_pages = kv_cache.get_num_pages(replica_idx=0) + 1

        device_buffers = [
            kv_cache.get_device_buffer(replica_idx) for replica_idx in range(dp)
        ]

        tensors: list[list[Buffer]] = []
        extra_tensor_groups: list[list[list[Buffer]]] = []
        child_keys: list[str] = []

        # Collect per-replica buffers. MultiKVCacheBuffer replicas are split
        # into per-child NIXL groups so each group is shape-homogeneous.
        # KVCacheBuffer replicas go into a single group as before.
        for r, buf in enumerate(device_buffers):
            if isinstance(buf, MultiKVCacheBuffer):
                if r == 0:
                    child_keys = list(buf.children.keys())
                    extra_tensor_groups = [[] for _ in child_keys[1:]]
                # Main group: this replica's buffers for the first child
                tensors.append(list(buf.children[child_keys[0]].all_buffers))
                # Extra groups: one entry per remaining child
                for g, key in enumerate(child_keys[1:]):
                    extra_tensor_groups[g].append(
                        list(buf.children[key].all_buffers)
                    )
            else:
                # Single-cache replica: flat buffer list
                tensors.append(list(buf.all_buffers))

        return cls(
            name=name,
            tensors=tensors,
            # Need to add 1 for the null block
            total_num_pages=total_num_pages,
            replicate_kv_across_tp=cache_params.replicates_kv_across_tp,
            extra_tensor_groups=extra_tensor_groups
            if extra_tensor_groups
            else None,
        )
