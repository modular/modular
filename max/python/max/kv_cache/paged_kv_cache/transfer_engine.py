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

import itertools
import logging
import os
import random
import socket
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeVar
from uuid import uuid4

import msgspec
from max._core import nixl
from max.driver import Buffer, Device
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams

from .cache_manager import PagedKVCacheManager

logger = logging.getLogger("max.pipelines")

NixlBackendType = Literal["ucx", "libfabric"]

_ShardT = TypeVar("_ShardT")

_NIXL_BACKEND_ENV_VAR = "MODULAR_NIXL_TRANSFER_BACKEND"
_SUPPORTED_BACKENDS: set[NixlBackendType] = {"ucx", "libfabric"}


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
        start_port (int): The lower bound of the port range (inclusive).
        end_port (int): The upper bound of the port range (inclusive).
        max_attempts (int): Maximum number of attempts to find a free port.

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


def _validate_device_type(
    devices: Sequence[Device], backend_type: NixlBackendType
) -> None:
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
    if first_device.api == "hip" and backend_type == "ucx":
        raise NotImplementedError("Currently UCX does not support HIP devices.")

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

    base_addr: int
    """Base memory address for this tensor."""

    device_id: int
    """Device ID for this tensor."""

    extra_groups: dict[str, tuple[int, int]] | None = None
    """Additional tensor groups: maps name → (base_addr, bytes_per_page).

    Present when the agent has registered extra tensor groups (e.g. draft
    KV cache for speculative decoding).
    """


@dataclass
class TensorGroupInfo:
    """Metadata for an additional tensor group registered on a TensorAgent."""

    name: str
    """Group name (e.g. ``"draft"``)."""

    base_addr: int
    """Base memory address for this group's buffer."""

    bytes_per_page: int
    """Bytes per page for this group."""

    reg_dlist: nixl.RegistrationDescriptorList
    """NIXL registration descriptor list for this group's memory."""


@dataclass
class TensorAgent:
    """Manages a single tensor and its associated NIXL agent for transfers.

    This class holds both the runtime state (live objects) and can generate
    the serializable metadata for communication between engines.

    Additional tensor groups (e.g. draft KV cache) can be registered via
    :meth:`register_extra_group`.  Their descriptors are appended to
    transfer requests alongside the primary tensor's descriptors so that
    all groups are bundled into a single NIXL transfer.
    """

    agent: nixl.Agent
    """NIXL agent for this tensor."""

    agent_name: str
    """Name of this agent."""

    tensor: Buffer
    """Tensor for this agent."""

    base_addr: int
    """Base memory address for this tensor."""

    backend: int
    """NIXL backend handle (UCX or libfabric)."""

    device_id: int
    """Device ID for this tensor."""

    agent_metadata: bytes
    """Metadata for this agent."""

    reg_dlist: nixl.RegistrationDescriptorList
    """Registration descriptor list for this tensor."""

    extra_groups: dict[str, TensorGroupInfo] = field(default_factory=dict)
    """Additional registered tensor groups keyed by name."""

    @classmethod
    def create_agent(
        cls,
        agent_name: str,
        listen_port: int,
        tensor: Buffer,
        total_num_pages: int,
        elts_per_page: int,
        memory_type: nixl.MemoryType,
        backend_type: NixlBackendType = "ucx",
    ) -> TensorAgent:
        """Creates and registers a NIXL agent for the given tensor.

        Args:
            agent_name: Unique name for this agent.
            listen_port: TCP port for the NIXL listener.
            tensor: GPU/CPU buffer to register.
            total_num_pages: Total KV cache pages in the tensor.
            elts_per_page: Elements per page.
            memory_type: NIXL memory segment type (DRAM or VRAM).
            backend_type: NIXL transport backend (``"ucx"`` or ``"libfabric"``).
        """
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

        # Reshape tensor to 2D view
        tensor_2d = tensor.view(tensor.dtype, (total_num_pages, elts_per_page))

        # Check backend availability
        available = agent.get_available_plugins()
        if backend_type not in available:
            raise RuntimeError(
                f"NIXL backend {backend_type!r} not available for agent "
                f"{agent_name}. Available plugins: {available}"
            )

        # Configure and create backend
        device = tensor.device
        backend_params = agent.get_plugin_params(backend_type)[0]
        if not device.is_host:
            backend_params["gpu_device_id"] = str(device.id)

        backend = agent.create_backend(
            type=backend_type,
            init_params=backend_params,
        )

        # Register memory
        base_addr = tensor._data_ptr()
        num_bytes = tensor.num_elements * tensor.dtype.size_in_bytes

        descs = [(base_addr, num_bytes, device.id, "")]
        reg_dlist = nixl.RegistrationDescriptorList(
            type=memory_type, descs=descs
        )

        status = agent.register_memory(reg_dlist, [backend])
        if status != nixl.Status.SUCCESS:
            raise ValueError(
                f"Failed to register memory for {agent_name}: {status}"
            )

        # Get metadata after registration
        agent_metadata = agent.get_local_metadata()

        # Create TensorAgent and add to list
        return TensorAgent(
            agent=agent,
            agent_name=agent_name,
            tensor=tensor_2d,
            base_addr=base_addr,
            backend=backend,
            device_id=device.id,
            agent_metadata=agent_metadata,
            reg_dlist=reg_dlist,
        )

    def register_extra_group(
        self,
        name: str,
        tensor: Buffer,
        total_num_pages: int,
        memory_type: nixl.MemoryType,
    ) -> None:
        """Register an additional tensor group on this agent.

        The tensor is registered as a new memory region on the existing
        NIXL agent.  Transfer descriptors for this group will be appended
        to future transfer requests alongside the primary tensor's
        descriptors.

        Args:
            name: Group name (e.g. ``"draft"``).
            tensor: Buffer to register.
            total_num_pages: Number of pages in the buffer.
            memory_type: NIXL memory type (DRAM or VRAM).
        """
        if name in self.extra_groups:
            raise ValueError(
                f"Extra group '{name}' already registered on agent "
                f"{self.agent_name}"
            )

        base_addr = tensor._data_ptr()
        num_bytes = tensor.num_elements * tensor.dtype.size_in_bytes
        bytes_per_page = num_bytes // total_num_pages

        descs = [(base_addr, num_bytes, self.device_id, "")]
        reg_dlist = nixl.RegistrationDescriptorList(
            type=memory_type, descs=descs
        )

        status = self.agent.register_memory(reg_dlist, [self.backend])
        if status != nixl.Status.SUCCESS:
            raise ValueError(
                f"Failed to register extra group '{name}' memory for "
                f"{self.agent_name}: {status}"
            )

        # Re-fetch metadata after new registration so remote agents
        # can see the newly registered memory region.
        self.agent_metadata = self.agent.get_local_metadata()

        self.extra_groups[name] = TensorGroupInfo(
            name=name,
            base_addr=base_addr,
            bytes_per_page=bytes_per_page,
            reg_dlist=reg_dlist,
        )

    def to_metadata(self) -> TensorAgentMetadata:
        """Convert to serializable metadata for communication."""
        extra_groups_meta: dict[str, tuple[int, int]] | None = None
        if self.extra_groups:
            extra_groups_meta = {
                name: (info.base_addr, info.bytes_per_page)
                for name, info in self.extra_groups.items()
            }
        return TensorAgentMetadata(
            agent_name=self.agent_name,
            metadata=self.agent_metadata,
            base_addr=self.base_addr,
            device_id=self.device_id,
            extra_groups=extra_groups_meta,
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
        f"has replicate_kv_across_tp=True (MLA) with TP>1 and its "
        f"DP*TP matches the other side's DP."
    )


class KVTransferEngineMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Metadata associated with a transfer engine.

    This is safe to send between threads/processes.
    """

    name: str
    """Base name of the transfer engine."""

    total_num_pages: int
    """Total number of pages in each tensor."""

    bytes_per_page: int
    """Bytes per page for each tensor."""

    memory_type: nixl.MemoryType
    """Memory type of the transfer engine."""

    hostname: str
    """Hostname of the machine that the transfer engine is running on."""

    agents_meta: list[list[TensorAgentMetadata]]
    """Metadata for each replica's agents: [replica][tp_shard]."""

    replicate_kv_across_tp: bool = False
    """True iff KV buffers are identical across TP ranks (e.g. MLA with
    num_kv_heads=1). When both sides declare different (dp, tp) but one
    replicates, the engine can reinterpret the replicating side as
    ``[dp*tp][1]`` to let a prefill worker at (DP=m, TP=n) connect to a
    decode worker at (DP=m*n, TP=1)."""


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


class KVTransferEngine:
    """KVCache Transfer Engine with support for Data Parallelism (DP) and Tensor Parallelism (TP).

    The engine accepts a 2D list of tensors: list[list[Buffer]] where the outer list
    represents DP replicas and the inner list represents TP shards within each replica.

    The TransferEngine communicates with other TransferEngines in other threads
    or processes. However, individual TransferEngines themselves are not
    thread-safe. It is intended to be used by MAX's single-threaded scheduler.
    """

    name: str
    """Name of transfer engine / nixl agent."""

    tensor_agents: list[list[TensorAgent]]
    """2D list of TensorAgent objects: [replica][tp_shard]."""

    total_num_pages: int
    """Total number of pages in each tensor (same across all replicas)."""

    bytes_per_page: int
    """Bytes per page for each tensor."""

    memory_type: nixl.MemoryType
    """Type of memory being managed (e.g. DRAM)."""

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

    replicate_kv_across_tp: bool
    """Whether KV is replicated across TP ranks (MLA)."""

    def __init__(
        self,
        name: str,
        tensors: Sequence[Sequence[Buffer]],
        *,
        total_num_pages: int,
        replicate_kv_across_tp: bool = False,
    ) -> None:
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
        self.tp = len(tensors[0])
        for replica_idx, replica_tensors in enumerate(tensors):
            if len(replica_tensors) != self.tp:
                raise ValueError(
                    f"All replicas must have the same number of tensors. "
                    f"Replica 0 has {self.tp} tensors, "
                    f"but replica {replica_idx} has {len(replica_tensors)} tensors"
                )

        self.dp = len(tensors)
        self.replicate_kv_across_tp = replicate_kv_across_tp and self.tp > 1

        backend_type = _get_nixl_backend_type()

        # Validate each replica independently
        bytes_per_page_list = []
        elts_per_page_list = []
        memory_types = []

        for replica_tensors in tensors:
            _validate_device_type(
                [t.device for t in replica_tensors], backend_type
            )
            bytes_per_page, elts_per_page = _validate_tensor_shape(
                replica_tensors, total_num_pages
            )
            bytes_per_page_list.append(bytes_per_page)
            elts_per_page_list.append(elts_per_page)

            is_cpu = replica_tensors[0].device.is_host
            memory_type = (
                nixl.MemoryType.DRAM if is_cpu else nixl.MemoryType.VRAM
            )
            memory_types.append(memory_type)

        # Validate all replicas have same bytes_per_page and memory_type
        if len(set(bytes_per_page_list)) != 1:
            raise ValueError(
                f"All replicas must have the same bytes_per_page. "
                f"Found: {bytes_per_page_list}"
            )

        if len(set(memory_types)) != 1:
            raise ValueError(
                f"All replicas must have the same memory type. "
                f"Found: {memory_types}"
            )

        # Set memory type and total pages
        self.total_num_pages = total_num_pages
        self.bytes_per_page = bytes_per_page_list[0]
        self.memory_type = memory_types[0]
        elts_per_page = elts_per_page_list[0]

        # Create agents for each tensor in 2D structure
        self.name = name
        self.tensor_agents = []
        for replica_idx, replica_tensors in enumerate(tensors):
            replica_agents = []
            for tp_idx, tensor in enumerate(replica_tensors):
                tensor_agent = TensorAgent.create_agent(
                    agent_name=f"{name}_{replica_idx}_{tp_idx}",
                    listen_port=available_port(),
                    tensor=tensor,
                    total_num_pages=total_num_pages,
                    elts_per_page=elts_per_page,
                    memory_type=self.memory_type,
                    backend_type=backend_type,
                )
                replica_agents.append(tensor_agent)
            self.tensor_agents.append(replica_agents)

        logger.info(
            "NIXL memory registration complete for %s (%s backend): "
            "%d agent(s) (dp=%d, tp=%d), %d bytes per agent.",
            self.name,
            backend_type,
            self.dp * self.tp,
            self.dp,
            self.tp,
            self.bytes_per_page * total_num_pages,
        )

        # Remote connections
        self.remote_connections: dict[str, KVTransferEngineMetadata] = {}

        # Per-peer routing view populated at connect().
        self._peer_views: dict[str, _PeerView] = {}

        # Map of agents to completed transfers
        self.completed_recv_transfers = defaultdict(lambda: defaultdict(int))

        # Map of remote agent names to their engine names
        self.remote_agent_to_engine = {}

        # All send transfers - maps transfer_name to list of (tensor_idx, transfer_id) tuples
        self.inflight_send_transfers = {}

        # All read transfers - maps transfer_name to TransferReqData
        self.inflight_read_transfers: dict[str, TransferReqData] = {}

    @classmethod
    def from_paged_kv_cache(
        cls, name: str, kv_cache: PagedKVCacheManager
    ) -> KVTransferEngine:
        """Construct an engine wired to a ``PagedKVCacheManager``.

        Pulls the per-replica device buffers, sets ``total_num_pages``, and
        derives ``replicate_kv_across_tp`` from ``is_mla`` on the primary
        cache params. Equivalent to constructing the engine manually but
        consolidates the boilerplate that prefill/decode schedulers share.
        """
        cache_params = kv_cache.params
        if isinstance(cache_params, MultiKVCacheParams):
            primary_params = cache_params.params[0]
        else:
            assert isinstance(cache_params, KVCacheParams)
            primary_params = cache_params
        dp = primary_params.data_parallel_degree
        # TODO: Also support scales tensors.
        tensors: list[list[Buffer]] = [
            list(kv_cache.get_device_buffer(replica_idx).values)
            for replica_idx in range(dp)
        ]
        return cls(
            name=name,
            tensors=tensors,
            total_num_pages=kv_cache.get_num_pages(replica_idx=0),
            replicate_kv_across_tp=primary_params.is_mla,
        )

    def register_tensor_group(
        self,
        name: str,
        tensors: Sequence[Sequence[Buffer]],
        total_num_pages: int,
    ) -> None:
        """Register an additional tensor group on all agents.

        The new buffers are registered as extra memory regions on the
        existing NIXL agents.  Future ``initiate_send_transfer`` calls
        will automatically include descriptors for this group alongside
        the primary tensor, bundling both into a single NIXL transfer.

        Args:
            name: Group name (e.g. ``"draft"``).
            tensors: 2D buffer grid ``[replica][tp_shard]`` matching the
                primary tensor layout.
            total_num_pages: Number of pages in each buffer (same page
                count as the primary tensor — page *size* may differ).
        """
        if len(tensors) != self.dp:
            raise ValueError(
                f"Extra group '{name}' has {len(tensors)} replicas, "
                f"expected {self.dp}"
            )
        for replica_idx, replica_tensors in enumerate(tensors):
            if len(replica_tensors) != self.tp:
                raise ValueError(
                    f"Extra group '{name}' replica {replica_idx} has "
                    f"{len(replica_tensors)} TP shards, expected {self.tp}"
                )

        for replica_idx, replica_tensors in enumerate(tensors):
            for tp_idx, tensor in enumerate(replica_tensors):
                self.tensor_agents[replica_idx][tp_idx].register_extra_group(
                    name=name,
                    tensor=tensor,
                    total_num_pages=total_num_pages,
                    memory_type=self.memory_type,
                )

        logger.info(
            "Registered extra tensor group '%s' on %s: %d agent(s).",
            name,
            self.name,
            self.dp * self.tp,
        )

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

    def _pick_transfer_shards(
        self,
        replica_agents: Sequence[_ShardT],
        flatten: bool,
        tp_shard_limit: int | None,
    ) -> list[_ShardT]:
        """Select which TP shards of a single replica participate in a transfer.

        Under ``flatten``, MLA KV is replicated across TP so shard 0 carries
        the full payload. Otherwise honor ``tp_shard_limit`` if set.
        """
        if flatten:
            return [replica_agents[0]]
        agents = list(replica_agents)
        if tp_shard_limit is not None:
            agents = agents[:tp_shard_limit]
        return agents

    def _effective_local_agents(self, flatten: bool) -> list[list[TensorAgent]]:
        """Return ``tensor_agents`` viewed as ``[effective_dp][effective_tp]`` for a peer.

        When ``flatten`` is True, the natural ``[dp][tp]`` is reinterpreted
        as ``[dp*tp][1]`` — each TP shard becomes its own single-shard
        replica. Otherwise returns the natural layout unchanged.
        """
        if flatten:
            return [
                [self.tensor_agents[r][s]]
                for r in range(self.dp)
                for s in range(self.tp)
            ]
        return [list(replica) for replica in self.tensor_agents]

    def _effective_remote_meta(
        self, remote: KVTransferEngineMetadata, flatten: bool
    ) -> list[list[TensorAgentMetadata]]:
        """Mirror of ``_effective_local_agents`` for a remote peer."""
        if flatten:
            return [
                [agent_meta]
                for replica_agents in remote.agents_meta
                for agent_meta in replica_agents
            ]
        return [list(replica) for replica in remote.agents_meta]

    def _effective_agents_for_peer(
        self,
        remote: KVTransferEngineMetadata,
        view: _PeerView | None,
    ) -> tuple[list[list[TensorAgent]], list[list[TensorAgentMetadata]]]:
        """Return the (local, remote) agent grids to iterate against a peer.

        Applies the peer view's flatten flags to align heterogeneous shapes;
        falls back to the natural ``[dp][tp]`` layout when ``view`` is None.
        """
        flatten_local = view.flatten_local if view is not None else False
        flatten_remote = view.flatten_remote if view is not None else False
        return (
            self._effective_local_agents(flatten_local),
            self._effective_remote_meta(remote, flatten_remote),
        )

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

        # Connect pairwise in the effective view, flattening [dp][tp] to
        # [dp*tp][1] on whichever side the peer view calls for.
        local_effective, remote_effective = self._effective_agents_for_peer(
            remote, view
        )
        assert (
            len(local_effective) == len(remote_effective) == view.effective_dp
        )
        for local_agents, remote_agents_meta in itertools.product(
            local_effective, remote_effective
        ):
            # Connect each TP shard within the replica
            for local_ta, remote_agent_meta in zip(
                local_agents,
                remote_agents_meta,
                strict=True,
            ):
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

        # Teardown must mirror the connect() iteration.
        local_eff, remote_eff = self._effective_agents_for_peer(remote, view)

        for local_agents, remote_agents_meta in itertools.product(
            local_eff, remote_eff
        ):
            for local_ta, remote_agent_meta in zip(
                local_agents, remote_agents_meta, strict=True
            ):
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
        tp_shard_limit: int | None = None,
    ) -> TransferReqData:
        """Initiate a transfer from current engine to remote engine.

        The same page indices are broadcast to all TP shards within the source and destination replicas.

        Args:
            remote_metadata: Metadata for the remote engine.
            src_idxs: List of indices of the source pages in the current engine.
            dst_idxs: List of indices of the destination pages in the remote engine.
            src_replica_idx: Index of the source replica to transfer from.
            dst_replica_idx: Index of the destination replica to transfer to.
            tp_shard_limit: Maximum number of TP shards to transfer. When set,
                only the first ``tp_shard_limit`` shards participate in the
                transfer. Useful for MLA models where KV data is identical
                across shards.
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

        # Source: pick which physical shard(s) source the bytes.
        # flatten_local picks shard 0 (MLA-replicated source saves bandwidth).
        # TODO(SERVOPT-1337): rotate shards to spread NIC/PCIe load.
        src_agents = self._pick_transfer_shards(
            self.tensor_agents[src_replica_idx],
            view.flatten_local,
            tp_shard_limit,
        )
        # Destination: always write to all TP shards on the chosen replica.
        # Each remote TP shard owns its own GPU memory and must receive a
        # copy. flatten_remote affects connect-time pairing only.
        remote_replica_agents_meta = list(remote.agents_meta[dst_replica_idx])
        if tp_shard_limit is not None:
            remote_replica_agents_meta = remote_replica_agents_meta[
                :tp_shard_limit
            ]
        # Fan out when src is one shard but dst has many (DP-prefill →
        # TP-decode): repeat the src so the loop pairs shard 0 with each
        # remote shard. All N transfers originate on the same source GPU.
        if len(src_agents) == 1 and len(remote_replica_agents_meta) > 1:
            src_agents = src_agents * len(remote_replica_agents_meta)
            local_shards_used = [0] * len(remote_replica_agents_meta)
        else:
            local_shards_used = list(range(len(src_agents)))

        for tp_idx, ta in enumerate(src_agents):
            remote_agent_meta = remote_replica_agents_meta[tp_idx]

            # Prepare source descriptor list (primary tensor)
            descs_src: list[tuple[int, int, int]] = []
            for src_idx in src_idxs:
                src_addr = ta.base_addr + src_idx * self.bytes_per_page
                descs_src.append((src_addr, self.bytes_per_page, ta.device_id))

            # Prepare destination descriptor list (primary tensor)
            descs_dst: list[tuple[int, int, int]] = []
            for dst_idx in dst_idxs:
                dst_addr = (
                    remote_agent_meta.base_addr + dst_idx * self.bytes_per_page
                )
                descs_dst.append(
                    (dst_addr, self.bytes_per_page, remote_agent_meta.device_id)
                )

            # Append descriptors for extra tensor groups (e.g. draft KV).
            # Both sides must have the same groups registered.
            remote_extra = remote_agent_meta.extra_groups or {}
            for group_name, group_info in ta.extra_groups.items():
                if group_name not in remote_extra:
                    raise ValueError(
                        f"Extra group '{group_name}' registered locally but "
                        f"not on remote agent {remote_agent_meta.agent_name}"
                    )
                remote_base, remote_bpp = remote_extra[group_name]
                local_bpp = group_info.bytes_per_page
                for src_idx in src_idxs:
                    src_addr = group_info.base_addr + src_idx * local_bpp
                    descs_src.append((src_addr, local_bpp, ta.device_id))
                for dst_idx in dst_idxs:
                    dst_addr = remote_base + dst_idx * remote_bpp
                    descs_dst.append(
                        (dst_addr, remote_bpp, remote_agent_meta.device_id)
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
                    f"Transfer request failed with status {status} for TP shard {tp_idx}"
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
        tp_shard_limit: int | None = None,
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
            tp_shard_limit: If set, only the first N TP shards transfer.
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

        # Local (destination): always use all TP shards on the chosen
        # replica. Each shard owns its own GPU memory and must land the
        # incoming bytes. flatten_local affects connect-time pairing only.
        dst_agents = list(self.tensor_agents[dst_replica_idx])
        if tp_shard_limit is not None:
            dst_agents = dst_agents[:tp_shard_limit]
        # Remote (source): flatten_remote picks shard 0 when the source is
        # MLA-replicated (any shard's copy works, saves bandwidth).
        # TODO(SERVOPT-1337): rotate shards to spread NIC/PCIe load.
        remote_replica_agents_meta = self._pick_transfer_shards(
            remote.agents_meta[src_replica_idx],
            view.flatten_remote,
            tp_shard_limit,
        )
        # Fan out when remote-source is one shard but local-dest has many
        # (DP-source → TP-dest read): repeat the remote source so each
        # local shard pulls a copy from the same remote GPU.
        if len(remote_replica_agents_meta) == 1 and len(dst_agents) > 1:
            remote_replica_agents_meta = remote_replica_agents_meta * len(
                dst_agents
            )
        local_shards_used = list(range(len(dst_agents)))

        for tp_idx, ta in enumerate(dst_agents):
            remote_agent_meta = remote_replica_agents_meta[tp_idx]

            # Local descriptors (destination: our GPU memory)
            descs_local: list[tuple[int, int, int]] = []
            for dst_idx in dst_idxs:
                local_addr = ta.base_addr + dst_idx * self.bytes_per_page
                descs_local.append(
                    (local_addr, self.bytes_per_page, ta.device_id)
                )

            # Remote descriptors (source: BlockStore DRAM)
            descs_remote: list[tuple[int, int, int]] = []
            for src_idx in src_idxs:
                remote_addr = (
                    remote_agent_meta.base_addr
                    + src_idx * remote.bytes_per_page
                )
                descs_remote.append(
                    (
                        remote_addr,
                        remote.bytes_per_page,
                        remote_agent_meta.device_id,
                    )
                )

            # Append descriptors for extra tensor groups.
            remote_extra = remote_agent_meta.extra_groups or {}
            for group_name, group_info in ta.extra_groups.items():
                if group_name not in remote_extra:
                    raise ValueError(
                        f"Extra group '{group_name}' registered locally but "
                        f"not on remote agent {remote_agent_meta.agent_name}"
                    )
                remote_base, remote_bpp = remote_extra[group_name]
                local_bpp = group_info.bytes_per_page
                for dst_idx in dst_idxs:
                    local_addr = group_info.base_addr + dst_idx * local_bpp
                    descs_local.append((local_addr, local_bpp, ta.device_id))
                for src_idx in src_idxs:
                    remote_addr = remote_base + src_idx * remote_bpp
                    descs_remote.append(
                        (remote_addr, remote_bpp, remote_agent_meta.device_id)
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
                    f"Read transfer request failed with status {status} for TP shard {tp_idx}"
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
            local_eff, remote_eff = self._effective_agents_for_peer(
                remote, self._peer_views.get(remote_name)
            )
            for local_agents, remote_agents_meta in itertools.product(
                local_eff, remote_eff
            ):
                for local_ta, remote_agent_meta in zip(
                    local_agents, remote_agents_meta, strict=True
                ):
                    status = local_ta.agent.invalidate_remote_metadata(
                        remote_agent_meta.agent_name
                    )
                    if status != nixl.Status.SUCCESS:
                        raise ValueError(
                            f"Failed to invalidate metadata: {status}"
                        )

        # Deregister NIXL memory for all tensors (all replicas)
        for replica_agents in self.tensor_agents:
            for ta in replica_agents:
                # Deregister extra groups first
                for group_info in ta.extra_groups.values():
                    status = ta.agent.deregister_memory(
                        group_info.reg_dlist, [ta.backend]
                    )
                    if status != nixl.Status.SUCCESS:
                        raise ValueError(
                            f"Failed to deregister extra group "
                            f"'{group_info.name}' memory: {status}"
                        )
                # Deregister primary tensor
                status = ta.agent.deregister_memory(ta.reg_dlist, [ta.backend])
                if status != nixl.Status.SUCCESS:
                    raise ValueError(f"Failed to deregister memory: {status}")
