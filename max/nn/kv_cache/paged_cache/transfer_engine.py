# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""KVCache Transfer Engine"""

from __future__ import annotations

import logging
import os
import random
import socket
import time
from collections import defaultdict
from uuid import uuid4

import msgspec
from max._core import nixl
from max.driver import Accelerator
from max.driver.tensor import Tensor

logger = logging.getLogger("max.pipelines")


def available_port(
    start_port: int = 8000, end_port: int = 9000, max_attempts: int = 100
) -> int:
    """
    Find an available TCP port in the given range.

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


class TensorAgentMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Metadata for a single tensor/agent in the transfer engine.

    This is used for serialization and communication between engines.
    """

    metadata: bytes  # Metadata for this agent
    agent_name: str  # Agent name for this tensor
    bytes_per_page: int  # Bytes per page for this tensor
    base_addr: int  # Base memory address for this tensor
    device_id: int  # Device ID for this tensor


class TensorAgent:
    """Manages a single tensor and its associated NIXL agent for transfers.

    This class holds both the runtime state (live objects) and can generate
    the serializable metadata for communication between engines.
    """

    def __init__(
        self,
        agent: nixl.Agent,
        agent_name: str,
        tensor: Tensor,
        base_addr: int,
        ucx_backend: int,
        bytes_per_page: int,
        device_id: int,
        agent_metadata: bytes,
        reg_dlist: nixl.RegistrationDescriptorList,
    ):
        self.agent = agent
        self.agent_name = agent_name
        self.tensor = tensor
        self.base_addr = base_addr
        self.ucx_backend = ucx_backend
        self.bytes_per_page = bytes_per_page
        self.device_id = device_id
        self.agent_metadata = agent_metadata
        self.reg_dlist = reg_dlist

    def to_metadata(self) -> TensorAgentMetadata:
        """Convert to serializable metadata for communication."""
        return TensorAgentMetadata(
            metadata=self.agent_metadata,
            agent_name=self.agent_name,
            bytes_per_page=self.bytes_per_page,
            base_addr=self.base_addr,
            device_id=self.device_id,
        )


class KVTransferEngineMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Metadata associated with a transfer engine.

    This is safe to send between threads/processes."""

    name: str  # Base name of the transfer engine
    total_num_pages: int
    memory_type: nixl.MemoryType
    agents_meta: list[TensorAgentMetadata]  # Metadata for each tensor/agent


class XferReqData(msgspec.Struct, tag=True, kw_only=True, omit_defaults=True):
    """Metadata associated with a transfer request.

    This is safe to send between threads/processes."""

    dst_name: str  # Base name of destination engine
    src_name: str  # Base name of source engine
    dst_agent_names: list[str]  # Agent names for destination (one per tensor)
    src_agent_names: list[str]  # Agent names for source (one per tensor)
    xfer_name: str  # Transfer name
    xfer_ids: list[int]  # Transfer IDs (one per tensor)
    src_idxs: list[
        int
    ]  # Length of source indices can differ from len(xfer_ids)
    dst_idxs: list[
        int
    ]  # Length of destination indices can differ from len(xfer_ids)


class KVTransferEngine:
    """KVCache Transfer Engine.

    Currently this is only tested on CPU and supports single or multiple Paged tensors.

    The TransferEngine communicates with other TransferEngines in other threads
    or processes. However, individual TransferEngines themselves are not
    thread-safe. It is intended to be used by MAX's single-threaded scheduler.
    """

    name: str
    """Name of transfer engine / nixl agent."""

    tensor_agents: list[TensorAgent]
    """List of TensorAgent objects containing all per-tensor data."""

    total_num_pages: int
    """Total number of pages in each tensor."""

    memory_type: nixl.MemoryType
    """Type of memory being managed (e.g. DRAM)."""

    remote_connections: dict[str, KVTransferEngineMetadata]
    """Map of remote engine names to their metadata."""

    completed_xfers: dict[str, set[str]]
    """Map of agent names to completed transfers."""

    remote_agent_to_engine: dict[str, str]
    """Map of remote agent names to their engine names."""

    def __init__(
        self,
        name: str,
        tensors: Tensor | list[Tensor],
        *,
        total_num_pages: int,
        listen_port: int,
    ) -> None:
        if total_num_pages <= 0:
            raise ValueError(
                f"Total number of pages {total_num_pages} must be greater than 0"
            )

        tensors = [tensors] if isinstance(tensors, Tensor) else tensors
        self.num_tensors = len(tensors)
        is_gpu = False
        is_cpu = False
        for t in tensors:
            if t.device.is_host:
                is_cpu = True
            else:
                is_gpu = True

            if is_cpu and is_gpu:
                raise ValueError(
                    "Mixed device tensors detected. All tensors must be either on CPU or GPU, not both."
                )

        if is_gpu:
            if len(tensors) != len(set(t.device.id for t in tensors)):
                raise ValueError("All tensors must be on different GPUs.")

        if is_cpu:
            if len(tensors) != 1:
                raise ValueError(
                    "CPU transfer engine must have exactly one tensor."
                )

        # Validate all tensors have the same shape
        if self.num_tensors > 1:
            first_shape = tensors[0].num_elements
            for i, tensor in enumerate(tensors[1:], 1):
                if tensor.num_elements != first_shape:
                    raise ValueError(
                        f"All tensors must have the same shape. Tensor 0 has {first_shape} elements, but tensor {i} has {tensor.num_elements} elements"
                    )

        for i, tensor in enumerate(tensors):
            if tensor.num_elements % total_num_pages != 0:
                raise ValueError(
                    f"Tensor {i} num elements {tensor.num_elements} must be divisible by total number of pages {total_num_pages}"
                )

        # Regardless of whether the tensor is on CPU / GPU, we must ensure that
        # CUDADriver.cpp is called which loads the libcuda.so.1 and libnvidia-ml.so.1
        # symbols PRIOR to loading the UCX CUDA backend.
        acc = Accelerator()
        if acc.api != "cuda":
            raise NotImplementedError(
                "Currently UCX only supports CUDA devices."
            )
        # This device is unused. It is created for the sole purpose of loading
        # the CUDA symbols.
        del acc

        # Initialize the transfer engine
        self.name = name
        self.tensor_agents = []

        # Set memory type and total pages
        self.total_num_pages = total_num_pages
        self.memory_type = (
            nixl.MemoryType.DRAM if is_cpu else nixl.MemoryType.VRAM
        )

        # Create agents and process each tensor
        use_prog_thread = (
            self.num_tensors > 1
        )  # in the multi tensor case, we need to use a progress thread to achieve overlapping transfers

        for i, tensor in enumerate(tensors):
            # Create agent name
            agent_name = f"{name}_{i}" if self.num_tensors > 1 else name

            # Create NIXL agent
            agent = nixl.Agent(
                agent_name,
                nixl.AgentConfig(
                    use_prog_thread=use_prog_thread,
                    use_listen_thread=True,
                    listen_port=listen_port + i,
                ),
            )

            # Calculate bytes per page
            bytes_per_page = (
                tensor.num_elements
                * tensor.dtype.size_in_bytes
                // total_num_pages
            )
            elts_per_page = tensor.num_elements // total_num_pages

            # Reshape tensor to 2D view
            tensor_2d = tensor.view(
                tensor.dtype, (self.total_num_pages, elts_per_page)
            )

            # Check UCX availability
            if "ucx" not in agent.get_available_plugins():
                raise RuntimeError(
                    f"UCX not currently available for agent {agent_name}, please ensure it is supported by your system."
                )

            # Configure UCX backend
            device = tensor.device
            ucx_params = agent.get_plugin_params("ucx")[0]
            if not device.is_host:
                ucx_params["gpu_device_id"] = str(device.id)

            # Create UCX backend
            ucx_backend = agent.create_backend(
                type="ucx",
                init_params=ucx_params,
            )

            if not device.is_host and (
                "MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE_PERCENT"
                not in os.environ
                and "BAZEL_TEST" not in os.environ
            ):
                # See GEX-2445 for more details.
                # We intentionally make falling back to the slower CUDA_COPY transport
                # a hard error. This check is best effort. Just because it is not
                # tripped does not guarantee that the we will end up using CUDA_IPC.
                # Note that we will use BufferCache regardless when running under
                # bazel test.
                raise ValueError(
                    "MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE_PERCENT must be set when using TransferEngine with GPU memory. "
                    "This flag enables the BufferCache which is required for the fast CUDA_IPC transport. "
                    "Try rerunning your command with MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE_PERCENT=99"
                )

            # Register memory
            base_addr = tensor._data_ptr()
            num_bytes = tensor.num_elements * tensor.dtype.size_in_bytes

            reg_dlist = nixl.RegistrationDescriptorList(
                type=self.memory_type,
                descs=[
                    (
                        base_addr,
                        num_bytes,
                        device.id,
                        "",
                    )
                ],
                sorted=True,
            )

            status = agent.register_memory(reg_dlist, [ucx_backend])
            if status != nixl.Status.SUCCESS:
                raise ValueError(
                    f"Failed to register memory for tensor {i}: {status}"
                )

            # Get metadata after registration
            agent_metadata = agent.get_local_metadata()

            # Create TensorAgent and add to list
            tensor_agent = TensorAgent(
                agent=agent,
                agent_name=agent_name,
                tensor=tensor_2d,
                base_addr=base_addr,
                ucx_backend=ucx_backend,
                bytes_per_page=bytes_per_page,
                device_id=device.id,
                agent_metadata=agent_metadata,
                reg_dlist=reg_dlist,
            )
            self.tensor_agents.append(tensor_agent)

        # Remote connections
        self.remote_connections: dict[str, KVTransferEngineMetadata] = {}

        # Map of agents to completed transfers
        self.completed_xfers: dict[str, set[str]] = defaultdict(set)

        # Map of remote agent names to their engine names
        self.remote_agent_to_engine: dict[str, str] = {}

        # All xfers - maps xfer_name to list of (tensor_idx, xfer_id) tuples
        self.inflight_xfers: dict[str, list[tuple[int, int]]] = {}

    @property
    def metadata(self) -> KVTransferEngineMetadata:
        """Get metadata for the current engine.

        Returns:
            Metadata for the current engine.
        """
        agents_meta = [ta.to_metadata() for ta in self.tensor_agents]

        return KVTransferEngineMetadata(
            name=self.name,
            total_num_pages=self.total_num_pages,
            memory_type=self.memory_type,
            agents_meta=agents_meta,
        )

    def connect(self, remote: KVTransferEngineMetadata) -> None:
        """Connect to a remote engine.

        Args:
            remote: Metadata for the remote engine.
        """
        if remote.name in self.remote_connections:
            raise ValueError(f"Agent {remote.name} already connected")

        if self.num_tensors != len(remote.agents_meta):
            raise ValueError(
                f"Number of tensors mismatch: {self.num_tensors} != {len(remote.agents_meta)}"
            )

        for i, (local_ta, remote_agent_meta) in enumerate(
            zip(self.tensor_agents, remote.agents_meta)
        ):
            if local_ta.bytes_per_page != remote_agent_meta.bytes_per_page:
                raise ValueError(
                    f"Bytes per page mismatch for tensor {i}: {local_ta.bytes_per_page} != {remote_agent_meta.bytes_per_page}"
                )

            loaded_bytes = local_ta.agent.load_remote_metadata(
                remote_agent_meta.metadata
            )
            try:
                loaded_remote_name = loaded_bytes.decode()
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"Metadata loading failed for agent {i}. Expected string, found {loaded_bytes!r}"
                ) from e
            if loaded_remote_name != remote_agent_meta.agent_name:
                raise ValueError(
                    f"Metadata loading failed for agent {i}. Expected {remote_agent_meta.agent_name}, got {loaded_remote_name}"
                )

        self.remote_connections[remote.name] = remote

        # Update the remote agent to engine mapping
        for agent_meta in remote.agents_meta:
            self.remote_agent_to_engine[agent_meta.agent_name] = remote.name

    def initiate_send_xfer(
        self,
        remote_metadata: KVTransferEngineMetadata,
        src_idxs: list[int],
        dst_idxs: list[int],
    ) -> XferReqData:
        """Initiate a transfer from current engine to remote engine for all tensors.

        Args:
            remote_metadata: Metadata for the remote engine.
            src_idxs: List of indices of the source pages in the current engine.
            dst_idxs: List of indices of the destination pages in the remote engine.
        """

        if remote_metadata.name not in self.remote_connections:
            raise ValueError(
                f"Remote connection {remote_metadata.name} not found"
            )

        remote = self.remote_connections[remote_metadata.name]

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

        # Create transfers for all tensors
        xfer_name = str(uuid4())
        self.inflight_xfers[xfer_name] = list()
        xfer_ids = []

        for tensor_idx, ta in enumerate(self.tensor_agents):
            # Prepare source descriptor list
            descs_src: list[tuple[int, int, int]] = []
            for src_idx in src_idxs:
                src_addr = ta.base_addr + src_idx * ta.bytes_per_page
                descs_src.append((src_addr, ta.bytes_per_page, ta.device_id))
            xfer_dlist_src = nixl.TransferDescriptorList(
                type=self.memory_type, descs=descs_src
            )

            # Prepare destination descriptor list
            descs_dst: list[tuple[int, int, int]] = []
            remote_agent_meta = remote.agents_meta[tensor_idx]
            for dst_idx in dst_idxs:
                dst_addr = (
                    remote_agent_meta.base_addr + dst_idx * ta.bytes_per_page
                )
                descs_dst.append(
                    (dst_addr, ta.bytes_per_page, remote_agent_meta.device_id)
                )
            xfer_dlist_dst = nixl.TransferDescriptorList(
                type=remote.memory_type, descs=descs_dst
            )

            # Use the appropriate agent for this tensor
            remote_agent_name = remote_agent_meta.agent_name

            xfer_id = ta.agent.create_transfer_request(
                operation=nixl.TransferOpType.WRITE,
                local_descs=xfer_dlist_src,
                remote_descs=xfer_dlist_dst,
                remote_agent=remote_agent_name,
                notif_msg=xfer_name,
            )
            status = ta.agent.post_transfer_request(xfer_id)

            if status not in [nixl.Status.SUCCESS, nixl.Status.IN_PROG]:
                raise ValueError(
                    f"Transfer request failed with status {status} for tensor {tensor_idx}"
                )

            xfer_ids.append(xfer_id)
            self.inflight_xfers[xfer_name].append((tensor_idx, xfer_id))

        xfer_req = XferReqData(
            dst_name=remote_metadata.name,
            src_name=self.name,
            dst_agent_names=[a.agent_name for a in remote.agents_meta],
            src_agent_names=[ta.agent_name for ta in self.tensor_agents],
            xfer_name=xfer_name,
            xfer_ids=xfer_ids,
            src_idxs=src_idxs,
            dst_idxs=dst_idxs,
        )
        return xfer_req

    def send_xfer_sync(self, xfer_req_id: XferReqData) -> None:
        """Wait for transfers initiated by current engine to complete for all tensors.

        WARNING, this method is prone to infinite loops. For the transfer to
        progress, the remote engine MUST call wait_recv_complete. As such, the
        following code will hang:

        ```
        xfer_req = engine_1.write_to(...)
        engine_1.wait_send_complete(xfer_req) # hangs!
        engine_2.wait_recv_complete(xfer_req) # never called
        ```
        """
        xfer_name = xfer_req_id.xfer_name
        completed_tensors: set[int] = set()

        while len(completed_tensors) < len(xfer_req_id.xfer_ids):
            for tensor_idx, xfer_id in enumerate(xfer_req_id.xfer_ids):
                if tensor_idx in completed_tensors:
                    continue

                agent = self.tensor_agents[tensor_idx].agent
                status = agent.get_transfer_status(xfer_id)

                if status == nixl.Status.SUCCESS:
                    agent.release_transfer_request(xfer_id)
                    completed_tensors.add(tensor_idx)
                elif status == nixl.Status.IN_PROG:
                    continue
                else:
                    raise ValueError(
                        f"Transfer request failed with status {status} for tensor {tensor_idx}"
                    )

            if len(completed_tensors) < len(xfer_req_id.xfer_ids):
                us = 1 / 1000 / 1000
                time.sleep(us)

        self.completed_xfers[xfer_req_id.dst_name].add(xfer_name)
        del self.inflight_xfers[xfer_name]

    def get_transfer_status(
        self, xfer_req_data: XferReqData
    ) -> list[nixl.Status]:
        """Get the current status of transfer requests for all tensors.

        This API can only be used by the initiating transfer engine.

        Returns:
            List of status values, one per tensor.
        """
        statuses = []
        for tensor_idx, xfer_id in enumerate(xfer_req_data.xfer_ids):
            agent = self.tensor_agents[tensor_idx].agent
            statuses.append(agent.get_transfer_status(xfer_id))
        return statuses

    def update_completed_xfers(self) -> None:
        """Process agent notifications and mark completed transfers."""
        # Collect and process all agent notifications
        for ta in self.tensor_agents:
            notifs = ta.agent.get_notifs()
            for remote_agent_name, notifications in notifs.items():
                engine_name = self.remote_agent_to_engine.get(remote_agent_name)
                if engine_name:
                    self.completed_xfers[engine_name].update(
                        notif.decode() for notif in notifications
                    )

    def is_complete(self, xfer_req_id: XferReqData) -> bool:
        """Check if all transfer requests have completed.

        This method is primarily expected to be used by the receiver of a transfer
        to check if data has been successfully transferred from the source.

        Args:
            xfer_req_id: The transfer request data containing transfer metadata.

        Returns:
            True if all transfers have completed, False otherwise.
        """
        return (
            xfer_req_id.xfer_name in self.completed_xfers[xfer_req_id.src_name]
        )

    def recv_xfer_sync(self, xfer_req_id: XferReqData) -> None:
        """Wait for transfers initiated by remote engine to complete for all tensors."""
        while not self.is_complete(xfer_req_id):
            self.update_completed_xfers()

        self.completed_xfers[xfer_req_id.src_name].discard(
            xfer_req_id.xfer_name
        )

    def cleanup(self) -> None:
        """Release all resources associated with the transfer engine.

        Should be called before the transfer engine is garbage collected.
        Moving this logic into the __del__ destructor does causes a UCX error for
        unknown reasons.
        """

        # Release all xfers (using the appropriate agent for each)
        for xfer_list in self.inflight_xfers.values():
            for tensor_idx, xfer_id in xfer_list:
                agent = self.tensor_agents[tensor_idx].agent
                status = agent.release_transfer_request(xfer_id)
                if status != nixl.Status.SUCCESS:
                    raise ValueError(
                        f"Failed to release transfer request: {status}"
                    )

        # Deregister NIXL memory for all tensors
        for i, ta in enumerate(self.tensor_agents):
            status = ta.agent.deregister_memory(ta.reg_dlist, [ta.ucx_backend])
            if status != nixl.Status.SUCCESS:
                raise ValueError(
                    f"Failed to deregister memory for tensor {i}: {status}"
                )

        # Invalidate metadata of other agents
        for remote_name in self.remote_connections:
            remote = self.remote_connections[remote_name]
            # Invalidate for each agent pair
            for i, (ta, remote_agent_meta) in enumerate(
                zip(self.tensor_agents, remote.agents_meta)
            ):
                status = ta.agent.invalidate_remote_metadata(
                    remote_agent_meta.agent_name
                )
                if status != nixl.Status.SUCCESS:
                    raise ValueError(
                        f"Failed to invalidate metadata for agent {i}: {status}"
                    )
