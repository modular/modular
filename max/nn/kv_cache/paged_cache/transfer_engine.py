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

    remote_agent_to_engine: dict[str, str]
    """Map of remote agent names to their engine names."""

    completed_recv_xfers: dict[str, dict[str, int]]
    """Map of agent names to completed recv transfers."""

    inflight_send_xfers: dict[str, XferReqData]
    """Map of transfer names to send transfer request data."""

    def __init__(
        self,
        name: str,
        tensors: Tensor | list[Tensor],
        *,
        total_num_pages: int,
        listen_port: int | None = None,
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

        device = tensors[0].device
        if device.api == "hip":
            raise NotImplementedError(
                "Currently UCX does not support HIP devices."
            )

        # Initialize the transfer engine
        self.name = name
        self.tensor_agents = []

        # Set memory type and total pages
        self.total_num_pages = total_num_pages
        self.memory_type = (
            nixl.MemoryType.DRAM if is_cpu else nixl.MemoryType.VRAM
        )

        # Create agents and process each tensor
        for i, tensor in enumerate(tensors):
            # Create agent name
            agent_name = f"{name}_{i}" if self.num_tensors > 1 else name

            # Create NIXL agent
            agent = nixl.Agent(
                agent_name,
                nixl.AgentConfig(
                    # Always use progress thread.
                    # - It helps with async notification delivery.
                    # - It enables overlapping transfers from multiple agents.
                    use_prog_thread=True,
                    use_listen_thread=True,
                    listen_port=listen_port + i
                    if listen_port
                    else available_port(),
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
        self.remote_connections = {}

        # Map of agents to completed transfers
        self.completed_recv_xfers = defaultdict(lambda: defaultdict(int))

        # Map of remote agent names to their engine names
        self.remote_agent_to_engine = {}

        # All send xfers - maps xfer_name to list of (tensor_idx, xfer_id) tuples
        self.inflight_send_xfers = {}

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
            zip(self.tensor_agents, remote.agents_meta, strict=True)
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

        xfer_req = XferReqData(
            dst_name=remote_metadata.name,
            src_name=self.name,
            xfer_name=xfer_name,
            xfer_ids=xfer_ids,
            src_idxs=src_idxs,
            dst_idxs=dst_idxs,
        )
        self.inflight_send_xfers[xfer_name] = xfer_req
        return xfer_req

    def _is_sender_of(self, xfer_req: XferReqData) -> bool:
        """Check if the current engine is the sender of a transfer."""
        return xfer_req.src_name == self.name

    def _is_send_complete(self, xfer_req: XferReqData) -> bool:
        """Check if a send transfer is complete.

        Args:
            xfer_req: The transfer request data containing transfer metadata.

        Returns:
            True if the send transfer is complete, False otherwise.
        """
        assert self._is_sender_of(xfer_req)

        is_complete = True
        for tensor_idx, xfer_id in enumerate(xfer_req.xfer_ids):
            agent = self.tensor_agents[tensor_idx].agent
            status = agent.get_transfer_status(xfer_id)

            if status == nixl.Status.SUCCESS:
                continue
            elif status == nixl.Status.IN_PROG:
                is_complete = False
                break
            else:
                raise ValueError(
                    f"Transfer request failed with status {status} for tensor {tensor_idx}"
                )

        return is_complete

    def _is_recv_complete(self, xfer_req: XferReqData) -> bool:
        """Check if a recv transfer is complete."""
        assert not self._is_sender_of(xfer_req)

        # Check what recv completion notifications have been received
        for ta in self.tensor_agents:
            notifs = ta.agent.get_notifs()
            for remote_agent_name, notifications in notifs.items():
                engine_name = self.remote_agent_to_engine[remote_agent_name]
                for notif in notifications:
                    notif_decoded = notif.decode()
                    self.completed_recv_xfers[engine_name][notif_decoded] += 1

        # A recv is complete when we get num_agents notifications about it
        num_agents = len(self.tensor_agents)
        xfer_name = xfer_req.xfer_name
        return (
            self.completed_recv_xfers[xfer_req.src_name][xfer_name]
            == num_agents
        )

    def is_complete(self, xfer_req: XferReqData) -> bool:
        """Check if a given send or recv transfer is completed.

        WARNING, this method is prone to infinite loops. For the transfer to
        progress, the remote engine MUST call wait_recv_complete. As such, the
        following code will hang:

        ```
        xfer_req = engine_1.write_to(...)
        while not engine_1.is_complete(xfer_req):
            pass
        while not engine_2.is_complete(xfer_req):
            pass
        ```

        Instead do:
        ```
        xfer_req = engine_1.write_to(...)
        while not engine_1.is_complete(xfer_req) or not engine_2.is_complete(xfer_req):
            pass
        ```

        Args:
            xfer_req: The transfer request.

        Returns:
            True if all transfers have completed, False otherwise.
        """
        if self._is_sender_of(xfer_req):
            return self._is_send_complete(xfer_req)
        else:
            return self._is_recv_complete(xfer_req)

    def _cleanup_recv_transfer(self, xfer_req: XferReqData) -> None:
        """Cleanup a transfer."""
        assert not self._is_sender_of(xfer_req)
        assert xfer_req.xfer_name not in self.inflight_send_xfers

        del self.completed_recv_xfers[xfer_req.src_name][xfer_req.xfer_name]

    def _cleanup_send_transfer(self, xfer_req: XferReqData) -> None:
        """Cleanup a send transfer."""
        assert self._is_sender_of(xfer_req)
        xfer_name = xfer_req.xfer_name
        assert xfer_name in self.inflight_send_xfers

        del self.inflight_send_xfers[xfer_name]

        for tensor_idx, xfer_id in enumerate(xfer_req.xfer_ids):
            agent = self.tensor_agents[tensor_idx].agent
            status = agent.release_transfer_request(xfer_id)
            if status != nixl.Status.SUCCESS:
                raise ValueError(
                    f"Failed to release transfer request: {status}"
                )

    def cleanup_transfer(self, xfer_req: XferReqData) -> None:
        """Cleanup a transfer. This should be called after a transfer is complete.

        Args:
            xfer_req: The transfer request to cleanup.
        """
        if not self.is_complete(xfer_req):
            raise ValueError(f"Transfer {xfer_req.xfer_name} is not complete")

        if self._is_sender_of(xfer_req):
            self._cleanup_send_transfer(xfer_req)
        else:
            self._cleanup_recv_transfer(xfer_req)

    def sync_and_release(self, xfer_req: XferReqData) -> None:
        """Wait for a transfer to complete and release the transfer after it completes."""
        while not self.is_complete(xfer_req):
            time.sleep(0.001)
        self.cleanup_transfer(xfer_req)

    def cleanup(self) -> None:
        """Release all resources associated with the transfer engine.

        Should be called before the transfer engine is garbage collected.
        Moving this logic into the __del__ destructor does causes a UCX error for
        unknown reasons.
        """

        # Release all xfers
        for send_xfer_req in list(self.inflight_send_xfers.values()):
            self._cleanup_send_transfer(send_xfer_req)

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
                zip(self.tensor_agents, remote.agents_meta, strict=True)
            ):
                status = ta.agent.invalidate_remote_metadata(
                    remote_agent_meta.agent_name
                )
                if status != nixl.Status.SUCCESS:
                    raise ValueError(
                        f"Failed to invalidate metadata for agent {i}: {status}"
                    )
