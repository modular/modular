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


class KVTransferEngineMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Metadata associated with a transfer engine.

    This is safe to send between threads/processes."""

    metadata: bytes
    name: str
    total_num_pages: int
    bytes_per_page: int
    base_addr: int
    memory_type: nixl.MemoryType
    device_id: int


class XferReqData(msgspec.Struct, tag=True, kw_only=True, omit_defaults=True):
    """Metadata associated with a transfer request.

    This is safe to send between threads/processes."""

    dst_name: str
    src_name: str
    xfer_name: str
    xfer_id: int
    src_idxs: list[int]
    dst_idxs: list[int]


class KVTransferEngine:
    """KVCache Transfer Engine.

    Currently this is only tested on CPU and supports a single Paged tensor.

    The TransferEngine communicates with other TransferEngines in other threads
    or processes. However, individual TransferEngines themselves are not
    thread-safe. It is intended to be used by MAX's single-threaded scheduler.
    """

    name: str
    """Name of transfer engine / nixl agent."""

    agent: nixl.Agent
    """NIXL agent for communication."""

    tensor: Tensor
    """Flattened tensor being managed."""

    base_addr: int
    """Base memory address of tensor / CPU staging buffer."""

    total_num_pages: int
    """Total number of pages in the tensor."""

    ucx_backend: int
    """UCX backend used for communication."""

    memory_type: nixl.MemoryType
    """Type of memory being managed (e.g. DRAM)."""

    remote_connections: dict[str, KVTransferEngineMetadata]
    """Map of remote engine names to their metadata."""

    completed_xfers: dict[str, set[str]]
    """Map of agent names to completed transfers."""

    def __init__(
        self,
        name: str,
        tensor: Tensor,
        *,
        total_num_pages: int,
        listen_port: int,
    ) -> None:
        if total_num_pages <= 0:
            raise ValueError(
                f"Total number of pages {total_num_pages} must be greater than 0"
            )

        if tensor.num_elements % total_num_pages != 0:
            raise ValueError(
                f"Tensor num elements {tensor.num_elements} must be divisible by total number of pages {total_num_pages}"
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

        # Create agent
        self.name = name
        self.agent = nixl.Agent(
            name,
            nixl.AgentConfig(
                use_prog_thread=False,
                use_listen_thread=True,
                listen_port=listen_port,
            ),
        )

        # Reshape tensor to 2D
        self.total_num_pages = total_num_pages
        self.bytes_per_page = (
            tensor.num_elements * tensor.dtype.size_in_bytes // total_num_pages
        )
        self.elts_per_page = tensor.num_elements // total_num_pages
        self.tensor = tensor.view(
            tensor.dtype, (self.total_num_pages, self.elts_per_page)
        )

        # Create UCX backend
        if "ucx" not in self.agent.get_available_plugins():
            raise RuntimeError(
                "UCX not currently available, please ensure it is supported by your system."
            )

        # Determine device ID
        device = self.tensor.device
        if not device.is_host:
            self.device_id = device.id
        else:
            self.device_id = 0

        ucx_params = self.agent.get_plugin_params("ucx")[0]
        if not device.is_host:
            ucx_params["gpu_device_id"] = str(self.device_id)
        self.ucx_backend = self.agent.create_backend(
            type="ucx",
            init_params=ucx_params,
        )

        # Register memory
        self.memory_type = (
            nixl.MemoryType.DRAM if device.is_host else nixl.MemoryType.VRAM
        )
        self.base_addr = self.tensor._data_ptr()
        num_bytes = self.tensor.num_elements * self.tensor.dtype.size_in_bytes
        self.reg_dlist = nixl.RegistrationDescriptorList(
            type=self.memory_type,
            descs=[
                (
                    self.base_addr,
                    num_bytes,
                    self.device_id,
                    "",
                )
            ],
            sorted=True,
        )
        status = self.agent.register_memory(self.reg_dlist, [self.ucx_backend])
        if status != nixl.Status.SUCCESS:
            raise ValueError(f"Failed to register memory: {status}")

        # Get metadata after registration
        self.agent_metadata = self.agent.get_local_metadata()

        # Remote connections
        self.remote_connections: dict[str, KVTransferEngineMetadata] = {}

        # Map of agents to completed transfers
        self.completed_xfers: dict[str, set[str]] = defaultdict(set)

        # All xfers
        self.inflight_xfers: dict[str, XferReqData] = {}

    @property
    def metadata(self) -> KVTransferEngineMetadata:
        """Get metadata for the current engine.

        Returns:
            Metadata for the current engine.
        """
        return KVTransferEngineMetadata(
            metadata=self.agent_metadata,
            name=self.name,
            total_num_pages=self.total_num_pages,
            bytes_per_page=self.bytes_per_page,
            base_addr=self.base_addr,
            memory_type=self.memory_type,
            device_id=self.device_id,
        )

    def connect(self, remote: KVTransferEngineMetadata) -> None:
        """Connect to a remote engine.

        Args:
            remote: Metadata for the remote engine.
        """
        if remote.name in self.remote_connections:
            raise ValueError(f"Agent {remote.name} already connected")
        if self.bytes_per_page != remote.bytes_per_page:
            raise ValueError(
                f"Bytes per page mismatch: {self.bytes_per_page} != {remote.bytes_per_page}"
            )
        loaded_bytes = self.agent.load_remote_metadata(remote.metadata)
        try:
            loaded_remote_name = loaded_bytes.decode()
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Metadata loading failed. Expected string, found {loaded_bytes!r}"
            ) from e
        if loaded_remote_name != remote.name:
            raise ValueError(
                f"Metadata loading failed. Expected {remote.name}, got {loaded_remote_name}"
            )
        self.remote_connections[remote.name] = remote

    def initiate_send_xfer(
        self,
        remote_metadata: KVTransferEngineMetadata,
        src_idxs: list[int],
        dst_idxs: list[int],
    ) -> XferReqData:
        """Initiate a transfer from current engine to remote engine.

        Args:
            remote: Metadata for the remote engine.
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

        bytes_per_page = self.bytes_per_page

        # Prepare source descriptor list
        descs_src: list[tuple[int, int, int]] = []
        for src_idx in src_idxs:
            src_addr = self.base_addr + src_idx * bytes_per_page
            descs_src.append((src_addr, bytes_per_page, self.device_id))
        xfer_dlist_src = nixl.TransferDescriptorList(
            type=self.memory_type, descs=descs_src
        )

        # Prepare destination descriptor list
        descs_dst: list[tuple[int, int, int]] = []
        for dst_idx in dst_idxs:
            dst_addr = remote.base_addr + dst_idx * bytes_per_page
            descs_dst.append((dst_addr, bytes_per_page, remote.device_id))
        xfer_dlist_dst = nixl.TransferDescriptorList(
            type=remote.memory_type, descs=descs_dst
        )

        xfer_name = str(uuid4())
        xfer_id = self.agent.create_transfer_request(
            operation=nixl.TransferOpType.WRITE,
            local_descs=xfer_dlist_src,
            remote_descs=xfer_dlist_dst,
            remote_agent=remote_metadata.name,
            notif_msg=xfer_name,
        )
        status = self.agent.post_transfer_request(xfer_id)

        if status not in [nixl.Status.SUCCESS, nixl.Status.IN_PROG]:
            raise ValueError(f"Transfer request failed with status {status}")

        xfer_req = XferReqData(
            dst_name=remote_metadata.name,
            src_name=self.name,
            xfer_name=xfer_name,
            xfer_id=xfer_id,
            src_idxs=src_idxs,
            dst_idxs=dst_idxs,
        )
        self.inflight_xfers[xfer_name] = xfer_req
        return xfer_req

    def send_xfer_sync(self, xfer_req_id: XferReqData) -> None:
        """Wait for a transfer initiated by current engine to complete.

        WARNING, this method is prone to infinite loops. For the transfer to
        progress, the remote engine MUST call wait_recv_complete. As such, the
        following code will hang:

        ```
        xfer_req = engine_1.write_to(...)
        engine_1.wait_send_complete(xfer_req) # hangs!
        engine_2.wait_recv_complete(xfer_req) # never called
        ```
        """
        while (
            xfer_req_id.xfer_name
            not in self.completed_xfers[xfer_req_id.dst_name]
        ):
            status = self.agent.get_transfer_status(xfer_req_id.xfer_id)

            if status == nixl.Status.SUCCESS:
                self.completed_xfers[xfer_req_id.dst_name].add(
                    xfer_req_id.xfer_name
                )
                self.agent.release_transfer_request(xfer_req_id.xfer_id)
                del self.inflight_xfers[xfer_req_id.xfer_name]
            elif status == nixl.Status.IN_PROG:
                us = 1 / 1000 / 1000
                time.sleep(us)
            else:
                raise ValueError(
                    f"Transfer request failed with status {status}"
                )

    def get_transfer_status(self, xfer_req_data: XferReqData) -> nixl.Status:
        """Get the current status of a transfer request.

        This API can only be used by the initiating transfer engine.
        """
        return self.agent.get_transfer_status(xfer_req_data.xfer_id)

    def update_completed_xfers(self) -> None:
        """Update the completed transfers by processing notifications from the agent.

        This method retrieves notifications from the transfer agent and updates
        the internal completed_xfers tracking for each remote connection.
        Notifications contain transfer names that have completed, which are
        decoded from bytes to strings and added to the appropriate remote's
        completed transfers set.
        """
        notifs = self.agent.get_notifs()

        for remote in notifs:
            completed_xfer_names = [x.decode() for x in notifs[remote]]
            self.completed_xfers[remote].update(completed_xfer_names)

    def is_complete(self, xfer_req_id: XferReqData) -> bool:
        """Check if a transfer request has completed.

        This method is primarily expected to be used by the receiver of a transfer
        to check if data has been successfully transferred from the source.

        Args:
            xfer_req_id: The transfer request data containing transfer metadata.

        Returns:
            True if the transfer has completed, False otherwise.
        """
        return (
            xfer_req_id.xfer_name in self.completed_xfers[xfer_req_id.src_name]
        )

    def recv_xfer_sync(self, xfer_req_id: XferReqData) -> None:
        """Wait for a transfer initiated by remote engine to complete."""
        while not self.is_complete(xfer_req_id):
            self.update_completed_xfers()

        self.completed_xfers[xfer_req_id.src_name].remove(xfer_req_id.xfer_name)

    def cleanup(self) -> None:
        """Release all resources associated with the transfer engine.

        Should be called before the transfer engine is garbage collected.
        Moving this logic into the __del__ destructor does causes a UCX error for
        unknown reasons.
        """

        # Release all xfers
        for xfer_req in self.inflight_xfers.values():
            status = self.agent.release_transfer_request(xfer_req.xfer_id)
            if status != nixl.Status.SUCCESS:
                raise ValueError(
                    f"Failed to release transfer request: {status}"
                )

        # Deregister NIXL memory
        status = self.agent.deregister_memory(
            self.reg_dlist, [self.ucx_backend]
        )
        if status != nixl.Status.SUCCESS:
            raise ValueError(f"Failed to deregister memory: {status}")

        # Invalidate metadata of other agents
        for remote_name in self.remote_connections:
            status = self.agent.invalidate_remote_metadata(remote_name)
            if status != nixl.Status.SUCCESS:
                raise ValueError(f"Failed to invalidate metadata: {status}")
