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
import time
from collections import defaultdict
from dataclasses import dataclass
from uuid import uuid4

import torch
from max import driver
from max._core import nixl
from max.driver import CPU
from max.driver.tensor import Tensor

logger = logging.getLogger("max.pipelines")


def _get_tensor_base_addr(tensor: Tensor) -> int:
    """Get the base address of a tensor."""
    return torch.from_dlpack(tensor).data_ptr()


@dataclass
class KVTransferEngineMetadata:
    """Metadata associated with a transfer engine.

    This is safe to send between threads/processes."""

    metadata: bytes
    name: str
    total_num_pages: int
    bytes_per_page: int
    base_addr: int
    memory_type: nixl.MemoryType


@dataclass
class XferReqData:
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

    cpu_staging_buffer: Tensor | None
    """CPU staging buffer for GPU->GPU transfers."""

    def __init__(
        self,
        name: str,
        tensor: Tensor,
        total_num_pages: int,
        *,
        listen_port: int = 8040,
    ):
        if total_num_pages <= 0:
            raise ValueError(
                f"Total number of pages {total_num_pages} must be greater than 0"
            )

        if tensor.num_elements % total_num_pages != 0:
            raise ValueError(
                f"Tensor num elements {tensor.num_elements} must be divisible by total number of pages {total_num_pages}"
            )

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

        ucx_params = self.agent.get_plugin_params("ucx")
        self.ucx_backend = self.agent.create_backend(
            type="ucx", init_params=ucx_params[0]
        )

        if not tensor.device.is_host and driver.accelerator_api() != "cuda":
            # TODO(E2EOPT-228): Support HIP
            raise ValueError(
                "Non-NVIDIA GPUs are not yet supported with NIXL Transfer Engine."
            )

        # TODO(E2EOPT-216) Delete CPU staging buffer after GPU->GPU is supported
        # Maybe allocate a CPU staging buffer for GPU->GPU transfers
        # So GPU->GPU would actually be GPU->CPU->CPU->GPU.
        self.cpu_staging_buffer = None
        if not tensor.device.is_host:
            logger.warning(
                "TransferEngine does not support GPUDirect, "
                "falling back to slow CPU TCP transfers until it is supported."
            )
            self.cpu_staging_buffer = Tensor(
                shape=(self.total_num_pages, self.elts_per_page),
                dtype=tensor.dtype,
                device=CPU(),
                pinned=True,
            )

        # Determine base address
        if self.cpu_staging_buffer is None:
            self.base_addr = _get_tensor_base_addr(self.tensor)
        else:
            self.base_addr = _get_tensor_base_addr(self.cpu_staging_buffer)

        # Register memory
        self.memory_type = (
            nixl.MemoryType.DRAM
            if self.tensor.device.is_host
            else nixl.MemoryType.VRAM
        )
        num_bytes = self.tensor.num_elements * self.tensor.dtype.size_in_bytes
        self.reg_dlist = nixl.RegistrationDescriptorList(
            type=self.memory_type,
            descs=[
                (
                    self.base_addr,
                    num_bytes,
                    self.memory_type.value,
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

        # Stage data to CPU staging buffer before xfer
        if self.cpu_staging_buffer is not None:
            for src_idx in src_idxs:
                self.cpu_staging_buffer[src_idx, :].inplace_copy_from(
                    self.tensor[src_idx, :]
                )
            self.cpu_staging_buffer.device.synchronize()

        bytes_per_page = self.bytes_per_page

        # Prepare source descriptor list
        descs_src: list[tuple[int, int, int]] = []
        for src_idx in src_idxs:
            src_addr = self.base_addr + src_idx * bytes_per_page
            descs_src.append((src_addr, bytes_per_page, self.memory_type.value))
        xfer_dlist_src = nixl.TransferDescriptorList(
            type=self.memory_type, descs=descs_src
        )

        # Prepare destination descriptor list
        descs_dst: list[tuple[int, int, int]] = []
        for dst_idx in dst_idxs:
            dst_addr = remote.base_addr + dst_idx * bytes_per_page
            descs_dst.append(
                (dst_addr, bytes_per_page, remote.memory_type.value)
            )
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

    def recv_xfer_sync(self, xfer_req_id: XferReqData) -> None:
        """Wait for a transfer initiated by remote engine to complete."""
        while (
            xfer_req_id.xfer_name
            not in self.completed_xfers[xfer_req_id.src_name]
        ):
            notifs = self.agent.get_notifs()

            for remote in notifs:
                completed_xfer_names = [x.decode() for x in notifs[remote]]
                self.completed_xfers[remote].update(completed_xfer_names)

        # move data from CPU staging buffer to tensor after xfer
        if self.cpu_staging_buffer is not None:
            for dst_idx in xfer_req_id.dst_idxs:
                self.tensor[dst_idx, :].inplace_copy_from(
                    self.cpu_staging_buffer[dst_idx, :]
                )
            self.cpu_staging_buffer.device.synchronize()

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
