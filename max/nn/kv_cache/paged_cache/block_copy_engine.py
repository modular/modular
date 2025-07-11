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

"""Facilitates copying of KVCache blocks."""

from __future__ import annotations

from enum import Enum

from max.driver import DeviceStream, Tensor


class BlockCopyType(Enum):
    """Type of block copy operation."""

    D2D = 1
    H2D = 2
    D2H = 3


class BlockCopyMetrics:
    def __init__(self) -> None:
        self.d2d = 0
        self.h2d = 0
        self.d2h = 0

    def reset(self) -> None:
        self.d2d = 0
        self.h2d = 0
        self.d2h = 0


class BlockCopyEngine:
    def __init__(
        self,
        block_size: int,
        num_device_blocks: int,
        device_tensors: list[Tensor],
        num_host_blocks: int,
        host_tensor: Tensor | None,
    ) -> None:
        if num_host_blocks > 0 and host_tensor is None:
            raise ValueError(
                "Host tensor must be non-null if there are host blocks"
            )
        if num_host_blocks <= 0 and host_tensor is not None:
            raise ValueError(
                "Host tensor must be null if there are no host blocks"
            )
        if num_device_blocks <= 0:
            raise ValueError("Number of device blocks must be non-zero")
        if block_size <= 0:
            raise ValueError("Block size must be positive")

        # There is at least 1 device tensors
        self.device_tensors = device_tensors
        # There can be 0 or 1 host tensors
        self.host_tensor = host_tensor

        self.block_size = block_size
        self.num_device_blocks = num_device_blocks
        self.num_host_blocks = num_host_blocks

        self.main_stream: DeviceStream | None = None
        self.d2h_auxiliary_stream: DeviceStream | None = None

        # Scheduling memory copies on separate stream is only useful if we have
        # pinned host memory.
        if self.host_tensor is not None and self.host_tensor.pinned:
            gpu_device = self.host_tensor.device
            self.main_stream = gpu_device.default_stream
            self.d2h_auxiliary_stream = DeviceStream(gpu_device)

        # Number of blocks that have been copied
        self.blocks_copied: BlockCopyMetrics = BlockCopyMetrics()

        self.staged_memcpy_ops: list[tuple[BlockCopyType, int, int]] = []

    def supports_multistream(self) -> bool:
        return self.d2h_auxiliary_stream is not None

    def memcpy_d2d(self, dst: int, src: int, num_tokens: int) -> None:
        """Copy a block between two blocks on the same device.

        This is primarily used for Copy-on-Write so a num_tokens argument can
        be used to copy only the relevant portion of the block.
        """
        # No need to copy if dst and src are the same
        if dst == src:
            return
        self.blocks_copied.d2d += 1

        # Currently we ignore num_tokens and just copy the entire block
        assert 0 < num_tokens <= self.block_size

        # For each device, copy the block from src to dst
        for device_tensor in self.device_tensors:
            device_tensor[dst, :, :, :, :, :].inplace_copy_from(
                device_tensor[src, :, :, :, :, :]
            )

    def memcpy_h2d(self, dst: int, src: int) -> None:
        if self.host_tensor is None:
            raise ValueError(
                "Attempted to enqueue h2d copy but there is no host tensor"
            )
        self.blocks_copied.h2d += 1

        # Copy block from host to each of the devices
        for device_tensor in self.device_tensors:
            device_tensor[dst, :, :, :, :, :].inplace_copy_from(
                self.host_tensor[src, :, :, :, :, :]
            )

    def memcpy_d2h(self, dst: int, src: int) -> None:
        if self.host_tensor is None:
            raise ValueError(
                "Attempted to enqueue d2h copy but there is no host tensor"
            )
        self.blocks_copied.d2h += 1

        # Copy the data from one device to the host. We assume that all
        # devices have the same data.
        device_tensor = self.device_tensors[0]
        src_block = device_tensor[src, :, :, :, :, :]
        dst_block = self.host_tensor[dst, :, :, :, :, :]

        if self.d2h_auxiliary_stream is not None:
            dst_block = dst_block.to(self.d2h_auxiliary_stream)

        dst_block.inplace_copy_from(src_block)

    def wait_for_completion(self) -> None:
        """Synchronize main stream with the auxiliary stream.

        This ensures that the d2h copies from BatchN completes before
        BatchN+1 begins. This is needed because BatchN+1 may write to the
        same blocks as BatchN is reading from.
        """
        if self.d2h_auxiliary_stream is not None:
            assert self.main_stream is not None
            self.main_stream.wait_for(self.d2h_auxiliary_stream)
