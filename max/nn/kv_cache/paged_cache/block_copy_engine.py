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

from max.driver import DeviceStream, Tensor


class BlockCopyEngine:
    def __init__(
        self,
        block_size: int,
        num_device_blocks: int,
        device_tensors: list[Tensor],
        num_host_blocks: int,
        host_tensors: list[Tensor] | None,
    ) -> None:
        if num_host_blocks > 0 and host_tensors is None:
            raise ValueError(
                "Host tensor must be non-null if there are host blocks"
            )
        if num_host_blocks <= 0 and host_tensors is not None:
            raise ValueError(
                "Host tensor must be null if there are no host blocks"
            )
        if num_device_blocks <= 0:
            raise ValueError("Number of device blocks must be non-zero")
        if block_size <= 0:
            raise ValueError("Block size must be positive")

        # There is at least 1 device tensors
        self.device_tensors = device_tensors
        # There can be 0 or len(self.device_tensors) host tensors
        self.host_tensors = host_tensors

        self.block_size = block_size
        self.num_device_blocks = num_device_blocks
        self.num_host_blocks = num_host_blocks

        self.main_streams: list[DeviceStream] | None = None
        self.d2h_auxiliary_streams: list[DeviceStream] | None = None

        # Scheduling memory copies on separate stream is only useful if we have
        # pinned host memory.

        if self.host_tensors is not None:
            self.main_streams = [
                self.host_tensors[i].device.default_stream
                for i in range(len(self.device_tensors))
            ]
            self.d2h_auxiliary_streams = [
                DeviceStream(self.host_tensors[i].device)
                for i in range(len(self.device_tensors))
            ]

    def supports_multistream(self) -> bool:
        return self.d2h_auxiliary_streams is not None

    def memcpy_h2d(self, dst: int, src: int) -> None:
        if self.host_tensors is None:
            raise ValueError(
                "Attempted to enqueue h2d copy but there is no host tensor"
            )

        # Copy block from host to each of the devices
        for device_tensor, host_tensor in zip(
            self.device_tensors, self.host_tensors, strict=True
        ):
            device_tensor[dst, :, :, :, :, :].inplace_copy_from(
                host_tensor[src, :, :, :, :, :]
            )

    def memcpy_d2h(self, dst: int, src: int) -> None:
        if self.host_tensors is None:
            raise ValueError(
                "Attempted to enqueue d2h copy but there is no host tensor"
            )

        # Copy the data from one device to the host.
        for i, (device_tensor, host_tensor) in enumerate(
            zip(self.device_tensors, self.host_tensors, strict=True)
        ):
            src_block = device_tensor[src, :, :, :, :, :]
            dst_block = host_tensor[dst, :, :, :, :, :]

            if self.d2h_auxiliary_streams is not None:
                dst_block = dst_block.to(self.d2h_auxiliary_streams[i])

            dst_block.inplace_copy_from(src_block)

    def wait_for_completion(self) -> None:
        """Synchronize main stream with the auxiliary stream.

        This ensures that the d2h copies from BatchN completes before
        BatchN+1 begins. This is needed because BatchN+1 may write to the
        same blocks as BatchN is reading from.
        """
        if self.d2h_auxiliary_streams is None:
            return
        assert self.main_streams is not None
        for main_stream, d2h_auxiliary_stream in zip(
            self.main_streams, self.d2h_auxiliary_streams, strict=True
        ):
            main_stream.wait_for(d2h_auxiliary_stream)
