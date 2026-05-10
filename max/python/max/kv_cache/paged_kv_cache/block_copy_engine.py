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

"""Facilitates copying of KVCache blocks."""

from __future__ import annotations

from max.driver import Buffer, DevicePinnedBuffer, DeviceStream
from max.dtype import DType


def _bytes_per_page(buffer: Buffer) -> int:
    num_pages = buffer.shape[0]
    return buffer.num_elements * buffer.dtype.size_in_bytes // num_pages


class BlockOffloadEngine:
    """Engine for offloading gpu KVCache blocks to host memory.

    This offload engine will allocate a DevicePinnedBuffer with the same shape
    as the gpu buffer. It uses auxiliary d2h streams to hide the latency of
    KV cache offloading copies on a stream detached from the main kernel exec
    stream. However, it still issues the h2d transfers on the same stream as
    kernel execution which is a major limitation (SERVOPT-1036).
    """

    def __init__(
        self, total_num_host_blocks: int, device_buffers: list[Buffer]
    ) -> None:
        num_device_pages = device_buffers[0].shape[0]
        self.device_buffers = [
            buffer.view(
                dtype=DType.uint8,
                shape=[num_device_pages, _bytes_per_page(buffer)],
            )
            for buffer in device_buffers
        ]
        gpu0 = device_buffers[0].device
        if gpu0.is_host:
            raise ValueError(
                "KVCacheBuffer is on the CPU. Unable to allocate host"
                " offload buffer for already-on-CPU buffers."
            )
        bytes_per_page = sum(
            _bytes_per_page(buffer) for buffer in device_buffers
        )
        self.host_buffer = DevicePinnedBuffer(
            shape=[total_num_host_blocks, bytes_per_page],
            dtype=DType.uint8,
            device=gpu0,
        )
        self.main_streams: dict[int, DeviceStream] = {
            buffer.device.id: buffer.device.default_stream
            for buffer in self.device_buffers
        }
        self.d2h_auxiliary_streams: dict[int, DeviceStream] = {
            buffer.device.id: DeviceStream(buffer.device)
            for buffer in self.device_buffers
        }

    def memcpy_h2d(self, dst: int, src: int) -> None:
        """Copies a block from host to device(s)."""
        # Copy block from host to each of the devices
        offset = 0
        for device_buffer in self.device_buffers:
            bytes_per_page = device_buffer.shape[1]
            dst_block = device_buffer[dst, :]
            src_block = self.host_buffer[src, offset : offset + bytes_per_page]
            dst_block.inplace_copy_from(src_block)
            offset += bytes_per_page

    def memcpy_d2h(self, dst: int, src: int) -> None:
        """Copies a block from device(s) to host."""
        offset = 0
        for device_buffer in self.device_buffers:
            bytes_per_page = device_buffer.shape[1]
            aux_stream = self.d2h_auxiliary_streams[device_buffer.device.id]
            host_block = self.host_buffer[
                dst, offset : offset + bytes_per_page
            ].to(aux_stream)
            host_block.inplace_copy_from(device_buffer[src, :])
            offset += bytes_per_page

    def wait_for_completion(self) -> None:
        """Synchronize main stream with the auxiliary stream.

        This ensures that the d2h copies from BatchN completes before
        BatchN+1 begins. This is needed because BatchN+1 may write to the
        same blocks as BatchN is reading from.

        Additionally, ensure that d2h offload of BatchN starts after BatchN
        completes. As such this needs to be a duplex sync.
        """
        for main_stream, d2h_auxiliary_stream in zip(
            self.main_streams.values(),
            self.d2h_auxiliary_streams.values(),
            strict=True,
        ):
            main_stream.wait_for(d2h_auxiliary_stream)
            d2h_auxiliary_stream.wait_for(main_stream)
