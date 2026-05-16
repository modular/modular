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

import logging

from max.driver import Buffer, DevicePinnedBuffer
from max.dtype import DType
from max.profiler import traced

logger = logging.getLogger("max.pipelines")


def _bytes_per_page(buffer: Buffer) -> int:
    num_pages = buffer.shape[0]
    return buffer.num_elements * buffer.dtype.size_in_bytes // num_pages


class DebugBlockOffloadEngine:
    """Debug class."""

    def __init__(
        self, total_num_host_blocks: int, device_buffers: list[Buffer]
    ) -> None:
        num_device_pages = device_buffers[0].shape[0]
        viewed = [
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

        self.device_buffers = viewed
        bytes_per_page = sum(_bytes_per_page(b) for b in self.device_buffers)
        self.host_buffer = DevicePinnedBuffer(
            shape=[total_num_host_blocks, bytes_per_page],
            dtype=DType.uint8,
            device=gpu0,
        )

    @traced
    def memcpy_h2d(self, dst: int, src: int) -> None:
        """Copies a block from host to device(s)."""
        offset = 0
        for buf in self.device_buffers:
            buf.device.synchronize()
            self.host_buffer.device.synchronize()

            page_bytes = buf.shape[1]
            buf[dst, :].inplace_copy_from(
                self.host_buffer[src, offset : offset + page_bytes]
            )
            offset += page_bytes

            buf.device.synchronize()
            self.host_buffer.device.synchronize()

    @traced
    def memcpy_d2h(self, dst: int, src: int) -> None:
        """Copies a block from device(s) to host."""
        offset = 0
        for buf in self.device_buffers:
            buf.device.synchronize()
            self.host_buffer.device.synchronize()

            page_bytes = buf.shape[1]
            self.host_buffer[
                dst, offset : offset + page_bytes
            ].inplace_copy_from(buf[src, :])
            offset += page_bytes

            buf.device.synchronize()
            self.host_buffer.device.synchronize()
