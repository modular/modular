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

from max.driver import DeviceStream
from max.nn.kv_cache.cache_params import KVCacheBuffer


class BlockOffloadEngine:
    """Engine for offloading gpu KVCache blocks to host memory.

    This offload engine will allocate a DevicePinnedBuffer with the same shape
    as the gpu buffer. It uses auxillary d2h streams to hide the latency of
    KV cache offloading copies on a stream detached from the main kernel exec
    stream. However, it still issues the h2d transfers on the same stream as
    kernel execution which is a major limitation (SERVOPT-1036).
    """

    def __init__(
        self, total_num_host_blocks: int, device_buffer: KVCacheBuffer
    ) -> None:
        self.device_buffer = device_buffer
        self.host_buffer = device_buffer.allocate_host_offload_buffer(
            total_num_host_blocks
        )

        self.main_streams: dict[int, DeviceStream] = {
            buffer.device.id: buffer.device.default_stream
            for buffer in device_buffer.all_buffers
        }
        self.d2h_auxiliary_streams: dict[int, DeviceStream] = {
            buffer.device.id: DeviceStream(buffer.device)
            for buffer in device_buffer.all_buffers
        }

    def memcpy_h2d(self, dst: int, src: int) -> None:
        """Copies a block from host to device(s)."""
        # Copy block from host to each of the devices
        for device_tensor, host_tensor in zip(
            self.device_buffer.all_buffers,
            self.host_buffer.all_buffers,
            strict=True,
        ):
            device_tensor[dst, :, :, :, :, :].inplace_copy_from(
                host_tensor[src, :, :, :, :, :]
            )

    def memcpy_d2h(self, dst: int, src: int) -> None:
        """Copies a block from device(s) to host."""
        # Copy the data from one device to the host.
        for device_buffer, host_buffer in zip(
            self.device_buffer.all_buffers,
            self.host_buffer.all_buffers,
            strict=True,
        ):
            src_block = device_buffer[src, :, :, :, :, :]
            dst_block = host_buffer[dst, :, :, :, :, :]

            device_id = device_buffer.device.id
            dst_block = dst_block.to(self.d2h_auxiliary_streams[device_id])

            dst_block.inplace_copy_from(src_block)

    def wait_for_completion(self) -> None:
        """Synchronize main stream with the auxiliary stream.

        This ensures that the d2h copies from BatchN completes before
        BatchN+1 begins. This is needed because BatchN+1 may write to the
        same blocks as BatchN is reading from.
        """
        for main_stream, d2h_auxiliary_stream in zip(
            self.main_streams.values(),
            self.d2h_auxiliary_streams.values(),
            strict=True,
        ):
            main_stream.wait_for(d2h_auxiliary_stream)
