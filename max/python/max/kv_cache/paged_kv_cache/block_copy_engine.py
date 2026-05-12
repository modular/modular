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

from max._distributed_ops import distributed_broadcast
from max.driver import Buffer, Device, DevicePinnedBuffer, DeviceStream
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.allreduce import Signals
from max.profiler import traced


def _bytes_per_page(buffer: Buffer) -> int:
    num_pages = buffer.shape[0]
    return buffer.num_elements * buffer.dtype.size_in_bytes // num_pages


def _group_by_device(buffers: list[Buffer]) -> list[list[Buffer]]:
    """Reshape a flat per-rank buffer list into per-group buffer lists.

    Buckets ``buffers`` by device in first-seen order, then transposes so
    each returned group is one buffer per device in canonical order. Every
    device must contribute the same number of buffers.
    """
    by_device: dict[int, list[Buffer]] = {}
    device_order: list[int] = []
    for buf in buffers:
        if buf.device.id not in by_device:
            by_device[buf.device.id] = []
            device_order.append(buf.device.id)
        by_device[buf.device.id].append(buf)

    counts = {len(b) for b in by_device.values()}
    if len(counts) != 1:
        raise ValueError(
            f"every device must contribute the same number of buffers; got "
            f"per-device counts {counts}"
        )

    num_groups = counts.pop()
    return [
        [by_device[dev_id][g] for dev_id in device_order]
        for g in range(num_groups)
    ]


class BlockOffloadEngine:
    """Engine for offloading gpu KVCache blocks to host memory.

    This offload engine will allocate a DevicePinnedBuffer with the same shape
    as the gpu buffer. It uses auxiliary d2h streams to hide the latency of
    KV cache offloading copies on a stream detached from the main kernel exec
    stream. However, it still issues the h2d transfers on the same stream as
    kernel execution which is a major limitation (SERVOPT-1036).

    With ``replicate_kv_across_tp=True``, the host buffer holds a single
    replica per logical group. ``self.device_buffers`` is the rank-0 buffer of
    each group (the H2D/D2H endpoints) and ``self.replicated_buffers[g]`` is
    the list of peer-rank buffers for group ``g``. H2D copies host → rank 0,
    then fans each group out to its peers via a broadcast.
    """

    def __init__(
        self,
        total_num_host_blocks: int,
        device_buffers: list[Buffer],
        *,
        replicate_kv_across_tp: bool = False,
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

        # Sharded layout: every buffer is its own group of one. Replicated
        # layout: bucket by device so each logical group's peers ride
        # alongside its rank-0 buffer.
        self.device_buffers: list[Buffer]
        self.replicated_buffers: list[list[Buffer]]
        if replicate_kv_across_tp:
            groups = _group_by_device(viewed)
            if len(groups[0]) > 1:
                self.device_buffers = [g[0] for g in groups]
                self.replicated_buffers = [g[1:] for g in groups]
            else:
                self.device_buffers = viewed
                self.replicated_buffers = []
        else:
            self.device_buffers = viewed
            self.replicated_buffers = []

        for root, peers in zip(
            self.device_buffers, self.replicated_buffers, strict=False
        ):
            page_bytes = _bytes_per_page(root)
            for peer in peers:
                if _bytes_per_page(peer) != page_bytes:
                    raise ValueError(
                        "replicate_kv_across_tp requires identical "
                        "bytes_per_page across replicas within a group"
                    )

        bytes_per_page = sum(_bytes_per_page(b) for b in self.device_buffers)
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

        self._signals: Signals | None = None
        self._signal_buffers: list[Buffer] = []
        self._broadcast_devices: list[Device] = []
        if self.replicated_buffers:
            # Every group shares the same TP devices in the same order.
            self._broadcast_devices = [
                self.device_buffers[0].device,
                *(p.device for p in self.replicated_buffers[0]),
            ]
            self._signals = Signals(
                devices=[
                    DeviceRef.GPU(id=d.id) for d in self._broadcast_devices
                ]
            )
            self._signal_buffers = self._signals.buffers()

    @traced
    def memcpy_h2d(self, dst: int, src: int) -> None:
        """Copies a block from host to device(s)."""
        offset = 0
        for buf in self.device_buffers:
            page_bytes = buf.shape[1]
            buf[dst, :].inplace_copy_from(
                self.host_buffer[src, offset : offset + page_bytes]
            )
            offset += page_bytes

        for root, peers in zip(
            self.device_buffers, self.replicated_buffers, strict=False
        ):
            distributed_broadcast(
                input_buffer=root[dst, :],
                output_buffers=[root[dst, :], *(p[dst, :] for p in peers)],
                signal_buffers=self._signal_buffers,
                devices=self._broadcast_devices,
                root=0,
            )

    @traced
    def memcpy_d2h(self, dst: int, src: int) -> None:
        """Copies a block from device(s) to host."""
        offset = 0
        for buf in self.device_buffers:
            page_bytes = buf.shape[1]
            # TODO(SERVOPT-1389): issue d2h on auxiliary stream after fixing driver bugs
            self.host_buffer[
                dst, offset : offset + page_bytes
            ].inplace_copy_from(buf[src, :])
            offset += page_bytes

    @traced
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
