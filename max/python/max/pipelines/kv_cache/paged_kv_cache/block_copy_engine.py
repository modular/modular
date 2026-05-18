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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from max._distributed_ops import distributed_broadcast
from max.driver import (
    Buffer,
    Device,
    DeviceEvent,
    DevicePinnedBuffer,
    DeviceStream,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.allreduce import Signals
from max.profiler import Tracer, traced
from max.support.math import ceildiv
from tqdm import tqdm


@dataclass
class DeviceEventBundle:
    """A bundle of device events."""

    events: list[DeviceEvent]

    @classmethod
    def record_on_streams(
        cls, streams: Sequence[DeviceStream]
    ) -> DeviceEventBundle:
        """Record an event on the given streams."""
        return cls(events=[stream.record_event() for stream in streams])

    def is_ready(self) -> bool:
        """Check if all events are ready."""
        return all(event.is_ready() for event in self.events)

    def synchronize(self) -> None:
        """Synchronize all events."""
        for event in self.events:
            event.synchronize()


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


def _2d_view(buffer: Buffer, num_pages: int) -> Buffer:
    if buffer.shape[0] != num_pages:
        raise ValueError(
            f"Expected first dimension of buffer to be {num_pages}, got {buffer.shape[0]}"
        )
    return buffer.view(
        dtype=DType.uint8,
        shape=[num_pages, _bytes_per_page(buffer)],
    )


_GIB = 1024**3
_MAX_PINNED_CHUNK_BYTES = 4 * _GIB


class PinnedHostKVCacheBuffer:
    """Chunked pinned host buffer for KV cache offloading.

    ``cuMemAllocHost`` can fail for very large contiguous allocations
    (e.g. >1 TiB) even when sufficient physical memory exists. This class
    splits the allocation into multiple ``DevicePinnedBuffer`` chunks and
    presents a unified interface keyed by block index.

    Args:
        total_num_blocks: Total number of KV cache blocks to host.
        bytes_per_block: Size of each block in bytes.
        device: GPU device the pinned memory is associated with.
        max_chunk_bytes: Upper bound on each pinned allocation in bytes.
    """

    def __init__(
        self,
        total_num_blocks: int,
        bytes_per_block: int,
        device: Device,
        max_chunk_bytes: int = _MAX_PINNED_CHUNK_BYTES,
    ) -> None:
        self._total_num_blocks = total_num_blocks
        self._bytes_per_block = bytes_per_block

        blocks_per_chunk = max(1, max_chunk_bytes // bytes_per_block)
        self._blocks_per_chunk = blocks_per_chunk

        total_gib = (total_num_blocks * bytes_per_block) / _GIB
        num_chunks = ceildiv(total_num_blocks, blocks_per_chunk)

        self._chunks: list[DevicePinnedBuffer] = []
        for i in tqdm(
            range(num_chunks),
            desc=f"Allocating {total_gib:.1f} GiB pinned host KV cache",
        ):
            start = i * blocks_per_chunk
            n = min(blocks_per_chunk, total_num_blocks - start)
            self._chunks.append(
                DevicePinnedBuffer(
                    shape=[n, bytes_per_block],
                    dtype=DType.uint8,
                    device=device,
                )
            )

    def _locate(self, block_id: int) -> tuple[int, int]:
        """Map a global *block_id* to ``(chunk_index, local_block_id)``."""
        if block_id < 0 or block_id >= self._total_num_blocks:
            raise IndexError(
                f"block_id {block_id} out of range "
                f"[0, {self._total_num_blocks})"
            )
        chunk_idx = block_id // self._blocks_per_chunk
        local_id = block_id % self._blocks_per_chunk
        return chunk_idx, local_id

    def page_view(self, block_id: int) -> Buffer:
        """Return a 1-D ``Buffer`` view of a given page."""
        chunk_idx, local_id = self._locate(block_id)
        return self._chunks[chunk_idx][local_id, :]

    def numpy_page_view(self, block_id: int) -> npt.NDArray[np.uint8]:
        """Return a 1-D NumPy view of a given page."""
        chunk_idx, local_id = self._locate(block_id)
        return self._chunks[chunk_idx].to_numpy()[local_id]

    @property
    def shape(self) -> list[int]:
        """Virtual ``[total_num_blocks, bytes_per_block]`` shape."""
        return [self._total_num_blocks, self._bytes_per_block]

    @property
    def dtype(self) -> DType:
        """Always ``DType.uint8``."""
        return DType.uint8

    @property
    def pinned(self) -> bool:
        """Always ``True``; every chunk is device-pinned."""
        return True


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
        non_replicated_device_buffers_to_offload: list[Buffer] | None = None,
    ) -> None:
        num_device_pages = device_buffers[0].shape[0]
        viewed = [
            _2d_view(buffer, num_device_pages) for buffer in device_buffers
        ]
        gpu0 = device_buffers[0].device
        if gpu0.is_host:
            raise ValueError(
                "KVCacheBuffer is on the CPU. Unable to allocate host"
                " offload buffer for already-on-CPU buffers."
            )

        if (
            non_replicated_device_buffers_to_offload
            and not replicate_kv_across_tp
        ):
            raise ValueError(
                "non_replicated_device_buffers_to_offload is only supported when replicate_kv_across_tp is True"
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

        # Special case for mla target + mha draft
        if non_replicated_device_buffers_to_offload is not None:
            non_replicated_viewed = [
                _2d_view(b, num_device_pages)
                for b in non_replicated_device_buffers_to_offload
            ]
            self.device_buffers.extend(non_replicated_viewed)

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
        self.host_buffer = PinnedHostKVCacheBuffer(
            total_num_blocks=total_num_host_blocks,
            bytes_per_block=bytes_per_page,
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
        self.device_buffers_on_aux_stream: list[Buffer] = [
            buffer.to(self.d2h_auxiliary_streams[buffer.device.id])
            for buffer in self.device_buffers
        ]

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
        # h2d on auxiliary stream.
        offset = 0
        for buf in self.device_buffers_on_aux_stream:
            page_bytes = buf.shape[1]
            buf[dst, :].inplace_copy_from(
                self.host_buffer.page_view(src)[offset : offset + page_bytes]
            )
            offset += page_bytes

        if not self.replicated_buffers:
            return

        # main stream waits for completion of d2h on auxiliary stream.
        for main_stream, d2h_auxiliary_stream in zip(
            self.main_streams.values(),
            self.d2h_auxiliary_streams.values(),
            strict=True,
        ):
            main_stream.wait_for(d2h_auxiliary_stream)

        # Broadcast the block to the other devices on main stream.
        num_replicated_buffers = len(self.replicated_buffers)
        for root, peers in zip(
            self.device_buffers[:num_replicated_buffers],
            self.replicated_buffers,
            strict=True,
        ):
            with Tracer("distributed_broadcast"):
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
        for buf in self.device_buffers_on_aux_stream:
            page_bytes = buf.shape[1]
            self.host_buffer.page_view(dst)[
                offset : offset + page_bytes
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

    @traced
    def record_d2h_event(self) -> DeviceEventBundle:
        """Record an event on all the d2h auxiliary streams."""
        return DeviceEventBundle.record_on_streams(
            list(self.d2h_auxiliary_streams.values())
        )
