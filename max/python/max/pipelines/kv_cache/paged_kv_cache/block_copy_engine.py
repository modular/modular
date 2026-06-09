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
from max.nn.kv_cache.cache_params import KVCacheMemory, ReplicatedKVCacheMemory
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

    For replicated KV caches (MLA), the host buffer holds a single replica per
    logical group. D2H copies from rank 0, then H2D fans back out to all peers
    via a broadcast. For sharded caches (MHA), every shard is its own unit.
    """

    def __init__(
        self,
        total_num_host_blocks: int,
        kv_memory: list[KVCacheMemory],
    ) -> None:
        gpu0 = kv_memory[0].buffer.device
        if gpu0.is_host:
            raise ValueError(
                "KVCacheMemory is on the CPU. Unable to allocate host"
                " offload buffer for already-on-CPU buffers."
            )

        self._units = kv_memory
        self._replicated_units: list[ReplicatedKVCacheMemory] = [
            u for u in self._units if isinstance(u, ReplicatedKVCacheMemory)
        ]

        # Validate device topology across all replicated units.
        unique_topologies: set[tuple[int, ...]] = {
            tuple(
                d.id
                for d in [unit.buffer.device, *(p.device for p in unit.peers)]
            )
            for unit in self._replicated_units
        }
        if len(unique_topologies) > 1:
            raise ValueError(
                "all replicated KVCacheMemory units must share the same "
                "TP device topology; mixed topologies are not supported"
            )

        # Broadcast devices: rank-0 + peers from the first replicated unit
        # (topology uniformity was validated above).
        self._broadcast_devices: list[Device] = (
            [
                self._replicated_units[0].buffer.device,
                *(p.device for p in self._replicated_units[0].peers),
            ]
            if self._replicated_units
            else []
        )

        # The D2H/H2D endpoints — one per unit (rank-0 for replicated units).
        self.device_buffers: list[Buffer] = [u.buffer for u in self._units]

        bytes_per_page = sum(b.shape[1] for b in self.device_buffers)
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
        if self._replicated_units:
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

        if not self._replicated_units:
            return

        # main stream waits for completion of d2h on auxiliary stream.
        for main_stream, d2h_auxiliary_stream in zip(
            self.main_streams.values(),
            self.d2h_auxiliary_streams.values(),
            strict=True,
        ):
            main_stream.wait_for(d2h_auxiliary_stream)

        # Broadcast the block to the other devices on main stream.
        for unit in self._replicated_units:
            root = unit.buffer
            with Tracer("distributed_broadcast"):
                distributed_broadcast(
                    input_buffer=root[dst, :],
                    output_buffers=[
                        root[dst, :],
                        *(p[dst, :] for p in unit.peers),
                    ],
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
