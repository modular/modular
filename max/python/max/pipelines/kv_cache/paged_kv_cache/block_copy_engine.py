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
import time
from collections.abc import Sequence
from dataclasses import dataclass

from max.driver import (
    Buffer,
    Device,
    DeviceEvent,
    DevicePinnedBuffer,
    DeviceStream,
    _unsafe_alloc_fast_pinned_buffer,
    _unsafe_free_fast_pinned_buffer,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.allreduce import Signals
from max.nn.kv_cache.cache_params import KVCacheMemory, ReplicatedKVCacheMemory
from max.profiler import Tracer, traced

_logger = logging.getLogger("max.pipelines")


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

        # Validate that all units have the same number of pages.
        unique_total_num_pages = {mem.total_num_pages for mem in kv_memory}
        if len(unique_total_num_pages) > 1:
            raise ValueError(
                "all kv_memory units must have the same total_num_pages; got "
                f"{unique_total_num_pages}"
            )

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
        # 2-D [num_host_blocks, bytes_per_page] page-locked host region; row
        # ``bid`` is block ``bid``. Not GC-freed -- close() releases it.
        total_bytes = total_num_host_blocks * bytes_per_page
        total_gib = total_bytes / _GIB
        # Large allocations take minutes; log before so the wait is explained.
        _logger.info(
            (
                "Allocating %.1f GiB pinned host KV cache (this can take"
                " several minutes for large sizes)..."
            ),
            total_gib,
        )
        start = time.perf_counter()
        self.host_buffer: DevicePinnedBuffer = _unsafe_alloc_fast_pinned_buffer(
            DType.uint8,
            [total_num_host_blocks, bytes_per_page],
            gpu0,
        )
        elapsed = time.perf_counter() - start
        _logger.info(
            "Allocated %.1f GiB pinned host KV cache in %.1f s (%.2f GiB/s)",
            total_gib,
            elapsed,
            total_gib / elapsed if elapsed > 0 else float("inf"),
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

        self._closed = False

    def close(self) -> None:
        """Host-synchronize the copy streams and free the host buffer.

        The host buffer is not GC-freed; it must be released explicitly, and
        only once the GPU is done with it. This belongs here, not in a
        destructor: the engine owns the streams that copy into the buffer (a
        destructor knows neither the streams nor when GC runs). Idempotent;
        forgetting to call it leaks (safe), freeing without the sync is a UAF.
        """
        if self._closed:
            return
        self._closed = True
        for stream in self.main_streams.values():
            stream.synchronize()
        for stream in self.d2h_auxiliary_streams.values():
            stream.synchronize()
        _unsafe_free_fast_pinned_buffer(self.host_buffer)

    @traced
    def memcpy_h2d(self, dsts: list[int], srcs: list[int]) -> None:
        """Copies blocks from host to device(s)."""
        if not dsts:
            return

        # h2d on auxiliary stream.
        for dst, src in zip(dsts, srcs, strict=True):
            offset = 0
            for buf in self.device_buffers_on_aux_stream:
                page_bytes = buf.shape[1]
                buf[dst, :].inplace_copy_from(
                    self.host_buffer[src, offset : offset + page_bytes]
                )
                offset += page_bytes

        if not self._replicated_units:
            return

        # Imported lazily: instantiating the GPU broadcast collective at module
        # load compiles it for the active GPU target, which fails on backends
        # without a GPUInfo entry. Only the multi-device replicated path needs
        # it, so defer the import (and its compile) to here.
        from max._distributed_ops import distributed_broadcast

        # main stream waits for completion of h2d on auxiliary stream.
        for main_stream, d2h_auxiliary_stream in zip(
            self.main_streams.values(),
            self.d2h_auxiliary_streams.values(),
            strict=True,
        ):
            main_stream.wait_for(d2h_auxiliary_stream)

        # Broadcast all blocks to the other devices on main stream.
        for dst in dsts:
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
    def memcpy_d2h(self, dsts: list[int], srcs: list[int]) -> None:
        """Copies blocks from device(s) to host."""
        for dst, src in zip(dsts, srcs, strict=True):
            offset = 0
            for buf in self.device_buffers_on_aux_stream:
                page_bytes = buf.shape[1]
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

    @traced
    def record_d2h_event(self) -> DeviceEventBundle:
        """Record an event on all the d2h auxiliary streams."""
        return DeviceEventBundle.record_on_streams(
            list(self.d2h_auxiliary_streams.values())
        )
