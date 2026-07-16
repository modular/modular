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

from max._distributed_ops import batched_copy_d2h, batched_copy_h2d
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


@dataclass
class _ReplicaOffloadState:
    """Per-replica device endpoints and copy streams for the offload engine."""

    device_buffers: list[Buffer]
    device_buffers_on_aux_stream: list[Buffer]
    main_streams: dict[int, DeviceStream]
    d2h_auxiliary_streams: dict[int, DeviceStream]
    replicated_units: list[ReplicatedKVCacheMemory]
    broadcast_devices: list[Device]
    signals: Signals | None
    signal_buffers: list[Buffer]


class BlockOffloadEngine:
    """Engine for offloading gpu KVCache blocks to host memory.

    This offload engine allocates a single ``DevicePinnedBuffer`` shared by
    every data-parallel (DP) replica and uses auxiliary d2h streams to hide the
    latency of KV cache offloading copies on a stream detached from the main
    kernel exec stream. However, it still issues the h2d transfers on the same
    stream as kernel execution which is a major limitation (SERVOPT-1036).

    The host buffer is replica-agnostic: it is keyed purely by host block id,
    so a block offloaded by one replica can be loaded back onto a *different*
    replica's device (SERVOPT-1501). ``memcpy_h2d`` / ``memcpy_d2h`` take a
    ``replica_idx`` selecting which replica's device buffers participate.

    For replicated KV caches (MLA), the host buffer holds a single replica per
    logical group. D2H copies from rank 0, then H2D fans back out to all peers
    via a broadcast. For sharded caches (MHA), every shard is its own unit.
    """

    def __init__(
        self,
        total_num_host_blocks: int,
        replica_kv_memory: Sequence[Sequence[KVCacheMemory]],
    ) -> None:
        if len(replica_kv_memory) < 1:
            raise ValueError("BlockOffloadEngine requires at least one replica")

        gpu0 = replica_kv_memory[0][0].buffer.device
        if gpu0.is_host:
            raise ValueError(
                "KVCacheMemory is on the CPU. Unable to allocate host"
                " offload buffer for already-on-CPU buffers."
            )

        self._num_replicas = len(replica_kv_memory)

        # ``bytes_per_page`` must match across replicas so a host block written
        # by one replica is layout-compatible when read back by another.
        bytes_per_page_per_replica = {
            sum(unit.buffer.shape[1] for unit in units)
            for units in replica_kv_memory
        }
        if len(bytes_per_page_per_replica) > 1:
            raise ValueError(
                "all replicas must have the same bytes-per-page; got "
                f"{bytes_per_page_per_replica}"
            )
        bytes_per_page = next(iter(bytes_per_page_per_replica))

        self._replicas: list[_ReplicaOffloadState] = [
            self._build_replica_state(units) for units in replica_kv_memory
        ]

        # 2-D [num_host_blocks, bytes_per_page] page-locked host region shared
        # by all replicas; row ``bid`` is block ``bid``. Not GC-freed --
        # close() releases it.
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

        self._closed = False

    @staticmethod
    def _build_replica_state(
        units: Sequence[KVCacheMemory],
    ) -> _ReplicaOffloadState:
        replicated_units = [
            u for u in units if isinstance(u, ReplicatedKVCacheMemory)
        ]

        # Validate that all units have the same number of pages.
        unique_total_num_pages = {mem.total_num_pages for mem in units}
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
            for unit in replicated_units
        }
        if len(unique_topologies) > 1:
            raise ValueError(
                "all replicated KVCacheMemory units must share the same "
                "TP device topology; mixed topologies are not supported"
            )

        # Broadcast devices: rank-0 + peers from the first replicated unit
        # (topology uniformity was validated above).
        broadcast_devices: list[Device] = (
            [
                replicated_units[0].buffer.device,
                *(p.device for p in replicated_units[0].peers),
            ]
            if replicated_units
            else []
        )

        # The D2H/H2D endpoints — one per unit (rank-0 for replicated units).
        device_buffers: list[Buffer] = [u.buffer for u in units]
        main_streams: dict[int, DeviceStream] = {
            buffer.device.id: buffer.device.default_stream
            for buffer in device_buffers
        }
        d2h_auxiliary_streams: dict[int, DeviceStream] = {
            buffer.device.id: DeviceStream(buffer.device)
            for buffer in device_buffers
        }
        device_buffers_on_aux_stream: list[Buffer] = [
            buffer.to(d2h_auxiliary_streams[buffer.device.id])
            for buffer in device_buffers
        ]

        signals: Signals | None = None
        signal_buffers: list[Buffer] = []
        if replicated_units:
            signals = Signals(
                devices=[DeviceRef.GPU(id=d.id) for d in broadcast_devices]
            )
            signal_buffers = signals.buffers()

        return _ReplicaOffloadState(
            device_buffers=device_buffers,
            device_buffers_on_aux_stream=device_buffers_on_aux_stream,
            main_streams=main_streams,
            d2h_auxiliary_streams=d2h_auxiliary_streams,
            replicated_units=replicated_units,
            broadcast_devices=broadcast_devices,
            signals=signals,
            signal_buffers=signal_buffers,
        )

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
        for replica in self._replicas:
            for stream in replica.main_streams.values():
                stream.synchronize()
            for stream in replica.d2h_auxiliary_streams.values():
                stream.synchronize()
        _unsafe_free_fast_pinned_buffer(self.host_buffer)

    def memcpy_h2d(
        self, dsts: list[int], srcs: list[int], replica_idx: int = 0
    ) -> None:
        """Copies blocks from host into ``replica_idx``'s device(s)."""
        if not dsts:
            return

        replica = self._replicas[replica_idx]

        if replica.replicated_units:
            root_and_peer_buffers = [
                [unit.buffer, *unit.peers] for unit in replica.replicated_units
            ]
            main_streams = list(replica.main_streams.values())
        else:
            root_and_peer_buffers = []
            main_streams = []

        with Tracer(f"memcpy_h2d of {len(dsts)} blocks"):
            batched_copy_h2d(
                self.host_buffer,
                replica.device_buffers_on_aux_stream,
                dsts,
                srcs,
                main_streams=main_streams,
                root_and_peer_buffers=root_and_peer_buffers,
                signal_buffers=replica.signal_buffers,
                broadcast_devices=replica.broadcast_devices,
            )

    def memcpy_d2h(
        self, dsts: list[int], srcs: list[int], replica_idx: int = 0
    ) -> None:
        """Copies blocks from ``replica_idx``'s device(s) to host."""
        if not dsts:
            return

        replica = self._replicas[replica_idx]

        with Tracer(f"memcpy_d2h of {len(dsts)} blocks"):
            batched_copy_d2h(
                self.host_buffer,
                replica.device_buffers_on_aux_stream,
                dsts,
                srcs,
            )

    @traced
    def wait_for_completion(self) -> None:
        """Synchronize main streams with the auxiliary streams (all replicas).

        This ensures that the d2h copies from BatchN completes before
        BatchN+1 begins. This is needed because BatchN+1 may write to the
        same blocks as BatchN is reading from.

        Additionally, ensure that d2h offload of BatchN starts after BatchN
        completes. As such this needs to be a duplex sync.
        """
        for replica in self._replicas:
            for main_stream, d2h_auxiliary_stream in zip(
                replica.main_streams.values(),
                replica.d2h_auxiliary_streams.values(),
                strict=True,
            ):
                main_stream.wait_for(d2h_auxiliary_stream)
                d2h_auxiliary_stream.wait_for(main_stream)

    @traced
    def record_d2h_event(self, replica_idx: int = 0) -> DeviceEventBundle:
        """Record an event on ``replica_idx``'s d2h auxiliary streams."""
        return DeviceEventBundle.record_on_streams(
            list(self._replicas[replica_idx].d2h_auxiliary_streams.values())
        )
