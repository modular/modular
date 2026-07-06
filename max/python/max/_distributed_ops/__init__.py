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

"""Python entry points for distributed kernels callable outside MAX graphs."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import mojo.importer
from max.driver import (
    Buffer,
    Device,
    DevicePinnedBuffer,
    DeviceStream,
    accelerator_api,
    accelerator_count,
)

if (
    sys.platform == "linux"
    and accelerator_count() > 0
    and accelerator_api() in ("cuda", "hip")
):
    try:
        from .block_offload_ops import (  # type: ignore[import-not-found]
            copy_d2h as _copy_d2h,
        )
        from .block_offload_ops import (
            copy_h2d as _copy_h2d,
        )
    except ImportError:
        _copy_h2d = None
        _copy_d2h = None

    try:
        from .distributed_ops import (  # type: ignore[import-not-found]
            broadcast_kernel as _broadcast_kernel,
        )
    except ImportError:
        _broadcast_kernel = None
else:
    _broadcast_kernel = None
    _copy_h2d = None
    _copy_d2h = None


def distributed_broadcast(
    input_buffer: Buffer,
    output_buffers: Sequence[Buffer],
    signal_buffers: Sequence[Buffer],
    devices: Sequence[Device],
    root: int,
) -> None:
    """Enqueues a broadcast of ``input_buffer`` to every output buffer.

    The transfer is byte-oriented — ``input_buffer.dtype`` only sets the
    payload size in bytes; the kernel itself broadcasts raw bytes.

    Peer access is established when signal buffers are allocated via
    :meth:`max.nn.comm.allreduce.Signals.buffers`; if you allocate signal
    buffers another way you must call
    :func:`max.driver.enable_all_peer_access` yourself.

    This function only enqueues work on each device's stream. The caller is
    responsible for keeping ``input_buffer``, ``output_buffers``, and
    ``signal_buffers`` alive until each participating device's stream has
    been synchronized — the kernel is enqueued, not executed, when this
    returns, so dropping any of them earlier yields a use-after-free in the
    queued work.

    Args:
        input_buffer: Source buffer resident on ``devices[root]``.
        output_buffers: One destination buffer per device, matching
            ``input_buffer`` in shape and dtype; ``output_buffers[i]`` must
            be on ``devices[i]``. ``output_buffers[root]`` may alias
            ``input_buffer`` for an in-place broadcast on the root rank.
        signal_buffers: Per-device ``uint8`` synchronization buffers, one
            per device, sized to :data:`max.nn.comm.allreduce.Signals.NUM_BYTES`.
            Obtain from :meth:`max.nn.comm.allreduce.Signals.buffers`.
        devices: Participating GPUs, indexed by rank.
        root: Index into ``devices`` of the source rank.

    Raises:
        ValueError: If lengths, devices, dtype, or shapes are inconsistent.
    """
    n = len(devices)
    if n < 2:
        raise ValueError(
            f"distributed_broadcast requires at least 2 devices; got {n}"
        )
    if len(output_buffers) != n:
        raise ValueError(
            f"len(output_buffers)={len(output_buffers)} must equal "
            f"len(devices)={n}"
        )
    if len(signal_buffers) != n:
        raise ValueError(
            f"len(signal_buffers)={len(signal_buffers)} must equal "
            f"len(devices)={n}"
        )
    if not (0 <= root < n):
        raise ValueError(f"root={root} out of range [0, {n})")
    if input_buffer.device != devices[root]:
        raise ValueError(
            f"input_buffer.device={input_buffer.device} must equal "
            f"devices[root]={devices[root]}"
        )

    dtype = input_buffer.dtype
    shape = tuple(input_buffer.shape)
    for i, (out, dev) in enumerate(zip(output_buffers, devices, strict=False)):
        if out.device != dev:
            raise ValueError(
                f"output_buffers[{i}].device={out.device} must equal "
                f"devices[{i}]={dev}"
            )
        if tuple(out.shape) != shape:
            raise ValueError(
                f"output_buffers[{i}].shape={tuple(out.shape)} must equal "
                f"input_buffer.shape={shape}"
            )
        if out.dtype != dtype:
            raise ValueError(
                f"output_buffers[{i}].dtype={out.dtype} must equal "
                f"input_buffer.dtype={dtype}"
            )

    if _broadcast_kernel is None:
        raise RuntimeError(
            "distributed_broadcast is unavailable: the broadcast kernel could "
            "not be loaded. This means the host has no GPU toolchain, or the "
            "broadcast kernel does not support this accelerator's GPU "
            "architecture."
        )

    num_bytes = input_buffer.num_elements * dtype.size_in_bytes
    _broadcast_kernel(
        input_buffer._data_ptr(),
        list(output_buffers),
        [b._data_ptr() for b in signal_buffers],
        int(num_bytes),
        n,
        int(root),
    )


def batched_copy_h2d(
    host_buf: DevicePinnedBuffer,
    device_bufs_on_aux_stream: list[Buffer],
    dsts: list[int],
    srcs: list[int],
    *,
    main_streams: list[DeviceStream],
    root_and_peer_buffers: list[list[Buffer]],
    signal_buffers: list[Buffer],
    broadcast_devices: Sequence[Device],
) -> None:
    """Enqueue H2D copies, stream waits, and peer broadcasts for all block pairs.

    A single Mojo call handles the full pipeline for every ``(dst, src)`` pair,
    looping inside Mojo with the GIL released for the entire batch:

    1. **H2D DMA** (aux stream): one async copy per device buffer per pair.
    2. **Stream wait** (device-side): ``main_ctx.enqueue_wait_for(aux_ctx)``
       per unit so the main stream will not advance until the DMA completes.
       Only issued when ``main_streams`` is non-empty.
    3. **Peer broadcast** (main stream, replicated units only): fans each block
       out to all peer devices via the ``comm.broadcast`` kernel.
       Only issued when ``root_and_peer_buffers`` is non-empty.

    The broadcast-related arguments are unconditional; pass empty lists for the
    non-replicated (MHA-only) path rather than ``None``.

    Args:
        host_buf:                 Pinned host buffer ``[num_host_blocks, bytes_per_block]``.
        device_bufs_on_aux_stream: Device buffers, each bound to their device's
                                   aux stream via ``Buffer.to(stream)``.
        dsts:                     Destination device block IDs.
        srcs:                     Source host block IDs.
        main_streams:             Main streams, one per unit, for the stream-wait
                                  barrier. Empty to skip.
        root_and_peer_buffers:    For replicated units: each element is
                                  ``[root_buf, peer_buf_0, ...]``. Empty for
                                  non-replicated (MHA-only) deployments.
        signal_buffers:           Per-device signal buffers for the broadcast.
                                  Empty when ``root_and_peer_buffers`` is empty.
        broadcast_devices:        Broadcast-group devices (root + peers). Empty
                                  when ``root_and_peer_buffers`` is empty.
    """
    if _copy_h2d is None:
        raise RuntimeError(
            "batched_copy_h2d is unavailable: the Mojo extension could not be "
            "loaded. This typically means the host has no GPU toolchain."
        )

    # Stream-wait contexts: main stream for each unit (device-side barrier).
    wait_main_ctx_ptrs = (
        [ms._device_context_ptr() for ms in main_streams]
        if main_streams
        else []
    )

    # Broadcast parameters: empty list signals "no broadcast" to copy_h2d.
    if root_and_peer_buffers:
        bcast_signal_ptrs = [sig._data_ptr() for sig in signal_buffers]
        # [num_units][ngpus]: base data ptr for each output buffer per unit.
        bcast_out_ptrs = [
            [b._data_ptr() for b in unit_bufs]
            for unit_bufs in root_and_peer_buffers
        ]
        bcast_out_strides = [
            unit_bufs[0].shape[1] for unit_bufs in root_and_peer_buffers
        ]
        # Main-stream contexts for all broadcast devices (root + peers).
        bcast_main_ctx_ptrs = [
            dev._device_context_ptr() for dev in broadcast_devices
        ]
    else:
        bcast_signal_ptrs = []
        bcast_out_ptrs = []
        bcast_out_strides = []
        bcast_main_ctx_ptrs = []

    _copy_h2d(
        [host_buf._data_ptr(), host_buf.shape[1]],
        [
            [buf._data_ptr() for buf in device_bufs_on_aux_stream],
            [buf.shape[1] for buf in device_bufs_on_aux_stream],
            # Stream-specific aux context — NOT buf.device._device_context_ptr().
            [
                buf.stream._device_context_ptr()
                for buf in device_bufs_on_aux_stream
            ],
        ],
        dsts,
        srcs,
        wait_main_ctx_ptrs,
        [
            bcast_signal_ptrs,
            bcast_out_ptrs,
            bcast_out_strides,
            bcast_main_ctx_ptrs,
        ]
        if root_and_peer_buffers
        else [],
    )


def batched_copy_d2h(
    host_buf: DevicePinnedBuffer,
    device_bufs_on_aux_stream: list[Buffer],
    dst_ids: Sequence[int],
    src_ids: Sequence[int],
) -> None:
    """Enqueue async D2H copies for all block pairs across all device buffers.

    For each ``(dst, src)`` pair and device buffer ``j``:

        host_buf[dst, offset_j : offset_j + strides[j]] ← device_bufs[j][src, :]

    where ``offset_j = sum(strides[0..j-1])``.  Copies land on each buffer's
    aux stream via its stream-specific DeviceContext.

    Args:
        host_buf:                 Pinned host buffer ``[num_host_blocks, bytes_per_block]``.
        device_bufs_on_aux_stream: Device buffers, each bound to their device's
                                   aux stream via ``Buffer.to(stream)``.
        dst_ids:                  Destination host block IDs.
        src_ids:                  Source device block IDs.
    """
    n = len(dst_ids)
    if n == 0:
        return
    if _copy_d2h is None:
        raise RuntimeError(
            "batched_copy_d2h is unavailable: the Mojo extension could not be "
            "loaded. This typically means the host has no GPU toolchain."
        )

    _copy_d2h(
        host_buf._data_ptr(),
        host_buf.shape[1],
        [buf._data_ptr() for buf in device_bufs_on_aux_stream],
        [buf.shape[1] for buf in device_bufs_on_aux_stream],
        [buf.stream._device_context_ptr() for buf in device_bufs_on_aux_stream],
        dst_ids,
        src_ids,
    )


__all__ = [
    "batched_copy_d2h",
    "batched_copy_h2d",
    "distributed_broadcast",
]
