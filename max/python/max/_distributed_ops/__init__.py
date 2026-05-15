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
from max.driver import Buffer, Device, accelerator_count

# The mojo source comptime-instantiates a GPU kernel, which fails to JIT on
# hosts without a GPU toolchain.
if sys.platform == "linux" and accelerator_count() > 0:
    from .distributed_ops import (  # type: ignore[import-not-found]
        broadcast_kernel as _broadcast_kernel,
    )
else:
    _broadcast_kernel = None


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
            "distributed_broadcast is unavailable: the mojo extension could "
            "not be loaded. This typically means the host has no GPU toolchain."
        )

    num_bytes = input_buffer.num_elements * dtype.size_in_bytes
    _broadcast_kernel(
        input_buffer._data_ptr(),
        [b._data_ptr() for b in output_buffers],
        [b._data_ptr() for b in signal_buffers],
        [d._device_context_ptr() for d in devices],
        int(num_bytes),
        n,
        int(root),
    )


__all__ = ["distributed_broadcast"]
