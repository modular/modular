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

"""Correctness tests for the batched H2D/D2H block-offload Mojo FFI kernels.

These exercise ``max._distributed_ops.batched_copy_h2d`` and
``batched_copy_d2h`` -- the block-granular pinned-host <-> device copy
primitives backing :class:`~max.pipelines.kv_cache.paged_kv_cache.block_copy_engine.BlockOffloadEngine`.

The host buffer is laid out as ``[num_blocks, bytes_per_block]`` where each
block row is the concatenation of the per-device-buffer slices, so a block
moved by buffer ``j`` lands at offset ``sum(strides[0..j-1])`` within the row.
The tests verify that mapping for single- and multi-buffer batches, in both
directions, plus the replicated (MLA) H2D broadcast fan-out path.
"""

from __future__ import annotations

import numpy as np
import pytest
from max._distributed_ops import batched_copy_d2h, batched_copy_h2d
from max.driver import (
    Accelerator,
    Buffer,
    DevicePinnedBuffer,
    DeviceStream,
    accelerator_count,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.allreduce import Signals


def _block_pattern(
    num_blocks: int, bytes_per_block: int, salt: int
) -> np.ndarray:
    """Return a deterministic, per-(block, byte) nonzero ``uint8`` pattern.

    Values stay in ``[1, 251]`` so they are never ``0``; untouched device or
    host blocks remain zero-filled, which keeps "not copied" assertions sharp.
    """
    rows = np.arange(num_blocks, dtype=np.uint64)[:, None]
    cols = np.arange(bytes_per_block, dtype=np.uint64)[None, :]
    return ((rows * 131 + cols * 7 + salt) % 251 + 1).astype(np.uint8)


def _ngpu_options() -> list[int]:
    n = accelerator_count()
    return [k for k in (2, 4, 8) if k <= n]


@pytest.fixture(scope="module")
def gpu_devices() -> list[Accelerator]:
    n = accelerator_count()
    if n < 1:
        pytest.skip("block-offload copy tests need at least 1 GPU")
    return [Accelerator(id=i) for i in range(n)]


@pytest.fixture(scope="module")
def gpu(gpu_devices: list[Accelerator]) -> Accelerator:
    return gpu_devices[0]


def test_h2d_single_buffer(gpu: Accelerator) -> None:
    """H2D copies ``host[src] -> device[dst]`` for every pair in one call."""
    num_host_blocks = num_device_blocks = 5
    bytes_per_block = 320
    aux = DeviceStream(gpu)

    # The host source is fully overwritten below, so allocate it without
    # zeroing -- this also avoids racing the async pinned-host zero against
    # the CPU pattern write.
    host_buf = DevicePinnedBuffer(
        dtype=DType.uint8, shape=[num_host_blocks, bytes_per_block], device=gpu
    )
    host_np = host_buf.to_numpy()
    host_pattern = _block_pattern(num_host_blocks, bytes_per_block, salt=11)
    host_np[:] = host_pattern

    # Keep the base buffer referenced; the aux-stream handle aliases its
    # storage and a dropped base could be GC-freed out from under the DMA.
    base = Buffer.zeros(
        shape=[num_device_blocks, bytes_per_block],
        dtype=DType.uint8,
        device=gpu,
    )
    device_buf = base.to(aux)

    # Drain the device destination's async zeroing before the aux-stream copy.
    gpu.synchronize()

    dsts = [0, 2, 4]
    srcs = [1, 3, 0]
    batched_copy_h2d(
        host_buf,
        [device_buf],
        dsts,
        srcs,
        main_streams=[],
        root_and_peer_buffers=[],
        signal_buffers=[],
        broadcast_devices=[],
    )
    aux.synchronize()
    gpu.synchronize()

    got = base.to_numpy()
    for dst, src in zip(dsts, srcs, strict=True):
        np.testing.assert_array_equal(got[dst], host_pattern[src])
    for dst in set(range(num_device_blocks)) - set(dsts):
        np.testing.assert_array_equal(
            got[dst], np.zeros(bytes_per_block, np.uint8)
        )


def test_d2h_single_buffer(gpu: Accelerator) -> None:
    """D2H copies ``device[src] -> host[dst]`` for every pair in one call."""
    num_host_blocks = num_device_blocks = 5
    bytes_per_block = 320
    aux = DeviceStream(gpu)

    device_pattern = _block_pattern(num_device_blocks, bytes_per_block, salt=23)
    # Retain the base device buffer; the aux-stream handle aliases its storage.
    base = Buffer.from_numpy(device_pattern).to(gpu)
    device_buf = base.to(aux)

    host_buf = DevicePinnedBuffer.zeros(
        shape=[num_host_blocks, bytes_per_block], dtype=DType.uint8, device=gpu
    )
    host_np = host_buf.to_numpy()

    # Drain the async device upload before the aux-stream copy reads it.
    gpu.synchronize()

    dsts = [4, 1, 0]
    srcs = [0, 2, 3]
    batched_copy_d2h(host_buf, [device_buf], dsts, srcs)
    aux.synchronize()
    gpu.synchronize()

    for dst, src in zip(dsts, srcs, strict=True):
        np.testing.assert_array_equal(host_np[dst], device_pattern[src])
    for dst in set(range(num_host_blocks)) - set(dsts):
        np.testing.assert_array_equal(
            host_np[dst], np.zeros(bytes_per_block, np.uint8)
        )


def test_h2d_multi_buffer_offsets(gpu: Accelerator) -> None:
    """Each device buffer receives its concatenated slice of the host block."""
    num_host_blocks = num_device_blocks = 4
    stride0, stride1 = 128, 192
    bytes_per_block = stride0 + stride1
    aux = DeviceStream(gpu)

    host_buf = DevicePinnedBuffer(
        dtype=DType.uint8, shape=[num_host_blocks, bytes_per_block], device=gpu
    )
    host_np = host_buf.to_numpy()
    host_pattern = _block_pattern(num_host_blocks, bytes_per_block, salt=7)
    host_np[:] = host_pattern

    # Retain the base buffers; the aux-stream handles alias their storage.
    base0 = Buffer.zeros(
        shape=[num_device_blocks, stride0], dtype=DType.uint8, device=gpu
    )
    base1 = Buffer.zeros(
        shape=[num_device_blocks, stride1], dtype=DType.uint8, device=gpu
    )
    dev0 = base0.to(aux)
    dev1 = base1.to(aux)

    # Drain the device destinations' async zeroing before the aux-stream copy.
    gpu.synchronize()

    dsts = [0, 3]
    srcs = [2, 1]
    batched_copy_h2d(
        host_buf,
        [dev0, dev1],
        dsts,
        srcs,
        main_streams=[],
        root_and_peer_buffers=[],
        signal_buffers=[],
        broadcast_devices=[],
    )
    aux.synchronize()
    gpu.synchronize()

    got0 = base0.to_numpy()
    got1 = base1.to_numpy()
    for dst, src in zip(dsts, srcs, strict=True):
        np.testing.assert_array_equal(got0[dst], host_pattern[src][:stride0])
        np.testing.assert_array_equal(
            got1[dst], host_pattern[src][stride0 : stride0 + stride1]
        )


def test_d2h_multi_buffer_offsets(gpu: Accelerator) -> None:
    """Each device buffer writes into its concatenated slice of the host block."""
    num_host_blocks = num_device_blocks = 4
    stride0, stride1 = 128, 192
    bytes_per_block = stride0 + stride1
    aux = DeviceStream(gpu)

    pattern0 = _block_pattern(num_device_blocks, stride0, salt=3)
    pattern1 = _block_pattern(num_device_blocks, stride1, salt=4)
    # Retain the base buffers; the aux-stream handles alias their storage.
    base0 = Buffer.from_numpy(pattern0).to(gpu)
    base1 = Buffer.from_numpy(pattern1).to(gpu)
    dev0 = base0.to(aux)
    dev1 = base1.to(aux)

    host_buf = DevicePinnedBuffer.zeros(
        shape=[num_host_blocks, bytes_per_block], dtype=DType.uint8, device=gpu
    )
    host_np = host_buf.to_numpy()

    # Drain the async device uploads before the aux-stream copy reads them.
    gpu.synchronize()

    dsts = [1, 2]
    srcs = [3, 0]
    batched_copy_d2h(host_buf, [dev0, dev1], dsts, srcs)
    aux.synchronize()
    gpu.synchronize()

    for dst, src in zip(dsts, srcs, strict=True):
        np.testing.assert_array_equal(host_np[dst][:stride0], pattern0[src])
        np.testing.assert_array_equal(
            host_np[dst][stride0 : stride0 + stride1], pattern1[src]
        )


def test_h2d_d2h_many_units(gpu: Accelerator) -> None:
    """More KV-cache units than GPUs must not overflow the pre-extraction arrays.

    MHA sharding produces one device buffer per unit, and the unit count is not
    bounded by the GPU count. Uses more device buffers than any single host has
    GPUs to guard against a fixed-size per-unit array regression.
    """
    num_units = 12
    num_host_blocks = num_device_blocks = 3
    stride = 64
    bytes_per_block = num_units * stride
    aux = DeviceStream(gpu)

    host_buf = DevicePinnedBuffer(
        dtype=DType.uint8, shape=[num_host_blocks, bytes_per_block], device=gpu
    )
    host_np = host_buf.to_numpy()
    host_pattern = _block_pattern(num_host_blocks, bytes_per_block, salt=13)
    host_np[:] = host_pattern

    # Retain the base buffers; the aux-stream handles alias their storage.
    bases = [
        Buffer.zeros(
            shape=[num_device_blocks, stride], dtype=DType.uint8, device=gpu
        )
        for _ in range(num_units)
    ]
    device_bufs = [base.to(aux) for base in bases]

    # Drain the device destinations' async zeroing before the aux-stream copy.
    gpu.synchronize()

    dsts = [0, 2]
    srcs = [1, 0]
    batched_copy_h2d(
        host_buf,
        device_bufs,
        dsts,
        srcs,
        main_streams=[],
        root_and_peer_buffers=[],
        signal_buffers=[],
        broadcast_devices=[],
    )
    aux.synchronize()
    gpu.synchronize()

    for unit, base in enumerate(bases):
        got = base.to_numpy()
        lo = unit * stride
        for dst, src in zip(dsts, srcs, strict=True):
            np.testing.assert_array_equal(
                got[dst], host_pattern[src][lo : lo + stride]
            )

    # D2H round-trip: copy the just-written device blocks back into a fresh
    # host buffer and confirm the full concatenated block matches.
    out_host = DevicePinnedBuffer.zeros(
        shape=[num_host_blocks, bytes_per_block], dtype=DType.uint8, device=gpu
    )
    out_np = out_host.to_numpy()
    # Drain out_host's async zeroing before the d2h copy writes into it.
    gpu.synchronize()
    batched_copy_d2h(out_host, device_bufs, dsts, dsts)
    aux.synchronize()
    gpu.synchronize()

    for dst, src in zip(dsts, srcs, strict=True):
        np.testing.assert_array_equal(out_np[dst], host_pattern[src])


def test_d2h_empty_is_noop(gpu: Accelerator) -> None:
    """An empty block list returns immediately without touching the buffers."""
    host_buf = DevicePinnedBuffer.zeros(
        shape=[2, 16], dtype=DType.uint8, device=gpu
    )
    gpu.synchronize()

    device_buf = Buffer.zeros(shape=[2, 16], dtype=DType.uint8, device=gpu).to(
        DeviceStream(gpu)
    )

    batched_copy_d2h(host_buf, [device_buf], [], [])

    np.testing.assert_array_equal(
        host_buf.to_numpy(), np.zeros((2, 16), np.uint8)
    )


@pytest.mark.parametrize("ngpus", _ngpu_options())
def test_h2d_replicated_broadcast(
    gpu_devices: list[Accelerator], ngpus: int
) -> None:
    """Replicated H2D writes the root block then fans it out to every peer.

    Mirrors ``BlockOffloadEngine.memcpy_h2d`` for a single replicated (MLA)
    unit: the host block is DMA'd onto the root device, a device-side stream
    wait gates the main stream on that DMA, and a peer broadcast replicates the
    block to all participating GPUs. Every device's block must equal the
    source host block afterwards.
    """
    devices = gpu_devices[:ngpus]
    num_host_blocks = num_device_blocks = 4
    bytes_per_block = 256

    signals = Signals(devices=[DeviceRef.GPU(id=d.id) for d in devices])
    signal_buffers = signals.buffers()

    host_buf = DevicePinnedBuffer(
        dtype=DType.uint8,
        shape=[num_host_blocks, bytes_per_block],
        device=devices[0],
    )
    host_np = host_buf.to_numpy()
    host_pattern = _block_pattern(num_host_blocks, bytes_per_block, salt=5)
    host_np[:] = host_pattern

    # One replicated unit: root buffer on devices[0], peer buffers on the rest.
    unit_buffers = [
        Buffer.zeros(
            shape=[num_device_blocks, bytes_per_block],
            dtype=DType.uint8,
            device=dev,
        )
        for dev in devices
    ]

    # Drain the per-device destination zeroing before the broadcast copy.
    for dev in devices:
        dev.synchronize()

    aux = DeviceStream(devices[0])
    device_bufs_on_aux = [unit_buffers[0].to(aux)]
    main_streams = [devices[0].default_stream]

    dsts = [0, 3]
    srcs = [1, 2]
    batched_copy_h2d(
        host_buf,
        device_bufs_on_aux,
        dsts,
        srcs,
        main_streams=main_streams,
        root_and_peer_buffers=[unit_buffers],
        signal_buffers=signal_buffers,
        broadcast_devices=devices,
    )
    for dev in devices:
        dev.synchronize()

    for rank, buf in enumerate(unit_buffers):
        got = buf.to_numpy()
        for dst, src in zip(dsts, srcs, strict=True):
            np.testing.assert_array_equal(
                got[dst],
                host_pattern[src],
                err_msg=f"rank {rank} block {dst} diverges from host block {src}",
            )
