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

from __future__ import annotations

import numpy as np
import pytest
from max._distributed_ops import distributed_broadcast
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.allreduce import Signals

_NUMPY_DTYPE = {
    DType.uint8: np.uint8,
    DType.bfloat16: None,  # numpy has no bf16; fill via uint16 view.
    DType.float16: np.float16,
    DType.float32: np.float32,
}


def _ngpu_options() -> list[int]:
    n = accelerator_count()
    return [k for k in (2, 4, 8) if k <= n]


def _make_pattern(num_elements: int, dtype: DType, root: int) -> np.ndarray:
    if dtype == DType.uint8:
        return (
            (np.arange(num_elements, dtype=np.uint64) + root + 1) % 251
        ).astype(np.uint8)
    if dtype == DType.bfloat16:
        # Mask to a finite positive bf16 bit pattern (clears the sign bit and
        # caps the exponent below NaN/Inf).
        return (
            (np.arange(num_elements, dtype=np.uint64) + root + 1) & 0x3F7F
        ).astype(np.uint16)
    np_dtype = _NUMPY_DTYPE[dtype]
    return ((np.arange(num_elements, dtype=np.uint64) + root + 1) % 251).astype(
        np_dtype
    )


def _allocate_filled(
    pattern: np.ndarray, dtype: DType, device: Accelerator
) -> Buffer:
    host = Buffer.from_numpy(pattern)
    if dtype == DType.bfloat16:
        host = host.view(dtype=DType.bfloat16, shape=list(pattern.shape))
    return host.to(device)


def _to_uint_bytes(buf: Buffer, length: int) -> np.ndarray:
    item_bytes = buf.dtype.size_in_bytes
    return buf.view(dtype=DType.uint8, shape=[length * item_bytes]).to_numpy()


@pytest.fixture(scope="module")
def _gpu_devices() -> list[Accelerator]:
    n = accelerator_count()
    if n < 2:
        pytest.skip(f"distributed_broadcast needs >=2 GPUs, found {n}")
    return [Accelerator(id=i) for i in range(n)]


@pytest.fixture(scope="module")
def _signal_buffers(_gpu_devices: list[Accelerator]) -> list[Buffer]:
    refs = [DeviceRef.GPU(id=d.id) for d in _gpu_devices]
    return Signals(devices=refs).buffers()


def _ngpu_root_options() -> list[tuple[int, int]]:
    """Return (ngpus, root) pairs covering both the first and last rank."""
    return [(n, r) for n in _ngpu_options() for r in {0, n - 1}]


@pytest.mark.parametrize(("ngpus", "root"), _ngpu_root_options())
@pytest.mark.parametrize(
    "dtype", [DType.uint8, DType.bfloat16, DType.float16, DType.float32]
)
@pytest.mark.parametrize("length", [1, 1024, 1 << 16])
def test_broadcast_correctness(
    _gpu_devices: list[Accelerator],
    _signal_buffers: list[Buffer],
    ngpus: int,
    root: int,
    dtype: DType,
    length: int,
) -> None:
    devices = _gpu_devices[:ngpus]
    signals = _signal_buffers[:ngpus]

    pattern = _make_pattern(length, dtype, root)
    in_buf = _allocate_filled(pattern, dtype, devices[root])

    out_bufs = [
        Buffer.zeros(shape=in_buf.shape, dtype=dtype, device=dev)
        for dev in devices
    ]

    distributed_broadcast(
        input_buffer=in_buf,
        output_buffers=out_bufs,
        signal_buffers=signals,
        devices=devices,
        root=root,
    )
    for dev in devices:
        dev.synchronize()

    expected_bytes = _to_uint_bytes(in_buf, length)
    for i, out in enumerate(out_bufs):
        got = _to_uint_bytes(out, length)
        assert np.array_equal(got, expected_bytes), (
            f"rank {i}: broadcast result diverges from root pattern"
        )


def test_rejects_unsupported_ngpus(
    _gpu_devices: list[Accelerator], _signal_buffers: list[Buffer]
) -> None:
    # ``ngpus=1`` is below the kernel's ``ngpus >= 2`` requirement.
    devices = _gpu_devices[:1]
    signals = _signal_buffers[:1]
    in_buf = Buffer.zeros(shape=[8], dtype=DType.uint8, device=devices[0])
    out_bufs = [Buffer.zeros(shape=[8], dtype=DType.uint8, device=devices[0])]
    with pytest.raises(ValueError, match="at least 2 devices"):
        distributed_broadcast(
            input_buffer=in_buf,
            output_buffers=out_bufs,
            signal_buffers=signals,
            devices=devices,
            root=0,
        )


def test_rejects_bad_root(
    _gpu_devices: list[Accelerator], _signal_buffers: list[Buffer]
) -> None:
    devices = _gpu_devices[:2]
    signals = _signal_buffers[:2]
    in_buf = Buffer.zeros(shape=[8], dtype=DType.uint8, device=devices[0])
    out_bufs = [
        Buffer.zeros(shape=[8], dtype=DType.uint8, device=d) for d in devices
    ]
    with pytest.raises(ValueError, match="root"):
        distributed_broadcast(
            input_buffer=in_buf,
            output_buffers=out_bufs,
            signal_buffers=signals,
            devices=devices,
            root=2,
        )


def test_rejects_input_on_non_root(
    _gpu_devices: list[Accelerator], _signal_buffers: list[Buffer]
) -> None:
    devices = _gpu_devices[:2]
    signals = _signal_buffers[:2]
    # Input lives on device 1 but root=0 says the source is device 0.
    in_buf = Buffer.zeros(shape=[8], dtype=DType.uint8, device=devices[1])
    out_bufs = [
        Buffer.zeros(shape=[8], dtype=DType.uint8, device=d) for d in devices
    ]
    with pytest.raises(ValueError, match="devices\\[root\\]"):
        distributed_broadcast(
            input_buffer=in_buf,
            output_buffers=out_bufs,
            signal_buffers=signals,
            devices=devices,
            root=0,
        )


def test_rejects_dtype_mismatch(
    _gpu_devices: list[Accelerator], _signal_buffers: list[Buffer]
) -> None:
    devices = _gpu_devices[:2]
    signals = _signal_buffers[:2]
    in_buf = Buffer.zeros(shape=[8], dtype=DType.uint8, device=devices[0])
    out_bufs = [
        Buffer.zeros(shape=[8], dtype=DType.uint8, device=devices[0]),
        Buffer.zeros(shape=[8], dtype=DType.float32, device=devices[1]),
    ]
    with pytest.raises(ValueError, match="dtype"):
        distributed_broadcast(
            input_buffer=in_buf,
            output_buffers=out_bufs,
            signal_buffers=signals,
            devices=devices,
            root=0,
        )


def test_rejects_output_buffers_length_mismatch(
    _gpu_devices: list[Accelerator], _signal_buffers: list[Buffer]
) -> None:
    devices = _gpu_devices[:2]
    signals = _signal_buffers[:2]
    in_buf = Buffer.zeros(shape=[8], dtype=DType.uint8, device=devices[0])
    out_bufs = [
        Buffer.zeros(shape=[8], dtype=DType.uint8, device=devices[0])
    ]  # only one output, but two devices
    with pytest.raises(ValueError, match="output_buffers"):
        distributed_broadcast(
            input_buffer=in_buf,
            output_buffers=out_bufs,
            signal_buffers=signals,
            devices=devices,
            root=0,
        )


def test_rejects_signal_buffers_length_mismatch(
    _gpu_devices: list[Accelerator], _signal_buffers: list[Buffer]
) -> None:
    devices = _gpu_devices[:2]
    in_buf = Buffer.zeros(shape=[8], dtype=DType.uint8, device=devices[0])
    out_bufs = [
        Buffer.zeros(shape=[8], dtype=DType.uint8, device=d) for d in devices
    ]
    with pytest.raises(ValueError, match="signal_buffers"):
        distributed_broadcast(
            input_buffer=in_buf,
            output_buffers=out_bufs,
            signal_buffers=_signal_buffers[:1],  # only one signal buffer
            devices=devices,
            root=0,
        )
