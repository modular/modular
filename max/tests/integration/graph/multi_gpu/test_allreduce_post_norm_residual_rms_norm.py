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
"""Multi-GPU wiring test for the fused allreduce + post-norm residual join op.

Checks that `ops.allreduce_post_norm_residual_rms_norm` hands its operands to
the kernel in order, against a numpy reference of the unfused ops: the peer
sum normed with the first weight, each of it and the residual weighed and
rounded to bfloat16 on its own, their bfloat16 sum, and that normed with the
second weight. The two weights differ, the mix weights are not 1 and the
residual is not small, so swapping operands or dropping a mix weight lands far
outside the gate. One row runs as one launch and 256 rows as several row
chunks; the kernel's numerics across shapes are
`test_lamport_post_norm_residual_rmsnorm.mojo`'s.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.legacy import Signals

NUM_GPUS = 2
COLS = 5120  # Beam-1.0's hidden size.
EPS = 1e-6
MIX = (0.37, 1.61)
# One bfloat16 rounding per kernel rounding; the f32 pipeline sits a
# rounding-boundary flip from the fp64 reference at most.
MAX_BF16_ULPS = 2


def _bf16(arr: np.ndarray) -> np.ndarray:
    return arr.astype(ml_dtypes.bfloat16).astype(np.float32)


def _to_device(arr: np.ndarray, dtype: DType, device: Accelerator) -> Buffer:
    if dtype == DType.bfloat16:
        bits = np.ascontiguousarray(
            arr.astype(ml_dtypes.bfloat16).view(np.uint16)
        )
        return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)
    return Buffer.from_numpy(np.ascontiguousarray(arr)).to(device)


def _from_device_bf16(buf: Buffer) -> np.ndarray:
    out = buf.copy(device=CPU())
    return (
        out.view(DType.uint16)
        .to_numpy()
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    )


def _ulps(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    def keys(values: np.ndarray) -> np.ndarray:
        bits = (
            values.astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.int32)
        )
        return np.where(bits & 0x8000, 0xFFFF - (bits & 0xFFFF), bits | 0x8000)

    return np.abs(keys(got) - keys(want))


def _graph(signals: Signals) -> Graph:
    devices = signals.devices
    row_type = lambda d: TensorType(DType.bfloat16, ["rows", COLS], device=d)
    col_type = lambda d: TensorType(DType.bfloat16, [COLS], device=d)
    mix_type = lambda d: TensorType(DType.float32, [2], device=d)
    input_types: list[TensorType | BufferType] = [
        *(row_type(d) for d in devices),
        *(row_type(d) for d in devices),
        *(col_type(d) for d in devices),
        *(col_type(d) for d in devices),
        *(mix_type(d) for d in devices),
        *signals.input_types(),
    ]
    with Graph(
        "allreduce_post_norm_residual_rms_norm", input_types=input_types
    ) as graph:
        values = list(graph.inputs)
        n = NUM_GPUS
        groups = [
            [v.tensor for v in values[i * n : (i + 1) * n]] for i in range(5)
        ]
        sigs = [v.buffer for v in values[5 * n :]]
        joined, normed = ops.allreduce_post_norm_residual_rms_norm(
            groups[0], sigs, groups[1], groups[2], groups[3], groups[4], EPS
        )
        graph.output(*joined, *normed)
    return graph


@pytest.fixture(scope="module")
def compiled() -> tuple[Model, Signals, list[Accelerator]]:
    if NUM_GPUS > accelerator_count():
        pytest.skip(f"Not enough GPUs ({NUM_GPUS}).")
    devices = [Accelerator(i) for i in range(NUM_GPUS)]
    signals = Signals(devices=[DeviceRef.GPU(i) for i in range(NUM_GPUS)])
    session = InferenceSession(devices=devices)
    return session.load(_graph(signals)), signals, devices


@pytest.mark.parametrize("rows", [1, 256])
def test_allreduce_post_norm_residual_rms_norm(
    compiled: tuple[Model, Signals, list[Accelerator]], rows: int
) -> None:
    model, signals, devices = compiled
    rng = np.random.default_rng(rows)
    shards = [_bf16(rng.uniform(-0.5, 0.5, (rows, COLS))) for _ in devices]
    residual = _bf16(rng.uniform(-2.0, 2.0, (rows, COLS)))
    gamma = _bf16(rng.uniform(0.5, 1.5, COLS))
    gamma2 = _bf16(rng.uniform(0.5, 1.5, COLS))
    mix = np.array(MIX, dtype=np.float32)

    total = np.zeros((rows, COLS), dtype=np.float64)
    for shard in shards:
        total += shard
    norm1 = 1.0 / np.sqrt(np.mean(total * total, axis=-1, keepdims=True) + EPS)
    normed1 = _bf16(total * norm1 * gamma)
    joined = _bf16(_bf16(normed1 * mix[0]) + _bf16(residual * mix[1]))
    norm2 = 1.0 / np.sqrt(
        np.mean(joined.astype(np.float64) ** 2, axis=-1, keepdims=True) + EPS
    )
    want_normed = _bf16(joined * norm2 * gamma2)

    args = [
        *(
            _to_device(s, DType.bfloat16, d)
            for s, d in zip(shards, devices, strict=True)
        ),
        *(_to_device(residual, DType.bfloat16, d) for d in devices),
        *(_to_device(gamma, DType.bfloat16, d) for d in devices),
        *(_to_device(gamma2, DType.bfloat16, d) for d in devices),
        *(_to_device(mix, DType.float32, d) for d in devices),
        *signals.buffers(),
    ]
    outputs = model.execute(*args)
    for i in range(NUM_GPUS):
        for name, out, want in (
            ("joined", outputs[i], joined),
            ("normed", outputs[NUM_GPUS + i], want_normed),
        ):
            assert isinstance(out, Buffer)
            got = _from_device_bf16(out)
            ulps = _ulps(got, want)
            assert int(ulps.max()) <= MAX_BF16_ULPS, (
                f"rows={rows} {name} on GPU {i}: max {int(ulps.max())} ULPs"
            )
