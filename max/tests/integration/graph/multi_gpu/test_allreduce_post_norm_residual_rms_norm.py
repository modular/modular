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

A captured decode graph holds this op, so it also has to give the same
values when a device graph replays on new inputs, with the generation flag
it keeps on the device carried from one replay to the next.
"""

from __future__ import annotations

from dataclasses import dataclass

import ml_dtypes
import numpy as np
import numpy.typing as npt
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    accelerator_api,
    accelerator_count,
)
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
# 256 rows of 5120 bfloat16 columns are three Lamport launches of at most
# 1 MiB, and one row is one launch, so the replays below wrap the three
# generations several times, and each 1-row replay after a 256-row one clears
# the larger call's leftover slots.
REPLAY_ROWS = (256, 1, 256, 256, 1, 256)
# Where the join's terms cancel, a one-ULP difference in a term of up to about
# 8 is many ULPs of the sum, so the replays are checked against the reference
# in absolute terms. Another step's values differ by O(1) almost everywhere,
# so this gate still fails a replay that reads them.
STEP_ATOL = 2**-2


def _bf16(arr: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    return arr.astype(ml_dtypes.bfloat16).astype(np.float32)


def _to_device(
    arr: npt.NDArray[np.float32], dtype: DType, device: Accelerator
) -> Buffer:
    if dtype == DType.bfloat16:
        bits = np.ascontiguousarray(
            arr.astype(ml_dtypes.bfloat16).view(np.uint16)
        )
        return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)
    return Buffer.from_numpy(np.ascontiguousarray(arr)).to(device)


def _bf16_bits(buf: Buffer) -> npt.NDArray[np.uint16]:
    return buf.copy(device=CPU()).view(DType.uint16).to_numpy()


def _bits_to_f32(bits: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    return bits.view(ml_dtypes.bfloat16).astype(np.float32)


def _from_device_bf16(buf: Buffer) -> npt.NDArray[np.float32]:
    return _bits_to_f32(_bf16_bits(buf))


def _ulps(
    got: npt.NDArray[np.float32], want: npt.NDArray[np.float32]
) -> npt.NDArray[np.int32]:
    def keys(values: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
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


@dataclass(frozen=True)
class _Case:
    """One call's inputs and the outputs the unfused ops give for them."""

    shards: list[npt.NDArray[np.float32]]
    residual: npt.NDArray[np.float32]
    gamma: npt.NDArray[np.float32]
    gamma2: npt.NDArray[np.float32]
    mix: npt.NDArray[np.float32]
    joined: npt.NDArray[np.float32]
    normed: npt.NDArray[np.float32]

    def inputs(self) -> list[tuple[npt.NDArray[np.float32], DType]]:
        """Returns each tensor input's value and dtype, in the graph's order."""
        n = len(self.shards)
        return [
            *((shard, DType.bfloat16) for shard in self.shards),
            *[(self.residual, DType.bfloat16)] * n,
            *[(self.gamma, DType.bfloat16)] * n,
            *[(self.gamma2, DType.bfloat16)] * n,
            *[(self.mix, DType.float32)] * n,
        ]


def _case(
    rng: np.random.Generator, rows: int, mix: npt.NDArray[np.float32]
) -> _Case:
    shards = [
        _bf16(rng.uniform(-0.5, 0.5, (rows, COLS))) for _ in range(NUM_GPUS)
    ]
    residual = _bf16(rng.uniform(-2.0, 2.0, (rows, COLS)))
    gamma = _bf16(rng.uniform(0.5, 1.5, COLS))
    gamma2 = _bf16(rng.uniform(0.5, 1.5, COLS))

    total = np.zeros((rows, COLS), dtype=np.float64)
    for shard in shards:
        total += shard
    norm1 = 1.0 / np.sqrt(np.mean(total * total, axis=-1, keepdims=True) + EPS)
    normed1 = _bf16(total * norm1 * gamma)
    joined = _bf16(_bf16(normed1 * mix[0]) + _bf16(residual * mix[1]))
    norm2 = 1.0 / np.sqrt(
        np.mean(joined.astype(np.float64) ** 2, axis=-1, keepdims=True) + EPS
    )
    normed = _bf16(joined * norm2 * gamma2)
    return _Case(shards, residual, gamma, gamma2, mix, joined, normed)


def _upload(case: _Case, devices: list[Accelerator]) -> list[Buffer]:
    return [
        _to_device(value, dtype, devices[i % NUM_GPUS])
        for i, (value, dtype) in enumerate(case.inputs())
    ]


def _check(outputs: list[Buffer], case: _Case, label: str) -> None:
    for i in range(NUM_GPUS):
        for name, out, want in (
            ("joined", outputs[i], case.joined),
            ("normed", outputs[NUM_GPUS + i], case.normed),
        ):
            ulps = _ulps(_from_device_bf16(out), want)
            assert int(ulps.max()) <= MAX_BF16_ULPS, (
                f"{label} {name} on GPU {i}: max {int(ulps.max())} ULPs"
            )


@pytest.mark.parametrize("rows", [1, 256])
def test_allreduce_post_norm_residual_rms_norm(
    compiled: tuple[Model, Signals, list[Accelerator]], rows: int
) -> None:
    model, signals, devices = compiled
    case = _case(np.random.default_rng(rows), rows, np.array(MIX, np.float32))
    outputs = model.execute(*_upload(case, devices), *signals.buffers())
    _check(outputs, case, f"rows={rows}")


@pytest.mark.skipif(
    accelerator_api() != "cuda",
    reason="Device-graph capture is fully supported on NVIDIA only.",
)
def test_captured_replays_follow_changing_inputs(
    compiled: tuple[Model, Signals, list[Accelerator]],
) -> None:
    """Replays captured 1- and 256-row graphs on fresh inputs each step.

    Each replay must match an eager call on the same inputs bit for bit. The
    generation flag lives in the signal buffers, so both graphs and the eager
    calls share one set, as a serving engine's steps do.
    """
    model, signals, devices = compiled
    rng = np.random.default_rng(0)
    sigs = signals.buffers()
    captured: dict[int, tuple[list[Buffer], list[Buffer]]] = {}
    for rows in sorted(set(REPLAY_ROWS)):
        inputs = _upload(_case(rng, rows, np.array(MIX, np.float32)), devices)
        captured[rows] = (inputs, model.capture(rows, *inputs, *sigs))

    for step, rows in enumerate(REPLAY_ROWS):
        inputs, outputs = captured[rows]
        case = _case(rng, rows, rng.uniform(0.25, 2.0, 2).astype(np.float32))
        fresh = _upload(case, devices)
        for buf, value in zip(inputs, fresh, strict=True):
            buf.inplace_copy_from(value)
        model.replay(rows, *inputs, *sigs)
        replayed = [_bf16_bits(out) for out in outputs]
        # The eager call shares the signal buffers too, as a serving
        # engine's uncaptured steps do, so it also advances the generation.
        eager = model.execute(*fresh, *sigs)
        label = f"replay {step} ({rows} rows)"
        for i, (got, out) in enumerate(zip(replayed, eager, strict=True)):
            np.testing.assert_array_equal(
                got, _bf16_bits(out), err_msg=f"{label} output {i}"
            )
        for i in range(NUM_GPUS):
            for name, got, want in (
                ("joined", replayed[i], case.joined),
                ("normed", replayed[NUM_GPUS + i], case.normed),
            ):
                np.testing.assert_allclose(
                    _bits_to_f32(got),
                    want,
                    rtol=0,
                    atol=STEP_ATOL,
                    err_msg=f"{label} {name} on GPU {i}",
                )
