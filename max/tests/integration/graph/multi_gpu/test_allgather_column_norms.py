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

"""The column-gather norm collectives against the ops they fuse.

``allgather_rms_norm`` over column shards with a rounded ``1 + w`` gamma is
compared with a column all-gather followed by ``rms_norm`` of ``1 + w``, and
``allgather_gated_residual_rms_norm`` with a column all-gather followed by
``gated_residual_rms_norm``. The fused ops reduce each row in a different
order, so outputs may differ by an ulp at a rounding boundary; a wrong gamma
rounding, a skipped scale or a misplaced shard changes far more.
"""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.legacy import Signals
from test_common.graph_utils import is_b100_b200

NUM_GPUS = 2
EPS = 1e-5
POST_EPS = 1e-6
TEMPERATURE = 2.0
GATE_EPS = 1e-3
# Decode row counts, up to the sharded path's token limit.
ROWS = [1, 7, 64, 256]
# Elements a fused output may differ from the unfused ops in.
MISMATCH_BAR = 1e-2
# Largest difference, in units of the row's RMS: two bf16 ulps at that
# magnitude.
ROW_RMS_BAR = 2.0**-6
# The gated boundary's: two bf16 ulps at four times that magnitude, where a
# rounding flip lands on an outlier element. Its faults change most elements.
GATED_ROW_RMS_BAR = 2.0**-5

pytestmark = pytest.mark.skipif(
    accelerator_count() < NUM_GPUS, reason=f"needs {NUM_GPUS} GPUs"
)
RNG = np.random.default_rng(0)


def _bf16(arr: np.ndarray) -> np.ndarray:
    return arr.astype(ml_dtypes.bfloat16)


def _to_device(arr: np.ndarray, device: Accelerator) -> Buffer:
    bits = np.ascontiguousarray(_bf16(arr).view(np.uint16))
    return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)


def _from_device(buf: Any) -> np.ndarray:
    out = buf.copy(device=CPU()).view(DType.uint16).to_numpy()
    return out.view(ml_dtypes.bfloat16).astype(np.float32)


def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def ordered(x: np.ndarray) -> np.ndarray:
        bits = _bf16(x).view(np.uint16).astype(np.int32)
        return np.where(bits & 0x8000, 0x8000 - bits, bits)

    return np.abs(ordered(a) - ordered(b))


def _check(
    name: str, out: np.ndarray, ref: np.ndarray, bar: float = ROW_RMS_BAR
) -> None:
    ulps = _ulps(out, ref)
    frac = float((ulps > 0).mean())
    rms = np.sqrt(np.mean(ref.astype(np.float64) ** 2, axis=-1, keepdims=True))
    worst = float((np.abs(out - ref) / rms).max())
    msg = (
        f"{name}: {frac:.2e} differ, max {int(ulps.max())} ulp, "
        f"max {worst:.2e} row RMS"
    )
    print(msg)
    assert np.isfinite(out).all(), name
    assert frac <= MISMATCH_BAR and worst <= bar, msg


def _session(devices: list[Accelerator]) -> InferenceSession:
    return InferenceSession(devices=[CPU(), *devices])


def _run(model: Model, *inputs: Buffer) -> list[Any]:
    return list(model.execute(*inputs))


def _per_device(
    shape: list[Any], dtype: DType = DType.bfloat16
) -> list[TensorType]:
    return [
        TensorType(dtype, shape, device=DeviceRef.GPU(i))
        for i in range(NUM_GPUS)
    ]


def _centered_norm_graphs(
    cols: int, round_offset_gamma: bool
) -> tuple[Graph, Graph, Signals, Signals]:
    shard = _per_device(["rows", cols // NUM_GPUS])
    gamma = _per_device([cols])
    fused_sig = Signals(devices=[DeviceRef.GPU(i) for i in range(NUM_GPUS)])
    with Graph(
        "fused", input_types=[*shard, *gamma, *fused_sig.input_types()]
    ) as fused:
        xs = [v.tensor for v in fused.inputs[:NUM_GPUS]]
        ws = [v.tensor for v in fused.inputs[NUM_GPUS : 2 * NUM_GPUS]]
        sigs = [v.buffer for v in fused.inputs[2 * NUM_GPUS :]]
        normed, gathered = ops.allgather_rms_norm(
            xs,
            sigs,
            ws,
            EPS,
            weight_offset=1.0,
            axis=1,
            round_offset_gamma=round_offset_gamma,
        )
        fused.output(*normed, *gathered)
    ref_sig = Signals(devices=[DeviceRef.GPU(i) for i in range(NUM_GPUS)])
    with Graph(
        "unfused", input_types=[*shard, *gamma, *ref_sig.input_types()]
    ) as unfused:
        xs = [v.tensor for v in unfused.inputs[:NUM_GPUS]]
        ws = [v.tensor for v in unfused.inputs[NUM_GPUS : 2 * NUM_GPUS]]
        sigs = [v.buffer for v in unfused.inputs[2 * NUM_GPUS :]]
        gathered = ops.allgather(xs, sigs, axis=1)
        # CenteredRMSNorm: 1 + w rounded to bf16, then the f32 multiply.
        normed = [
            ops.rms_norm(g, w + 1.0, EPS, multiply_before_cast=True)
            for g, w in zip(gathered, ws, strict=True)
        ]
        unfused.output(*normed, *gathered)
    return fused, unfused, fused_sig, ref_sig


@pytest.mark.parametrize("rows", ROWS)
def test_column_allgather_centered_norm(rows: int) -> None:
    """The pooled-projection gather: 2 x 2048 columns of a 4096-wide row."""
    cols = 4096
    devices = [Accelerator(i) for i in range(NUM_GPUS)]
    fused, unfused, fused_sig, ref_sig = _centered_norm_graphs(cols, True)
    fault, _, fault_sig, _ = _centered_norm_graphs(cols, False)
    session = _session(devices)
    unfused_model, fused_model, fault_model = [
        session.load(graph) for graph in (unfused, fused, fault)
    ]
    x = RNG.standard_normal((rows, cols)).astype(np.float32)
    w = (RNG.standard_normal(cols) * 0.2).astype(np.float32)
    shards = np.split(x, NUM_GPUS, axis=1)
    inputs = [
        *(_to_device(s, d) for s, d in zip(shards, devices, strict=True)),
        *(_to_device(w, d) for d in devices),
    ]
    ref = _run(unfused_model, *inputs, *ref_sig.buffers())
    out = _run(fused_model, *inputs, *fused_sig.buffers())
    for i in range(NUM_GPUS):
        gathered = _from_device(out[NUM_GPUS + i])
        np.testing.assert_array_equal(gathered, _bf16(x).astype(np.float32))
        _check(
            f"centered norm {rows} rows, GPU {i}",
            _from_device(out[i]),
            _from_device(ref[i]),
        )
    # Planted fault: 1 + w kept in f32 (the op's default) is a different norm.
    bad = _run(fault_model, *inputs, *fault_sig.buffers())
    frac = float((_ulps(_from_device(bad[0]), _from_device(ref[0])) > 0).mean())
    print(f"f32 offset {rows} rows: {frac:.2e} differ")
    assert frac > MISMATCH_BAR, f"an f32 gamma offset passes: {frac:.2e}"


def _boundary_graphs(
    cols: int, scale_branch: bool, fused_temperature: float
) -> tuple[Graph, Graph, Signals, Signals]:
    shard = _per_device(["rows", cols // NUM_GPUS])
    full = _per_device(["rows", cols])
    weights = _per_device([cols]) * 3
    types = [*shard, *full, *weights]

    def unpack(graph: Graph) -> tuple[list[TensorValue], ...]:
        values = [v.tensor for v in graph.inputs[: 5 * NUM_GPUS]]
        return tuple(
            values[k * NUM_GPUS : (k + 1) * NUM_GPUS] for k in range(5)
        )

    fused_sig = Signals(devices=[DeviceRef.GPU(i) for i in range(NUM_GPUS)])
    with Graph(
        "fused", input_types=[*types, *fused_sig.input_types()]
    ) as fused:
        branch, residual, post_w, gate, norm_w = unpack(fused)
        stream, normed = ops.allgather_gated_residual_rms_norm(
            branch,
            [v.buffer for v in fused.inputs[5 * NUM_GPUS :]],
            residual,
            post_w,
            gate,
            norm_w,
            POST_EPS,
            EPS,
            fused_temperature,
            GATE_EPS,
            scale_branch,
        )
        fused.output(*stream, *normed)
    ref_sig = Signals(devices=[DeviceRef.GPU(i) for i in range(NUM_GPUS)])
    with Graph(
        "unfused", input_types=[*types, *ref_sig.input_types()]
    ) as unfused:
        branch, residual, post_w, gate, norm_w = unpack(unfused)
        gathered = ops.allgather(
            branch, [v.buffer for v in unfused.inputs[5 * NUM_GPUS :]], axis=1
        )
        outs = [
            ops.custom(
                "gated_residual_rms_norm",
                device=b.device,
                values=[
                    b,
                    r,
                    p,
                    g,
                    n,
                    *(
                        ops.constant(v, DType.float32, device=DeviceRef.CPU())
                        for v in (POST_EPS, EPS, TEMPERATURE, GATE_EPS)
                    ),
                ],
                out_types=[b.type, b.type],
                parameters={"scale_branch": True},
            )
            for b, r, p, g, n in zip(
                gathered, residual, post_w, gate, norm_w, strict=True
            )
        ]
        unfused.output(
            *(o[0].tensor for o in outs), *(o[1].tensor for o in outs)
        )
    return fused, unfused, fused_sig, ref_sig


@pytest.mark.skipif(
    accelerator_count() < NUM_GPUS or not is_b100_b200(),
    reason="gated_residual_rms_norm runs on B200 only",
)
@pytest.mark.parametrize("rows", ROWS)
def test_column_allgather_gated_residual_norm(rows: int) -> None:
    """The unpooled-projection boundary: 2 x 4096 columns of an 8192 row."""
    cols = 8192
    devices = [Accelerator(i) for i in range(NUM_GPUS)]
    fused, unfused, fused_sig, ref_sig = _boundary_graphs(
        cols, True, TEMPERATURE
    )
    session = _session(devices)
    unfused_model, fused_model = [
        session.load(graph) for graph in (unfused, fused)
    ]
    # Planted faults: an unscaled branch, a wrong gate temperature.
    faults = []
    for label, scale, temperature in (
        ("unscaled branch", False, TEMPERATURE),
        ("temperature 1", True, 1.0),
    ):
        fault, _, fault_sig, _ = _boundary_graphs(cols, scale, temperature)
        faults.append((label, session.load(fault), fault_sig))
    branch = RNG.standard_normal((rows, cols)).astype(np.float32)
    residual = RNG.standard_normal((rows, cols)).astype(np.float32)
    weights = [
        (RNG.standard_normal(cols) * std).astype(np.float32)
        for std in (0.2, 0.5, 0.2)
    ]
    inputs = [
        *(
            _to_device(s, d)
            for s, d in zip(
                np.split(branch, NUM_GPUS, axis=1), devices, strict=True
            )
        ),
        *(_to_device(residual, d) for d in devices),
        *(_to_device(w, d) for w in weights for d in devices),
    ]
    ref = _run(unfused_model, *inputs, *ref_sig.buffers())
    out = _run(fused_model, *inputs, *fused_sig.buffers())
    for i in range(NUM_GPUS):
        for k, name in enumerate(("stream", "normed")):
            _check(
                f"boundary {name} {rows} rows, GPU {i}",
                _from_device(out[k * NUM_GPUS + i]),
                _from_device(ref[k * NUM_GPUS + i]),
                GATED_ROW_RMS_BAR,
            )
    for label, fault_model, fault_sig in faults:
        bad = _run(fault_model, *inputs, *fault_sig.buffers())
        frac = float(
            (_ulps(_from_device(bad[0]), _from_device(ref[0])) > 0).mean()
        )
        print(f"{label} {rows} rows: {frac:.2e} differ")
        assert frac > MISMATCH_BAR, f"{label} passes: {frac:.2e}"
