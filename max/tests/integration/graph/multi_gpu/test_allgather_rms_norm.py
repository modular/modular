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

"""Multi-GPU smoke test for the fused all-gather + RMSNorm graph op.

Exercises both dispatch branches: the fused kernel (decode M, below the fuse
threshold) and the two-launch fallback (prefill M, standalone all-gather +
rms_norm_gpu with the @__copy_capture closures). Verifies the gathered residual
is bit-identical to a plain all-gather (concat) and the normed output matches a
host RMSNorm reference (mbc=True, weight_offset=1.0, full-H divisor).
"""

from __future__ import annotations

from typing import Any, cast

import ml_dtypes
import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Type, ops
from max.nn import Signals

COLS = 6144  # M3 hidden size; the fuse threshold is bytes = rows*COLS*2.
EPS = 1e-6
WEIGHT_OFFSET = 1.0  # M3 Gemma-style: gamma_eff = gamma + 1.0.


def _to_device_bf16(arr: np.ndarray, device: Accelerator) -> Buffer:
    """Host float array -> bf16 device buffer (uint16 view: DLPack-safe)."""
    bits = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16).view(np.uint16))
    return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)


def _from_device_bf16(buf: Buffer) -> np.ndarray:
    """bf16 device buffer -> host float32 (via the uint16 view)."""
    out = buf.copy(device=CPU())
    return (
        out.view(DType.uint16)
        .to_numpy()
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    )


def _ag_rms_norm_graph(signals: Signals, shard_rows: int) -> Graph:
    devices = signals.devices
    num_devices = len(devices)
    with Graph(
        "allgather_rms_norm",
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(
                    dtype=DType.bfloat16,
                    shape=[shard_rows, COLS],
                    device=device,
                )
                for device in devices
            ]
            + [
                TensorType(dtype=DType.bfloat16, shape=[COLS], device=device)
                for device in devices
            ]
            + signals.input_types(),
        ),
    ) as graph:
        inputs = [v.tensor for v in graph.inputs[:num_devices]]
        gammas = [v.tensor for v in graph.inputs[num_devices : 2 * num_devices]]
        sigs = [v.buffer for v in graph.inputs[2 * num_devices :]]
        normed, residual = ops.allgather_rms_norm(
            inputs=inputs,
            signal_buffers=sigs,
            gammas=gammas,
            epsilon=EPS,
            weight_offset=WEIGHT_OFFSET,
        )
        graph.output(*normed, *residual)
        return graph


def _host_rmsnorm(gathered_f32: np.ndarray) -> np.ndarray:
    """Reference RMSNorm: gamma=0 so gamma_eff=WEIGHT_OFFSET(=1.0); mbc=True."""
    m2 = np.mean(gathered_f32**2, axis=-1, keepdims=True)
    nf = 1.0 / np.sqrt(m2 + EPS)
    gamma_eff = 0.0 + WEIGHT_OFFSET
    return (
        (gathered_f32 * nf * gamma_eff)
        .astype(ml_dtypes.bfloat16)
        .astype(np.float32)
    )


@pytest.mark.parametrize(
    "num_gpus, shard_rows, regime",
    [
        (4, 2, "fused"),  # 8 total rows -> below threshold -> fused kernel
        (4, 128, "two_launch"),  # 512 total rows -> above -> two-launch path
    ],
)
def test_allgather_rms_norm_execution(
    num_gpus: int, shard_rows: int, regime: str
) -> None:
    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs ({num_gpus}) for {regime} allgather_rms_norm."
        )

    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _ag_rms_norm_graph(signals, shard_rows)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    # Positive, varied shard data (avoids all-equal rows / pow-of-two aliasing).
    numpy_shards = []
    tensor_inputs = []
    offset = 0
    for i in range(num_gpus):
        size = shard_rows * COLS
        arr = (
            ((np.arange(size) + offset) % 251 + 1).reshape(shard_rows, COLS)
        ).astype(np.float32)
        numpy_shards.append(arr.astype(ml_dtypes.bfloat16).astype(np.float32))
        tensor_inputs.append(_to_device_bf16(arr, devices[i]))
        offset += size
    gammas = [
        _to_device_bf16(np.zeros(COLS, dtype=np.float32), devices[i])
        for i in range(num_gpus)
    ]

    outputs = compiled.execute(*tensor_inputs, *gammas, *signals.buffers())

    gathered = np.concatenate(
        numpy_shards, axis=0
    )  # bit-exact gather ref (f32)
    normed_ref = _host_rmsnorm(gathered)

    normed_out = outputs[:num_gpus]
    residual_out = outputs[num_gpus : 2 * num_gpus]

    for n in range(num_gpus):
        res = _from_device_bf16(cast(Buffer, residual_out[n]))
        # Residual is a verbatim gather -> bit-identical to concat on every GPU.
        assert np.array_equal(res, gathered), (
            f"residual mismatch on GPU {n} ({regime})"
        )

        nrm = _from_device_bf16(cast(Buffer, normed_out[n]))
        max_abs = float(np.max(np.abs(nrm - normed_ref)))
        # Loose bf16 reduction-order tolerance (values are O(1) after norm); a
        # wrong eps/cols-divisor/gamma-order bug is far larger than this. This is
        # NOT the real numeric gate -- the tight one (frac>1ULP <= 1%, max_ulp
        # <= 4, plus bit-identity vs `rms_norm_gpu`) lives in the kernel-level
        # test/gpu/comm/test_allgather_rmsnorm.mojo; here we only sanity-check the
        # op lowers and runs end-to-end through both dispatch branches.
        assert np.allclose(nrm, normed_ref, rtol=2e-2, atol=2e-2), (
            f"normed mismatch on GPU {n} ({regime}): max_abs={max_abs}"
        )
