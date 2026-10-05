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
"""Graph-level test for the ``mamba2_ssd_chunk_scan_varlen_fwd_inplace`` op.

Runs a ragged prefill step and then a state-carrying decode step through the
registered op on every available device and checks the output and the
in-place state pool against a NumPy reference. ``x``, ``B`` and ``C`` are
column slices of one wide tensor, as the Nemotron-H mixer feeds them.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.state_space import mamba2_ssd_chunk_scan_varlen_fwd_inplace

_NHEADS = 4
# Not a multiple of the 16 channels per block of the split kernel, so the
# out-of-range channel guard runs.
_HEAD_DIM = 24
_NGROUPS = 2
_DSTATE = 16
_MAX_SLOTS = 4
_SLOTS = [3, 1]
_BATCH = len(_SLOTS)
_X_WIDTH = _NHEADS * _HEAD_DIM
_BC_WIDTH = _NGROUPS * _DSTATE
_WIDE = _X_WIDTH + 2 * _BC_WIDTH


def _build_graph(device: DeviceRef) -> Graph:
    with Graph(
        "mamba2_ssd_scan",
        input_types=[
            TensorType(DType.float32, ["total_len", _WIDE], device=device),
            TensorType(DType.float32, ["total_len", _NHEADS], device=device),
            TensorType(DType.float32, [_NHEADS], device=device),
            TensorType(DType.float32, [_NHEADS], device=device),
            TensorType(DType.float32, [_NHEADS], device=device),
            BufferType(
                DType.float32,
                [_MAX_SLOTS, _NHEADS, _HEAD_DIM, _DSTATE],
                device=device,
            ),
            TensorType(DType.int32, [_BATCH + 1], device=device),
            TensorType(DType.bool, [_BATCH], device=device),
            TensorType(DType.uint32, [_BATCH], device=device),
        ],
    ) as graph:
        xbc, dt, A, D, dt_bias, pool, qsl, his, slots = graph.inputs
        x, B, C = ops.split(
            xbc.tensor, [_X_WIDTH, _BC_WIDTH, _BC_WIDTH], axis=1
        )
        y = mamba2_ssd_chunk_scan_varlen_fwd_inplace(
            x=ops.reshape(x, [-1, _NHEADS, _HEAD_DIM]),
            dt=dt.tensor,
            A=A.tensor,
            B=ops.reshape(B, [-1, _NGROUPS, _DSTATE]),
            C=ops.reshape(C, [-1, _NGROUPS, _DSTATE]),
            D=D.tensor,
            dt_bias=dt_bias.tensor,
            ssm_pool=pool.buffer,
            query_start_loc=qsl.tensor,
            has_initial_state=his.tensor,
            cache_indices=slots.tensor,
        )
        graph.output(y)
    return graph


def _reference(
    xbc: npt.NDArray[np.float32],
    dt: npt.NDArray[np.float32],
    A: npt.NDArray[np.float32],
    D: npt.NDArray[np.float32],
    dt_bias: npt.NDArray[np.float32],
    pool: npt.NDArray[np.float32],
    offsets: list[int],
    has_init: bool,
) -> npt.NDArray[np.float32]:
    """Returns ``y`` and updates ``pool`` in place."""
    x = xbc[:, :_X_WIDTH].reshape(-1, _NHEADS, _HEAD_DIM).astype(np.float64)
    B = xbc[:, _X_WIDTH : _X_WIDTH + _BC_WIDTH].reshape(-1, _NGROUPS, _DSTATE)
    C = xbc[:, _X_WIDTH + _BC_WIDTH :].reshape(-1, _NGROUPS, _DSTATE)
    group = np.arange(_NHEADS) // (_NHEADS // _NGROUPS)
    y = np.zeros_like(x)
    for b, slot in enumerate(_SLOTS):
        state = (
            pool[slot].astype(np.float64)
            if has_init
            else np.zeros((_NHEADS, _HEAD_DIM, _DSTATE))
        )
        for t in range(offsets[b], offsets[b + 1]):
            delta = np.log1p(np.exp(dt[t] + dt_bias))
            dA = np.exp(A * delta)
            state = (
                state * dA[:, None, None]
                + (delta[:, None] * x[t])[:, :, None] * B[t, group][:, None, :]
            )
            y[t] = (state * C[t, group][:, None, :]).sum(-1) + D[:, None] * x[t]
        pool[slot] = state
    return y.astype(np.float32)


@pytest.mark.parametrize("on_gpu", [False, True])
def test_mamba2_ssd_scan_prefill_then_decode(
    session: InferenceSession, on_gpu: bool
) -> None:
    if on_gpu and accelerator_count() == 0:
        pytest.skip("Requires GPU")
    device_ref = DeviceRef.GPU() if on_gpu else DeviceRef.CPU()
    model = session.load(_build_graph(device_ref))
    device = model.input_devices[0]

    rng = np.random.default_rng(0)
    A = -(rng.random(_NHEADS, dtype=np.float32) + 0.1)
    D = rng.random(_NHEADS, dtype=np.float32)
    dt_bias = rng.random(_NHEADS, dtype=np.float32)
    pool_initial = rng.standard_normal(
        (_MAX_SLOTS, _NHEADS, _HEAD_DIM, _DSTATE)
    ).astype(np.float32)
    pool_ref = pool_initial.copy()
    pool_buf = md.Buffer.from_numpy(pool_initial.copy()).to(device)

    def run(
        offsets: list[int], has_init: bool
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        total_len = offsets[-1]
        xbc = rng.standard_normal((total_len, _WIDE)).astype(np.float32)
        dt = (rng.random((total_len, _NHEADS), dtype=np.float32) - 0.5).astype(
            np.float32
        )
        y_ref = _reference(xbc, dt, A, D, dt_bias, pool_ref, offsets, has_init)
        (y,) = model.execute(
            *(
                md.Buffer.from_numpy(a).to(device)
                for a in (xbc, dt, A, D, dt_bias)
            ),
            pool_buf,
            md.Buffer.from_numpy(np.asarray(offsets, dtype=np.int32)).to(
                device
            ),
            md.Buffer.from_numpy(np.asarray([has_init] * _BATCH)).to(device),
            md.Buffer.from_numpy(np.asarray(_SLOTS, dtype=np.uint32)).to(
                device
            ),
        )
        return y_ref, torch.from_dlpack(y).cpu().numpy()

    def check(
        y_ref: npt.NDArray[np.float32], y: npt.NDArray[np.float32]
    ) -> None:
        np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)
        pool = torch.from_dlpack(pool_buf).cpu().numpy()
        np.testing.assert_allclose(pool, pool_ref, rtol=1e-4, atol=1e-4)
        for s in range(_MAX_SLOTS):
            if s not in _SLOTS:
                np.testing.assert_array_equal(pool[s], pool_initial[s])

    # Ragged prefill of two fresh sequences, then one decode token each that
    # continues the pooled state.
    check(*run([0, 7, 12], has_init=False))
    check(*run([0, 1, 2], has_init=True))
