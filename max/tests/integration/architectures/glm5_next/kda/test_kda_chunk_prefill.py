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
"""``kda_chunk`` against ``kda_decode`` at GLM-5.3-Flash's prefill contract.

GLM's KDA layer routes prefill through ``kda_chunk`` and decode through
``kda_decode``, so the two must agree on everything the layer passes: float32
q/k/v, the bounded ``"safe"`` gate, sigmoid-applied beta logits, a float32
``K_FIRST`` pool, and one TP8 rank's 8 heads of 128. The op's own suite checks
sequences of a few dozen tokens; these run thousands, from a non-zero state, so
a chunked prefill that resumes a prefix-cache checkpoint is covered too.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import accelerator_api, accelerator_architecture_name
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.state_space import kda_chunk, kda_decode

HEADS = 8
HEAD_DIM = 128
MAX_SLOTS = 6
REPEATED_LAUNCHES = 8

# The same bounds the op suite holds kda_chunk to against its fp64 oracle.
REL_RMSE_TOL = 5e-3
MAX_ABS_TOL = 1e-2


def _is_sm100_gpu() -> bool:
    name = accelerator_architecture_name()
    digits = name.removeprefix("sm_").rstrip("abcdefghijklmnopqrstuvwxyz")
    return (
        accelerator_api() == "cuda" and digits.isdigit() and int(digits) >= 100
    )


def _rel_rmse(actual: np.ndarray, desired: np.ndarray) -> float:
    actual64, desired64 = actual.astype(np.float64), desired.astype(np.float64)
    diff = np.sqrt(np.mean((actual64 - desired64) ** 2))
    return float(diff / (np.sqrt(np.mean(desired64**2)) + 1e-8))


def _assert_close(actual: np.ndarray, desired: np.ndarray, what: str) -> None:
    rel_rmse = _rel_rmse(actual, desired)
    max_abs = float(np.abs(actual - desired).max())
    assert rel_rmse < REL_RMSE_TOL, f"{what}: rel_rmse={rel_rmse:.3e}"
    assert max_abs < MAX_ABS_TOL, f"{what}: max_abs={max_abs:.3e}"


def _load(
    session: InferenceSession,
    chunked: bool,
    total_t: int,
    batch: int,
    pool_shape: tuple[int, ...],
) -> Model:
    """Compiles one op at fixed shapes."""
    gpu = DeviceRef.GPU()
    f32 = DType.float32
    with Graph(
        f"glm_kda_{'chunk' if chunked else 'decode'}_{total_t}",
        input_types=[
            TensorType(f32, [total_t, HEADS, HEAD_DIM], device=gpu),
            TensorType(f32, [total_t, HEADS, HEAD_DIM], device=gpu),
            TensorType(f32, [total_t, HEADS, HEAD_DIM], device=gpu),
            TensorType(f32, [total_t, HEADS, HEAD_DIM], device=gpu),
            TensorType(f32, [total_t, HEADS], device=gpu),
            TensorType(f32, [HEADS], device=gpu),
            TensorType(f32, [HEADS, HEAD_DIM], device=gpu),
            TensorType(DType.int32, [batch + 1], device=gpu),
            BufferType(f32, list(pool_shape), device=gpu),
            TensorType(DType.int32, [batch], device=gpu),
        ],
    ) as graph:
        t = [inp.tensor for inp in graph.inputs[:8]]
        op = kda_chunk if chunked else kda_decode
        graph.output(
            op(
                t[0],
                t[1],
                t[2],
                t[3],
                t[4],
                t[5],
                t[6],
                t[7],
                graph.inputs[8].buffer,
                graph.inputs[9].tensor,
                output_dtype=f32,
                gate_mode="safe",
                beta_mode="logits",
                state_layout="K_FIRST",
            )
        )
    return session.load(graph)


def _run(
    session: InferenceSession,
    chunked: bool,
    inputs: list[np.ndarray],
    pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs one op over ``inputs``, returning its output and the mutated pool."""
    model = _load(
        session, chunked, inputs[0].shape[0], inputs[-1].shape[0], pool.shape
    )
    device = model.input_devices[0]
    pool_buf = md.Buffer.from_numpy(pool.copy()).to(device)
    args = [md.Buffer.from_numpy(a).to(device) for a in inputs]
    out = model.execute(*args[:-1], pool_buf, args[-1])[0]
    return (
        torch.from_dlpack(out).cpu().numpy(),
        torch.from_dlpack(pool_buf).cpu().numpy(),
    )


@pytest.mark.skipif(
    not _is_sm100_gpu(), reason="kda_chunk kernels are NVIDIA SM100-only"
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([8192], id="one_long_prefill"),
        # A mixed batch takes the prefill branch, decode rows and all, and
        # 4099 and 1000 end in partial chunks.
        pytest.param([4099, 17, 1, 1000], id="ragged_mixed"),
    ],
)
def test_chunk_matches_decode(seq_lens: list[int]) -> None:
    session = InferenceSession(devices=[md.Accelerator()])
    rng = np.random.default_rng(907)
    total_t = sum(seq_lens)
    batch = len(seq_lens)

    def normal(*shape: int, scale: float = 1.0) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    # The layer feeds the recurrence silu(conv(...)) outputs; unit-scale
    # inputs over 8K steps keep the state well away from both zero and
    # overflow, which is where a carry bug shows.
    q = normal(total_t, HEADS, HEAD_DIM)
    k = normal(total_t, HEADS, HEAD_DIM)
    v = normal(total_t, HEADS, HEAD_DIM)
    raw_gate = normal(total_t, HEADS, HEAD_DIM)
    beta_logits = normal(total_t, HEADS)
    a_log = normal(HEADS, scale=0.5)
    dt_bias = normal(HEADS, HEAD_DIM)
    cu_seqlens = np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32)
    # Rows out of order, with untouched rows between them, and a non-zero
    # starting state as a resumed checkpoint has.
    state_indices = np.array([4, 1, 5, 2][:batch], dtype=np.int32)
    pool = normal(MAX_SLOTS, HEADS, HEAD_DIM, HEAD_DIM, scale=0.1)

    inputs = [
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
    ]
    chunk_out, chunk_pool = _run(session, True, inputs, pool)
    decode_out, decode_pool = _run(session, False, inputs, pool)

    assert np.all(np.isfinite(chunk_out))
    for b in range(batch):
        rows = slice(cu_seqlens[b], cu_seqlens[b + 1])
        _assert_close(chunk_out[rows], decode_out[rows], f"output of seq {b}")
        slot = state_indices[b]
        _assert_close(chunk_pool[slot], decode_pool[slot], f"state of seq {b}")
    untouched = sorted(set(range(MAX_SLOTS)) - set(state_indices.tolist()))
    np.testing.assert_array_equal(chunk_pool[untouched], pool[untouched])


@pytest.mark.skipif(
    not _is_sm100_gpu(), reason="kda_chunk kernels are NVIDIA SM100-only"
)
def test_tiny_ragged_prefill_launches() -> None:
    """Launches ``kda_chunk`` repeatedly on a sub-chunk ragged batch.

    The launcher reads ``cu_seqlens`` back to the host to build its chunk map.
    That readback once failed with CUDA_ERROR_INVALID_VALUE on the first launch
    of roughly 40% of processes and never on a later one, so only a fresh
    process exposes it; run this target with ``--runs_per_test``. The extra
    launches hold every result equal to the first.
    """
    session = InferenceSession(devices=[md.Accelerator()])
    rng = np.random.default_rng(907)
    seq_lens = [5, 3]
    total_t, batch = sum(seq_lens), len(seq_lens)

    def normal(*shape: int, scale: float = 1.0) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    inputs = [
        normal(total_t, HEADS, HEAD_DIM),
        normal(total_t, HEADS, HEAD_DIM),
        normal(total_t, HEADS, HEAD_DIM),
        normal(total_t, HEADS, HEAD_DIM),
        normal(total_t, HEADS),
        normal(HEADS, scale=0.5),
        normal(HEADS, HEAD_DIM),
        np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32),
        np.array([1, 0], dtype=np.int32),
    ]
    pool = normal(MAX_SLOTS, HEADS, HEAD_DIM, HEAD_DIM, scale=0.1)

    model = _load(session, True, total_t, batch, pool.shape)
    device = model.input_devices[0]
    args = [md.Buffer.from_numpy(a).to(device) for a in inputs]
    first = None
    for launch in range(REPEATED_LAUNCHES):
        pool_buf = md.Buffer.from_numpy(pool.copy()).to(device)
        try:
            out = model.execute(*args[:-1], pool_buf, args[-1])[0]
        except ValueError as e:
            raise AssertionError(f"launch {launch} failed: {e}") from e
        got = torch.from_dlpack(out).cpu().numpy()
        assert np.all(np.isfinite(got))
        if first is None:
            first = got
        else:
            np.testing.assert_array_equal(got, first)
