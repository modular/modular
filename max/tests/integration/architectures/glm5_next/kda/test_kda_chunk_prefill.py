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

On gfx950 the chunk op takes bf16 q/k/v and output, so the layer casts at the
op boundary while decode stays float32. The AMD cases grade that cast chunk
path against float32 decode at one TP2 rank's 32 heads, out to 80K tokens of
carried state.
"""

from __future__ import annotations

from dataclasses import dataclass

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import accelerator_api, accelerator_architecture_name
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.legacy.state_space import kda_chunk, kda_decode
from max.pipelines.architectures.glm5_next_modulev3.layers import (
    kda as modulev3_kda,
)

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


AMD_HEADS = 32
AMD_SLOTS = 6
CONTINUITY_TOKENS = 64
CONTINUATION_CALLS = 5
CONTINUATION_TOKENS = 16384

# Chunk (bf16 q/k/v and output) against float32 decode, at most 1.5x the worst
# rel-L2 measured on MI355X over every case below: output 5.37e-3 (80K
# continuation), state 4.24e-3, and decode after the prefill 4.18e-3 (output)
# and 4.15e-3 (state). Rounding A's output to bf16 alone costs 1.66e-3. The
# AMD op suite's ceilings against its fp64 oracle (`test_kda_ops.py`) are
# 2e-2 and 1e-2.
AMD_OUTPUT_REL_L2_TOL = 8e-3
AMD_STATE_REL_L2_TOL = 6.3e-3


def _is_gfx950_gpu() -> bool:
    try:
        return accelerator_architecture_name().startswith("gfx950")
    except Exception:
        return False


def _normal(
    rng: np.random.Generator,
    *shape: int,
    mean: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    return rng.standard_normal(shape, dtype=np.float32) * scale + mean


@dataclass
class _KdaInputs:
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    raw_gate: np.ndarray
    beta_logits: np.ndarray
    a_log: np.ndarray
    dt_bias: np.ndarray
    cu_seqlens: np.ndarray
    state_indices: np.ndarray

    def args(self) -> list[np.ndarray]:
        return [
            self.q,
            self.k,
            self.v,
            self.raw_gate,
            self.beta_logits,
            self.a_log,
            self.dt_bias,
            self.cu_seqlens,
            self.state_indices,
        ]

    def bf16_qkv(self) -> _KdaInputs:
        """Rounds q/k/v through bf16, the chunk path's only input rounding."""

        def rnd(x: np.ndarray) -> np.ndarray:
            return torch.from_numpy(x).to(torch.bfloat16).float().numpy()

        return _KdaInputs(
            rnd(self.q),
            rnd(self.k),
            rnd(self.v),
            *self.args()[3:],
        )


def _carrying_inputs(
    rng: np.random.Generator,
    seq_lens: list[int],
    state_indices: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
) -> _KdaInputs:
    """Random activations whose state survives thousands of tokens.

    Unit-normal gate logits decay the state to nothing within one chunk, which
    would hide any drift in it. Logits about -6 decay it by about e^-0.01 per
    token, and beta logits shifted by -5 keep the delta rule from overwriting
    it.
    """
    total_t = sum(seq_lens)
    return _KdaInputs(
        q=_normal(rng, total_t, AMD_HEADS, HEAD_DIM, scale=0.5),
        k=_normal(rng, total_t, AMD_HEADS, HEAD_DIM, scale=0.5),
        v=_normal(rng, total_t, AMD_HEADS, HEAD_DIM, scale=0.5),
        raw_gate=_normal(
            rng, total_t, AMD_HEADS, HEAD_DIM, mean=-6.0, scale=0.5
        ),
        beta_logits=_normal(rng, total_t, AMD_HEADS, mean=-5.0),
        a_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32),
        state_indices=state_indices,
    )


def _load_amd(session: InferenceSession, chunked: bool) -> Model:
    """Compiles one op for any token and sequence count.

    The chunk arm casts q/k/v to bf16 and its output back to float32 inside
    the graph, as the layer does on gfx950.
    """
    gpu = DeviceRef.GPU()
    f32 = DType.float32
    act = TensorType(f32, ["total_tokens", AMD_HEADS, HEAD_DIM], device=gpu)
    with Graph(
        f"glm_kda_amd_{'chunk' if chunked else 'decode'}",
        input_types=[
            act,
            act,
            act,
            act,
            TensorType(f32, ["total_tokens", AMD_HEADS], device=gpu),
            TensorType(f32, [AMD_HEADS], device=gpu),
            TensorType(f32, [AMD_HEADS, HEAD_DIM], device=gpu),
            TensorType(DType.int32, ["num_seqs_plus_one"], device=gpu),
            BufferType(
                f32, [AMD_SLOTS, AMD_HEADS, HEAD_DIM, HEAD_DIM], device=gpu
            ),
            TensorType(DType.int32, ["num_seqs"], device=gpu),
        ],
    ) as graph:
        t = [inp.tensor for inp in graph.inputs[:8]]
        io_dtype = DType.bfloat16 if chunked else f32
        op = kda_chunk if chunked else kda_decode
        out = op(
            ops.cast(t[0], io_dtype),
            ops.cast(t[1], io_dtype),
            ops.cast(t[2], io_dtype),
            t[3],
            t[4],
            t[5],
            t[6],
            t[7],
            graph.inputs[8].buffer,
            graph.inputs[9].tensor,
            output_dtype=io_dtype,
            gate_mode="safe",
            beta_mode="logits",
            state_layout="K_FIRST",
        )
        graph.output(ops.cast(out, f32))
    return session.load(graph)


class _AmdOps:
    def __init__(self) -> None:
        session = InferenceSession(devices=[md.Accelerator()])
        self._models = {
            chunked: _load_amd(session, chunked) for chunked in (True, False)
        }

    def run(
        self, chunked: bool, inputs: _KdaInputs, pool: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs one op from a copy of ``pool``; returns output and new pool."""
        model = self._models[chunked]
        device = model.input_devices[0]
        pool_buf = md.Buffer.from_numpy(pool.copy()).to(device)
        args = [md.Buffer.from_numpy(a).to(device) for a in inputs.args()]
        out = model.execute(*args[:-1], pool_buf, args[-1])[0]
        return (
            torch.from_dlpack(out).cpu().numpy(),
            torch.from_dlpack(pool_buf).cpu().numpy(),
        )


@pytest.fixture(scope="module")
def amd_ops() -> _AmdOps:
    return _AmdOps()


class _Grader:
    """Prints every comparison, then fails on the ones over tolerance.

    Arms: A is float32 decode, B is float32 decode on bf16-rounded q/k/v, C is
    the bf16 chunk path, and D is A's output rounded to bf16. A-B is the cost
    of rounding the inputs, B-C what the chunk kernel adds, A-D the floor that
    rounding the output alone sets. Only A-C is gated.
    """

    def __init__(self) -> None:
        self.failures: list[str] = []

    def compare(
        self,
        tag: str,
        actual: np.ndarray,
        desired: np.ndarray,
        tol: float | None = None,
    ) -> None:
        rel_l2 = _rel_rmse(actual, desired)
        max_abs = float(
            np.abs(actual.astype(np.float64) - desired.astype(np.float64)).max()
        )
        print(f"KDA_AMD {tag} rel_l2={rel_l2:.3e} max_abs={max_abs:.3e}")
        if tol is not None and not rel_l2 <= tol:
            self.failures.append(f"{tag}: rel_l2={rel_l2:.3e} > {tol:.0e}")

    def prefill(
        self,
        tag: str,
        inputs: _KdaInputs,
        outs: dict[str, np.ndarray],
        pools: dict[str, np.ndarray],
    ) -> None:
        out_d = torch.from_numpy(outs["A"]).to(torch.bfloat16).float().numpy()
        for b, slot in enumerate(inputs.state_indices.tolist()):
            rows = slice(inputs.cu_seqlens[b], inputs.cu_seqlens[b + 1])
            seq = f"{tag} seq={b}"
            out_a, out_b, out_c = (outs[x][rows] for x in "ABC")
            self.compare(f"{seq} output A-B", out_b, out_a)
            self.compare(f"{seq} output B-C", out_c, out_b)
            self.compare(
                f"{seq} output A-C", out_c, out_a, AMD_OUTPUT_REL_L2_TOL
            )
            self.compare(f"{seq} output A-D", out_d[rows], out_a)
            st_a, st_b, st_c = (pools[x][slot] for x in "ABC")
            self.compare(f"{seq} state A-B", st_b, st_a)
            self.compare(f"{seq} state B-C", st_c, st_b)
            self.compare(f"{seq} state A-C", st_c, st_a, AMD_STATE_REL_L2_TOL)

    def check(self) -> None:
        assert not self.failures, "\n".join(self.failures)


def _prefill_arms(
    amd_ops: _AmdOps, inputs: _KdaInputs, pools: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Runs arms A, B and C, each from its own pool."""
    rounded = inputs.bf16_qkv()
    out_a, pool_a = amd_ops.run(False, inputs, pools["A"])
    out_b, pool_b = amd_ops.run(False, rounded, pools["B"])
    out_c, pool_c = amd_ops.run(True, inputs, pools["C"])
    return (
        {"A": out_a, "B": out_b, "C": out_c},
        {"A": pool_a, "B": pool_b, "C": pool_c},
    )


def _grade_continuity(
    amd_ops: _AmdOps,
    grader: _Grader,
    tag: str,
    rng: np.random.Generator,
    template: _KdaInputs,
    pools: dict[str, np.ndarray],
) -> None:
    """Decodes the same float32 suffix from arm A's and arm C's states."""
    n = len(template.state_indices)
    suffix = _carrying_inputs(
        rng,
        [CONTINUITY_TOKENS] * n,
        template.state_indices,
        template.a_log,
        template.dt_bias,
    )
    out_a, pool_a = amd_ops.run(False, suffix, pools["A"])
    out_c, pool_c = amd_ops.run(False, suffix, pools["C"])
    for b, slot in enumerate(suffix.state_indices.tolist()):
        rows = slice(suffix.cu_seqlens[b], suffix.cu_seqlens[b + 1])
        seq = f"{tag} seq={b} continuity"
        grader.compare(
            f"{seq} output A-C", out_c[rows], out_a[rows], AMD_OUTPUT_REL_L2_TOL
        )
        grader.compare(
            f"{seq} state A-C", pool_c[slot], pool_a[slot], AMD_STATE_REL_L2_TOL
        )


def _layer_weights(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``a_log``, ``dt_bias`` and a non-zero initial pool."""
    return (
        _normal(rng, AMD_HEADS, scale=0.5),
        _normal(rng, AMD_HEADS, HEAD_DIM, scale=0.5),
        _normal(rng, AMD_SLOTS, AMD_HEADS, HEAD_DIM, HEAD_DIM, scale=0.1),
    )


@pytest.mark.skipif(
    not _is_gfx950_gpu(), reason="AMD kda_chunk path is gfx950-only"
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([64], id="one_chunk"),
        pytest.param([65], id="one_token_tail"),
        pytest.param([4099, 17, 1, 1000], id="ragged_mixed"),
        pytest.param([8192], id="8k"),
        pytest.param([8192, 8192], id="2x8k"),
        pytest.param([16384], id="16k"),
    ],
)
def test_amd_bf16_chunk_matches_fp32_decode(
    amd_ops: _AmdOps, seq_lens: list[int]
) -> None:
    """The layer's bf16 chunk path against float32 decode, then decode on."""
    rng = np.random.default_rng(907)
    a_log, dt_bias, pool = _layer_weights(rng)
    # Rows out of order, with untouched rows between them.
    state_indices = np.array([4, 1, 5, 2][: len(seq_lens)], dtype=np.int32)
    inputs = _carrying_inputs(rng, seq_lens, state_indices, a_log, dt_bias)

    outs, pools = _prefill_arms(amd_ops, inputs, dict.fromkeys("ABC", pool))
    assert np.all(np.isfinite(outs["C"]))
    untouched = sorted(set(range(AMD_SLOTS)) - set(state_indices.tolist()))
    np.testing.assert_array_equal(pools["C"][untouched], pool[untouched])

    tag = f"case={'+'.join(map(str, seq_lens))}"
    grader = _Grader()
    grader.prefill(tag, inputs, outs, pools)
    _grade_continuity(amd_ops, grader, tag, rng, inputs, pools)
    grader.check()


@pytest.mark.skipif(
    not _is_gfx950_gpu(), reason="AMD kda_chunk path is gfx950-only"
)
def test_amd_bf16_chunk_long_continuation(amd_ops: _AmdOps) -> None:
    """One 80K-token sequence prefilled as five 16K calls, graded at each.

    Each arm carries its own state through the pool, so error the chunk path
    leaves in the state compounds over the calls if it compounds at all.
    """
    rng = np.random.default_rng(1907)
    a_log, dt_bias, pool = _layer_weights(rng)
    state_indices = np.array([3], dtype=np.int32)
    pools = dict.fromkeys("ABC", pool)
    grader = _Grader()
    for call in range(CONTINUATION_CALLS):
        inputs = _carrying_inputs(
            rng, [CONTINUATION_TOKENS], state_indices, a_log, dt_bias
        )
        outs, pools = _prefill_arms(amd_ops, inputs, pools)
        assert np.all(np.isfinite(outs["C"]))
        tag = (
            f"continuation call={call} t_end={(call + 1) * CONTINUATION_TOKENS}"
        )
        grader.prefill(tag, inputs, outs, pools)
        _grade_continuity(amd_ops, grader, tag, rng, inputs, pools)
    grader.check()


# At most 1.5x the worst measured on MI355X over the stress cases below:
# per-head rel-L2 4.23e-3 (output) and 4.41e-3 (state), max-abs 1.86e-3
# (output, against values up to 0.24) and 7.06e-3 (state, up to 0.97).
AMD_STRESS_HEAD_OUTPUT_REL_L2_TOL = 6.3e-3
AMD_STRESS_HEAD_STATE_REL_L2_TOL = 6.6e-3
AMD_STRESS_OUTPUT_MAX_ABS_TOL = 2.7e-3
AMD_STRESS_STATE_MAX_ABS_TOL = 1.05e-2


def _stressing_inputs(
    rng: np.random.Generator,
    seq_lens: list[int],
    state_indices: np.ndarray,
    dt_bias: np.ndarray,
) -> _KdaInputs:
    """Inputs that lean on the delta rule.

    Unit-normal beta logits let every token rewrite the state, and q and k
    share a mean direction, so after the kernel's L2 norm consecutive keys
    overlap and the correction term is large. A large ``a_log`` drives each
    channel's gate to one of its two ends: some channels forget within a
    token, the rest barely decay at all.
    """
    total_t = sum(seq_lens)
    direction = _normal(rng, AMD_HEADS, HEAD_DIM)

    def aligned() -> np.ndarray:
        return direction + _normal(rng, total_t, AMD_HEADS, HEAD_DIM)

    return _KdaInputs(
        q=aligned(),
        k=aligned(),
        v=_normal(rng, total_t, AMD_HEADS, HEAD_DIM),
        raw_gate=_normal(rng, total_t, AMD_HEADS, HEAD_DIM),
        beta_logits=_normal(rng, total_t, AMD_HEADS),
        a_log=_normal(rng, AMD_HEADS, mean=2.0, scale=0.1),
        dt_bias=dt_bias,
        cu_seqlens=np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32),
        state_indices=state_indices,
    )


def _worst_head(
    tag: str, actual: np.ndarray, desired: np.ndarray, head_axis: int
) -> tuple[float, float]:
    """Prints and returns the worst per-head rel-L2 and the max-abs error."""
    heads = [
        _rel_rmse(
            np.take(actual, h, axis=head_axis),
            np.take(desired, h, axis=head_axis),
        )
        for h in range(actual.shape[head_axis])
    ]
    max_abs = float(
        np.abs(actual.astype(np.float64) - desired.astype(np.float64)).max()
    )
    print(
        f"KDA_AMD_STRESS {tag} worst_head_rel_l2={max(heads):.3e}"
        f" head={int(np.argmax(heads))} max_abs={max_abs:.3e}"
        f" ref_max_abs={float(np.abs(desired).max()):.3e}"
    )
    return max(heads), max_abs


@pytest.mark.skipif(
    not _is_gfx950_gpu(), reason="AMD kda_chunk path is gfx950-only"
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([65], id="one_token_tail"),
        pytest.param([4099, 17, 1, 1000], id="ragged_mixed"),
        pytest.param([8192], id="8k"),
    ],
)
def test_amd_bf16_chunk_delta_rule_stress(
    amd_ops: _AmdOps, seq_lens: list[int]
) -> None:
    """bf16 chunk against float32 decode where the delta rule dominates.

    The other cases keep beta near zero so the state survives; this one is
    graded per head and by max-abs error, so one bad head cannot hide in the
    average.
    """
    rng = np.random.default_rng(2907)
    _, dt_bias, pool = _layer_weights(rng)
    state_indices = np.array([4, 1, 5, 2][: len(seq_lens)], dtype=np.int32)
    inputs = _stressing_inputs(rng, seq_lens, state_indices, dt_bias)

    out_a, pool_a = amd_ops.run(False, inputs, pool)
    out_c, pool_c = amd_ops.run(True, inputs, pool)
    assert np.all(np.isfinite(out_c))

    tag = f"case={'+'.join(map(str, seq_lens))}"
    rows = state_indices.tolist()
    out_head, out_abs = _worst_head(f"{tag} output", out_c, out_a, 1)
    st_head, st_abs = _worst_head(f"{tag} state", pool_c[rows], pool_a[rows], 1)
    for value, tol, what in (
        (out_head, AMD_STRESS_HEAD_OUTPUT_REL_L2_TOL, "output head rel-L2"),
        (st_head, AMD_STRESS_HEAD_STATE_REL_L2_TOL, "state head rel-L2"),
        (out_abs, AMD_STRESS_OUTPUT_MAX_ABS_TOL, "output max-abs"),
        (st_abs, AMD_STRESS_STATE_MAX_ABS_TOL, "state max-abs"),
    ):
        assert value <= tol, f"{tag} {what} {value:.3e} > {tol:.1e}"


MODULEV3_CONV_KERNEL = 4
MODULEV3_DECODE_STEPS = 8


def _load_modulev3_core(session: InferenceSession, routed: bool) -> Model:
    """Compiles ModuleV3's per-device KDA core for one TP2 rank.

    ``routed=False`` builds it as if ``kda_chunk`` were unservable, so every
    call takes float32 ``kda_decode``.
    """
    gpu = DeviceRef.GPU()
    f32 = DType.float32
    qkv = AMD_HEADS * HEAD_DIM
    act = TensorType(f32, ["total_tokens", qkv], device=gpu)
    conv = TensorType(f32, [qkv, MODULEV3_CONV_KERNEL], device=gpu)
    rows = TensorType(DType.int32, [1, "batch"], device=gpu)
    with pytest.MonkeyPatch.context() as patch:
        if not routed:
            patch.setattr(
                modulev3_kda, "kda_chunk_servable", lambda *args: False
            )
        with Graph(
            f"glm_kda_modulev3_core_{'routed' if routed else 'decode'}",
            input_types=[
                act,
                act,
                act,
                conv,
                conv,
                conv,
                BufferType(
                    f32,
                    [AMD_SLOTS, 3 * qkv, MODULEV3_CONV_KERNEL - 1],
                    device=gpu,
                ),
                rows,
                act,
                TensorType(f32, ["total_tokens", AMD_HEADS], device=gpu),
                act,
                TensorType(f32, [AMD_HEADS], device=gpu),
                TensorType(f32, [qkv], device=gpu),
                TensorType(f32, [HEAD_DIM], device=gpu),
                BufferType(
                    f32, [AMD_SLOTS, AMD_HEADS, HEAD_DIM, HEAD_DIM], device=gpu
                ),
                rows,
                TensorType(DType.uint32, ["batch_plus_one"], device=gpu),
            ],
        ) as graph:
            i = graph.inputs
            graph.output(
                modulev3_kda._kda_core_graph(
                    q=i[0].tensor,
                    k=i[1].tensor,
                    v=i[2].tensor,
                    q_conv=i[3].tensor,
                    k_conv=i[4].tensor,
                    v_conv=i[5].tensor,
                    conv_pool=i[6].buffer,
                    conv_rows=i[7].tensor,
                    raw_gate=i[8].tensor,
                    beta_logits=i[9].tensor,
                    output_gate=i[10].tensor,
                    a_log=i[11].tensor,
                    dt_bias=i[12].tensor,
                    norm_weight=i[13].tensor,
                    recurrent_pool=i[14].buffer,
                    recurrent_rows=i[15].tensor,
                    input_row_offsets=i[16].tensor,
                    layer=ops.constant(0, DType.int64, device=DeviceRef.CPU()),
                    head_dim=HEAD_DIM,
                    norm_eps=1e-5,
                    gate_mode="safe",
                )
            )
    return session.load(graph)


@pytest.mark.skipif(
    not _is_gfx950_gpu(), reason="AMD kda_chunk path is gfx950-only"
)
def test_amd_modulev3_core_prefill_matches_decode() -> None:
    """ModuleV3's KDA core: routed prefill, then float32 decode steps.

    The routed graph's prefill takes the bf16 chunk path and is graded
    against the same graph built decode-only, at the op's bounds, which the
    gated norm's output meets in practice. Each graph
    then decodes from the pools its own prefill left. One extra step from a
    shared pool must agree bit for bit, which shows decode is not routed.
    """
    session = InferenceSession(devices=[md.Accelerator()])
    models = {
        routed: _load_modulev3_core(session, routed) for routed in (True, False)
    }
    device = models[True].input_devices[0]
    rng = np.random.default_rng(2207)
    qkv = AMD_HEADS * HEAD_DIM
    state_indices = np.array([[4, 1]], dtype=np.int32)

    weights = [
        _normal(rng, qkv, MODULEV3_CONV_KERNEL, scale=0.5) for _ in range(3)
    ]
    a_log = _normal(rng, AMD_HEADS, scale=0.5)
    dt_bias = _normal(rng, qkv, scale=0.5)
    norm_weight = 1.0 + _normal(rng, HEAD_DIM, scale=0.05)
    conv_pool = _normal(
        rng, AMD_SLOTS, 3 * qkv, MODULEV3_CONV_KERNEL - 1, scale=0.5
    )
    rec_pool = _normal(rng, AMD_SLOTS, AMD_HEADS, HEAD_DIM, HEAD_DIM, scale=0.1)

    def run(
        routed: bool, seq_lens: list[int], pools: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        t = sum(seq_lens)
        conv_buf, rec_buf = (
            md.Buffer.from_numpy(p.copy()).to(device) for p in pools
        )

        def put(a: np.ndarray) -> md.Buffer:
            return md.Buffer.from_numpy(a).to(device)

        offsets = np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.uint32)
        out = models[routed].execute(
            *(put(_normal(rng, t, qkv, scale=0.5)) for _ in range(3)),
            *(put(w) for w in weights),
            conv_buf,
            put(state_indices),
            put(_normal(rng, t, qkv, mean=-6.0, scale=0.5)),
            put(_normal(rng, t, AMD_HEADS, mean=-5.0)),
            put(_normal(rng, t, qkv)),
            put(a_log),
            put(dt_bias),
            put(norm_weight),
            rec_buf,
            put(state_indices),
            put(offsets),
        )[0]
        return torch.from_dlpack(out).cpu().numpy(), (
            torch.from_dlpack(conv_buf).cpu().numpy(),
            torch.from_dlpack(rec_buf).cpu().numpy(),
        )

    # Each call draws fresh activations, so both arms must see the same draw.
    def both(
        seq_lens: list[int],
        pools: dict[bool, tuple[np.ndarray, np.ndarray]],
    ) -> dict[bool, tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]]:
        state = rng.bit_generator.state
        results: dict[bool, tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]]
        results = {}
        for routed in (True, False):
            rng.bit_generator.state = state
            results[routed] = run(routed, seq_lens, pools[routed])
        return results

    grader = _Grader()
    seq_lens = [1000, 300]
    cu = np.concatenate([[0], np.cumsum(seq_lens)])
    results = both(
        seq_lens, dict.fromkeys((True, False), (conv_pool, rec_pool))
    )
    (out_c, pools_c), (out_a, pools_a) = results[True], results[False]
    assert not np.array_equal(out_c, out_a), "prefill did not take kda_chunk"
    for b, slot in enumerate(state_indices[0].tolist()):
        rows = slice(cu[b], cu[b + 1])
        grader.compare(
            f"modulev3 prefill seq={b} output A-C",
            out_c[rows],
            out_a[rows],
            AMD_OUTPUT_REL_L2_TOL,
        )
        grader.compare(
            f"modulev3 prefill seq={b} state A-C",
            pools_c[1][slot],
            pools_a[1][slot],
            AMD_STATE_REL_L2_TOL,
        )

    decode = [1] * len(seq_lens)
    shared = both(decode, dict.fromkeys((True, False), pools_a))
    np.testing.assert_array_equal(shared[True][0], shared[False][0])
    np.testing.assert_array_equal(shared[True][1][1], shared[False][1][1])

    pools = {True: pools_c, False: pools_a}
    for step in range(MODULEV3_DECODE_STEPS):
        results = both(decode, pools)
        pools = {routed: results[routed][1] for routed in (True, False)}
        grader.compare(
            f"modulev3 decode step={step} output A-C",
            results[True][0],
            results[False][0],
            AMD_OUTPUT_REL_L2_TOL,
        )
    for slot in state_indices[0].tolist():
        grader.compare(
            f"modulev3 after decode slot={slot} state A-C",
            pools[True][1][slot],
            pools[False][1][slot],
            AMD_STATE_REL_L2_TOL,
        )
    grader.check()
