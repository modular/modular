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
"""``kda_chunk`` against ``kda_decode`` at Kimi K3's prefill contract.

K3's KDA layer calls ``kda_decode`` for every batch, and on gfx950 the op
itself picks the recurrence: a prefill batch under the AMD chunk contract
runs the chunk-parallel pipeline -- so it must come back bit for bit
identical to the ``kda_chunk`` op on the same operands (pinned below) --
while a decode batch, and any batch the contract does not cover, runs the
token-sequential kernel at its own tighter error. Both ops take the SAME
tensors: the layer
pre-folds the
bounded gate into the kernel's raw input (``(raw_gate + dt_bias) *
exp(A_log)`` with K3's per-channel ``A_log``, zero ``a_log`` and zero
``dt_bias``), so the op never sees K3's per-channel gate parameters. These
tests feed the op exactly that folded convention, at one TP8 rank's 12 heads
of 128, from a non-zero float32 ``K_FIRST`` pool, and grade the bf16 chunk
path against float32 decode and against an fp64 token-serial oracle.

Each arm is held to the acceptance bound against the ORACLE on the
production-like cases, and to the stress allowance on the delta-rule-
dominated regimes; the two arms are also held to a looser mutual bound,
because they round independently (the chunk path takes bf16 q/k/v and
output, the decode reference stays float32), so their distance carries
roughly both arms' oracle errors at once.

The op's own suite (``max/tests/integration/graph/test_kda_ops.py``) covers
the raw per-head gate convention; nothing else grades the chunk op at K3's
folded one. The oracle is copied here rather than shared, matching the
parity tests' stated reasoning: a shared mistake would not be caught.
"""

from __future__ import annotations

from dataclasses import dataclass

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import accelerator_architecture_name
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.legacy.state_space import kda_chunk, kda_decode
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
)
from max.pipelines.architectures.kimi_k3_modulev3.weight_adapters import (
    interleave_for_tp,
)
from max.tensor import Tensor, default_device

HEADS = 12
"""One TP8 rank's KDA heads: K3's 96 heads split eight ways."""

HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
MAX_SLOTS = 6
CONV_KERNEL = 4
HIDDEN = 64
GATE_LOWER_BOUND = -5.0

# The acceptance bounds, held by each arm against the fp64 oracle on the
# production-like cases: the slow-gate continuation below measures output
# 4.85e-3 and state 3.90e-3 at worst.
ORACLE_OUTPUT_TOL = 5e-3
ORACLE_STATE_TOL = 4e-3

# The delta-rule-dominated regimes take 1.25x on the oracle bounds:
# unit-normal beta logits let every token rewrite the state -- the same
# construction GLM's stress cases use -- and with K3's folded gate at H=12
# the bf16 chunk path measures a 5.30e-3 output floor against fp64 there
# (state 3.84e-3), the rounding the AMD chunk contract bakes in. Measured
# on MI355X over every such case below.
STRESS_FACTOR = 1.25

# The mutual bounds between the two arms. Measured on MI355X at H=12 over
# every case below: output 5.32e-3 and state 4.17e-3, i.e. roughly the
# quadrature sum of the two arms' oracle errors, as the arms round
# independently (fp32 decode sits 2.5e-7 from the oracle, so the mutual
# distance is the chunk path's alone). The bounds sit about 1.2x over the
# measured worst.
MUTUAL_OUTPUT_TOL = 6.5e-3
MUTUAL_STATE_TOL = 5e-3

# The token-sequential kernel against the fp64 oracle: fp32 inputs and
# accumulation put its decode batches at 2.5e-7 from the oracle at H=12,
# and a 2048-token prefill compounds to 1.3e-5 in the state (both measured
# on the cases below), so these hold with two orders of margin -- and the
# chunk path, at 4-5e-3, cannot squeeze under either. This is the bound
# that tells the two recurrences apart inside the ops.
RECURRENT_OUTPUT_TOL = 1e-5
RECURRENT_STATE_TOL = 1e-4


def _is_gfx950_gpu() -> bool:
    try:
        return accelerator_architecture_name().startswith("gfx950")
    except Exception:
        return False


_requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950_gpu(), reason="K3 routes prefill to kda_chunk on gfx950"
)


def _rel_rmse(actual: np.ndarray, desired: np.ndarray) -> float:
    actual64 = actual.astype(np.float64)
    desired64 = desired.astype(np.float64)
    diff = np.sqrt(np.mean((actual64 - desired64) ** 2))
    return float(diff / (np.sqrt(np.mean(desired64**2)) + 1e-8))


def _normal(
    rng: np.random.Generator,
    *shape: int,
    mean: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    return rng.standard_normal(shape, dtype=np.float32) * scale + mean


@dataclass
class _KdaInputs:
    """One batch's op inputs, in the layer's pre-folded gate convention."""

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


def _fold_k3_gate(rng: np.random.Generator, raw_gate: np.ndarray) -> np.ndarray:
    """Folds K3's per-channel gate parameters into the kernel's raw input.

    The layer computes ``(raw_gate + dt_bias) * exp(A_log)`` and passes zero
    ``a_log`` and zero ``dt_bias``, so the kernel's own activation reduces to
    the published bounded gate at ``A = 1``.
    """
    a_log = _normal(rng, HEADS, HEAD_DIM, scale=0.5)
    dt_bias = _normal(rng, HEADS, HEAD_DIM, scale=0.5)
    return ((raw_gate + dt_bias) * np.exp(a_log.astype(np.float64))).astype(
        np.float32
    )


def _k3_inputs(
    rng: np.random.Generator,
    seq_lens: list[int],
    state_indices: np.ndarray,
    *,
    gate_mean: float = 0.0,
    gate_scale: float = 1.0,
    beta_shift: float = 0.0,
    aligned_keys: bool = False,
) -> _KdaInputs:
    """Random op inputs whose state survives, in K3's gate convention.

    A unit-normal gate decays the state to nothing within a chunk, which
    would hide drift in it, so the carried cases shift the gate to about -6
    (a decay of e^-0.01 per token) and the beta logits by -5, keeping the
    delta rule from overwriting the state either.
    """
    total_t = sum(seq_lens)
    direction = _normal(rng, HEADS, HEAD_DIM)

    def keys() -> np.ndarray:
        if aligned_keys:
            return direction + _normal(rng, total_t, HEADS, HEAD_DIM)
        return _normal(rng, total_t, HEADS, HEAD_DIM, scale=0.5)

    raw_gate = _normal(
        rng, total_t, HEADS, HEAD_DIM, mean=gate_mean, scale=gate_scale
    )
    return _KdaInputs(
        q=_normal(rng, total_t, HEADS, HEAD_DIM, scale=0.5),
        k=keys(),
        v=_normal(rng, total_t, HEADS, HEAD_DIM, scale=0.5),
        raw_gate=_fold_k3_gate(rng, raw_gate),
        beta_logits=_normal(rng, total_t, HEADS, mean=beta_shift),
        a_log=np.zeros(HEADS, dtype=np.float32),
        dt_bias=np.zeros((HEADS, HEAD_DIM), dtype=np.float32),
        cu_seqlens=np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32),
        state_indices=state_indices,
    )


def _round_bf16(a: np.ndarray) -> np.ndarray:
    """Rounds an array to bf16 and back, the op boundary's own rounding.

    Feeding the rounded values to both the ops and the oracle takes the
    input rounding out of the comparison, so the remaining distance is the
    recurrence path's alone.
    """
    return torch.from_numpy(a).to(torch.bfloat16).to(torch.float32).numpy()


def _bf16_inputs(inputs: _KdaInputs) -> _KdaInputs:
    """Rounds q/k/v to bf16 in the op inputs, as the graph's cast does."""
    inputs.q = _round_bf16(inputs.q)
    inputs.k = _round_bf16(inputs.k)
    inputs.v = _round_bf16(inputs.v)
    return inputs


def _random_pool(rng: np.random.Generator) -> np.ndarray:
    """A non-zero ``K_FIRST`` pool matching the op's pool buffer."""
    return _normal(rng, MAX_SLOTS, HEADS, HEAD_DIM, HEAD_DIM, scale=0.1)


def _load(
    session: InferenceSession,
    chunked: bool,
    io_dtype: DType | None = None,
    out_dtype: DType | None = None,
) -> Model:
    """Compiles one op for any token and sequence count.

    The chunk arm rounds q/k/v to bf16 and its output back to float32 inside
    the graph, as the layer does; decode stays float32 throughout unless
    ``io_dtype`` overrides its operand dtypes -- the decode op at the
    chunk arm's bf16, which the bitwise comparisons below need: the two
    ops see identical inputs only when both round them the same way.
    ``out_dtype`` overrides the op's output dtype, which defaults to
    ``io_dtype``.
    """
    gpu = DeviceRef.GPU()
    f32 = DType.float32
    act = TensorType(f32, ["total_tokens", HEADS, HEAD_DIM], device=gpu)
    name = f"k3_kda_{'chunk' if chunked else 'decode'}"
    if io_dtype is not None:
        name += f"_{io_dtype.name}"
    if out_dtype is not None:
        name += f"_out_{out_dtype.name}"
    with Graph(
        name,
        input_types=[
            act,
            act,
            act,
            act,
            TensorType(f32, ["total_tokens", HEADS], device=gpu),
            TensorType(f32, [HEADS], device=gpu),
            TensorType(f32, [HEADS, HEAD_DIM], device=gpu),
            TensorType(DType.int32, ["num_seqs_plus_one"], device=gpu),
            BufferType(f32, [MAX_SLOTS, HEADS, HEAD_DIM, HEAD_DIM], device=gpu),
            TensorType(DType.int32, ["num_seqs"], device=gpu),
        ],
    ) as graph:
        t = [inp.tensor for inp in graph.inputs[:8]]
        io_dtype = io_dtype or (DType.bfloat16 if chunked else f32)
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
            output_dtype=out_dtype or io_dtype,
            gate_mode="safe",
            beta_mode="logits",
            state_layout="K_FIRST",
        )
        graph.output(ops.cast(out, f32))
    return session.load(graph)


class _Ops:
    """Both ops, compiled once and run over the folded K3 convention."""

    def __init__(self) -> None:
        session = InferenceSession(devices=[md.Accelerator()])
        self._models = {
            chunked: _load(session, chunked) for chunked in (True, False)
        }
        # The bitwise dispatch comparison needs the decode op at the chunk
        # op's bf16 operands, so the two launch the same kernel
        # instantiation on the same values; again at the float32 output
        # the K3 layer takes.
        bf16, f32 = DType.bfloat16, DType.float32
        self._decode_bf16 = _load(session, False, bf16)
        self._dispatch = {
            bf16: (self._models[True], self._decode_bf16),
            f32: (
                _load(session, True, bf16, f32),
                _load(session, False, bf16, f32),
            ),
        }

    def run(
        self, chunked: bool, inputs: _KdaInputs, pool: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs one op from a copy of ``pool``; returns output and new pool."""
        return self._run(self._models[chunked], inputs, pool)

    def run_decode_bf16(
        self, inputs: _KdaInputs, pool: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs the decode op at the chunk op's bf16 operand dtypes."""
        return self._run(self._decode_bf16, inputs, pool)

    def run_dispatch_pair(
        self, out_dtype: DType, inputs: _KdaInputs, pool: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Runs both ops at bf16 operands and ``out_dtype`` output.

        Returns the chunk op's (output, pool), then the decode op's.
        """
        chunk, decode = self._dispatch[out_dtype]
        return self._run(chunk, inputs, pool), self._run(decode, inputs, pool)

    def _run(
        self, model: Model, inputs: _KdaInputs, pool: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        device = model.input_devices[0]
        pool_buf = md.Buffer.from_numpy(pool.copy()).to(device)
        args = [md.Buffer.from_numpy(a).to(device) for a in inputs.args()]
        out = model.execute(*args[:-1], pool_buf, args[-1])[0]
        return (
            torch.from_dlpack(out).cpu().numpy(),
            torch.from_dlpack(pool_buf).cpu().numpy(),
        )


@pytest.fixture(scope="module")
def ops_runner() -> _Ops:
    return _Ops()


def _oracle(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    raw_gate: np.ndarray,
    beta_logits: np.ndarray,
    cu_seqlens: np.ndarray,
    pool: np.ndarray,
    state_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The K3-convention recurrence in fp64 NumPy, token by token.

    With the layer's zero ``a_log`` and ``dt_bias`` the kernel's bounded gate
    reduces to ``alpha = exp(-5 * sigmoid(raw_gate))``.
    """
    total_t, num_heads, key_head_dim = q.shape
    query_scale = 1.0 / np.sqrt(np.float64(key_head_dim))
    q64 = q.astype(np.float64)
    k64 = k.astype(np.float64)
    v64 = v.astype(np.float64)
    gate64 = raw_gate.astype(np.float64)
    beta64 = beta_logits.astype(np.float64)
    pool64 = pool.astype(np.float64)
    out = np.zeros((total_t, num_heads, v.shape[2]), dtype=np.float64)
    for b in range(len(state_indices)):
        slot = int(state_indices[b])
        for h in range(num_heads):
            state = pool64[slot, h].copy()
            for t in range(int(cu_seqlens[b]), int(cu_seqlens[b + 1])):
                qv, kv = q64[t, h], k64[t, h]
                q_n = qv / np.sqrt((qv * qv).sum() + 1e-6) * query_scale
                k_n = kv / np.sqrt((kv * kv).sum() + 1e-6)
                alpha = np.exp(GATE_LOWER_BOUND / (1.0 + np.exp(-gate64[t, h])))
                beta = 1.0 / (1.0 + np.exp(-beta64[t, h]))
                state = alpha[:, None] * state
                err = v64[t, h] - k_n @ state
                state = state + beta * np.outer(k_n, err)
                out[t, h] = q_n @ state
            pool64[slot, h] = state
    return out, pool64


class _Grader:
    """Prints every comparison, then fails on the ones over tolerance."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def compare(
        self, tag: str, actual: np.ndarray, desired: np.ndarray, tol: float
    ) -> None:
        # Every graded array flows through here, so finiteness is checked
        # once, per array: a NaN also has to fail the tolerance, which a
        # bare `>` comparison would let through.
        assert np.all(np.isfinite(actual)), f"{tag}: non-finite values"
        rel_rmse = _rel_rmse(actual, desired)
        max_abs = float(np.abs(actual.astype(np.float64) - desired).max())
        print(f"K3_KDA {tag} rel_rmse={rel_rmse:.3e} max_abs={max_abs:.3e}")
        if not (rel_rmse <= tol):
            self.failures.append(f"{tag}: rel_rmse {rel_rmse:.3e} > {tol:.0e}")

    def done(self) -> None:
        assert not self.failures, "\n".join(self.failures)

    def batch(
        self,
        tag: str,
        inputs: _KdaInputs,
        arms: dict[str, tuple[np.ndarray, np.ndarray]],
        start_pools: dict[str, np.ndarray],
        *,
        output_tol: float,
        state_tol: float,
        mutual_output_tol: float = MUTUAL_OUTPUT_TOL,
        mutual_state_tol: float = MUTUAL_STATE_TOL,
    ) -> None:
        """Grades every arm of one batch against the oracle and each other.

        ``arms`` holds, per arm name, its output and its mutated pool;
        ``start_pools`` holds the pool each arm started from, which the
        carried continuation cases differ on. The oracle bounds gate each
        arm alone; the mutual bound gates the two arms' distance, at the
        looser tolerance their independent rounding sets (see the module
        docstring).
        """
        refs: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]] = []

        def ref_for(pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            for start, ref in refs:
                if np.array_equal(start, pool):
                    return ref
            ref = _oracle(
                inputs.q,
                inputs.k,
                inputs.v,
                inputs.raw_gate,
                inputs.beta_logits,
                inputs.cu_seqlens,
                pool,
                inputs.state_indices,
            )
            refs.append((pool, ref))
            return ref

        for b, slot in enumerate(inputs.state_indices.tolist()):
            rows = slice(inputs.cu_seqlens[b], inputs.cu_seqlens[b + 1])
            seq = f"{tag} seq={b}"
            for name, (out, out_pool) in arms.items():
                ref_out, ref_pool = ref_for(start_pools[name])
                self.compare(
                    f"{seq} {name} output-vs-oracle",
                    out[rows],
                    ref_out[rows],
                    output_tol,
                )
                self.compare(
                    f"{seq} {name} state-vs-oracle",
                    out_pool[slot],
                    ref_pool[slot],
                    state_tol,
                )
            if len(arms) == 2:
                chunk_out, chunk_pool = arms["chunk"]
                decode_out, decode_pool = arms["decode"]
                self.compare(
                    f"{seq} chunk-vs-decode output",
                    chunk_out[rows],
                    decode_out[rows],
                    mutual_output_tol,
                )
                self.compare(
                    f"{seq} chunk-vs-decode state",
                    chunk_pool[slot],
                    decode_pool[slot],
                    mutual_state_tol,
                )


def _assert_untouched(
    pool: np.ndarray, initial: np.ndarray, used: list[int]
) -> None:
    untouched = sorted(set(range(MAX_SLOTS)) - set(used))
    np.testing.assert_array_equal(pool[untouched], initial[untouched])


def _run_arms(
    ops_runner: _Ops, inputs: _KdaInputs, pool: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    chunk_out, chunk_pool = ops_runner.run(True, inputs, pool)
    decode_out, decode_pool = ops_runner.run(False, inputs, pool)
    _assert_untouched(chunk_pool, pool, inputs.state_indices.tolist())
    return {
        "chunk": (chunk_out, chunk_pool),
        "decode": (decode_out, decode_pool),
    }


@_requires_gfx950
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([2048], id="one_long_prefill"),
        pytest.param([300, 65, 1], id="ragged"),
        pytest.param([2048, 1, 1, 1], id="mixed_decode_rows"),
    ],
)
def test_chunk_matches_decode_at_k3_shapes(
    ops_runner: _Ops, seq_lens: list[int]
) -> None:
    """The folded-gate chunk path against float32 decode at H=12, D=128.

    Gate and beta logits both unit-normal, so the state decays within a
    chunk and the delta rule dominates -- the case that pushes the chunk
    path's intra-chunk solve (the kernel's ``(I + Akk)^-1``) hardest, and
    the regime the stress allowance covers.
    """
    rng = np.random.default_rng(3909)
    state_indices = np.array([4, 1, 5, 2][: len(seq_lens)], dtype=np.int32)
    inputs = _k3_inputs(rng, seq_lens, state_indices)
    pool = _random_pool(rng)

    grader = _Grader()
    grader.batch(
        f"case={'+'.join(map(str, seq_lens))}",
        inputs,
        _run_arms(ops_runner, inputs, pool),
        {"chunk": pool, "decode": pool},
        output_tol=STRESS_FACTOR * ORACLE_OUTPUT_TOL,
        state_tol=STRESS_FACTOR * ORACLE_STATE_TOL,
    )
    grader.done()


@_requires_gfx950
def test_chunk_carries_state_across_four_prefill_calls(
    ops_runner: _Ops,
) -> None:
    """Four chained 1024-token calls, each reading the last one's state.

    The slow gate and the -5 beta shift keep the state alive, so an error a
    path leaves in the pool compounds here if it compounds at all -- the
    shape of every chunked-prefill step in serving.
    """
    rng = np.random.default_rng(13909)
    state_indices = np.array([3], dtype=np.int32)
    pool = _random_pool(rng)
    pools = {"chunk": pool.copy(), "decode": pool.copy()}
    grader = _Grader()
    for call in range(4):
        inputs = _k3_inputs(
            rng,
            [1024],
            state_indices,
            gate_mean=-6.0,
            gate_scale=0.5,
            beta_shift=-5.0,
        )
        start_pools = dict(pools)
        arms = {}
        for name, chunked in (("chunk", True), ("decode", False)):
            out, pools[name] = ops_runner.run(chunked, inputs, pools[name])
            arms[name] = (out, pools[name])
        grader.batch(
            f"continuation call={call} t_end={(call + 1) * 1024}",
            inputs,
            arms,
            start_pools,
            output_tol=ORACLE_OUTPUT_TOL,
            state_tol=ORACLE_STATE_TOL,
        )
    grader.done()


@_requires_gfx950
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([1024], id="aligned_keys"),
        pytest.param([300, 65, 1], id="aligned_keys_ragged"),
    ],
)
def test_chunk_survives_stress_inputs(
    ops_runner: _Ops, seq_lens: list[int]
) -> None:
    """Near-duplicate keys and a gate driven onto the -5 bound.

    Keys sharing a mean direction overlap after the kernel's L2 norm, so
    consecutive corrections are large; the wide folded gate saturates
    ``sigmoid(pre)`` onto both ends, so some channels forget within a token
    (``alpha = exp(-5)``) and the rest barely decay. Every bound takes the
    stress allowance here.
    """
    rng = np.random.default_rng(23909)
    state_indices = np.array([4, 1, 5, 2][: len(seq_lens)], dtype=np.int32)
    inputs = _k3_inputs(
        rng,
        seq_lens,
        state_indices,
        gate_mean=0.0,
        gate_scale=4.0,
        aligned_keys=True,
    )
    pool = _random_pool(rng)

    grader = _Grader()
    grader.batch(
        f"stress={'+'.join(map(str, seq_lens))}",
        inputs,
        _run_arms(ops_runner, inputs, pool),
        {"chunk": pool, "decode": pool},
        output_tol=STRESS_FACTOR * ORACLE_OUTPUT_TOL,
        state_tol=STRESS_FACTOR * ORACLE_STATE_TOL,
        mutual_output_tol=STRESS_FACTOR * MUTUAL_OUTPUT_TOL,
        mutual_state_tol=STRESS_FACTOR * MUTUAL_STATE_TOL,
    )
    grader.done()


@_requires_gfx950
@pytest.mark.parametrize(
    "out_dtype",
    [
        pytest.param(DType.bfloat16, id="bf16_out"),
        pytest.param(DType.float32, id="fp32_out"),
    ],
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([2048], id="one_long_prefill"),
        pytest.param([300, 65, 1], id="ragged"),
    ],
)
def test_decode_prefill_takes_chunk_pipeline(
    ops_runner: _Ops, seq_lens: list[int], out_dtype: DType
) -> None:
    """A prefill batch through ``kda_decode`` runs the ``kda_chunk`` pipeline.

    On gfx950, at the chunk op's own operand dtypes (bf16 q/k/v, fp32
    gates, state and pool) and either output dtype -- float32 is the K3
    layer's -- a prefill-shaped batch (more tokens than sequences) must take
    the chunk-parallel pipeline inside the decode op, so its output and
    state come back bit for bit identical to the ``kda_chunk`` op on the
    same operands. A dead dispatch, one that ran the token-sequential
    kernel instead, differs in the pipeline's bf16 intermediates and misses.
    """
    rng = np.random.default_rng(63909)
    state_indices = np.asarray([4, 1, 5, 2][: len(seq_lens)], dtype=np.int32)
    inputs = _k3_inputs(rng, seq_lens, state_indices)
    pool = _random_pool(rng)
    (chunk_out, chunk_pool), (decode_out, decode_pool) = (
        ops_runner.run_dispatch_pair(out_dtype, inputs, pool)
    )
    # Bitwise equality is NaN-blind: NaN equals NaN, so finiteness is what
    # catches an arm that produced garbage in unison.
    assert np.all(np.isfinite(decode_out))
    assert np.all(np.isfinite(decode_pool))
    np.testing.assert_array_equal(
        decode_out,
        chunk_out,
        err_msg="a prefill batch through kda_decode must take the chunk"
        " pipeline and match kda_chunk bit for bit",
    )
    np.testing.assert_array_equal(
        decode_pool,
        chunk_pool,
        err_msg="a prefill batch's state must match kda_chunk's bit for bit",
    )
    _assert_untouched(decode_pool, pool, state_indices.tolist())


@_requires_gfx950
def test_decode_fp32_prefill_falls_back(ops_runner: _Ops) -> None:
    """A prefill batch at float32 q/k/v keeps the token-sequential kernel.

    The chunk pipeline's comptime contract takes bf16 q/k/v, so a float32
    batch cannot run it: the op must fall back to the recurrence
    it runs everywhere else, not raise and not silently round its operands
    into the pipeline. The recurrence sits two orders under the chunk path's
    oracle distance, and the pipeline's bf16 rounding makes the two arms'
    bits differ.
    """
    rng = np.random.default_rng(73909)
    state_indices = np.asarray([4], dtype=np.int32)
    inputs = _k3_inputs(
        rng,
        [2048],
        state_indices,
        gate_mean=-6.0,
        gate_scale=0.5,
        beta_shift=-5.0,
    )
    pool = _random_pool(rng)
    decode_out, decode_pool = ops_runner.run(False, inputs, pool)
    chunk_out, _ = ops_runner.run(True, inputs, pool)
    ref_out, ref_pool = _oracle(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.raw_gate,
        inputs.beta_logits,
        inputs.cu_seqlens,
        pool,
        inputs.state_indices,
    )
    grader = _Grader()
    grader.compare(
        "fp32-prefill decode output-vs-oracle",
        decode_out,
        ref_out,
        RECURRENT_OUTPUT_TOL,
    )
    grader.compare(
        "fp32-prefill decode state-vs-oracle",
        decode_pool[state_indices[0]],
        ref_pool[state_indices[0]],
        RECURRENT_STATE_TOL,
    )
    grader.done()
    assert not np.array_equal(decode_out, chunk_out), (
        "a float32 prefill batch must not round through the chunk"
        " pipeline's bf16 intermediates"
    )


@_requires_gfx950
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([1], id="b1"),
        pytest.param([1, 1, 1, 1], id="b4"),
        pytest.param([0, 1, 1], id="empty_seq"),
    ],
)
def test_decode_decode_batch_runs_recurrence(
    ops_runner: _Ops, seq_lens: list[int]
) -> None:
    """A decode batch through ``kda_decode`` runs the token-sequential kernel.

    ``T <= N`` identifies a decode batch: no sequence holds more than one
    token -- the ``empty_seq`` case holds one sequence at zero, taking
    ``T`` below ``N`` -- which the chunk split cannot help, every chunk
    holding at most one token. Both the float32 arm and the bf16 arm (where
    the comptime chunk contract holds, so only the runtime shape check can
    route) must run the recurrence: its state sits two orders under the
    chunk pipeline's oracle distance, and the chunk op -- which keeps its
    own pipeline for such a batch -- must differ from both.

    The slow gate and shifted beta logits keep the state the batch reads
    alive, so the comparison cannot pass on a state that decayed to zero.
    """
    rng = np.random.default_rng(43909)
    state_indices = np.asarray([4, 1, 5, 2][: len(seq_lens)], dtype=np.int32)
    inputs = _k3_inputs(
        rng,
        seq_lens,
        state_indices,
        gate_mean=-6.0,
        gate_scale=0.5,
        beta_shift=-5.0,
    )
    pool = _random_pool(rng)
    decode_out, decode_pool = ops_runner.run(False, inputs, pool)
    grader = _Grader()
    ref_out, ref_pool = _oracle(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.raw_gate,
        inputs.beta_logits,
        inputs.cu_seqlens,
        pool,
        inputs.state_indices,
    )
    for b, slot in enumerate(state_indices.tolist()):
        rows = slice(int(inputs.cu_seqlens[b]), int(inputs.cu_seqlens[b + 1]))
        # An empty sequence writes no output rows; its pool row must come
        # back unchanged, which the state grade below covers.
        if rows.stop > rows.start:
            grader.compare(
                f"decode fp32 seq={b} output-vs-oracle",
                decode_out[rows],
                ref_out[rows],
                RECURRENT_OUTPUT_TOL,
            )
        grader.compare(
            f"decode fp32 seq={b} state-vs-oracle",
            decode_pool[slot],
            ref_pool[slot],
            RECURRENT_STATE_TOL,
        )
    grader.done()

    # The bf16 arm: the chunk contract holds comptime, so this grades the
    # runtime shape check. Its output rounds to bf16, so the state -- fp32
    # in both paths -- carries the discrimination.
    bf16_inputs = _bf16_inputs(inputs)
    bf16_out, bf16_pool = ops_runner.run_decode_bf16(bf16_inputs, pool)
    assert np.all(np.isfinite(bf16_out))
    assert np.all(np.isfinite(bf16_pool))
    _, bf16_ref_pool = _oracle(
        bf16_inputs.q,
        bf16_inputs.k,
        bf16_inputs.v,
        bf16_inputs.raw_gate,
        bf16_inputs.beta_logits,
        bf16_inputs.cu_seqlens,
        pool,
        bf16_inputs.state_indices,
    )
    grader = _Grader()
    for b, slot in enumerate(state_indices.tolist()):
        grader.compare(
            f"decode bf16 seq={b} state-vs-oracle",
            bf16_pool[slot],
            bf16_ref_pool[slot],
            RECURRENT_STATE_TOL,
        )
    grader.done()

    chunk_out, chunk_pool = ops_runner.run(True, bf16_inputs, pool)
    assert not np.array_equal(bf16_out, chunk_out), (
        "a decode batch through kda_decode must run the token-sequential"
        " kernel, not the chunk pipeline"
    )
    assert not np.array_equal(bf16_pool, chunk_pool)


@_requires_gfx950
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([300, 1, 1], id="short"),
        pytest.param([2048, 1], id="long"),
        pytest.param([2, 1, 1], id="decode_boundary"),
    ],
)
def test_decode_mixed_batch_takes_chunk_pipeline(
    ops_runner: _Ops, seq_lens: list[int]
) -> None:
    """A mixed batch -- prefill rows with decode rows -- takes the chunk path.

    ``T > N`` is the chunk side, the length-1 sequences riding along; the
    ``decode_boundary`` case sits one token past it, at ``T == N + 1``, the
    first batch the split must still give the chunk pipeline. Through the
    decode op at the chunk op's bf16 operands the result must match the
    chunk op bit for bit; a split that fell every mixed batch back to the
    token-sequential kernel would differ in the pipeline's bf16
    intermediates.
    """
    rng = np.random.default_rng(53909)
    state_indices = np.asarray([4, 1, 5, 2][: len(seq_lens)], dtype=np.int32)
    inputs = _k3_inputs(rng, seq_lens, state_indices)
    pool = _random_pool(rng)
    decode_out, decode_pool = ops_runner.run_decode_bf16(inputs, pool)
    chunk_out, chunk_pool = ops_runner.run(True, inputs, pool)
    assert np.all(np.isfinite(decode_out))
    np.testing.assert_array_equal(
        decode_out,
        chunk_out,
        err_msg="a mixed batch through kda_decode must take the chunk"
        " pipeline and match kda_chunk bit for bit",
    )
    np.testing.assert_array_equal(
        decode_pool,
        chunk_pool,
        err_msg="a mixed batch's state must match kda_chunk's bit for bit",
    )
    _assert_untouched(decode_pool, pool, state_indices.tolist())


def _layer(weights: dict[str, np.ndarray]) -> KimiK3KDAAttention:
    layer = KimiK3KDAAttention(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL,
        gate_lower_bound=GATE_LOWER_BOUND,
    )
    # The merged q/k/v projection holds bf16 weights, as production's
    # does, so it emits the bf16 q/k/v the op's chunk contract takes.
    layer.qkv_proj.weight = Tensor(
        interleave_for_tp(
            [weights["q_proj"], weights["k_proj"], weights["v_proj"]], 1
        ),
        dtype=DType.float32,
    ).cast(DType.bfloat16)
    for name in (
        "f_a_proj",
        "f_b_proj",
        "b_proj",
        "g_proj",
        "o_proj",
        "o_norm",
    ):
        getattr(layer, name).weight = Tensor(weights[name], dtype=DType.float32)
    for name in ("conv_weight", "a_log", "dt_bias"):
        setattr(layer, name, Tensor(weights[name], dtype=DType.float32))
    return layer


def _layer_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Random signed weights, scaled like the parity tests'."""
    shapes = {
        "q_proj": (INNER, HIDDEN),
        "k_proj": (INNER, HIDDEN),
        "v_proj": (INNER, HIDDEN),
        "conv_weight": (3 * INNER, CONV_KERNEL),
        "f_a_proj": (HEAD_DIM, HIDDEN),
        "f_b_proj": (INNER, HEAD_DIM),
        "b_proj": (HEADS, HIDDEN),
        "a_log": (HEADS, HEAD_DIM),
        "dt_bias": (HEADS, HEAD_DIM),
        "g_proj": (INNER, HIDDEN),
        "o_norm": (HEAD_DIM,),
        "o_proj": (HIDDEN, INNER),
    }
    return {
        name: (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(
            np.float32
        )
        for name, shape in shapes.items()
    }


def _run_layer(
    layer: KimiK3KDAAttention,
    x_np: np.ndarray,
    state_pool: np.ndarray,
    slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One eager layer call; returns the output and the mutated pool."""
    total_seq_len = x_np.shape[0]
    batch = len(slots)
    with default_device(md.Accelerator()):
        x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        conv_pool = Tensor.zeros(
            [MAX_SLOTS, 3 * INNER, CONV_KERNEL - 1], dtype=DType.float32
        )
        pool = Tensor(state_pool, dtype=DType.float32)
        row_ids = Tensor(slots, dtype=DType.uint32)
        offsets = Tensor(
            np.array(
                [i * (total_seq_len // batch) for i in range(batch + 1)],
                dtype=np.uint32,
            ),
            dtype=DType.uint32,
        )
        out = layer(x, conv_pool, row_ids, pool, row_ids, offsets)
        return (
            out.cast(DType.float32).to_numpy(),
            pool.cast(DType.float32).to_numpy(),
        )


@pytest.mark.compile_on_gpu(
    "gated_delta_conv1d_fwd and the KDA ops mutate their pools in place"
)
@_requires_gfx950
def test_layer_runs_prefill_and_decode() -> None:
    """The layer's own graph: one ``kda_decode`` call serving both regimes.

    The layer hands every batch to ``kda_decode`` and the op picks the
    recurrence, so a prefill batch and a decode batch must both run end to
    end -- proving the layer's zero ``a_log`` and zero ``dt_bias`` reach the
    op as the dense tensors the chunk contract requires -- with the state
    written only at the slots the batch names. Which pipeline each batch
    took is pinned op by op above (bitwise against the ``kda_chunk`` op) and
    in the built graph by ``test_kda_prefill_routing.py``; this test holds
    the layer's wiring.
    """
    rng = np.random.default_rng(33909)
    weights = _layer_weights(rng)
    pool = _random_pool(rng)
    slots = np.array([4, 1], dtype=np.uint32)
    batches = [
        ("prefill", _normal(rng, 300, HIDDEN)),
        ("decode", _normal(rng, 2, HIDDEN)),
    ]
    for name, x_np in batches:
        out, state = _run_layer(_layer(weights), x_np, pool.copy(), slots)
        assert np.all(np.isfinite(out)), f"the {name} batch output"
        for slot in slots.tolist():
            assert not np.array_equal(state[slot], pool[slot]), (
                f"the {name} batch must advance the state at slot {slot}"
            )
        _assert_untouched(state, pool, slots.tolist())
