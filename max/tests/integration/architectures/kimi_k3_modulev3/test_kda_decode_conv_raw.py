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
"""``kda_decode``'s ``input_mode="conv_raw"`` against the activated path.

K3's decode graph otherwise runs its own small elementwise launches before
``kda_decode``: SiLU plus the model-dtype rounding over the conv's raw fp32
q/k/v, and the safe gate's fp32 per-channel pre-activation
``(raw_gate + dt_bias) * exp(a_log)``. The fused mode moves all of that into
the kernel's prologue, so its contract is agreement with that activated
graph path up to fp32 rounding: both spell SiLU in fp32 and round to the
model dtype once, but they are separately compiled code, and a value that
lands an ulp from a bf16 rounding tie can round differently. So the decode
arms hold every fp32 output and state element within one bf16 quantum at
that element's own magnitude -- a different SiLU spelling, a missing
rounding or a reordered pre-activation lands far outside that bound. The
bound is taken on the fp32 values themselves, never after rounding them to
bf16, so a small systematic drift stays visible; the fp64 oracle tests
carry that half of the contract, and no cross-op comparison here is
promised exact.

The decode arms run at K3's own geometry (one TP8 rank's 12 heads of 128,
fp32 pool) at batch 1, 4 and 64, plus one multi-token sequence held to a
relative bound instead: it stays fp32 past the SiLU, keeping both arms off
gfx950's chunk pipeline but dropping the bf16 rounding that absorbs the
last-ulp difference between the graph's elementwise SiLU and the kernel's
own fold. The inputs are signed and reach both SiLU rails and both gate
rails, including pre-activations that pin the bounded gate at its -5.0
decay floor.

The layer wiring is covered here for eager batches -- a decode batch must
reach the op with the raw operands and match the same layer with the fusion
switched off within the same rounding bound, and every batch reaches the op
with the raw operands, the routing living inside the op.
``test_kda_prefill_routing_layer.py`` holds the staged-graph half of the
rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from max import functional as F
from max.driver import Buffer
from max.driver import driver as md
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    ops,
)
from max.nn._graph_ops.state_space import kda_decode
from max.pipelines.architectures.kimi_k3_modulev3.layers import kda_attention
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
)
from max.tensor import Tensor, default_device

HEADS = 12
"""One TP8 rank's KDA heads at K3's production split."""

HEAD_DIM = 128
"""K3's KDA head width; also the state pool's inner axes."""

MAX_SLOTS = 64
"""State pool rows the op-level arms allocate."""

GATE_LOWER_BOUND = -5
"""The bound the "safe" gate hardcodes; see ``kimi_delta.KDA_GATE_LOWER_BOUND``."""

ORACLE_OUTPUT_TOL = 1e-05
"""Rel-RMSE the fused op may sit from the fp64 oracle's readout."""

ORACLE_STATE_TOL = 0.0001
"""Rel-RMSE the fused op may sit from the fp64 oracle's committed state."""


def _load(
    session: InferenceSession,
    fused: bool,
    round_to_bf16: bool,
) -> Model:
    """Compiles the decode op over raw conv operands, fused or not.

    The unfused arm is the layer's activated graph path: SiLU and the bf16
    rounding per q/k/v slice, the fp32 pre-activation over the gate, then
    the op over the activated operands with zeroed ``a_log`` and
    ``dt_bias``. The fused arm passes every operand raw under
    ``input_mode="conv_raw"``; the per-channel ``a_log`` is packed flat by
    the wrapper. ``round_to_bf16=False`` keeps both arms in fp32 past the
    SiLU, which also keeps the unfused arm off gfx950's chunk pipeline (it
    takes bf16 q/k/v), so the two run the same token-serial kernel on a
    multi-token sequence.
    """
    gpu = DeviceRef.GPU()
    f32 = DType.float32
    act = TensorType(f32, ["total_tokens", HEADS, HEAD_DIM], device=gpu)
    with Graph(
        f"k3_kda_conv_{'raw' if fused else 'activated'}",
        input_types=[
            act,
            act,
            act,
            act,
            TensorType(f32, ["total_tokens", HEADS], device=gpu),
            TensorType(f32, [HEADS, HEAD_DIM], device=gpu),
            TensorType(f32, [HEADS, HEAD_DIM], device=gpu),
            TensorType(DType.int32, ["num_seqs_plus_one"], device=gpu),
            BufferType(f32, [MAX_SLOTS, HEADS, HEAD_DIM, HEAD_DIM], device=gpu),
            TensorType(DType.int32, ["num_seqs"], device=gpu),
        ],
    ) as graph:
        t = [inp.tensor for inp in graph.inputs[:8]]
        a_log, dt_bias = t[5], t[6]
        model_dtype: DType = DType.bfloat16 if round_to_bf16 else f32

        def activated(x: Any) -> Any:
            return ops.cast(ops.silu(x), model_dtype)

        if fused:
            out = kda_decode(
                t[0],
                t[1],
                t[2],
                t[3],
                t[4],
                a_log,
                dt_bias,
                t[7],
                graph.inputs[8].buffer,
                graph.inputs[9].tensor,
                output_dtype=f32,
                gate_mode="safe",
                beta_mode="logits",
                state_layout="K_FIRST",
                input_mode="conv_raw",
                activation_dtype=model_dtype,
            )
        else:
            zero_heads = ops.constant(
                np.zeros(HEADS, dtype=np.float32), dtype=f32, device=gpu
            )
            zero_channels = ops.constant(
                np.zeros((HEADS, HEAD_DIM), dtype=np.float32),
                dtype=f32,
                device=gpu,
            )
            out = kda_decode(
                activated(t[0]),
                activated(t[1]),
                activated(t[2]),
                (t[3] + dt_bias) * ops.exp(a_log),
                t[4],
                zero_heads,
                zero_channels,
                t[7],
                graph.inputs[8].buffer,
                graph.inputs[9].tensor,
                output_dtype=f32,
                gate_mode="safe",
                beta_mode="logits",
                state_layout="K_FIRST",
            )
        graph.output(out)
    return session.load(graph)


class _Ops:
    """Both arms of the op, compiled once and shared by the tests."""

    def __init__(self) -> None:
        session = InferenceSession(devices=[md.Accelerator()])
        self._models = {
            (fused, rounded): _load(session, fused, rounded)
            for fused in (True, False)
            for rounded in (True, False)
        }

    def run(
        self,
        fused: bool,
        inputs: list[np.ndarray],
        pool: np.ndarray,
        round_to_bf16: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs one arm from a copy of the pool; returns output and pool.

        The op's operand order interleaves the pool between ``cu_seqlens``
        and ``state_indices``, matching the graph's input list.
        """
        model = self._models[(fused, round_to_bf16)]
        device = model.input_devices[0]
        buffers = [
            Buffer.from_numpy(a).to(device)
            for a in [*inputs[:8], pool, inputs[8]]
        ]
        out = model.execute(*buffers)[0]
        return (
            torch.from_dlpack(out).cpu().numpy(),
            torch.from_dlpack(buffers[8]).cpu().numpy(),
        )


@pytest.fixture(scope="module")
def ops_runner() -> _Ops:
    return _Ops()


def _spiky(rng: np.random.Generator, total: int, scale: float) -> np.ndarray:
    """Signed values with a few +-30 outliers, so SiLU runs on both rails."""
    base = rng.standard_normal((total, HEADS, HEAD_DIM)) * scale
    spikes = rng.choice([-30, 30], base.shape)
    return (base + np.where(rng.random(base.shape) < 0.05, spikes, 0)).astype(
        np.float32
    )


def _raw_inputs(
    rng: np.random.Generator, seq_lens: list[int]
) -> list[np.ndarray]:
    """Raw conv-side operands at the layer's own convention.

    ``a_log`` in [-2, 1] and ``dt_bias`` in [-5, 5] put the folded
    pre-activation as far as +-60 out, pinning ``sigmoid`` at 0 and 1 -- the
    latter is the -5.0 decay floor the bounded gate clamps at.
    """
    total = sum(seq_lens)
    a_log = rng.uniform(-2, 1, (HEADS, HEAD_DIM)).astype(np.float32)
    dt_bias = rng.uniform(-5, 5, (HEADS, HEAD_DIM)).astype(np.float32)
    raw_gate = (rng.standard_normal((total, HEADS, HEAD_DIM)) * 15).astype(
        np.float32
    )
    beta_logits = rng.standard_normal((total, HEADS)).astype(np.float32)
    cu_seqlens = np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32)
    state_indices = np.arange(len(seq_lens), dtype=np.int32)
    return [
        _spiky(rng, total, 2),
        _spiky(rng, total, 2),
        _spiky(rng, total, 2),
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
    ]


def _initial_pool(rng: np.random.Generator) -> np.ndarray:
    return (
        rng.standard_normal((MAX_SLOTS, HEADS, HEAD_DIM, HEAD_DIM)) * 0.1
    ).astype(np.float32)


def _bf16_quantum(magnitude: np.ndarray) -> np.ndarray:
    """One bf16 step at each element's magnitude.

    bf16 keeps fp32's exponent and 7 mantissa bits, so values in
    ``[2**e, 2**(e+1))`` sit ``2**(e-7)`` apart; below the normal range
    the subnormal grid is flat at the smallest step, ``2**-133``.
    """
    _, exponent = np.frexp(magnitude)
    return np.maximum(np.exp2(exponent - 8), 2.0**-133)


def _assert_within_one_bf16_quantum(
    actual: np.ndarray, desired: np.ndarray, what: str
) -> None:
    """Holds every fp32 element of `actual` within one bf16 quantum of `desired`.

    Both operands round to the model dtype in production, so a last-ulp fp32
    difference in the SiLU spelling is absorbed by the rounding except at a
    tie, where it can flip one bf16 quantum -- the bound here, taken on the
    fp32 values themselves at each element's own magnitude. Comparing the
    raw values rather than their bf16 roundings keeps a localized error of a
    few quanta and any drift the fp64-oracle tests would catch from being
    rounded away ahead of the comparison; a different SiLU spelling, a
    missing rounding or a reordered pre-activation lands orders outside
    that bound.
    """
    a = np.ascontiguousarray(actual, dtype=np.float64)
    b = np.ascontiguousarray(desired, dtype=np.float64)
    magnitude = np.maximum(np.abs(a), np.abs(b))
    diff = np.abs(a - b)
    quantum = _bf16_quantum(magnitude)
    # Elements where both arms are zero must agree exactly: there is no
    # magnitude to scale a rounding allowance against.
    ok = np.where(magnitude == 0, diff == 0, diff <= quantum)
    violations = int(np.count_nonzero(~ok))
    assert violations == 0, (
        f"fused {what} differs from the activated path by more than one"
        f" bf16 quantum per element: {violations} of {magnitude.size}"
        f" elements, worst {float(np.max(diff / quantum)):.2f} bf16 quanta"
    )


def _assert_matches_activated_graph_path(
    ops_runner: _Ops,
    seq_lens: list[int],
    seed: int,
    round_to_bf16: bool = True,
) -> None:
    rng = np.random.default_rng(seed)
    inputs = _raw_inputs(rng, seq_lens)
    pool = _initial_pool(rng)
    fused_out, fused_pool = ops_runner.run(
        True, inputs, pool.copy(), round_to_bf16
    )
    plain_out, plain_pool = ops_runner.run(
        False, inputs, pool.copy(), round_to_bf16
    )
    _assert_within_one_bf16_quantum(fused_out, plain_out, "output")
    _assert_within_one_bf16_quantum(fused_pool, plain_pool, "state")


def test_conv_raw_decode_shaped_multitoken_batch_matches_oracle(
    ops_runner: _Ops,
) -> None:
    """The prologue fires per token inside the loop, not once per launch.

    ``total_tokens == num_seqs`` routes to the fused kernel even when a
    sequence holds more than one token (or none), so a batch like this one
    -- two single tokens, one two-token sequence, one empty -- exercises
    the per-token SiLU and pre-activation folds that a one-token decode
    batch cannot tell apart from a prologue that ran once per launch. The
    fp64 oracle over the raw operands fails a fold that dropped a token's
    activation.
    """
    rng = np.random.default_rng(99)
    inputs = _raw_inputs(rng, [1, 1, 2, 0])
    pool = _initial_pool(rng)
    fused_out, fused_pool = ops_runner.run(True, inputs, pool.copy())
    q, k, v, raw_gate, beta, a_log, dt_bias, cu, slots = inputs
    ref_out, ref_pool = _oracle(
        q, k, v, raw_gate, a_log, dt_bias, beta, cu, pool, slots
    )
    assert _rel_rmse(fused_out, ref_out) < ORACLE_OUTPUT_TOL
    assert _rel_rmse(fused_pool, ref_pool) < ORACLE_STATE_TOL


@pytest.mark.parametrize("batch", [1, 4, 64])
def test_conv_raw_matches_activated_graph_path_at_decode(
    ops_runner: _Ops, batch: int
) -> None:
    """The fused op agrees with the activated path within one bf16 quantum.

    Both round SiLU's fp32 result to the model dtype once, so the agreement
    is every fp32 element within one bf16 quantum at its own magnitude; the
    separately compiled elementwise passes cannot be promised exact bytes.
    """
    _assert_matches_activated_graph_path(ops_runner, [1] * batch, 1234 + batch)


def test_conv_raw_multi_token_sequence_matches_activated_graph_path(
    ops_runner: _Ops,
) -> None:
    """A prefill-shaped batch keeps the activated path's numerics.

    A single 8-token sequence is not decode-shaped, so the fused mode
    activates the operands inside the op and runs the same activated
    recurrence the old graph path ran. Both arms stay fp32 past the SiLU:
    the bf16-rounded activated arm would take gfx950's chunk pipeline at
    this shape and round differently, and neither fused route runs it at
    fp32 q/k/v.

    That also removes the bf16 rounding that absorbs the last-ulp
    difference between the graph's elementwise SiLU lowering and the
    kernel's own fold, so this arm sits further from the activated path
    than the decode arms; it is held to a tight relative bound instead,
    with the fp64 oracle below carrying the correctness claim.
    """
    rng = np.random.default_rng(4321)
    inputs = _raw_inputs(rng, [8])
    pool = _initial_pool(rng)
    fused_out, fused_pool = ops_runner.run(
        True, inputs, pool.copy(), round_to_bf16=False
    )
    plain_out, plain_pool = ops_runner.run(
        False, inputs, pool.copy(), round_to_bf16=False
    )
    for name, fused, plain in (
        ("output", fused_out, plain_out),
        ("state", fused_pool, plain_pool),
    ):
        rel = _rel_rmse(fused, plain)
        assert rel < 1e-06, f"{name} rel rmse {rel}"
    q, k, v, raw_gate, beta, a_log, dt_bias, cu, slots = inputs
    ref_out, ref_pool = _oracle(
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu,
        pool,
        slots,
        round_to_bf16=False,
    )
    assert _rel_rmse(fused_out, ref_out) < ORACLE_OUTPUT_TOL
    assert _rel_rmse(fused_pool, ref_pool) < ORACLE_STATE_TOL


def _rel_rmse(actual: np.ndarray, desired: np.ndarray) -> float:
    actual64, desired64 = (
        actual.astype(np.float64),
        desired.astype(np.float64),
    )
    diff = np.sqrt(np.mean((actual64 - desired64) ** 2))
    return float(diff / (np.sqrt(np.mean(desired64**2)) + 1e-08))


def _round_bf16(a: np.ndarray) -> np.ndarray:
    return torch.from_numpy(a).to(torch.bfloat16).to(torch.float32).numpy()


def _silu(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    return (x32 * (1 / (1 + np.exp(-x32)))).astype(np.float32)


def _oracle(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    raw_gate: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    beta_logits: np.ndarray,
    cu_seqlens: np.ndarray,
    pool: np.ndarray,
    state_indices: np.ndarray,
    round_to_bf16: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """The fused mode's recurrence in fp64, on the raw conv operands.

    Models the kernel's prologue: SiLU in fp32 then one bf16 rounding per
    q/k/v (skipped when ``round_to_bf16=False``, the arm that stays fp32),
    and the fp32 per-channel pre-activation before the bounded gate.
    """

    def _silu_round(x: np.ndarray) -> np.ndarray:
        y = _silu(x)
        return _round_bf16(y) if round_to_bf16 else y

    q64 = _silu_round(q).astype(np.float64)
    k64 = _silu_round(k).astype(np.float64)
    v64 = _silu_round(v).astype(np.float64)
    pre = ((raw_gate + dt_bias) * np.exp(a_log)).astype(np.float32)
    beta64 = beta_logits.astype(np.float64)
    pool64 = pool.astype(np.float64)
    total_t, num_heads, key_head_dim = q.shape
    query_scale = 1 / np.sqrt(np.float64(key_head_dim))
    out = np.zeros((total_t, num_heads, v.shape[2]), dtype=np.float64)
    for b in range(len(state_indices)):
        slot = int(state_indices[b])
        for h in range(num_heads):
            state = pool64[slot, h].copy()
            for t in range(int(cu_seqlens[b]), int(cu_seqlens[b + 1])):
                qv, kv = q64[t, h], k64[t, h]
                q_n = qv / np.sqrt((qv * qv).sum() + 1e-06) * query_scale
                k_n = kv / np.sqrt((kv * kv).sum() + 1e-06)
                alpha = np.exp(
                    GATE_LOWER_BOUND
                    / (1 + np.exp(-pre[t, h].astype(np.float64)))
                )
                beta = 1 / (1 + np.exp(-beta64[t, h]))
                state = state * alpha[:, None]
                err = v64[t, h] - k_n @ state
                state = state + beta * np.outer(k_n, err)
                out[t, h] = q_n @ state
            pool64[slot, h] = state
    return out, pool64


def test_conv_raw_matches_fp64_oracle(ops_runner: _Ops) -> None:
    """The fused op is graded against an independent reference, not itself."""
    rng = np.random.default_rng(2026)
    inputs = _raw_inputs(rng, [1, 1, 1, 1])
    pool = _initial_pool(rng)
    out, new_pool = ops_runner.run(True, inputs, pool.copy())
    q, k, v, raw_gate, beta, a_log, dt_bias, cu, slots = inputs
    ref_out, ref_pool = _oracle(
        q, k, v, raw_gate, a_log, dt_bias, beta, cu, pool, slots
    )
    got_out = _rel_rmse(out, ref_out)
    got_pool = _rel_rmse(new_pool, ref_pool)
    assert got_out < ORACLE_OUTPUT_TOL, f"output rel rmse {got_out}"
    assert got_pool < ORACLE_STATE_TOL, f"state rel rmse {got_pool}"


_HIDDEN = 64
_NUM_HEADS = 4
_HEAD_DIM = 32
_CONV_KERNEL = 4
_MAX_SLOTS = 4


def _make_layer(rng: np.random.Generator) -> KimiK3KDAAttention:
    """Builds a KDA layer whose fused-mode parameters are all nonzero.

    The module's defaults leave ``conv_weight``, ``a_log`` and ``dt_bias`` at
    zero, which makes every operand the fused prologue is responsible for
    constant: SiLU(0) is 0 whatever the fold does, and any ``a_log``
    packing reads the same zeros. Distinct per-channel values make those
    defects observable, so the equality test seeds them here.
    """
    layer = KimiK3KDAAttention(
        hidden_size=_HIDDEN,
        num_heads=_NUM_HEADS,
        head_dim=_HEAD_DIM,
        conv_kernel_size=_CONV_KERNEL,
        gate_lower_bound=GATE_LOWER_BOUND,
    )
    layer.conv_weight = Tensor(
        rng.standard_normal([3 * _NUM_HEADS * _HEAD_DIM, _CONV_KERNEL]).astype(
            np.float32
        ),
        dtype=DType.float32,
    )
    # `a_log` in [-2, 1] and `dt_bias` in [-5, 5] drive the folded
    # pre-activation across both sigmoid rails, as the op-level arms do.
    layer.a_log = Tensor(
        rng.uniform(-2, 1, (_NUM_HEADS, _HEAD_DIM)).astype(np.float32),
        dtype=DType.float32,
    )
    layer.dt_bias = Tensor(
        rng.uniform(-5, 5, (_NUM_HEADS, _HEAD_DIM)).astype(np.float32),
        dtype=DType.float32,
    )
    return layer


def _layer_inputs(
    seq_lens: list[int],
    rng: np.random.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Layer inputs whose carried pools are nonzero, unlike the defaults.

    A zero conv pool makes the conv output independent of its history, and a
    zero state pool makes the recurrence read zeros, so both arms of an
    equality test would agree on constants. The pools are drawn from ``rng``
    so both arms of a comparison can restart from identical carried state.
    """
    inner = _NUM_HEADS * _HEAD_DIM
    batch = len(seq_lens)
    if rng is None:
        rng = np.random.default_rng(7)
    conv_pool = Tensor(
        (
            rng.standard_normal((_MAX_SLOTS, 3 * inner, _CONV_KERNEL - 1)) * 0.5
        ).astype(np.float32),
        dtype=DType.float32,
    )
    state_pool = Tensor(
        (
            rng.standard_normal((_MAX_SLOTS, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM))
            * 0.25
        ).astype(np.float32),
        dtype=DType.float32,
    )
    slot_idx = Tensor(np.arange(batch, dtype=np.uint32), dtype=DType.uint32)
    input_row_offsets = Tensor(
        np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.uint32),
        dtype=DType.uint32,
    )
    x_np = (rng.standard_normal((sum(seq_lens), _HIDDEN)) * 2).astype(
        np.float32
    )
    x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
    return x, conv_pool, slot_idx, state_pool, slot_idx, input_row_offsets


def _shapes_of(tensor: Any) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def _run_layer(
    seq_lens: list[int],
    layer: KimiK3KDAAttention | None = None,
    input_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs one eager layer call over ``seq_lens``; returns the fp32 output
    and the state pool as numpy arrays.

    ``layer`` and ``input_rng`` let a caller hold one layer fixed across
    several runs and redraw identical inputs, which an equality test needs:
    both pools are mutated in place, so compared arms must restart from
    equal carried state.
    """
    with default_device(md.Accelerator()):
        if layer is None or input_rng is None:
            input_rng = np.random.default_rng(7)
            layer = _make_layer(np.random.default_rng(11))
        x, conv_pool, slot_idx, state_pool, row_id, offsets = _layer_inputs(
            seq_lens, input_rng
        )
        out = layer(x, conv_pool, slot_idx, state_pool, row_id, offsets)
        return out.cast(DType.float32).to_numpy(), state_pool.to_numpy()


@pytest.fixture
def captured_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Records every ``kda_decode`` call the layer makes."""
    calls: list[dict[str, Any]] = []
    real = kda_attention.kda_decode

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, "kwargs": kwargs})
        return real(*args, **kwargs)

    monkeypatch.setattr(kda_attention, "kda_decode", spy)
    return calls


@pytest.mark.compile_on_gpu("gated_delta_conv1d_fwd mutates its state in place")
def test_decode_batch_reaches_op_with_raw_operands(
    captured_calls: list[dict[str, Any]],
) -> None:
    """A one-token-per-sequence batch traces the fused op over raw operands."""
    _run_layer([1, 1, 1, 1])
    assert len(captured_calls) == 1
    kwargs = captured_calls[0]["kwargs"]
    args = captured_calls[0]["args"]
    assert kwargs["input_mode"] == "conv_raw"
    assert kwargs["activation_dtype"] == DType.bfloat16
    assert kwargs["gate_mode"] == "safe"
    assert args[0].dtype == DType.float32
    assert args[3].dtype == DType.float32
    assert _shapes_of(args[5]) == [_NUM_HEADS * _HEAD_DIM]


@pytest.mark.compile_on_gpu("gated_delta_conv1d_fwd mutates its state in place")
def test_prefill_batch_reaches_op_with_raw_operands(
    captured_calls: list[dict[str, Any]],
) -> None:
    """A multi-token batch also traces the op over the raw operands.

    The layer does not gate on batch shape: whatever the batch looks like,
    the op receives the conv's raw fp32 operands and routes at run time
    itself, since a trace-time shape gate cannot see a staged batch's
    symbolic extents.
    """
    _run_layer([3, 5])
    assert len(captured_calls) == 1
    kwargs = captured_calls[0]["kwargs"]
    args = captured_calls[0]["args"]
    assert kwargs["input_mode"] == "conv_raw"
    assert kwargs["activation_dtype"] == DType.bfloat16
    assert args[0].dtype == DType.float32
    assert _shapes_of(args[5]) == [_NUM_HEADS * _HEAD_DIM]


@pytest.mark.compile_on_gpu("gated_delta_conv1d_fwd mutates its state in place")
def test_symbolic_graph_matches_old_path_at_decode_and_prefill(
    ops_runner: _Ops,
) -> None:
    """A symbolic-extent graph routes per executed batch, correctly.

    The same compiled graph -- built over the symbolic dims a served graph
    carries -- runs a decode batch through the fused prologue and a
    prefill-shaped batch through the op's in-run activation and the same
    recurrence an activated call takes, so both must match the old graph
    path: within one bf16 quantum per element at decode, where both round to
    the model dtype, and within the fp32 rounding tolerance on the prefill
    batch.
    """
    for seq_lens, decode_shaped in (([1, 1, 1, 1], True), ([3, 5], False)):
        rng = np.random.default_rng(810)
        inputs = _raw_inputs(rng, seq_lens)
        pool = _initial_pool(rng)
        fused_out, fused_pool = ops_runner.run(True, inputs, pool.copy())
        plain_out, plain_pool = ops_runner.run(False, inputs, pool.copy())
        if decode_shaped:
            _assert_within_one_bf16_quantum(fused_out, plain_out, "output")
            _assert_within_one_bf16_quantum(fused_pool, plain_pool, "state")
        else:
            for name, fused, plain in (
                ("output", fused_out, plain_out),
                ("state", fused_pool, plain_pool),
            ):
                rel = _rel_rmse(fused, plain)
                assert rel < 1e-06, f"prefill {name} rel rmse {rel}"


@pytest.mark.compile_on_gpu("gated_delta_conv1d_fwd mutates its state in place")
def test_fused_decode_layer_output_matches_activated_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused layer decode equals the same layer with fusion forced off,
    within one bf16 quantum per element.

    One seeded layer, redrawn identical inputs: one run takes the fused
    prologue, the other replaces the op call with the activated operand
    convention the fold stands in for -- SiLU plus the model-dtype rounding
    over q/k/v, the fp32 gate pre-activation, zeroed ``a_log`` and
    ``dt_bias``. The replayed pre-activation is derived from the layer's
    raw ``[H, K]`` ``a_log`` parameter, not from the flat operand the op
    call carries, so a wrong packing in the layer disagrees here instead
    of cancelling against itself. The operands must survive that swap
    within one bf16 quantum, on both the output and the state pool: the
    fused and replayed SiLU are separately compiled, so a last-ulp fp32
    difference can flip a bf16 tie, but nothing further.

    The inputs are chosen so the comparison can bite: the module defaults
    zero ``conv_weight``, ``a_log``, ``dt_bias`` and both pools, which
    makes q/k/v and the recurrence constants and lets a dropped SiLU or a
    mis-packed ``a_log`` survive any comparison. Here the conv weights are
    nonzero, the gate parameters are distinct per channel, and the carried
    conv and state pools are nonzero.
    """
    real = kda_attention.kda_decode
    raw_a_log: Tensor | None = None

    def activated_spy(*args: Any, **kwargs: Any) -> Any:
        """Replays one op call over the activated operand convention."""
        assert kwargs["input_mode"] == "conv_raw"
        assert raw_a_log is not None
        q, k, v, raw_gate, beta_logits, _packed_a_log, dt_bias = args[:7]
        model_dtype = kwargs["activation_dtype"]
        activated = [F.silu(t).cast(model_dtype) for t in (q, k, v)]
        pre_gate = (raw_gate + dt_bias) * F.exp(raw_a_log)
        rest = {
            key: value
            for key, value in kwargs.items()
            if key not in ("input_mode", "activation_dtype")
        }
        return real(
            *activated,
            pre_gate,
            beta_logits,
            F.zeros_like(raw_a_log[:, 0]),
            F.zeros_like(dt_bias),
            *args[7:],
            **rest,
        )

    outputs: list[np.ndarray] = []
    pools: list[np.ndarray] = []
    with default_device(md.Accelerator()):
        layer = _make_layer(np.random.default_rng(2026))
        raw_a_log = layer.a_log
        for force_off in (False, True):
            if force_off:
                monkeypatch.setattr(kda_attention, "kda_decode", activated_spy)
            out, state_pool = _run_layer(
                [1, 1, 1, 1], layer, np.random.default_rng(31)
            )
            outputs.append(out)
            pools.append(state_pool)
    _assert_within_one_bf16_quantum(outputs[0], outputs[1], "layer output")
    _assert_within_one_bf16_quantum(pools[0], pools[1], "layer state pool")
