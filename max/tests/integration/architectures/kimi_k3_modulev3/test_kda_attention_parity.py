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
"""V2-vs-V3 numerical parity for :class:`KimiK3KDAAttention`.

Reuses the independent float64 NumPy replay of the ``kda_decode`` recurrence
already built and trusted for V2
(``kimi_k3/tests/test_kda_attention_gpu.py``'s ``_reference_kda`` -- copied
here rather than imported, matching that file's own stated reasoning for not
sharing a helper between the layer and its reference: a shared mistake would
not be caught). That reference correctly captures ``kda_decode``'s internal
q/k L2-normalization and folded query scale, so using it for V3 avoids
re-deriving that math by hand.

Weights are constructed once as NumPy arrays and assigned directly onto the
V3 layer's attributes (``Linear.weight``, ``conv_weight``, ``a_log``,
``dt_bias``, ``o_norm.weight``) rather than letting the layer randomly
initialize its own. All in float32, matching V2's
``test_kda_attention_gpu.py``'s own float32 case and its 1e-2 tolerance,
derived there from the hardware's tf32 GEMM path, not from this reference.

This file tests the ``-5.0`` "safe" gate arm specifically -- the actual
production path (published K3 sets this bound; V2's own comment warns the
"original" arm was once a silent ~850x whole-model regression before being
caught). The ``None`` "original" arm is tested separately, in its own file
and bazel target (``test_kda_attention_parity_original_gate.py``): running
both arms in one pytest process produces a spurious O(1) failure on
whichever runs second, a cross-call caching artifact in the experimental
Tensor runtime's compiled-graph path, not a real divergence -- each arm
passes cleanly alone.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.experimental.tensor import Tensor, default_device
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
)

# Same geometry as V2's test_kda_attention_gpu.py: the kernel is compiled for
# (key_head_dim, value_head_dim) == (32, 32) or (128, 128).
HIDDEN = 64
NUM_HEADS = 8
HEAD_DIM = 32
CONV_KERNEL = 4
INNER = NUM_HEADS * HEAD_DIM
CONV_DIM = 3 * INNER
MAX_SLOTS = 2

_WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "q_proj_weight": (INNER, HIDDEN),
    "k_proj_weight": (INNER, HIDDEN),
    "v_proj_weight": (INNER, HIDDEN),
    "conv_weight": (CONV_DIM, CONV_KERNEL),
    "f_a_proj_weight": (HEAD_DIM, HIDDEN),
    "f_b_proj_weight": (INNER, HEAD_DIM),
    "b_proj_weight": (NUM_HEADS, HIDDEN),
    "a_log": (NUM_HEADS, HEAD_DIM),
    "dt_bias": (NUM_HEADS, HEAD_DIM),
    "g_proj_weight": (INNER, HIDDEN),
    "o_norm_weight": (HEAD_DIM,),
    "o_proj_weight": (HIDDEN, INNER),
}


def _random_weights(gate_lower_bound: float | None) -> dict[str, np.ndarray]:
    """Copied from V2's test_kda_attention_gpu.py::_random_weights verbatim
    (same seed), so V2 and V3 start from identical weights."""
    rng = np.random.default_rng(20260821)
    weights = {
        name: (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(
            np.float32
        )
        for name, shape in _WEIGHT_SHAPES.items()
    }
    ramp = np.linspace(-1.0, 0.5, NUM_HEADS * HEAD_DIM, dtype=np.float32)
    weights["a_log"] = (
        np.broadcast_to(
            np.linspace(-1.0, 0.5, NUM_HEADS, dtype=np.float32)[:, None],
            (NUM_HEADS, HEAD_DIM),
        ).copy()
        if gate_lower_bound is None
        else ramp.reshape(NUM_HEADS, HEAD_DIM)
    )
    weights["o_norm_weight"] = (
        1.0 + 0.1 * weights["o_norm_weight"] * np.sqrt(HEAD_DIM)
    ).astype(np.float32)
    return weights


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _reference_kda(
    x: np.ndarray,
    weights: dict[str, np.ndarray],
    gate_lower_bound: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Copied from V2's test_kda_attention_gpu.py::_reference_kda verbatim.

    Deliberately not imported/shared with V2's test module (matching that
    file's own stated reasoning): a shared helper would move any wiring
    error in the reference itself onto both sides, hiding a real divergence.
    """
    w = {k: v.astype(np.float64) for k, v in weights.items()}
    x64 = x.astype(np.float64)
    seq_len = x64.shape[0]

    qkv = np.concatenate(
        [
            x64 @ w["q_proj_weight"].T,
            x64 @ w["k_proj_weight"].T,
            x64 @ w["v_proj_weight"].T,
        ],
        axis=1,
    )
    padded = np.concatenate(
        [np.zeros((CONV_KERNEL - 1, CONV_DIM)), qkv], axis=0
    )
    conv = np.stack(
        [
            sum(
                w["conv_weight"][:, j] * padded[t + j]
                for j in range(CONV_KERNEL)
            )
            for t in range(seq_len)
        ]
    )
    conv = conv / (1.0 + np.exp(-conv))

    q, k, v = (
        conv[:, i * INNER : (i + 1) * INNER].reshape(
            seq_len, NUM_HEADS, HEAD_DIM
        )
        for i in range(3)
    )
    raw_gate = (
        (x64 @ w["f_a_proj_weight"].T) @ w["f_b_proj_weight"].T
    ).reshape(seq_len, NUM_HEADS, HEAD_DIM)
    beta = 1.0 / (1.0 + np.exp(-(x64 @ w["b_proj_weight"].T)))

    query_scale = HEAD_DIM**-0.5
    out = np.zeros((seq_len, NUM_HEADS, HEAD_DIM))
    state = np.zeros((NUM_HEADS, HEAD_DIM, HEAD_DIM))
    states = np.zeros((seq_len, NUM_HEADS, HEAD_DIM, HEAD_DIM))
    for t in range(seq_len):
        for h in range(NUM_HEADS):
            pre = raw_gate[t, h] + w["dt_bias"][h]
            a = np.exp(w["a_log"][h])
            if gate_lower_bound is None:
                alpha = np.exp(-a * _softplus(pre))
            else:
                alpha = np.exp(gate_lower_bound / (1.0 + np.exp(-(a * pre))))
            q_factor = query_scale / np.sqrt(q[t, h] @ q[t, h] + 1e-6)
            k_inv = 1.0 / np.sqrt(k[t, h] @ k[t, h] + 1e-6)
            s = state[h] * alpha[:, None]
            kv = (k[t, h] @ s) * k_inv
            s += np.outer(k[t, h], beta[t, h] * (v[t, h] - kv) * k_inv)
            state[h] = s
            out[t, h] = (q[t, h] @ s) * q_factor
        states[t] = state

    gate = (x64 @ w["g_proj_weight"].T).reshape(seq_len, NUM_HEADS, HEAD_DIM)
    KIMI_RMS_NORM_EPS = 1e-5
    rms = np.sqrt(np.mean(out**2, axis=-1, keepdims=True) + KIMI_RMS_NORM_EPS)
    normed = (out / rms) * w["o_norm_weight"]
    normed = normed / (1.0 + np.exp(-gate))
    return normed.reshape(seq_len, INNER) @ w["o_proj_weight"].T, states


def _make_v3_layer_with_weights(
    weights: dict[str, np.ndarray], gate_lower_bound: float | None
) -> KimiK3KDAAttention:
    layer = KimiK3KDAAttention(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL,
        gate_lower_bound=gate_lower_bound,
    )
    layer.q_proj.weight = Tensor(weights["q_proj_weight"], dtype=DType.float32)
    layer.k_proj.weight = Tensor(weights["k_proj_weight"], dtype=DType.float32)
    layer.v_proj.weight = Tensor(weights["v_proj_weight"], dtype=DType.float32)
    layer.conv_weight = Tensor(weights["conv_weight"], dtype=DType.float32)
    layer.f_a_proj.weight = Tensor(
        weights["f_a_proj_weight"], dtype=DType.float32
    )
    layer.f_b_proj.weight = Tensor(
        weights["f_b_proj_weight"], dtype=DType.float32
    )
    layer.b_proj.weight = Tensor(weights["b_proj_weight"], dtype=DType.float32)
    layer.a_log = Tensor(weights["a_log"], dtype=DType.float32)
    layer.dt_bias = Tensor(weights["dt_bias"], dtype=DType.float32)
    layer.g_proj.weight = Tensor(weights["g_proj_weight"], dtype=DType.float32)
    layer.o_norm.weight = Tensor(weights["o_norm_weight"], dtype=DType.float32)
    layer.o_proj.weight = Tensor(weights["o_proj_weight"], dtype=DType.float32)
    return layer


def _run_v3(
    layer: KimiK3KDAAttention, x_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    seq_len = x_np.shape[0]
    with default_device(Accelerator()):
        x = Tensor(x_np, dtype=DType.float32)
        conv_pool = Tensor.zeros(
            [MAX_SLOTS, CONV_DIM, CONV_KERNEL - 1], dtype=DType.float32
        )
        state_pool = Tensor.zeros(
            [MAX_SLOTS, NUM_HEADS, HEAD_DIM, HEAD_DIM], dtype=DType.float32
        )
        slot_idx = Tensor(np.array([0], dtype=np.uint32), dtype=DType.uint32)
        input_row_offsets = Tensor(
            np.array([0, seq_len], dtype=np.uint32), dtype=DType.uint32
        )
        out = layer(
            x, conv_pool, slot_idx, state_pool, slot_idx, input_row_offsets
        )
        out_np = out.cast(DType.float32).to_numpy().astype(np.float64)
        state_np = state_pool.cast(DType.float32).to_numpy().astype(np.float64)
    return out_np, state_np


def _rel_error(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-12))


# V2's float32 test case bounds its tolerance at 1e-2, set by the hardware's
# tf32 GEMM path, not by reference precision (NumPy float64 is far tighter).
# A wiring error scores O(1) there, so this leaves ample room.
TOL = 1e-2


@pytest.mark.compile_on_gpu("gated_delta_conv1d_fwd mutates its state in place")
def test_kda_v3_matches_v2_reference_safe_gate() -> None:
    """The ``-5.0`` "safe" gate arm -- published K3's actual production
    path -- end to end: projections, fused short conv, recurrence, output
    gate, o_norm, o_proj."""
    gate_lower_bound = -5.0
    weights = _random_weights(gate_lower_bound)
    rng = np.random.default_rng(11)
    seq_len = 5
    x_np = rng.standard_normal((seq_len, HIDDEN)).astype(np.float32)

    expected_out, expected_states = _reference_kda(
        x_np, weights, gate_lower_bound
    )

    layer = _make_v3_layer_with_weights(weights, gate_lower_bound)
    out_np, state_np = _run_v3(layer, x_np)

    out_err = _rel_error(out_np, expected_out)
    assert out_err < TOL, (
        f"V3 KDA output diverges from the independent reference:"
        f" rel_error={out_err}"
    )

    # slot 0 is the only one used; must hold the reference's final state.
    state_err = _rel_error(state_np[0], expected_states[seq_len - 1])
    assert state_err < TOL, (
        "V3 KDA's recurrent pool after the call diverges from the"
        f" reference's final state: rel_error={state_err}"
    )
    assert not state_np[1].any(), "slot 1 (unused) must not have moved"
