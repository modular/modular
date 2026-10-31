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
"""Numerical verification for :class:`KimiK3LatentMoE`.

The reference below is written from the official reference's own recipe
(``modeling_kimi_linear.py``'s ``KimiSparseMoeBlock.forward``/``moe_infer``),
not from this port's implementation, so a shared mistake could not hide a
bug. It exercises every non-obvious thing the module docstring calls out:
the entry/exit projections are inverted from what their names suggest, the
norm sits between the experts and the exit over the LATENT, the routing
weight multiplies in the latent before the norm, and the shared branch runs
on the original full-width input.

``num_experts=64`` stands in for a smaller count: the router's Mojo kernel
binds its thread count to ``num_experts``, which must be a whole number of
warps (64-wide on this box's AMD GPU) -- the same constraint the full-model
verification test documents.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
)
from max.tensor import Tensor, default_device

_HIDDEN = 16
_LATENT = 12
_MOE_INTERMEDIATE = 10
_NUM_SHARED_EXPERTS = 2
_NUM_EXPERTS = 64
_TOP_K = 4
_SEQ_LEN = 6
_EPS = 1e-5
_BETA = 1.3
_LINEAR_BETA = 0.9
_SCALE = 1.7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _situ_and_mul(gate_up: np.ndarray) -> np.ndarray:
    width = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :width], gate_up[..., width:]
    situ = _BETA * np.tanh(gate / _BETA) * _sigmoid(gate)
    up = _LINEAR_BETA * np.tanh(up / _LINEAR_BETA)
    return situ * up


def _dense_mlp(
    x: np.ndarray, gate_up_weight: np.ndarray, down_weight: np.ndarray
) -> np.ndarray:
    gate_up = x @ gate_up_weight.T
    return _situ_and_mul(gate_up) @ down_weight.T


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    rstd = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x * rstd) * weight


def _reference(
    x: np.ndarray,
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    x64 = x.astype(np.float64)
    w = {k: v.astype(np.float64) for k, v in weights.items()}

    logits = x64 @ w["gate_score_weight"].T
    scores = _sigmoid(logits)
    scores_for_choice = scores + w["e_score_correction_bias"][None, :]
    topk_idx = np.argsort(-scores_for_choice, axis=-1)[:, :_TOP_K]
    topk_weight = np.take_along_axis(scores, topk_idx, axis=-1)
    topk_weight = topk_weight / (
        topk_weight.sum(axis=-1, keepdims=True) + 1e-20
    )
    topk_weight = topk_weight * _SCALE

    # ENTRY: hidden -> latent, despite the "down_proj" name.
    latent = x64 @ w["routed_expert_down_proj_weight"].T

    combined = np.zeros((_SEQ_LEN, _LATENT), dtype=np.float64)
    for t in range(_SEQ_LEN):
        for k in range(_TOP_K):
            e = topk_idx[t, k]
            gate_up = latent[t] @ w["experts_gate_up_proj"][e].T
            activated = _situ_and_mul(gate_up[None, :])[0]
            expert_out = activated @ w["experts_down_proj"][e].T
            # The weighted combine happens IN THE LATENT, before the norm
            # and before the exit projection.
            combined[t] += topk_weight[t, k] * expert_out

    normed = _rms_norm(combined, w["routed_expert_norm_weight"], _EPS)
    # EXIT: latent -> hidden, despite the "up_proj" name.
    routed = normed @ w["routed_expert_up_proj_weight"].T

    # The shared branch reads the ORIGINAL full-width input, never the
    # latent, and is added last.
    shared = _dense_mlp(
        x64, w["shared_gate_up_weight"], w["shared_down_weight"]
    )
    return routed + shared


def _random_weights(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shared_intermediate = _MOE_INTERMEDIATE * _NUM_SHARED_EXPERTS

    def arr(*shape: int, scale: float = 1.0) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    return {
        "gate_score_weight": arr(_NUM_EXPERTS, _HIDDEN, scale=0.1),
        "e_score_correction_bias": arr(_NUM_EXPERTS, scale=0.1),
        "routed_expert_down_proj_weight": arr(
            _LATENT, _HIDDEN, scale=1 / np.sqrt(_HIDDEN)
        ),
        "routed_expert_up_proj_weight": arr(
            _HIDDEN, _LATENT, scale=1 / np.sqrt(_LATENT)
        ),
        "routed_expert_norm_weight": np.ones(_LATENT, dtype=np.float32),
        "experts_gate_up_proj": arr(
            _NUM_EXPERTS,
            2 * _MOE_INTERMEDIATE,
            _LATENT,
            scale=1 / np.sqrt(_LATENT),
        ),
        "experts_down_proj": arr(
            _NUM_EXPERTS,
            _LATENT,
            _MOE_INTERMEDIATE,
            scale=1 / np.sqrt(_MOE_INTERMEDIATE),
        ),
        "shared_gate_up_weight": arr(
            2 * shared_intermediate, _HIDDEN, scale=1 / np.sqrt(_HIDDEN)
        ),
        "shared_down_weight": arr(
            _HIDDEN, shared_intermediate, scale=1 / np.sqrt(shared_intermediate)
        ),
    }


@pytest.mark.compile_on_gpu(
    "mo.moe.single.group.router is staged as an undeclared custom op the interpreter can't bind"
)
def test_matches_independent_reference() -> None:
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((_SEQ_LEN, _HIDDEN)).astype(np.float32)
    weights = _random_weights(7)
    expected = _reference(x_np, weights)

    with default_device(Accelerator()):
        layer = KimiK3LatentMoE(
            hidden_size=_HIDDEN,
            latent_size=_LATENT,
            moe_intermediate_size=_MOE_INTERMEDIATE,
            shared_intermediate_size=_MOE_INTERMEDIATE * _NUM_SHARED_EXPERTS,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_TOP_K,
            beta=_BETA,
            linear_beta=_LINEAR_BETA,
            rms_norm_eps=_EPS,
            routed_scaling_factor=_SCALE,
            num_expert_group=1,
            topk_group=1,
            moe_renormalize=True,
        )
        layer.gate.gate_score.weight = Tensor(
            weights["gate_score_weight"], dtype=DType.float32
        )
        layer.gate.e_score_correction_bias = Tensor(
            weights["e_score_correction_bias"], dtype=DType.float32
        )
        layer.routed_expert_down_proj.weight = Tensor(
            weights["routed_expert_down_proj_weight"], dtype=DType.float32
        )
        layer.routed_expert_up_proj.weight = Tensor(
            weights["routed_expert_up_proj_weight"], dtype=DType.float32
        )
        layer.routed_expert_norm.weight = Tensor(
            weights["routed_expert_norm_weight"], dtype=DType.float32
        )
        layer.experts_gate_up_proj = Tensor(
            weights["experts_gate_up_proj"], dtype=DType.float32
        )
        layer.experts_down_proj = Tensor(
            weights["experts_down_proj"], dtype=DType.float32
        )
        layer.shared_experts.gate_up_proj.weight = Tensor(
            weights["shared_gate_up_weight"], dtype=DType.float32
        )
        layer.shared_experts.down_proj.weight = Tensor(
            weights["shared_down_weight"], dtype=DType.float32
        )

        x = Tensor(x_np, dtype=DType.float32)
        out_np = layer(x).to_numpy().astype(np.float64)

    assert out_np.shape == (_SEQ_LEN, _HIDDEN)
    assert np.all(np.isfinite(out_np))
    rel_error = np.max(np.abs(out_np - expected)) / max(
        np.max(np.abs(expected)), 1e-12
    )
    assert rel_error < 1e-2, f"KimiK3LatentMoE diverges: rel_error={rel_error}"
