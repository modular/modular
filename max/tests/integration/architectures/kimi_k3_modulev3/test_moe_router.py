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
"""Numerical verification for :func:`make_kimi_k3_moe_router`.

The reference below is adapted from V2's own trusted transcription of the
official reference (``kimi_k3/tests/test_moe_router.py``), copied here rather
than imported so a shared mistake between the layer and its reference could
not hide a bug.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_router import (
    make_kimi_k3_moe_router,
)
from max.tensor import Tensor, default_device, default_dtype

_TOKENS = 12
_HIDDEN = 32
_EXPERTS = 64
_TOP_K = 4
_SCALE = 2.5


def _reference(
    hidden: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    bias: npt.NDArray[np.float32],
    num_expert_group: int,
    topk_group: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Transcribes ``KimiMoEGate.forward``, ``modeling_kimi_linear.py:775-838``.

    Run in float64 so a disagreement is the gate's logic and not the
    reference's own rounding.
    """
    logits = hidden.astype(np.float64) @ weight.astype(np.float64).T
    scores = 1.0 / (1.0 + np.exp(-logits))
    scores_for_choice = scores + bias.astype(np.float64)[None, :]
    if num_expert_group > 1 and num_expert_group > topk_group:
        grouped = scores_for_choice.reshape(_TOKENS, num_expert_group, -1)
        group_scores = np.sort(grouped, axis=-1)[..., -2:].sum(axis=-1)
        group_idx = np.argsort(-group_scores, axis=-1)[:, :topk_group]
        keep = np.zeros_like(group_scores, dtype=bool)
        np.put_along_axis(keep, group_idx, True, axis=-1)
        keep = np.repeat(keep, _EXPERTS // num_expert_group, axis=-1)
        tmp_scores = np.where(keep, scores_for_choice, -np.inf)
    else:
        tmp_scores = scores_for_choice
    topk_idx = np.argsort(-tmp_scores, axis=-1)[:, :_TOP_K]
    topk_weight = np.take_along_axis(scores, topk_idx, axis=-1)
    topk_weight /= topk_weight.sum(axis=-1, keepdims=True) + 1e-20
    return topk_idx, topk_weight * _SCALE


def _random_inputs(
    seed: int,
) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
]:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((_TOKENS, _HIDDEN)).astype(np.float32),
        rng.standard_normal((_EXPERTS, _HIDDEN)).astype(np.float32),
        rng.standard_normal(_EXPERTS).astype(np.float32),
    )


@pytest.mark.compile_on_gpu(
    "the mo.moe router kernels are staged as an undeclared custom op the interpreter can't bind"
)
@pytest.mark.parametrize(
    ("num_expert_group", "topk_group"),
    [(1, 1), (8, 4)],
    ids=["ungrouped", "group_limited"],
)
def test_matches_independent_reference(
    num_expert_group: int, topk_group: int
) -> None:
    """Published K3 runs ungrouped (``num_expert_group=1``); the
    group-limited case is checked too so a mask that silently selected
    everything would not pass by coincidence."""
    hidden_np, weight_np, bias_np = _random_inputs(0)
    expected_idx, expected_weight = _reference(
        hidden_np, weight_np, bias_np, num_expert_group, topk_group
    )

    with default_device(Accelerator()):
        router = make_kimi_k3_moe_router(
            hidden_dim=_HIDDEN,
            num_experts=_EXPERTS,
            num_experts_per_token=_TOP_K,
            routed_scaling_factor=_SCALE,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            moe_renormalize=True,
        )
        router.gate_score.weight = Tensor(weight_np, dtype=DType.float32)
        router.e_score_correction_bias = Tensor(bias_np, dtype=DType.float32)

        hidden = Tensor(hidden_np, dtype=DType.float32)
        idx, weight = router(hidden)
        idx_np = idx.cast(DType.int64).to_numpy()
        weight_np_out = weight.to_numpy().astype(np.float64)

    # Selection must match exactly: a biased-vs-unbiased gather mistake still
    # produces plausible-looking weights at the WRONG indices.
    np.testing.assert_array_equal(
        np.sort(idx_np, axis=-1), np.sort(expected_idx, axis=-1)
    )
    rel_error = np.max(np.abs(weight_np_out - expected_weight)) / max(
        np.max(np.abs(expected_weight)), 1e-12
    )
    # The shared `group_limited_router_kernel`'s GPU sigmoid/exp carries
    # about this much deviation from a float64 CPU reference -- DeepSeek
    # V2's own test of the same kernel (`test_moe_gate.py`) uses the same
    # `1e-3` rtol, not a tighter one, for this reason.
    assert rel_error < 1e-3, f"router weights diverge: rel_error={rel_error}"


def test_gate_weight_is_forced_to_fp32_under_a_different_ambient_dtype() -> (
    None
):
    """The gate matmul must run in fp32 even when constructed under a
    non-fp32 ambient default dtype -- the whole reason this file wraps
    ``DeepseekV3TopKRouter`` in a ``default_dtype(DType.float32)`` context."""
    with default_device(Accelerator()), default_dtype(DType.bfloat16):
        router = make_kimi_k3_moe_router(
            hidden_dim=_HIDDEN,
            num_experts=_EXPERTS,
            num_experts_per_token=_TOP_K,
            routed_scaling_factor=_SCALE,
            num_expert_group=1,
            topk_group=1,
            moe_renormalize=True,
        )
        assert router.gate_score.weight.dtype == DType.float32
