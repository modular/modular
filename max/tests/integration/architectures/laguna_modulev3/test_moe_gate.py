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
"""Tests the ModuleV3 Laguna router resolves its top-k in float32.

``poolside/Laguna-S-2.1`` ships ``e_score_correction_bias`` in bfloat16. In
bfloat16, sigmoid scores across 256 experts collapse into tie groups and top-k
fills its last slots arbitrarily. The logits below are three distinct
bfloat16 values whose float32 sigmoids are distinct and whose bfloat16 sigmoids
are one value; see ``laguna/test_moe_gate.py`` for the derivation.
"""

from __future__ import annotations

import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.laguna_modulev3.layers.moe_gate import (
    LagunaTopKRouter,
)

NUM_EXPERTS = 256
TOP_K = 10
ROUTED_SCALING_FACTOR = 2.5

_HIGH_LOGIT = -0.933594
_MID_LOGIT = -0.937500
_LOW_LOGIT = -0.941406
_CLEAR_WINNERS = tuple(range(8))
# The float32-correct answer is the tie family's two highest indices, so a
# top-k that breaks ties by index order cannot pass by accident.
_TIE_FAMILY_HIGH = (205, 206)
_TIE_FAMILY_MID = (207,)
_TIE_FAMILY_LOW = tuple(range(200, 205))


def _logits() -> torch.Tensor:
    logits = torch.full((1, NUM_EXPERTS), -5.0, dtype=torch.float32)
    for rank, expert in enumerate(_CLEAR_WINNERS):
        logits[0, expert] = 1.0 - 0.1 * rank
    for expert in _TIE_FAMILY_LOW:
        logits[0, expert] = _LOW_LOGIT
    for expert in _TIE_FAMILY_MID:
        logits[0, expert] = _MID_LOGIT
    for expert in _TIE_FAMILY_HIGH:
        logits[0, expert] = _HIGH_LOGIT
    return logits.to(torch.bfloat16)


def test_router_resolves_the_top_k_in_float32() -> None:
    with F.lazy(), default_dtype(DType.bfloat16):
        router = LagunaTopKRouter(
            hidden_dim=NUM_EXPERTS,
            num_experts=NUM_EXPERTS,
            num_experts_per_token=TOP_K,
            norm_topk_prob=True,
            gate_dtype=DType.bfloat16,
            correction_bias_dtype=DType.bfloat16,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
        )
        router.to(CPU())

    # An identity gate makes the router's input its own logits, so the tie
    # structure under test is stated directly.
    compiled = router.compile(
        TensorType(DType.bfloat16, (1, NUM_EXPERTS), device=DeviceRef.CPU()),
        weights={
            "gate_score.weight": torch.eye(
                NUM_EXPERTS, dtype=torch.bfloat16
            ).contiguous(),
            "e_score_correction_bias": torch.zeros(
                NUM_EXPERTS, dtype=torch.bfloat16
            ),
        },
    )
    topk_idx, topk_weight = compiled.execute_raw(Buffer.from_dlpack(_logits()))
    assert isinstance(topk_idx, Buffer)
    assert isinstance(topk_weight, Buffer)

    selected = set(topk_idx.to_numpy()[0].tolist())
    assert selected == set(_CLEAR_WINNERS) | set(_TIE_FAMILY_HIGH), selected
    assert topk_weight.dtype == DType.bfloat16
    total = float(torch.from_dlpack(topk_weight).to(torch.float32).sum().item())
    assert abs(total - ROUTED_SCALING_FACTOR) < 1e-2, total
