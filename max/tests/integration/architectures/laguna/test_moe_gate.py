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
"""The router resolves its top-k in float32, whatever dtype the checkpoint stores.

The dtype a checkpoint happens to store ``e_score_correction_bias`` in says
nothing about the precision needed to rank experts. bfloat16 carries about three
significant decimal digits, so across 256 experts the scores collapse into ties
and top-k fills its last slots arbitrarily. ``poolside/Laguna-S-2.1-NVFP4`` is
one such checkpoint.

The logits below are three bfloat16 values whose float32 sigmoids differ but
whose bfloat16 sigmoids are one value.
"""

from __future__ import annotations

import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.laguna.layers.moe_gate import LagunaTopKRouter

NUM_EXPERTS = 256
TOP_K = 10
ROUTED_SCALING_FACTOR = 2.5

# Each of these is bfloat16-exact. In float32 their sigmoids are 0.282196,
# 0.281406 and 0.280616; in bfloat16 all three are 0.281250.
_HIGH_LOGIT = -0.933594
_MID_LOGIT = -0.937500
_LOW_LOGIT = -0.941406

# Eight experts that win on any reading, so the tie family decides slots 9-10.
_CLEAR_WINNERS = tuple(range(8))
# The tie family, arranged so the float32-correct answer is its two *highest*
# indices: a top-k that breaks ties by index order cannot pass by accident.
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
    router = LagunaTopKRouter(
        num_experts_per_token=TOP_K,
        num_experts=NUM_EXPERTS,
        norm_topk_prob=True,
        hidden_dim=NUM_EXPERTS,
        dtype=DType.bfloat16,
        gate_dtype=DType.bfloat16,
        # The dtype whose precision the routing must not inherit.
        correction_bias_dtype=DType.bfloat16,
        devices=[DeviceRef.CPU()],
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )
    # An identity gate makes the router's input its own logits, so the tie
    # structure under test is stated directly instead of through a matmul.
    router.load_state_dict(
        {
            "gate_score.weight": torch.eye(
                NUM_EXPERTS, dtype=torch.bfloat16
            ).contiguous(),
            "e_score_correction_bias": torch.zeros(
                NUM_EXPERTS, dtype=torch.bfloat16
            ),
        }
    )

    with Graph(
        "laguna_router",
        input_types=[
            TensorType(DType.bfloat16, (1, NUM_EXPERTS), device=DeviceRef.CPU())
        ],
    ) as graph:
        (x,) = graph.inputs
        assert isinstance(x, TensorValue)
        graph.output(*router(x))

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry=router.state_dict())
    topk_idx, topk_weight = model.execute(Buffer.from_dlpack(_logits()))
    assert isinstance(topk_idx, Buffer)
    assert isinstance(topk_weight, Buffer)

    selected = set(topk_idx.to_numpy()[0].tolist())
    assert selected == set(_CLEAR_WINNERS) | set(_TIE_FAMILY_HIGH), selected

    # The weights are the unbiased sigmoid scores, renormalised to sum to one
    # and then scaled by ``moe_routed_scaling_factor``.
    total = float(torch.from_dlpack(topk_weight).to(torch.float32).sum().item())
    assert abs(total - ROUTED_SCALING_FACTOR) < 1e-2, total
