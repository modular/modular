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
"""Tests the MoE router ops with an elementwise producer fused into their
scores input."""

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import moe_router_group_limited
from torch.utils.dlpack import from_dlpack

TOPK = 8
SCALE = 2.5


def _scores_type(n_experts: int) -> TensorType:
    return TensorType(
        DType.float32, ["num_tokens", n_experts], device=DeviceRef.GPU()
    )


def _bias_type(n_experts: int) -> TensorType:
    return TensorType(DType.float32, [n_experts], device=DeviceRef.GPU())


def _inputs(
    num_tokens: int, n_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(num_tokens, n_experts, device="cuda")
    bias = 0.1 * torch.randn(n_experts, device="cuda")
    return logits, bias


def _reference_weights(
    scores: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    weights = scores.gather(-1, indices)
    return SCALE * weights / weights.sum(-1, keepdim=True)


def _assert_same_routing(
    got_indices: torch.Tensor,
    got_weights: torch.Tensor,
    want_indices: torch.Tensor,
    want_weights: torch.Tensor,
) -> None:
    # Neither side guarantees an order within a token's top-k.
    got_order = torch.argsort(got_indices, dim=-1)
    want_order = torch.argsort(want_indices, dim=-1)
    torch.testing.assert_close(
        got_indices.gather(-1, got_order).long(),
        want_indices.gather(-1, want_order).long(),
    )
    torch.testing.assert_close(
        got_weights.gather(-1, got_order),
        want_weights.gather(-1, want_order),
    )


def _build_group_limited_graph(
    n_experts: int, n_groups: int, topk_group: int
) -> Graph:
    with Graph(
        "moe_router_group_limited",
        input_types=(_scores_type(n_experts), _bias_type(n_experts)),
    ) as graph:
        logits, bias = (v.tensor for v in graph.inputs)
        indices, weights = moe_router_group_limited(
            ops.sigmoid(logits),
            bias,
            n_routed_experts=n_experts,
            n_experts_per_tok=TOPK,
            n_groups=n_groups,
            topk_group=topk_group,
            norm_weights=True,
            routed_scaling_factor=SCALE,
        )
        graph.output(indices, weights)
    return graph


@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_single_group_router_with_fused_scores(
    gpu_session: InferenceSession, num_tokens: int
) -> None:
    n_experts = 384
    logits, bias = _inputs(num_tokens, n_experts)

    compiled = gpu_session.load(
        _build_group_limited_graph(n_experts, n_groups=1, topk_group=1)
    )
    got_indices, got_weights = (
        from_dlpack(t) for t in compiled.execute(logits, bias)
    )

    scores = torch.sigmoid(logits)
    want_indices = torch.topk(scores + bias, TOPK, dim=-1).indices
    _assert_same_routing(
        got_indices,
        got_weights,
        want_indices,
        _reference_weights(scores, want_indices),
    )
