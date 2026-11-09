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
"""Tests the packed routing ops end to end in a graph: the router writes the
packed rows, the index build reads them."""

from __future__ import annotations

import pytest
import torch
from max.driver import accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import (
    moe_create_indices_packed,
    moe_router_group_limited,
    moe_single_group_router_packed,
)
from torch.utils.dlpack import from_dlpack

N_EXPERTS = 128
TOPK = 8
SCALE = 2.5
ALIGN = 128


def _build_graph(expert_offset: int, local_experts: int) -> Graph:
    with Graph(
        "moe_routing_packed",
        input_types=(
            TensorType(
                DType.float32,
                ["num_tokens", N_EXPERTS],
                device=DeviceRef.GPU(),
            ),
            TensorType(DType.float32, [N_EXPERTS], device=DeviceRef.GPU()),
        ),
    ) as graph:
        scores, bias = (v.tensor for v in graph.inputs)
        ids, weights, packed = moe_single_group_router_packed(
            scores, bias, N_EXPERTS, TOPK, SCALE
        )
        want_ids, want_weights = moe_router_group_limited(
            scores,
            bias,
            n_routed_experts=N_EXPERTS,
            n_experts_per_tok=TOPK,
            n_groups=1,
            topk_group=1,
            norm_weights=True,
            routed_scaling_factor=SCALE,
        )
        graph.output(
            ids,
            weights,
            packed,
            want_ids,
            want_weights,
            *moe_create_indices_packed(
                packed,
                TOPK,
                expert_offset=expert_offset,
                local_experts=local_experts,
            ),
        )
    return graph


@pytest.mark.skipif(
    accelerator_api() != "cuda",
    reason="the packed index build is tested on NVIDIA GPUs only",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("expert_offset, local_experts", [(0, 128), (32, 16)])
def test_packed_routing_matches_the_router_and_a_stable_sort(
    gpu_session: InferenceSession,
    num_tokens: int,
    expert_offset: int,
    local_experts: int,
) -> None:
    torch.manual_seed(0)
    scores = torch.rand(num_tokens, N_EXPERTS, device="cuda")
    bias = 0.1 * torch.randn(N_EXPERTS, device="cuda")
    compiled = gpu_session.load(_build_graph(expert_offset, local_experts))
    (
        ids,
        weights,
        packed,
        want_ids,
        want_weights,
        order,
        start,
        restore,
        expert_ids,
        usage,
        scales_offset,
        dense_weights,
    ) = (from_dlpack(t).cpu() for t in compiled.execute(scores, bias))

    # The packed op's ids and weights are the plain router's, bit for bit,
    # and each packed row is those ids as f32, then those weights.
    assert torch.equal(ids, want_ids)
    assert torch.equal(weights, want_weights)
    assert torch.equal(packed[:, :TOPK], ids.float())
    assert torch.equal(packed[:, TOPK:], weights)

    # The index build: other devices' slots go to one trailing group, and
    # slots keep their order within a group.
    groups = local_experts + 1
    group = ids.flatten().long() - expert_offset
    group[(group < 0) | (group >= local_experts)] = local_experts
    counts = torch.bincount(group, minlength=groups)
    csr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    aligned = torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            ((counts + ALIGN - 1) // ALIGN * ALIGN).cumsum(0),
        ]
    )
    want_order = torch.argsort(group, stable=True)
    assert torch.equal(order.long(), want_order)
    assert torch.equal(restore.long()[want_order], torch.arange(group.numel()))
    assert torch.equal(start.long(), csr)
    assert torch.equal(expert_ids.long(), torch.arange(groups))
    assert torch.equal(
        scales_offset.long(), aligned[:-1] // ALIGN - csr[:-1] // ALIGN
    )
    assert usage.tolist() == [counts.max().item(), groups]
    assert torch.equal(dense_weights, weights)
