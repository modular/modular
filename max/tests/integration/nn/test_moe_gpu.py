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

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import moe_create_indices
from torch.utils.dlpack import from_dlpack


def test_moe_create_indices() -> None:
    host = CPU(0)
    device0 = Accelerator(0)
    devices = [device0]
    session = InferenceSession(devices=devices)

    NUM_EXPERTS = 16

    # set MLIR type for the graph.
    topk_ids_type = TensorType(
        DType.int32, ["num_tokens"], device=DeviceRef.GPU()
    )

    def construct() -> Graph:
        with Graph(
            "test_moe_create_indices",
            input_types=(topk_ids_type,),
        ) as g:
            topk_ids = g.inputs[0].tensor

            (
                token_expert_order,
                expert_start_indices,
                restore_token_order,
                expert_ids,
                expert_usage_stats,
            ) = moe_create_indices(
                topk_ids,
                NUM_EXPERTS,
            )

            g.output(
                token_expert_order,
                expert_start_indices,
                restore_token_order,
                expert_ids,
                expert_usage_stats,
            )
        return g

    graph = construct()
    model = session.load(graph)

    def validate_moe_indices(
        results: Sequence[Buffer],
        topk_ids: np.ndarray,
        NUM_TOKENS: int,
    ) -> None:
        # check output 0
        assert isinstance(results[0], Buffer)
        token_expert_order = from_dlpack(results[0]).cpu().numpy()

        experts_for_tokens = topk_ids[token_expert_order]

        # check that sorted_ids is unique and ranges from 0 to NUM_TOKENS - 1
        assert np.unique(token_expert_order).size == NUM_TOKENS
        assert np.all(token_expert_order >= 0)
        assert np.all(token_expert_order < NUM_TOKENS)

        # check that tokens for the same expert are consecutive
        # this array should be monotonic increasing

        current_expert = experts_for_tokens[0]
        current_count = 1

        unique_experts, counts = np.unique(
            experts_for_tokens, return_counts=True
        )
        expert_counts = dict(zip(unique_experts, counts, strict=True))

        for exp in experts_for_tokens[1:]:
            if exp != current_expert:
                assert expert_counts[current_expert] == current_count
                expert_counts[current_expert] = 0
                current_expert = exp
                current_count = 1
            else:
                current_count += 1

        # check output 2
        assert isinstance(results[2], Buffer)
        restore_token_order = from_dlpack(results[2]).cpu().numpy()
        # check that unperm_ids is the inverse of sorted_ids
        assert np.all(
            token_expert_order[restore_token_order] == np.arange(NUM_TOKENS)
        )

        bin_counts = np.bincount(topk_ids, minlength=NUM_EXPERTS)

        # check output 4
        assert isinstance(results[4], Buffer)
        expert_usage_stats = from_dlpack(results[4]).cpu().numpy()
        max_M_among_experts = expert_usage_stats[0]
        num_experts_used = expert_usage_stats[1]
        # check that max_M_among_experts is the maximum of bin_counts
        assert max_M_among_experts == np.max(bin_counts)
        assert num_experts_used == np.sum(bin_counts > 0)

        expert_ids = from_dlpack(results[3]).cpu().numpy()
        expert_start_indices = from_dlpack(results[1]).cpu().numpy()

        for i in range(num_experts_used):
            start_idx = expert_start_indices[i]
            end_idx = expert_start_indices[i + 1]

            assert np.all(
                experts_for_tokens[start_idx:end_idx] == expert_ids[i]
            )

        assert end_idx == NUM_TOKENS

    topk_ids_0 = torch.randint(
        0, NUM_EXPERTS, size=(2500,), dtype=torch.int32
    ).numpy()
    results_0 = model.execute(Buffer.from_numpy(topk_ids_0).to(device0))
    validate_moe_indices(results_0, topk_ids_0, 2500)

    topk_ids_1 = torch.randint(
        0, NUM_EXPERTS, size=(11,), dtype=torch.int32
    ).numpy()
    results_1 = model.execute(Buffer.from_numpy(topk_ids_1).to(device0))
    validate_moe_indices(results_1, topk_ids_1, 11)

    topk_ids_2 = torch.randint(
        0, NUM_EXPERTS, size=(1,), dtype=torch.int32
    ).numpy()
    results_2 = model.execute(Buffer.from_numpy(topk_ids_2).to(device0))
    validate_moe_indices(results_2, topk_ids_2, 1)
