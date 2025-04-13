# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import torch
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.nn.kernels import moe_create_indices
from torch.utils.dlpack import from_dlpack


def test_moe_create_indices() -> None:
    host = CPU(0)
    device0 = Accelerator(0)
    devices = [device0]
    session = InferenceSession(devices=devices)

    NUM_EXPERTS = 16

    # set MLIR type for the graph.
    topk_ids_type = TensorType(DType.uint32, ["num_tokens"])

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

    def validate_moe_indices(results, topk_ids, NUM_TOKENS):
        # check output 0
        token_expert_order = from_dlpack(results[0]).cpu().numpy()

        experts_for_tokens = topk_ids[token_expert_order]

        # check that sorted_ids is unique and ranges from 0 to NUM_TOKENS - 1
        assert np.unique(token_expert_order).size == NUM_TOKENS
        assert np.all(token_expert_order >= 0)
        assert np.all(token_expert_order < NUM_TOKENS)

        # check that tokens for the same expert are consecutive
        # this array should be monotonic increasing
        assert np.all(experts_for_tokens == np.sort(experts_for_tokens))

        # check output 2
        restore_token_order = from_dlpack(results[2]).cpu().numpy()
        # check that unperm_ids is the inverse of sorted_ids
        assert np.all(
            token_expert_order[restore_token_order] == np.arange(NUM_TOKENS)
        )

        bin_counts = np.bincount(topk_ids, minlength=NUM_EXPERTS)

        # check output 4
        expert_usage_stats = from_dlpack(results[4]).cpu().numpy()
        max_M_among_experts = expert_usage_stats[0]
        num_experts_used = expert_usage_stats[1]
        # check that max_M_among_experts is the maximum of bin_counts
        assert max_M_among_experts == np.max(bin_counts)
        assert num_experts_used == np.sum(bin_counts > 0)

        # check output 1
        expert_start_indices = from_dlpack(results[1]).cpu().numpy()
        # check that expert_offsets is the prefix sum of bin_counts
        # Get indices of non-zero bin counts
        non_zero_indices = np.nonzero(bin_counts)[0]
        # Calculate cumulative sum of non-zero bin counts
        cumsum_non_zero = np.cumsum(bin_counts[non_zero_indices])
        # Create expected start indices
        expected_indices = np.concatenate([[0], cumsum_non_zero])
        # Verify expert_start_indices matches expected values
        assert np.all(
            expert_start_indices[: num_experts_used + 1] == expected_indices
        )

        # check output 3
        expert_ids = from_dlpack(results[3]).cpu().numpy()
        # it should be the non-zero bin_counts indices
        assert np.all(expert_ids[:num_experts_used] == non_zero_indices)

    topk_ids_0 = torch.randint(
        0, NUM_EXPERTS, size=(2500,), dtype=torch.uint32
    ).numpy()
    results_0 = model.execute(Tensor.from_numpy(topk_ids_0).to(device0))
    validate_moe_indices(results_0, topk_ids_0, 2500)

    topk_ids_1 = torch.randint(
        0, NUM_EXPERTS, size=(11,), dtype=torch.uint32
    ).numpy()
    results_1 = model.execute(Tensor.from_numpy(topk_ids_1).to(device0))
    validate_moe_indices(results_1, topk_ids_1, 11)

    topk_ids_2 = torch.randint(
        0, NUM_EXPERTS, size=(1,), dtype=torch.uint32
    ).numpy()
    results_2 = model.execute(Tensor.from_numpy(topk_ids_2).to(device0))
    validate_moe_indices(results_2, topk_ids_2, 1)
