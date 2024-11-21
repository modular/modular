# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests the RandomWeight class."""

import torch
from max.dtype import DType
from max.graph.weights import RandomWeights


def test_random_weight() -> None:
    """Tests that random weight creation works, checking shape and dtype."""
    weights = RandomWeights()
    _ = weights.vision_model.gated_positional_embedding.gate.allocate(
        DType.bfloat16, [1]
    )
    materialized_weights = weights.allocated_weights
    weight_name = "vision_model.gated_positional_embedding.gate"
    assert weight_name in materialized_weights
    assert materialized_weights[weight_name].dtype == torch.bfloat16
