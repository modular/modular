# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for InternVL weight adapters."""

from __future__ import annotations

import numpy as np
from max.graph.weights import WeightData
from max.pipelines.architectures.internvl.weight_adapters import (
    convert_internvl_language_model_state_dict,
)


class MockWeight:
    """Mock Weight object for testing."""

    def __init__(self, data_value: str) -> None:
        self._data_value = data_value

    def data(self) -> WeightData:
        # Create a simple numpy array for testing
        test_array = np.array([1.0, 2.0], dtype=np.float32)
        return WeightData.from_numpy(test_array, self._data_value)


def test_weight_filtering_and_mapping():
    """Test that weights are filtered and mapped correctly."""
    test_weights = {
        "language_model.model.layers.0.input_layernorm.weight": MockWeight(
            "llm_data"
        ),
        "language_model.model.layers.0.self_attn.q_proj.bias": MockWeight(
            "bias_data"
        ),
        "language_model.lm_head.weight": MockWeight("output_data"),
        "vision_model.embeddings.patch_embedding.weight": MockWeight(
            "vision_data"
        ),
        "mlp1.0.weight": MockWeight("projection_data"),
    }

    # Use None for unused config parameters in our test
    result = convert_internvl_language_model_state_dict(
        test_weights,  # type: ignore[arg-type]
    )

    expected_outputs = {
        "layers.0.input_layernorm.weight",
        "layers.0.self_attn.q_proj.bias",
        "lm_head.weight",
    }

    assert set(result.keys()) == expected_outputs
