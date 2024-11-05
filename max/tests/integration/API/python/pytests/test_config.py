# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.pipelines.config import WeightsFormat, PipelineConfig


def test_config_weights_format__raises_with_no_weights_path():
    config = PipelineConfig(architecture="test", weight_path=None)

    with pytest.raises(ValueError):
        config.weights_format


def test_config_weights_format__raises_with_bad_weights_path():
    config = PipelineConfig(
        architecture="test",
        weight_path="this_is_a_random_weight_path_without_extension",
    )

    with pytest.raises(ValueError):
        config.weights_format


def test_config_weights_format__correct_weights_format():
    config = PipelineConfig(
        architecture="test",
        weight_path="model_a.gguf",
    )

    assert config.weights_format == WeightsFormat.gguf

    config.weight_path = "model_b.safetensors"
    assert config.weights_format == WeightsFormat.safetensors
