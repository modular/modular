# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from pathlib import Path
from max.pipelines.config import WeightsFormat, PipelineConfig


def test_config_init__raises_with_none_weights_path():
    # We expect this to fail.
    with pytest.raises(ValueError):
        _ = PipelineConfig(architecture="test", weight_path=None)


def test_config_init__reformats_with_str_weights_path():
    # We expect this to convert the string.
    config = PipelineConfig(architecture="test", weight_path="file.path")

    assert isinstance(config.weight_path, list)
    assert len(config.weight_path) == 1
    assert isinstance(config.weight_path[0], Path)


def test_config_weights_format__raises_with_no_weights_path():
    config = PipelineConfig(architecture="test", weight_path=[])

    with pytest.raises(ValueError):
        config.weights_format


def test_config_weights_format__raises_with_bad_weights_path():
    config = PipelineConfig(
        architecture="test",
        weight_path=[Path("this_is_a_random_weight_path_without_extension")],
    )

    with pytest.raises(ValueError):
        config.weights_format


def test_config_weights_format__raises_with_conflicting_weights_path():
    config = PipelineConfig(
        architecture="test",
        weight_path=[
            Path("this_is_a_random_weight_path_without_extension"),
            Path("this_is_a_gguf_file.gguf"),
        ],
    )

    with pytest.raises(ValueError):
        config.weights_format


def test_config_weights_format__correct_weights_format():
    config = PipelineConfig(
        architecture="test",
        weight_path=[Path("model_a.gguf")],
    )

    assert config.weights_format == WeightsFormat.gguf

    config.weight_path = [
        Path("model_b.safetensors"),
        Path("model_c.safetensors"),
    ]
    assert config.weights_format == WeightsFormat.safetensors


def test_validate_huggingface_repo_id__model_id_provided():
    config = PipelineConfig(
        architecture="test",
        huggingface_repo_id="bert-base-uncased",
    )

    assert config.huggingface_repo_id == "bert-base-uncased"


def test_validate_huggingface_repo_id__correct_repo_id_provided():
    config = PipelineConfig(
        architecture="test",
        huggingface_repo_id="modularai/llama-3.1",
    )

    assert config.huggingface_repo_id == "modularai/llama-3.1"


def test_validate_huggingface_repo_id__bad_repo_provided():
    with pytest.raises(Exception):
        _ = PipelineConfig(
            architecture="test",
            huggingface_repo_id="bert-base-asdfasdf",
        )


def test_hf_config_retrieval():
    config = PipelineConfig(
        architecture="test",
        huggingface_repo_id="modularai/llama-3.1",
    )

    assert config.huggingface_config is not None


def test_hf_architecture():
    config = PipelineConfig(
        architecture=None,
        huggingface_repo_id="modularai/llama-3.1",
    )

    assert config.architecture == "LlamaForCausalLM"

    config = PipelineConfig(
        architecture=None,
        huggingface_repo_id="replit/replit-code-v1_5-3b",
        trust_remote_code=True,
    )

    assert config.architecture == "MPTForCausalLM"
