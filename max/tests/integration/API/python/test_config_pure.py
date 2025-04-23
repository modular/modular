# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pickle
from pathlib import Path

import pytest
from max.driver import DeviceSpec, accelerator_count
from max.pipelines import (
    PIPELINE_REGISTRY,
    SupportedEncoding,
)
from max.pipelines.lib import PipelineConfig
from test_common.mocks import mock_pipeline_config_hf_dependencies
from test_common.pipeline_model_dummy import DUMMY_ARCH
from test_common.registry import prepare_registry


@mock_pipeline_config_hf_dependencies
def test_config_init__raises_with_no_model_path():
    # We expect this to fail.
    with pytest.raises(ValueError):
        _ = PipelineConfig(weight_path="file.gguf")


@mock_pipeline_config_hf_dependencies
def test_config_post_init__with_weight_path_but_no_model_path():
    config = PipelineConfig(
        trust_remote_code=True,
        weight_path=[
            Path("modularai/replit-code-1.5/replit-code-v1_5-3b-f32.gguf")
        ],
    )

    assert config.model_config.model_path == "modularai/replit-code-1.5"
    assert config.model_config.weight_path == [
        Path("replit-code-v1_5-3b-f32.gguf")
    ]


@mock_pipeline_config_hf_dependencies
def test_config_init__reformats_with_str_weights_path():
    # We expect this to convert the string.
    config = PipelineConfig(
        model_path="modularai/llama-3.1",
        weight_path="file.gguf",
    )

    assert isinstance(config.model_config.weight_path, list)
    assert len(config.model_config.weight_path) == 1
    assert isinstance(config.model_config.weight_path[0], Path)


@mock_pipeline_config_hf_dependencies
def test_validate_model_path__correct_repo_id_provided():
    config = PipelineConfig(
        model_path="modularai/llama-3.1",
    )

    assert config.model_config.model_path == "modularai/llama-3.1"


class LimitedPickler(pickle.Unpickler):
    """A custom Unpickler class that checks for transformer modules."""

    def find_class(self, module, name):
        if module.startswith("transformers"):
            raise AssertionError(
                "Tried to unpickle class from transformers module, raising an "
                "error because this may break in serving."
            )
        return super().find_class(module, name)


@mock_pipeline_config_hf_dependencies
def test_config_is_picklable(tmp_path):
    config = PipelineConfig(
        model_path="modularai/llama-3.1",
    )

    pickle_path = tmp_path / "config.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(config, f)

    with open(pickle_path, "rb") as f:
        limited_pickler = LimitedPickler(f)
        loaded_config = limited_pickler.load()

    assert loaded_config == config


@mock_pipeline_config_hf_dependencies
def test_config__validate_devices():
    # This test should always have a cpu available.
    _ = PipelineConfig(
        model_path="HuggingFaceTB/SmolLM-135M",
        device_specs=[DeviceSpec.cpu()],
    )

    if accelerator_count() == 0:
        with pytest.raises(ValueError):
            _ = PipelineConfig(
                model_path="HuggingFaceTB/SmolLM-135M",
                device_specs=[DeviceSpec.accelerator()],
            )
    else:
        _ = PipelineConfig(
            model_path="HuggingFaceTB/SmolLM-135M",
            device_specs=[DeviceSpec.accelerator()],
        )


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_config__validates_supported_device():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    # Valid device/encoding combinations.
    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding=SupportedEncoding.float32,
        max_length=1,
    )

    if accelerator_count() > 0:
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding=SupportedEncoding.bfloat16,
            max_length=1,
        )
