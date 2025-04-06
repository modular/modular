# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import pickle
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from max.driver import DeviceSpec
from max.pipelines.config import PipelineConfig
from test_common.pipeline_model import mock_estimate_memory_footprint


@mock_estimate_memory_footprint
def test_config_init__raises_with_no_model_path():
    # We expect this to fail.
    with pytest.raises(ValueError):
        _ = PipelineConfig(weight_path="file.gguf")


@mock_estimate_memory_footprint
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


@mock_estimate_memory_footprint
def test_config_init__reformats_with_str_weights_path():
    # We expect this to convert the string.
    config = PipelineConfig(
        model_path="modularai/llama-3.1",
        weight_path="file.gguf",
    )

    assert isinstance(config.model_config.weight_path, list)
    assert len(config.model_config.weight_path) == 1
    assert isinstance(config.model_config.weight_path[0], Path)


@mock_estimate_memory_footprint
def test_validate_model_path__correct_repo_id_provided():
    config = PipelineConfig(
        model_path="modularai/llama-3.1",
    )

    assert config.model_config.model_path == "modularai/llama-3.1"


@mock_estimate_memory_footprint
def test_validate_model_path__bad_repo_provided():
    with pytest.raises(Exception):
        _ = PipelineConfig(
            model_path="bert-base-asdfasdf",
        )


class LimitedPickler(pickle.Unpickler):
    """A custom Unpickler class that checks for transformer modules."""

    def find_class(self, module, name):
        if module.startswith("transformers"):
            raise AssertionError(
                "Tried to unpickle class from transformers module, raising an "
                "error because this may break in serving."
            )
        return super().find_class(module, name)


@mock_estimate_memory_footprint
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


@pytest.mark.skip("huggingface download is flaky")
def test_config__with_local_huggingface_repo():
    # Download huggingface repo to local path.
    target_path = os.path.join(os.getcwd(), "tmp_repo")
    downloaded_path = snapshot_download(
        repo_id="trl-internal-testing/tiny-LlamaForCausalLM-3.2",
        local_dir=target_path,
        revision="main",
    )

    # Load pipeline config with downloaded_path.
    # This should not raise, as the path should be available locally.
    _ = PipelineConfig(
        model_path=downloaded_path,
    )


@mock_estimate_memory_footprint
def test_config__validate_devices():
    # This test should always have a cpu available.
    _ = PipelineConfig(
        model_path="HuggingFaceTB/SmolLM-135M",
        device_specs=[DeviceSpec.cpu()],
    )

    # This test should never have a gpu available.
    with pytest.raises(ValueError):
        _ = PipelineConfig(
            model_path="HuggingFaceTB/SmolLM-135M",
            device_specs=[DeviceSpec.accelerator()],
        )


@mock_estimate_memory_footprint
def test_config_post_init__other_repo_weights():
    config = PipelineConfig(
        model_path="replit/replit-code-v1_5-3b",
        trust_remote_code=True,
        weight_path=[
            Path("modularai/replit-code-1.5/replit-code-v1_5-3b-f32.gguf")
        ],
    )

    assert config.model_config._weights_repo_id == "modularai/replit-code-1.5"
    assert config.model_config.weight_path == [
        Path("replit-code-v1_5-3b-f32.gguf")
    ]
