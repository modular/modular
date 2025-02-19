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
from max.pipelines.config import (
    PipelineConfig,
    SupportedEncoding,
    WeightsFormat,
)


def test_config_init__raises_with_no_huggingface_repo_id():
    # We expect this to fail.
    with pytest.raises(ValueError):
        _ = PipelineConfig(weight_path="file.gguf")  # type: ignore


def test_config_post_init__with_weight_path_but_no_huggingface_repo_id():
    config = PipelineConfig(
        trust_remote_code=True,
        weight_path=[
            Path("modularai/replit-code-1.5/replit-code-v1_5-3b-f32.gguf")
        ],
    )

    assert config.huggingface_repo_id == "modularai/replit-code-1.5"
    assert config.weight_path == [Path("replit-code-v1_5-3b-f32.gguf")]


def test_config_init__reformats_with_str_weights_path():
    # We expect this to convert the string.
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        weight_path="file.path",  # type: ignore
    )

    assert isinstance(config.weight_path, list)
    assert len(config.weight_path) == 1
    assert isinstance(config.weight_path[0], Path)


def test_config_weights_format__raises_with_no_weights_path():
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1", weight_path=[]
    )

    with pytest.raises(ValueError):
        config.weights_format


def test_config_weights_format__raises_with_bad_weights_path():
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        weight_path=[Path("this_is_a_random_weight_path_without_extension")],
    )

    with pytest.raises(ValueError):
        config.weights_format


def test_config_weights_format__raises_with_conflicting_weights_path():
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        weight_path=[
            Path("this_is_a_random_weight_path_without_extension"),
            Path("this_is_a_gguf_file.gguf"),
        ],
    )

    with pytest.raises(ValueError):
        config.weights_format


def test_config_weights_format__correct_weights_format():
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        weight_path=[Path("model_a.gguf")],
    )

    assert config.weights_format == WeightsFormat.gguf

    config.weight_path = [
        Path("model_b.safetensors"),
        Path("model_c.safetensors"),
    ]
    assert config.weights_format == WeightsFormat.safetensors


def test_validate_huggingface_repo_id__correct_repo_id_provided():
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
    )

    assert config.huggingface_repo_id == "modularai/llama-3.1"


def test_validate_huggingface_repo_id__bad_repo_provided():
    with pytest.raises(Exception):
        _ = PipelineConfig(
            huggingface_repo_id="bert-base-asdfasdf",
        )


def test_hf_config_retrieval():
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
    )

    assert config.huggingface_config is not None


class LimitedPickler(pickle.Unpickler):
    """A custom Unpickler class that that checks for transformer modules."""

    def find_class(self, module, name):
        if module.startswith("transformers"):
            raise AssertionError(
                "Tried to unpickle class from transformers module, raising an "
                "error because this may break in serving."
            )
        return super().find_class(module, name)


def test_config_is_picklable(tmp_path):
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
    )
    assert config.huggingface_config is not None

    pickle_path = tmp_path / "config.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(config, f)

    with open(pickle_path, "rb") as f:
        limited_pickler = LimitedPickler(f)
        loaded_config = limited_pickler.load()

    assert loaded_config._huggingface_config is None
    assert loaded_config != config

    # Now try loading the Hugging Face config
    assert loaded_config.huggingface_config is not None
    # The configs should now be equivalent.
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
        huggingface_repo_id=downloaded_path,
    )


def test_config_post_init__other_repo_weights():
    config = PipelineConfig(
        huggingface_repo_id="replit/replit-code-v1_5-3b",
        trust_remote_code=True,
        weight_path=[
            Path("modularai/replit-code-1.5/replit-code-v1_5-3b-f32.gguf")
        ],
    )

    assert config._weights_repo_id == "modularai/replit-code-1.5"
    assert config.weight_path == [Path("replit-code-v1_5-3b-f32.gguf")]

    # This example, should not set the _weights_repo_id.
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        weight_path=[
            Path(
                "SDK/integration-test/pipelines/python/llama3/testdata/tinyllama_f32.gguf"
            )
        ],
        quantization_encoding=SupportedEncoding.float32,
    )

    assert config._weights_repo_id is None
    weights_repo = config.huggingface_weights_repo()
    assert weights_repo.repo_id == "modularai/llama-3.1"
    assert config.weight_path == [
        Path(
            "SDK/integration-test/pipelines/python/llama3/testdata/tinyllama_f32.gguf"
        )
    ]
