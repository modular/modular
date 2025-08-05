# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from max.driver import CPU
from max.pipelines.lib.lora import LoRAManager, LoRAModel, _validate_lora_path


@pytest.fixture
def mock_lora_model() -> Generator[MagicMock, None, None]:
    with patch("max.pipelines.lib.lora.LoRAModel") as MockLoRAModel:
        yield MockLoRAModel


@pytest.fixture
def configured_mock_lora(mock_lora_model: MagicMock) -> MagicMock:
    """
    Sets up LoRAModel mock to return a mock instance with a .name attribute.
    Usage: test can just rely on the LoRAModel mock working correctly.
    """

    def _factory(name: str) -> MagicMock:
        instance = MagicMock()
        instance.name = name
        instance.rank = 16
        return instance

    mock_lora_model.side_effect = lambda name, path: _factory(name)
    return mock_lora_model


@pytest.fixture
def mock_weights() -> MagicMock:
    return MagicMock(name="Weights")


@pytest.fixture
def mock_validate_lora_path() -> Generator[MagicMock, None, None]:
    """Mock _validate_lora_path to bypass filesystem checks during testing."""
    with patch("max.pipelines.lib.lora._validate_lora_path") as mock_validate:
        yield mock_validate


def test_load_single_adapter(
    mock_weights: MagicMock,
    configured_mock_lora: MagicMock,
    mock_validate_lora_path: MagicMock,
) -> None:
    manager = LoRAManager(
        base_model_path="a-name/best-ai-model",
        base_weights=mock_weights,
        max_num_loras=2,
        max_lora_rank=16,
    )

    name = manager.load_adapter("my_cool_lora=/path/to/lora")

    assert name == "my_cool_lora"
    assert "my_cool_lora" in manager._loras
    assert manager._lora_index_to_id[0] == "my_cool_lora"


def test_load_adapter_no_equals(
    mock_weights: MagicMock,
    configured_mock_lora: MagicMock,
    mock_validate_lora_path: MagicMock,
) -> None:
    manager = LoRAManager(
        base_model_path="a-name/best-ai-model",
        base_weights=mock_weights,
        max_num_loras=2,
        max_lora_rank=16,
    )

    name = manager.load_adapter("/path/to/lora")

    assert name == "/path/to/lora"
    assert "/path/to/lora" in manager._loras


def test_load_adapters_bulk(
    mock_weights: MagicMock,
    configured_mock_lora: MagicMock,
    mock_validate_lora_path: MagicMock,
) -> None:
    manager = LoRAManager(
        base_model_path="a-name/best-ai-model",
        base_weights=mock_weights,
        max_num_loras=3,
        max_lora_rank=16,
    )

    names = manager.load_adapters(["a=/path/a", "b=/path/b"])

    assert names == ["a", "b"]
    assert "a" in manager._loras
    assert "b" in manager._loras


def test_load_adapter_limit_exceeded(
    mock_weights: MagicMock,
    configured_mock_lora: MagicMock,
    mock_validate_lora_path: MagicMock,
) -> None:
    manager = LoRAManager(
        base_model_path="a-name/best-ai-model",
        base_weights=mock_weights,
        max_num_loras=1,
        max_lora_rank=16,
    )

    manager.load_adapter("first=/path/first")

    with pytest.raises(RuntimeError, match="No available LoRA slots left"):
        manager.load_adapter("second=/path/second")


def test_reloading_existing_adapter_raises(
    mock_weights: MagicMock,
    configured_mock_lora: MagicMock,
    mock_validate_lora_path: MagicMock,
) -> None:
    manager = LoRAManager(
        base_model_path="a-name/best-ai-model",
        base_weights=mock_weights,
        max_num_loras=2,
        max_lora_rank=16,
    )

    manager.load_adapter("existing=/path/existing")

    with pytest.raises(
        RuntimeError,
        match="LoRA with name existing already exists in LoRA registry",
    ):
        manager.load_adapter("existing=/another/path")


def test_get_lora_graph_inputs(
    mock_weights: MagicMock,
    configured_mock_lora: MagicMock,
    mock_validate_lora_path: MagicMock,
) -> None:
    # Configure the mock to not raise an exception
    mock_validate_lora_path.return_value = None

    manager = LoRAManager(
        base_model_path="a-name/best-ai-model",
        base_weights=mock_weights,
        max_num_loras=2,
        max_lora_rank=16,
    )

    # Load a LoRA
    manager.load_adapter("loaded_lora=/path/loaded")

    # Create requests
    device = CPU()

    # Get LoRA graph inputs
    lora_ids, _ = manager.get_lora_graph_inputs(["loaded_lora", None], device)

    lora_ids_np = lora_ids.to_numpy()

    # Assertions
    assert np.all(lora_ids_np == [0, -1])


def test_lora_remote_hf_repo_validation(mock_weights: MagicMock) -> None:
    """Test that remote HuggingFace repositories are rejected for LoRA adapters."""
    # Test _validate_lora_path function with HF-style paths
    fake_hf_repo = "username/my-lora-model"
    with pytest.raises(
        ValueError, match="appears to be a HuggingFace repository identifier"
    ):
        _validate_lora_path(fake_hf_repo)

    # Test _validate_lora_path function with non-existent local paths
    non_existent = "/path/that/does/not/exist"
    with pytest.raises(ValueError, match="LoRA adapter path does not exist"):
        _validate_lora_path(non_existent)

    # Test LoRAModel initialization with HF-style path
    fake_hf_repo = "mistralai/my-lora-adapter"
    with pytest.raises(
        ValueError, match="appears to be a HuggingFace repository identifier"
    ):
        LoRAModel("test", fake_hf_repo)

    # Test LoRAManager.load_adapter with HF-style paths
    manager = LoRAManager(
        base_model_path="a-name/best-ai-model",
        base_weights=mock_weights,
        max_num_loras=5,
        max_lora_rank=16,
    )

    # Test with HF-style path
    fake_hf_repo = "meta-llama/llama-lora-adapter"
    with pytest.raises(
        ValueError, match="appears to be a HuggingFace repository identifier"
    ):
        manager.load_adapter(fake_hf_repo)

    # Test with name=path format where path is HF repo
    fake_path_with_name = f"my_lora={fake_hf_repo}"
    with pytest.raises(
        ValueError, match="appears to be a HuggingFace repository identifier"
    ):
        manager.load_adapter(fake_path_with_name)

    # Test with non-existent local path
    non_existent_path = "/this/path/does/not/exist"
    with pytest.raises(ValueError, match="LoRA adapter path does not exist"):
        manager.load_adapter(non_existent_path)

    # Test that relative paths would get "does not exist" error, not HF error
    relative_path = "relative/path/to/lora"
    with pytest.raises(ValueError, match="LoRA adapter path does not exist"):
        _validate_lora_path(relative_path)
