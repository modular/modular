# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from unittest.mock import MagicMock, patch

import pytest
from max.pipelines.lib.lora import LoRAManager


@pytest.fixture
def mock_lora_model():
    with patch("max.pipelines.lib.lora.LoRAModel") as MockLoRAModel:
        yield MockLoRAModel


@pytest.fixture
def configured_mock_lora(mock_lora_model):  # noqa: ANN001
    """
    Sets up LoRAModel mock to return a mock instance with a .name attribute.
    Usage: test can just rely on the LoRAModel mock working correctly.
    """

    def _factory(name):  # noqa: ANN001
        instance = MagicMock()
        instance.name = name
        return instance

    mock_lora_model.side_effect = lambda name, path: _factory(name)
    return mock_lora_model


@pytest.fixture
def mock_weights():
    return MagicMock(name="Weights")


def test_load_single_adapter(mock_weights, configured_mock_lora) -> None:  # noqa: ANN001
    manager = LoRAManager(base_weights=mock_weights, max_num_loras=2)

    name = manager.load_adapter("my_cool_lora=/path/to/lora")

    assert name == "my_cool_lora"
    assert "my_cool_lora" in manager._loras
    assert manager._lora_index_to_id[0] == "my_cool_lora"


def test_load_adapter_no_equals(mock_weights, configured_mock_lora) -> None:  # noqa: ANN001
    manager = LoRAManager(base_weights=mock_weights, max_num_loras=1)

    name = manager.load_adapter("/path/to/lora")

    assert name == "/path/to/lora"
    assert "/path/to/lora" in manager._loras


def test_load_adapters_bulk(mock_weights, configured_mock_lora) -> None:  # noqa: ANN001
    manager = LoRAManager(base_weights=mock_weights, max_num_loras=2)

    names = manager.load_adapters(["a=/path/a", "b=/path/b"])

    assert names == ["a", "b"]
    assert "a" in manager._loras
    assert "b" in manager._loras


def test_load_adapter_limit_exceeded(
    mock_weights,  # noqa: ANN001
    configured_mock_lora,  # noqa: ANN001
) -> None:
    manager = LoRAManager(base_weights=mock_weights, max_num_loras=1)

    manager.load_adapter("first=/path/first")

    with pytest.raises(RuntimeError, match="No available LoRA slots left"):
        manager.load_adapter("second=/path/second")


def test_reloading_existing_adapter_warns(
    mock_weights,  # noqa: ANN001
    configured_mock_lora,  # noqa: ANN001
) -> None:
    manager = LoRAManager(base_weights=mock_weights, max_num_loras=1)

    manager.load_adapter("existing=/path/existing")

    with patch("max.pipelines.lib.lora.logger") as mock_logger:
        name = manager.load_adapter("existing=/another/path")
        mock_logger.warning.assert_called_once()
        assert name == "existing"
