# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections.abc import Generator, Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.interfaces import LoRAOperation, LoRARequest, LoRAResponse, LoRAStatus
from max.pipelines.lib.lora import LoRAManager
from max.pipelines.lib.lora_config import LoRAConfig


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

    mock_lora_model.side_effect = (
        lambda name, path, base_dtype, strict=True: _factory(name)
    )
    return mock_lora_model


class MockLoRARequestProcessor:
    """Mock LoRARequestProcessor that doesn't create ZMQ sockets or threads."""

    def __init__(
        self,
        manager: LoRAManager,
        zmq_request_endpoint: str,
        zmq_response_endpoint: str,
    ) -> None:
        self.manager = manager

    def _handle_lora_request(self, request: LoRARequest) -> LoRAResponse:
        """Mock request handler for testing."""

        if request.operation == LoRAOperation.LOAD:
            status = self.manager.load_adapter(
                f"{request.lora_name}={request.lora_path}"
            )
            return LoRAResponse(
                status=status,
                message=f"LoRA '{request.lora_name}' loaded successfully"
                if status == LoRAStatus.SUCCESS
                else "Failed to load",
            )
        elif request.operation == LoRAOperation.UNLOAD:
            status = self.manager.unload_adapter(request.lora_name)  # type: ignore
            return LoRAResponse(
                status=status,
                message=f"LoRA '{request.lora_name}' unloaded successfully"
                if status == LoRAStatus.SUCCESS
                else "Failed to unload",
            )
        elif request.operation == LoRAOperation.LIST:
            return LoRAResponse(
                status=LoRAStatus.SUCCESS,
                message=f"Loaded LoRAs: {', '.join(self.manager.loras)}",
            )
        else:
            return LoRAResponse(
                status=LoRAStatus.LOAD_ERROR, message="Unknown operation"
            )


@pytest.fixture
def lora_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[LoRAManager]:
    """Create a LoRAManager instance with mocked ZMQ handler and locks disabled."""
    monkeypatch.setattr(
        "max.pipelines.lib.lora.LoRARequestProcessor", MockLoRARequestProcessor
    )

    mock_load_weights = MagicMock()
    monkeypatch.setattr(
        "max.pipelines.lib.lora.load_weights", mock_load_weights
    )

    config = LoRAConfig(
        enable_lora=True, max_num_loras=5, max_lora_rank=16, lora_paths=[]
    )

    manager = LoRAManager(
        config=config,
        base_model_path="/mock/path",
        base_dtype=DType.float32,
    )

    manager._validate_lora_path = lambda path: LoRAStatus.SUCCESS  # type: ignore

    yield manager


def test_load_single_adapter(
    lora_manager: LoRAManager, configured_mock_lora: MagicMock
) -> None:
    status = lora_manager.load_adapter("my_cool_lora=/path/to/lora")

    assert status == LoRAStatus.SUCCESS
    assert "my_cool_lora" in lora_manager._loras


def test_load_adapter_no_equals(
    lora_manager: LoRAManager, configured_mock_lora: MagicMock
) -> None:
    status = lora_manager.load_adapter("/path/to/lora")

    assert status == LoRAStatus.SUCCESS
    assert "/path/to/lora" in lora_manager._loras


def test_load_adapters_bulk(
    lora_manager: LoRAManager, configured_mock_lora: MagicMock
) -> None:
    # Load adapters one by one since load_adapters doesn't exist anymore
    statuses = []
    for path in ["a=/path/a", "b=/path/b"]:
        status = lora_manager.load_adapter(path)
        statuses.append(status)

    assert all(status == LoRAStatus.SUCCESS for status in statuses)
    assert "a" in lora_manager._loras
    assert "b" in lora_manager._loras


def test_reloading_existing_adapter_raises(
    lora_manager: LoRAManager, configured_mock_lora: MagicMock
) -> None:
    status = lora_manager.load_adapter("existing=/path/existing")
    assert status == LoRAStatus.SUCCESS

    # Try to load with same name but different path
    status = lora_manager.load_adapter("existing=/another/path")
    assert status == LoRAStatus.LOAD_NAME_EXISTS


def test_get_lora_graph_inputs(
    lora_manager: LoRAManager, configured_mock_lora: MagicMock
) -> None:
    # Load a LoRA
    lora_manager.load_adapter("loaded_lora=/path/loaded")

    # Activate the LoRA to assign it a slot
    lora_manager.activate_adapter("loaded_lora")

    # Create requests
    device = CPU()
    input_row_offsets = np.array([0, 8, 16])

    # Get LoRA graph inputs
    lora_ids, _, _ = lora_manager.get_lora_graph_inputs(
        [
            MagicMock(model_name="loaded_lora"),
            MagicMock(model_name=lora_manager.base_model_path),
        ],
        input_row_offsets,
        device,
    )

    lora_ids_np = lora_ids.to_numpy()

    # Assertions
    assert np.all(lora_ids_np == [0, -1])


def test_lora_invalid_path_validation(lora_manager: LoRAManager) -> None:
    """Test that non-existent paths return appropriate error status."""
    # Test with non-existent local path
    # Note: The test fixture mocks _validate_lora_path to return SUCCESS,
    # so this fails during LoRAModel construction and returns LOAD_INVALID_ADAPTER
    status = lora_manager.load_adapter("/this/path/does/not/exist")
    assert status == LoRAStatus.LOAD_INVALID_ADAPTER

    # Test with name=path format where path doesn't exist
    status = lora_manager.load_adapter("my_lora=/non/existent/path")
    assert status == LoRAStatus.LOAD_INVALID_ADAPTER


def test_model_name_base_model_mapping(
    lora_manager: LoRAManager, configured_mock_lora: MagicMock
) -> None:
    """Test that empty model_name and base_model_path both map to base model."""
    # Load a LoRA adapter
    lora_manager.load_adapter("test_lora=/path/to/lora")
    lora_manager.activate_adapter("test_lora")

    # Test that empty string maps to base model (-1)
    assert lora_manager._model_name_to_id("") == lora_manager._NO_ACTIVE_LORA
    assert lora_manager._model_name_to_id(None) == lora_manager._NO_ACTIVE_LORA

    # Test that base_model_path also maps to base model
    assert (
        lora_manager._model_name_to_id(lora_manager.base_model_path)
        == lora_manager._NO_ACTIVE_LORA
    )

    # Test that the loaded LoRA gets a valid slot ID
    lora_id = lora_manager._model_name_to_id("test_lora")
    assert lora_id >= 0  # Should have a valid slot

    # Test batch handling with various model_name values
    device = CPU()
    input_row_offsets = np.array([0, 8, 16, 24, 32])

    # Create contexts with different model_name values
    context_empty = MagicMock()
    context_empty.model_name = ""

    context_none = MagicMock()
    # Explicitly delete model_name to ensure getattr returns None
    del context_none.model_name

    context_base = MagicMock()
    context_base.model_name = lora_manager.base_model_path

    context_lora = MagicMock()
    context_lora.model_name = "test_lora"

    contexts = [
        context_empty,  # Empty string -> base model
        context_none,  # None (no attribute) -> base model
        context_base,  # Base path -> base model
        context_lora,  # LoRA adapter
    ]

    # Get LoRA graph inputs
    lora_ids, _, _ = lora_manager.get_lora_graph_inputs(
        contexts,
        input_row_offsets,
        device,
    )

    lora_ids_np = lora_ids.to_numpy()

    # The method groups consecutive contexts with the same LoRA ID
    # So we expect: [-1, 0] where:
    # - First group: contexts 0,1,2 all use base model (ID -1)
    # - Second group: context 3 uses LoRA adapter (ID >= 0)
    assert len(lora_ids_np) == 2
    assert lora_ids_np[0] == -1  # Base model group (contexts 0,1,2)
    assert lora_ids_np[1] >= 0  # LoRA adapter group (context 3)
