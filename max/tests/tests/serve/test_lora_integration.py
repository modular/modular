# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Integration tests for LoRA functionality."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from max.dtype import DType
from max.interfaces import LoRAOperation, LoRARequest, LoRAResponse, LoRAStatus
from max.pipelines.lib.lora import LoRAManager
from max.pipelines.lib.max_config import LoRAConfig


class MockLoRARequestProcessor:
    """Mock LoRARequestProcessor that doesn't create ZMQ sockets or threads."""

    def __init__(
        self,
        manager: LoRAManager,
        zmq_request_endpoint: str,
        zmq_response_endpoint: str,
    ) -> None:
        self.manager = manager
        self._zmq_running = False

    def _handle_zmq_request(self, request: LoRARequest) -> LoRAResponse:
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

    def stop(self) -> None:
        """Mock stop method."""
        self._zmq_running = False


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

    class NoOpLock:
        def __enter__(self) -> NoOpLock:
            return self

        def __exit__(self, *args) -> None:
            pass

    manager._lora_lock = NoOpLock()  # type: ignore
    manager._validate_lora_path = lambda path: LoRAStatus.SUCCESS  # type: ignore

    yield manager


@pytest.fixture
def temp_adapter() -> Iterator[str]:
    """Create a temporary LoRA adapter directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "adapter_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "v_proj"],
                    "task_type": "CAUSAL_LM",
                }
            )
        )

        (Path(tmpdir) / "adapter_model.safetensors").touch()

        yield tmpdir


@pytest.fixture
def temp_adapters() -> Iterator[list[str]]:
    """Create multiple temporary LoRA adapter directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapters = []
        for i in range(3):
            adapter_dir = Path(tmpdir) / f"adapter_{i}"
            adapter_dir.mkdir()

            config_path = adapter_dir / "adapter_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "r": 16,
                        "lora_alpha": 32,
                        "target_modules": ["q_proj", "v_proj"],
                        "task_type": "CAUSAL_LM",
                    }
                )
            )

            (adapter_dir / "adapter_model.safetensors").touch()
            adapters.append(str(adapter_dir))

        yield adapters


def test_lora_manager_load_unload(
    lora_manager: LoRAManager, temp_adapter: str
) -> None:
    """Test LoRAManager load and unload functionality directly."""

    status = lora_manager.load_adapter(f"test_adapter={temp_adapter}")
    assert status == LoRAStatus.SUCCESS
    assert "test_adapter" in lora_manager.loras
    assert len(lora_manager.loras) == 1

    status = lora_manager.unload_adapter("test_adapter")
    assert status == LoRAStatus.SUCCESS
    assert "test_adapter" not in lora_manager.loras
    assert len(lora_manager.loras) == 0


def test_zmq_handler_direct(
    lora_manager: LoRAManager, temp_adapter: str
) -> None:
    """Test the ZMQ handler functionality directly."""

    handler = lora_manager._request_processor

    load_request = LoRARequest(
        operation=LoRAOperation.LOAD,
        lora_name="test_adapter",
        lora_path=temp_adapter,
    )

    response = handler._handle_zmq_request(load_request)
    assert response.status == LoRAStatus.SUCCESS
    assert "loaded successfully" in response.message

    unload_request = LoRARequest(
        operation=LoRAOperation.UNLOAD,
        lora_name="test_adapter",
    )

    response = handler._handle_zmq_request(unload_request)
    assert response.status == LoRAStatus.SUCCESS
    assert "unloaded successfully" in response.message

    list_request = LoRARequest(operation=LoRAOperation.LIST)
    response = handler._handle_zmq_request(list_request)
    assert response.status == LoRAStatus.SUCCESS


def test_lora_error_handling(lora_manager: LoRAManager) -> None:
    """Test error handling when loading invalid LoRA paths."""

    status = lora_manager.load_adapter("nonexistent=/invalid/path")
    assert status == LoRAStatus.LOAD_INVALID_ADAPTER

    status = lora_manager.unload_adapter("nonexistent")
    assert status == LoRAStatus.UNLOAD_NAME_NONEXISTENT
