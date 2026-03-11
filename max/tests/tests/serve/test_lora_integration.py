# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Integration tests for LoRA functionality."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, NonCallableMock

import numpy as np
import pytest
from max.dtype import DType
from max.interfaces import (
    LoRAOperation,
    LoRARequest,
    LoRAResponse,
    LoRAStatus,
)
from max.pipelines.core import TextContext
from max.pipelines.lib.config.lora_config import LoRAConfig
from max.pipelines.lib.lora import LoRAManager
from safetensors.numpy import save_file


class MockLoRARequestProcessor:
    """Mock LoRARequestProcessor that doesn't create ZMQ sockets or threads."""

    def __init__(
        self,
        manager: LoRAManager,
        zmq_endpoint_base: str,
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
            status = self.manager.unload_adapter(request.lora_name)
            return LoRAResponse(
                status=status,
                message=f"LoRA '{request.lora_name}' unloaded successfully"
                if status == LoRAStatus.SUCCESS
                else "Failed to unload",
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
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        zmq_endpoint_base="fake",
    )

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

    response = handler._handle_lora_request(load_request)
    assert response.status == LoRAStatus.SUCCESS
    assert "loaded successfully" in response.message

    unload_request = LoRARequest(
        operation=LoRAOperation.UNLOAD,
        lora_name="test_adapter",
    )

    response = handler._handle_lora_request(unload_request)
    assert response.status == LoRAStatus.SUCCESS
    assert "unloaded successfully" in response.message


def test_lora_error_handling(lora_manager: LoRAManager) -> None:
    """Test error handling when loading invalid LoRA paths."""

    status = lora_manager.load_adapter("nonexistent=/invalid/path")
    assert status == LoRAStatus.LOAD_INVALID_ADAPTER

    status = lora_manager.unload_adapter("nonexistent")
    assert status == LoRAStatus.UNLOAD_NAME_NONEXISTENT


def test_lora_unsupported_target_modules(lora_manager: LoRAManager) -> None:
    """Test that loading LoRA adapters with unsupported target modules fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create adapter config with unsupported target modules
        config_data = {
            "base_model_name_or_path": "test_model",
            "lora_alpha": 16,
            "r": 8,
            "target_modules": [
                "embed_tokens",
                "lm_head",
            ],  # Unsupported modules
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        }

        config_path = Path(tmpdir) / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Create minimal weights
        weights = {
            "base_model.model.embed_tokens.lora_A.weight": np.random.randn(
                8, 100
            ).astype(np.float32),
            "base_model.model.embed_tokens.lora_B.weight": np.random.randn(
                100, 8
            ).astype(np.float32),
        }
        weights_path = Path(tmpdir) / "adapter_model.safetensors"
        save_file(weights, weights_path)

        # Attempt to load the adapter should fail
        status = lora_manager.load_adapter(f"unsupported={tmpdir}")
        assert status == LoRAStatus.LOAD_INVALID_ADAPTER


def test_lora_lru_eviction(
    lora_manager: LoRAManager, temp_adapters: list[str]
) -> None:
    """Test LRU eviction behavior when slots are full."""

    for i in range(6):
        adapter_path = temp_adapters[i % len(temp_adapters)]
        status = lora_manager.load_adapter(f"adapter_{i}={adapter_path}")
        assert status == LoRAStatus.SUCCESS

    assert len(lora_manager.loras) == 6

    cache_size = len(lora_manager._active_loras)
    assert cache_size < 6, (
        f"Expected LRU cache to evict some items, but has {cache_size} items"
    )

    evicted_count = 0
    for i in range(3):
        if lora_manager._active_loras.get_slot(f"adapter_{i}") is None:
            evicted_count += 1

    assert evicted_count > 0, (
        "Expected at least some early items to be evicted due to LRU policy"
    )


def test_lru_cache_manual_activation(
    lora_manager: LoRAManager, temp_adapters: list[str]
) -> None:
    """Test that LoRAs can be manually activated and sorted correctly."""

    for i in range(3):
        status = lora_manager.load_adapter(f"adapter_{i}={temp_adapters[i]}")
        assert status == LoRAStatus.SUCCESS

    lora_manager._active_loras.clear()
    assert len(lora_manager._active_loras) == 0

    # Manually activate the adapters (as would be done by the scheduler)
    lora_manager.activate_adapter("adapter_0")
    lora_manager.activate_adapter("adapter_1")

    mock_contexts = []
    for name in ["adapter_0", "adapter_1", "/mock/path"]:
        ctx = NonCallableMock(spec=TextContext)
        ctx.model_name = name
        mock_contexts.append(ctx)

    _sorted_batch: list[TextContext] = lora_manager.sort_lora_batch(
        cast(list[TextContext], mock_contexts)
    )

    assert len(lora_manager._active_loras) == 2
    assert lora_manager._active_loras.get_slot("adapter_0") is not None
    assert lora_manager._active_loras.get_slot("adapter_1") is not None
    assert lora_manager._active_loras.get_slot("adapter_2") is None

    slot_0 = lora_manager._model_name_to_id("adapter_0")
    slot_1 = lora_manager._model_name_to_id("adapter_1")
    base_slot = lora_manager._model_name_to_id("/mock/path")

    # Check that LoRA slots are assigned (>= 0) and base model slot is -1
    assert slot_0 >= 0
    assert slot_1 >= 0
    assert base_slot == -1

    # Check that slots are sorted
    assert slot_0 < slot_1


def test_lora_bias_config_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LoRA adapters with bias configuration are rejected."""

    monkeypatch.setattr(
        "max.pipelines.lib.lora.LoRARequestProcessor", MockLoRARequestProcessor
    )

    bias_configs_to_test = ["all", "lora_only"]

    for bias_config in bias_configs_to_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_data = {
                "base_model_name_or_path": "test_model",
                "lora_alpha": 16,
                "r": 8,
                "target_modules": ["q_proj", "v_proj"],
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "bias": bias_config,
            }

            config_path = Path(tmpdir) / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            weights = {
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.random.randn(
                    8, 100
                ).astype(np.float32),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": np.random.randn(
                    100, 8
                ).astype(np.float32),
            }

            weights_path = Path(tmpdir) / "adapter_model.safetensors"
            save_file(weights, weights_path)

            config = LoRAConfig(
                enable_lora=True,
                max_num_loras=5,
                max_lora_rank=16,
                lora_paths=[],
            )

            manager = LoRAManager(
                config=config,
                base_model_path="/mock/path",
                base_dtype=DType.float32,
                n_heads=32,
                n_kv_heads=8,
                head_dim=128,
                zmq_endpoint_base="fake",
            )

            manager._validate_lora_path = lambda path: LoRAStatus.SUCCESS  # type: ignore

            status = manager.load_adapter(
                f"bias_adapter_{bias_config}={tmpdir}"
            )
            assert status == LoRAStatus.LOAD_INVALID_ADAPTER


def test_lora_bias_none_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LoRA adapters with bias='none' are accepted."""
    import numpy as np
    from safetensors.numpy import save_file

    # Mock the LoRARequestProcessor to avoid ZMQ setup
    monkeypatch.setattr(
        "max.pipelines.lib.lora.LoRARequestProcessor", MockLoRARequestProcessor
    )

    mock_lora_model = MagicMock()
    monkeypatch.setattr("max.pipelines.lib.lora.LoRAModel", mock_lora_model)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_data = {
            "base_model_name_or_path": "test_model",
            "lora_alpha": 16,
            "r": 8,
            "target_modules": ["q_proj", "v_proj"],
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "bias": "none",
        }

        config_path = Path(tmpdir) / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        weights = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.random.randn(
                8, 100
            ).astype(np.float32),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": np.random.randn(
                100, 8
            ).astype(np.float32),
        }

        weights_path = Path(tmpdir) / "adapter_model.safetensors"
        save_file(weights, weights_path)

        # Create LoRA manager
        config = LoRAConfig(
            enable_lora=True, max_num_loras=5, max_lora_rank=16, lora_paths=[]
        )

        manager = LoRAManager(
            config=config,
            base_model_path="/mock/path",
            base_dtype=DType.float32,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            zmq_endpoint_base="fake",
        )

        # Mock path validation to pass initial checks
        manager._validate_lora_path = lambda path: LoRAStatus.SUCCESS  # type: ignore

        # Attempt to load the adapter should succeed with bias='none'
        status = manager.load_adapter(f"good_adapter={tmpdir}")
        assert status == LoRAStatus.SUCCESS


def test_lora_allocation_respects_protected_tg_loras(
    monkeypatch: pytest.MonkeyPatch, temp_adapters: list[str]
) -> None:
    """Test that can_allocate_lora_request correctly handles protected TG LoRAs.

    This test verifies the fix for a scheduler deadlock where:
    1. TG batch has multiple LoRAs active (protected)
    2. LoRA capacity is reached
    3. CE requests want a different LoRA
    4. can_allocate_lora_request must return False to defer the request
    5. Without proper deferral, empty CE batches are created indefinitely

    Scenarios tested:
    - Can allocate when protected LoRAs + new LoRA <= capacity
    - Cannot allocate when protected LoRAs + new LoRA > capacity
    - Can allocate when LoRA is already active (just refreshes LRU)
    - Non-protected globally active LoRAs can be evicted
    """
    from max.pipelines.core.context import TextContext
    from max.serve.scheduler.lora_scheduler_utils import (
        can_allocate_lora_request,
    )

    # Mock the LoRARequestProcessor to avoid ZMQ setup
    monkeypatch.setattr(
        "max.pipelines.lib.lora.LoRARequestProcessor", MockLoRARequestProcessor
    )

    mock_load_weights = MagicMock()
    monkeypatch.setattr(
        "max.pipelines.lib.lora.load_weights", mock_load_weights
    )

    # Create a LoRAManager with max_num_loras=2 (simulating the deadlock scenario)
    config = LoRAConfig(
        enable_lora=True,
        max_num_loras=2,
        max_lora_rank=16,
        lora_paths=[],
    )

    manager = LoRAManager(
        config=config,
        base_model_path="/mock/path",
        base_dtype=DType.float32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        zmq_endpoint_base="fake",
    )

    manager._validate_lora_path = lambda _: LoRAStatus.SUCCESS  # type: ignore

    # Load three LoRA adapters
    for i in range(3):
        adapter_path = temp_adapters[i % len(temp_adapters)]
        status = manager.load_adapter(f"adapter_{i}={adapter_path}")
        assert status == LoRAStatus.SUCCESS

    # === Scenario 1: TG has 2 protected LoRAs, CE wants a 3rd ===
    # This reproduces the deadlock from the logs:
    # - decode_reqs has adapter_0 and adapter_1 (protected)
    # - pending_reqs want adapter_2
    # - All CE requests should be deferred

    # Activate adapter_0 and adapter_1 (simulating TG batch)
    manager.activate_adapter("adapter_0")
    manager.activate_adapter("adapter_1")

    # Track protected LoRAs from TG batch
    active_loras = {"adapter_0", "adapter_1"}

    # Try to allocate a CE request with adapter_2
    ce_ctx = NonCallableMock(spec=TextContext)
    ce_ctx.model_name = "adapter_2"

    # Should return False: 2 protected + 1 new = 3 > 2 capacity
    can_allocate = can_allocate_lora_request(ce_ctx, active_loras, manager)
    assert not can_allocate, (
        "Should defer CE request when protected LoRAs + new LoRA exceeds capacity. "
        "This prevents the deadlock where empty CE batches are created indefinitely."
    )

    # === Scenario 2: TG has 1 protected LoRA, CE wants a 2nd (can allocate) ===
    active_loras = {"adapter_0"}
    manager.activate_adapter("adapter_0")

    ce_ctx.model_name = "adapter_1"

    # Should return False because adapter_1 is already globally active
    # (it will just refresh LRU, no eviction needed)
    can_allocate = can_allocate_lora_request(ce_ctx, active_loras, manager)
    assert can_allocate, (
        "Should allow allocation when LoRA is already globally active "
        "(just refreshes LRU)"
    )

    # === Scenario 3: Non-protected LoRAs can be evicted ===
    # Start fresh: activate adapter_0 (protected) and adapter_1 (not protected)
    manager.activate_adapter("adapter_0")
    manager.activate_adapter("adapter_1")

    active_loras = {"adapter_0"}  # Only adapter_0 is protected (from TG)

    # Refresh adapter_0 to make it most recently used
    manager.activate_adapter("adapter_0")

    # Try to allocate adapter_2
    ce_ctx.model_name = "adapter_2"
    can_allocate = can_allocate_lora_request(ce_ctx, active_loras, manager)

    # Should return True: 1 protected + 1 new = 2 <= 2 capacity
    assert can_allocate, (
        "Should allow allocation when protected + new fits within capacity"
    )

    # Activate adapter_2 (will evict adapter_1, not adapter_0)
    manager.activate_adapter("adapter_2")

    # Verify adapter_0 (protected) is still active
    assert manager.is_active_lora("adapter_0"), (
        "Protected TG LoRA should not be evicted"
    )

    # Verify adapter_1 (non-protected) was evicted
    assert not manager.is_active_lora("adapter_1"), (
        "Non-protected LoRA should be evicted to make room"
    )

    # Verify adapter_2 is now active
    assert manager.is_active_lora("adapter_2"), "New LoRA should be active"
