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
from collections.abc import Generator, Iterator
from unittest.mock import MagicMock, NonCallableMock, patch

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph.weights import WeightData
from max.interfaces import LoRAOperation, LoRARequest, LoRAResponse, LoRAStatus
from max.pipelines.core import TextContext
from max.pipelines.lib.config.lora_config import LoRAConfig
from max.pipelines.lib.lora import LoRAManager


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
        lambda name,
        path,
        base_dtype,
        max_lora_rank,
        n_heads,
        n_kv_heads,
        head_dim,
        strict=True: _factory(name)
    )
    return mock_lora_model


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

    status = lora_manager.load_adapter("existing=/another/path")
    assert status == LoRAStatus.LOAD_NAME_EXISTS


def test_get_lora_graph_inputs(
    lora_manager: LoRAManager, configured_mock_lora: MagicMock
) -> None:
    lora_manager.load_adapter("loaded_lora=/path/loaded")
    lora_manager.activate_adapter("loaded_lora")

    device = CPU()
    input_row_offsets = np.array([0, 8, 16])

    lora_ids, _, _, num_active_loras, _, _, _, _ = (
        lora_manager.get_lora_graph_inputs(
            [
                MagicMock(model_name="loaded_lora"),
                MagicMock(model_name=lora_manager.base_model_path),
            ],
            input_row_offsets,
            device,
        )
    )

    lora_ids_np = lora_ids.to_numpy()
    num_active_loras_np = num_active_loras.to_numpy()

    assert np.all(lora_ids_np == [0])
    assert num_active_loras_np[0] == 1


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
    lora_manager.load_adapter("test_lora=/path/to/lora")
    lora_manager.activate_adapter("test_lora")

    assert lora_manager._model_name_to_id("") == lora_manager._NO_ACTIVE_LORA
    assert lora_manager._model_name_to_id(None) == lora_manager._NO_ACTIVE_LORA
    assert (
        lora_manager._model_name_to_id(lora_manager.base_model_path)
        == lora_manager._NO_ACTIVE_LORA
    )

    lora_id = lora_manager._model_name_to_id("test_lora")
    assert lora_id >= 0

    device = CPU()
    input_row_offsets = np.array([0, 8, 16, 24, 32])

    context_empty = NonCallableMock(spec=TextContext)
    context_empty.model_name = ""

    context_none = NonCallableMock(spec=TextContext)
    del context_none.model_name

    context_base = NonCallableMock(spec=TextContext)
    context_base.model_name = lora_manager.base_model_path

    context_lora = NonCallableMock(spec=TextContext)
    context_lora.model_name = "test_lora"

    contexts = [
        context_lora,
        context_empty,
        context_none,
        context_base,
    ]

    lora_ids, _, _, num_active_loras, _, _, _, _ = (
        lora_manager.get_lora_graph_inputs(
            contexts,
            input_row_offsets,
            device,
        )
    )

    lora_ids_np = lora_ids.to_numpy()
    num_active_loras_np = num_active_loras.to_numpy()

    assert len(lora_ids_np) == 1
    assert lora_ids_np[0] >= 0
    assert num_active_loras_np[0] == 1


def test_served_model_name_base_model_mapping(
    lora_manager: LoRAManager, configured_mock_lora: MagicMock
) -> None:
    """Test that served_model_name is recognized as the base model."""
    served_name = "my-custom-model"
    original_base_path = lora_manager.base_model_path
    lora_manager.base_model_path = served_name

    lora_manager.load_adapter("test_lora=/path/to/lora")
    lora_manager.activate_adapter("test_lora")

    assert (
        lora_manager._model_name_to_id(served_name)
        == lora_manager._NO_ACTIVE_LORA
    )
    assert lora_manager._model_name_to_id("") == lora_manager._NO_ACTIVE_LORA
    assert lora_manager._model_name_to_id(None) == lora_manager._NO_ACTIVE_LORA

    lora_id = lora_manager._model_name_to_id("test_lora")
    assert lora_id >= 0

    device = CPU()
    input_row_offsets = np.array([0, 8, 16, 24])

    context_served = NonCallableMock(spec=TextContext)
    context_served.model_name = served_name

    context_lora = NonCallableMock(spec=TextContext)
    context_lora.model_name = "test_lora"

    contexts = [context_lora, context_served]

    lora_ids, _, _, num_active_loras, _, _, _, _ = (
        lora_manager.get_lora_graph_inputs(
            contexts,
            input_row_offsets,
            device,
        )
    )

    lora_ids_np = lora_ids.to_numpy()
    num_active_loras_np = num_active_loras.to_numpy()

    assert len(lora_ids_np) == 1
    assert lora_ids_np[0] >= 0
    assert num_active_loras_np[0] == 1

    lora_manager.base_model_path = original_base_path


# =============================================================================
# Tests for _update_alias_buffers_for_lora
# =============================================================================


def test_update_alias_buffers_copies_lora_a_weight(
    lora_manager: LoRAManager,
) -> None:
    """Test that _update_alias_buffers_for_lora correctly copies LoRA A weights."""
    max_num_loras = lora_manager.max_num_loras
    max_rank = lora_manager.max_lora_rank
    in_features = 64

    buffer_key = "layers.0.self_attn.qkv_lora.lora_A.weight"
    buffer = Buffer.zeros(
        (max_num_loras, 3 * max_rank, in_features), dtype=DType.float32
    )
    lora_manager._alias_buffers[buffer_key] = buffer

    mock_lora = MagicMock()
    weight_data = np.random.randn(3 * max_rank, in_features).astype(np.float32)
    weight_tensor = Buffer.from_numpy(weight_data)

    mock_weight = NonCallableMock(spec=WeightData)
    mock_weight.data = weight_tensor
    mock_lora.get.side_effect = lambda key: (
        mock_weight if key == buffer_key else None
    )

    slot = 2
    lora_manager._update_alias_buffers_for_lora(mock_lora, slot)

    result = Buffer.from_dlpack(lora_manager._alias_buffers[buffer_key])
    result_np = result.to_numpy()

    assert np.allclose(result_np[slot, :, :], weight_data)
    assert np.allclose(result_np[0, :, :], 0.0)
    assert np.allclose(result_np[1, :, :], 0.0)


def test_update_alias_buffers_copies_lora_b_kv_weight(
    lora_manager: LoRAManager,
) -> None:
    """Test that _update_alias_buffers_for_lora correctly splits and copies B_KV weights."""
    max_num_loras = lora_manager.max_num_loras
    max_rank = lora_manager.max_lora_rank
    kv_features = 32

    buffer_key = "layers.0.self_attn.qkv_lora.lora_B_kv.weight"
    buffer = Buffer.zeros(
        (2 * max_num_loras, kv_features, max_rank), dtype=DType.float32
    )
    lora_manager._alias_buffers[buffer_key] = buffer

    mock_lora = MagicMock()
    k_weight = np.random.randn(kv_features, max_rank).astype(np.float32)
    v_weight = np.random.randn(kv_features, max_rank).astype(np.float32)
    weight_data = np.stack([k_weight, v_weight])
    weight_tensor = Buffer.from_numpy(weight_data)

    mock_weight = NonCallableMock(spec=WeightData)
    mock_weight.data = weight_tensor
    mock_lora.get.side_effect = lambda key: (
        mock_weight if key == buffer_key else None
    )

    slot = 1
    lora_manager._update_alias_buffers_for_lora(mock_lora, slot)

    result = Buffer.from_dlpack(lora_manager._alias_buffers[buffer_key])
    result_np = result.to_numpy()

    assert np.allclose(result_np[slot, :, :], k_weight)
    assert np.allclose(result_np[slot + max_num_loras, :, :], v_weight)
    assert np.allclose(result_np[0, :, :], 0.0)


def test_update_alias_buffers_zeros_missing_weight(
    lora_manager: LoRAManager,
) -> None:
    """Test that _update_alias_buffers_for_lora zeros out slots for missing weights."""
    max_num_loras = lora_manager.max_num_loras
    max_rank = lora_manager.max_lora_rank
    in_features = 64

    buffer_key = "layers.0.self_attn.qkv_lora.lora_A.weight"
    initial_data = np.ones(
        (max_num_loras, 3 * max_rank, in_features), dtype=np.float32
    )
    buffer = Buffer.from_numpy(initial_data.copy())
    lora_manager._alias_buffers[buffer_key] = buffer

    mock_lora = MagicMock()
    mock_lora.get.return_value = None

    slot = 2
    lora_manager._update_alias_buffers_for_lora(mock_lora, slot)

    result = Buffer.from_dlpack(lora_manager._alias_buffers[buffer_key])
    result_np = result.to_numpy()

    assert np.allclose(result_np[slot, :, :], 0.0)
    assert np.allclose(result_np[0, :, :], 1.0)
    assert np.allclose(result_np[1, :, :], 1.0)


def test_update_alias_buffers_zeros_missing_b_kv_weight(
    lora_manager: LoRAManager,
) -> None:
    """Test that _update_alias_buffers_for_lora zeros both K and V slots for missing B_KV."""
    max_num_loras = lora_manager.max_num_loras
    max_rank = lora_manager.max_lora_rank
    kv_features = 32

    buffer_key = "layers.0.self_attn.qkv_lora.lora_B_kv.weight"
    initial_data = np.ones(
        (2 * max_num_loras, kv_features, max_rank), dtype=np.float32
    )
    buffer = Buffer.from_numpy(initial_data.copy())
    lora_manager._alias_buffers[buffer_key] = buffer

    mock_lora = MagicMock()
    mock_lora.get.return_value = None

    slot = 1
    lora_manager._update_alias_buffers_for_lora(mock_lora, slot)

    result = Buffer.from_dlpack(lora_manager._alias_buffers[buffer_key])
    result_np = result.to_numpy()

    assert np.allclose(result_np[slot, :, :], 0.0)
    assert np.allclose(result_np[slot + max_num_loras, :, :], 0.0)
    assert np.allclose(result_np[0, :, :], 1.0)


def test_update_alias_buffers_full_qkv_combination(
    lora_manager: LoRAManager,
) -> None:
    """Test _update_alias_buffers_for_lora with all QKV weight types together."""
    max_num_loras = lora_manager.max_num_loras
    max_rank = lora_manager.max_lora_rank
    in_features = 64
    q_features = 64
    kv_features = 32

    lora_a_key = "layers.0.self_attn.qkv_lora.lora_A.weight"
    lora_b_q_key = "layers.0.self_attn.qkv_lora.lora_B_q.weight"
    lora_b_kv_key = "layers.0.self_attn.qkv_lora.lora_B_kv.weight"

    lora_a_buffer = Buffer.zeros(
        (max_num_loras, 3 * max_rank, in_features), dtype=DType.float32
    )
    lora_b_q_buffer = Buffer.zeros(
        (max_num_loras, q_features, max_rank), dtype=DType.float32
    )
    lora_b_kv_buffer = Buffer.zeros(
        (2 * max_num_loras, kv_features, max_rank), dtype=DType.float32
    )

    lora_manager._alias_buffers[lora_a_key] = lora_a_buffer
    lora_manager._alias_buffers[lora_b_q_key] = lora_b_q_buffer
    lora_manager._alias_buffers[lora_b_kv_key] = lora_b_kv_buffer

    np.random.seed(42)
    lora_a_data = np.random.randn(3 * max_rank, in_features).astype(np.float32)
    lora_b_q_data = np.random.randn(q_features, max_rank).astype(np.float32)
    k_data = np.random.randn(kv_features, max_rank).astype(np.float32)
    v_data = np.random.randn(kv_features, max_rank).astype(np.float32)
    lora_b_kv_data = np.stack([k_data, v_data])

    mock_lora = MagicMock()

    def get_weight(key: str) -> NonCallableMock | None:
        if key == lora_a_key:
            mock_weight = NonCallableMock(spec=WeightData)
            mock_weight.data = Buffer.from_numpy(lora_a_data)
            return mock_weight
        elif key == lora_b_q_key:
            mock_weight = NonCallableMock(spec=WeightData)
            mock_weight.data = Buffer.from_numpy(lora_b_q_data)
            return mock_weight
        elif key == lora_b_kv_key:
            mock_weight = NonCallableMock(spec=WeightData)
            mock_weight.data = Buffer.from_numpy(lora_b_kv_data)
            return mock_weight
        return None

    mock_lora.get.side_effect = get_weight

    slot = 0
    lora_manager._update_alias_buffers_for_lora(mock_lora, slot)

    result_a = Buffer.from_dlpack(lora_manager._alias_buffers[lora_a_key])
    assert np.allclose(result_a.to_numpy()[slot, :, :], lora_a_data)

    result_b_q = Buffer.from_dlpack(lora_manager._alias_buffers[lora_b_q_key])
    assert np.allclose(result_b_q.to_numpy()[slot, :, :], lora_b_q_data)

    result_b_kv = Buffer.from_dlpack(lora_manager._alias_buffers[lora_b_kv_key])
    result_b_kv_np = result_b_kv.to_numpy()
    assert np.allclose(result_b_kv_np[slot, :, :], k_data)
    assert np.allclose(result_b_kv_np[slot + max_num_loras, :, :], v_data)

    np.random.seed(123)
    lora_a_data_2 = np.random.randn(3 * max_rank, in_features).astype(
        np.float32
    )
    lora_b_q_data_2 = np.random.randn(q_features, max_rank).astype(np.float32)
    k_data_2 = np.random.randn(kv_features, max_rank).astype(np.float32)
    v_data_2 = np.random.randn(kv_features, max_rank).astype(np.float32)
    lora_b_kv_data_2 = np.stack([k_data_2, v_data_2])

    def get_weight_2(key: str) -> NonCallableMock | None:
        if key == lora_a_key:
            mock_weight = NonCallableMock(spec=WeightData)
            mock_weight.data = Buffer.from_numpy(lora_a_data_2)
            return mock_weight
        elif key == lora_b_q_key:
            mock_weight = NonCallableMock(spec=WeightData)
            mock_weight.data = Buffer.from_numpy(lora_b_q_data_2)
            return mock_weight
        elif key == lora_b_kv_key:
            mock_weight = NonCallableMock(spec=WeightData)
            mock_weight.data = Buffer.from_numpy(lora_b_kv_data_2)
            return mock_weight
        return None

    mock_lora_2 = MagicMock()
    mock_lora_2.get.side_effect = get_weight_2

    slot_2 = 2
    lora_manager._update_alias_buffers_for_lora(mock_lora_2, slot_2)

    result_a = Buffer.from_dlpack(lora_manager._alias_buffers[lora_a_key])
    assert np.allclose(result_a.to_numpy()[slot, :, :], lora_a_data)
    assert np.allclose(result_a.to_numpy()[slot_2, :, :], lora_a_data_2)
    assert np.allclose(result_a.to_numpy()[1, :, :], 0.0)

    result_b_kv = Buffer.from_dlpack(lora_manager._alias_buffers[lora_b_kv_key])
    result_b_kv_np = result_b_kv.to_numpy()

    assert np.allclose(result_b_kv_np[slot, :, :], k_data)
    assert np.allclose(result_b_kv_np[slot + max_num_loras, :, :], v_data)
    assert np.allclose(result_b_kv_np[slot_2, :, :], k_data_2)
    assert np.allclose(result_b_kv_np[slot_2 + max_num_loras, :, :], v_data_2)
