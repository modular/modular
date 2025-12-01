# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test Suite for LoRA graph inputs calculation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.pipelines.lib.lora import LoRAManager
from max.pipelines.lib.lora_config import LoRAConfig


class MockTextContext:
    """Mock text generation context for testing."""

    def __init__(self, model_name: str | None, request_id: str = "test"):
        self.model_name = model_name
        self.request_id = request_id


class MockLoRAModel:
    """Mock LoRA model for testing."""

    def __init__(self, name: str, rank: int):
        self.name = name
        self.rank = rank
        self._lora_A: dict = {}
        self._lora_B: dict = {}
        self._lora_bias: dict = {}

    def get(self, key: str) -> None:
        return None


def create_test_lora_manager(
    max_num_loras: int = 4,
    max_lora_rank: int = 16,
    lora_configs: dict[str, int] | None = None,
) -> LoRAManager:
    """Create a LoRAManager with mocked internals for testing.

    Args:
        max_num_loras: Maximum number of LoRAs
        max_lora_rank: Maximum LoRA rank
        lora_configs: Dict mapping lora name -> rank

    Returns:
        LoRAManager with mocked LoRAs registered
    """
    config = LoRAConfig(
        enable_lora=True,
        max_num_loras=max_num_loras,
        max_lora_rank=max_lora_rank,
        lora_paths=[],
    )

    with patch("max.pipelines.lib.lora.LoRARequestProcessor") as mock_processor:
        mock_processor.return_value = MagicMock()
        manager = LoRAManager(
            config=config,
            base_model_path="base_model",
            base_dtype=DType.bfloat16,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            zmq_endpoint_base="tcp://127.0.0.1:5555",
        )

    if lora_configs:
        for lora_name, rank in lora_configs.items():
            mock_lora = MockLoRAModel(lora_name, rank)
            manager._loras[lora_name] = mock_lora  # type: ignore
            manager._active_loras.put(lora_name, mock_lora)  # type: ignore

    return manager


def test_single_lora_request() -> None:
    """Test graph inputs with a single LoRA request."""
    manager = create_test_lora_manager(
        max_num_loras=4, lora_configs={"lora_a": 8}
    )

    context_batch: Any = [MockTextContext(model_name="lora_a")]
    input_row_offsets = np.array([0, 10], dtype=np.uint32)
    device = CPU()

    result = manager.get_lora_graph_inputs(
        context_batch, input_row_offsets, device
    )

    (
        lora_ids,
        lora_ranks,
        lora_grouped_offsets,
        num_active_loras,
        lora_end_idx,
        batch_seq_len,
        lora_ids_kv,
        lora_grouped_offsets_kv,
    ) = result

    assert list(lora_ids.to_numpy()) == [0]
    assert list(lora_ranks.to_numpy()) == [8]
    assert list(lora_grouped_offsets.to_numpy()) == [0, 10]

    assert num_active_loras.to_numpy()[0] == 1
    assert lora_end_idx.to_numpy()[0] == 10
    assert batch_seq_len.to_numpy()[0] == 10

    ids_kv = list(lora_ids_kv.to_numpy())
    offsets_kv = list(lora_grouped_offsets_kv.to_numpy())

    assert ids_kv == [0, 4]
    assert offsets_kv == [0, 10, 20]


def test_multiple_different_loras() -> None:
    """Test graph inputs with multiple different LoRAs."""
    manager = create_test_lora_manager(
        max_num_loras=4, lora_configs={"lora_a": 8, "lora_b": 4}
    )

    context_batch: Any = [
        MockTextContext(model_name="lora_a"),
        MockTextContext(model_name="lora_b"),
    ]
    input_row_offsets = np.array([0, 5, 15], dtype=np.uint32)
    device = CPU()

    result = manager.get_lora_graph_inputs(
        context_batch, input_row_offsets, device
    )

    (
        lora_ids,
        lora_ranks,
        lora_grouped_offsets,
        num_active_loras,
        lora_end_idx,
        batch_seq_len,
        lora_ids_kv,
        lora_grouped_offsets_kv,
    ) = result

    assert list(lora_ids.to_numpy()) == [0, 1]
    assert list(lora_ranks.to_numpy()) == [8, 4]
    assert list(lora_grouped_offsets.to_numpy()) == [0, 5, 15]

    assert num_active_loras.to_numpy()[0] == 2
    assert lora_end_idx.to_numpy()[0] == 15
    assert batch_seq_len.to_numpy()[0] == 15

    ids_kv = list(lora_ids_kv.to_numpy())
    offsets_kv = list(lora_grouped_offsets_kv.to_numpy())

    assert ids_kv == [0, 1, 4, 5]
    batch_end = 15
    assert offsets_kv == [0, 5, 15, batch_end + 5, batch_end + 15]


def test_grouped_same_lora() -> None:
    """Test graph inputs where consecutive requests use the same LoRA."""
    manager = create_test_lora_manager(
        max_num_loras=4, lora_configs={"lora_a": 8, "lora_b": 4}
    )

    context_batch: Any = [
        MockTextContext(model_name="lora_a"),
        MockTextContext(model_name="lora_a"),  # Same as previous
        MockTextContext(model_name="lora_b"),
    ]
    input_row_offsets = np.array([0, 5, 10, 20], dtype=np.uint32)
    device = CPU()

    result = manager.get_lora_graph_inputs(
        context_batch, input_row_offsets, device
    )

    (
        lora_ids,
        lora_ranks,
        lora_grouped_offsets,
        num_active_loras,
        _,
        _,
        _,
        _,
    ) = result

    assert list(lora_ids.to_numpy()) == [0, 1]
    assert list(lora_ranks.to_numpy()) == [8, 4]
    assert list(lora_grouped_offsets.to_numpy()) == [0, 10, 20]

    assert num_active_loras.to_numpy()[0] == 2


def test_base_model_only() -> None:
    """Test graph inputs with only base model (no LoRA)."""
    manager = create_test_lora_manager(
        max_num_loras=4, lora_configs={"lora_a": 8}
    )

    context_batch: Any = [
        MockTextContext(model_name=None),
        MockTextContext(model_name=None),
    ]
    input_row_offsets = np.array([0, 5, 15], dtype=np.uint32)
    device = CPU()

    result = manager.get_lora_graph_inputs(
        context_batch, input_row_offsets, device
    )

    (
        lora_ids,
        lora_ranks,
        lora_grouped_offsets,
        num_active_loras,
        lora_end_idx,
        batch_seq_len,
        _,
        _,
    ) = result

    assert list(lora_ids.to_numpy()) == []
    assert list(lora_ranks.to_numpy()) == []
    assert list(lora_grouped_offsets.to_numpy()) == [0]

    assert num_active_loras.to_numpy()[0] == 0
    assert lora_end_idx.to_numpy()[0] == 0
    assert batch_seq_len.to_numpy()[0] == 15


def test_lora_then_base() -> None:
    """Test graph inputs with LoRA followed by base model."""
    manager = create_test_lora_manager(
        max_num_loras=4, lora_configs={"lora_a": 8}
    )

    context_batch: Any = [
        MockTextContext(model_name="lora_a"),
        MockTextContext(model_name=None),
    ]
    input_row_offsets = np.array([0, 10, 25], dtype=np.uint32)
    device = CPU()

    result = manager.get_lora_graph_inputs(
        context_batch, input_row_offsets, device
    )

    (
        lora_ids,
        lora_ranks,
        lora_grouped_offsets,
        num_active_loras,
        lora_end_idx,
        batch_seq_len,
        lora_ids_kv,
        lora_grouped_offsets_kv,
    ) = result

    assert list(lora_ids.to_numpy()) == [0]
    assert list(lora_ranks.to_numpy()) == [8]
    assert list(lora_grouped_offsets.to_numpy()) == [0, 10]

    assert num_active_loras.to_numpy()[0] == 1
    assert lora_end_idx.to_numpy()[0] == 10
    assert batch_seq_len.to_numpy()[0] == 25

    batch_end = 25
    ids_kv = list(lora_ids_kv.to_numpy())
    offsets_kv = list(lora_grouped_offsets_kv.to_numpy())

    assert ids_kv == [0, 4]
    assert offsets_kv == [0, 10, batch_end + 10]


def test_separate_lora_tensors() -> None:
    """Test that num_active_loras, lora_end_idx, and batch_seq_len are separate tensors."""
    manager = create_test_lora_manager(
        max_num_loras=4, lora_configs={"lora_a": 8, "lora_b": 4}
    )

    context_batch: Any = [
        MockTextContext(model_name="lora_a"),
        MockTextContext(model_name="lora_b"),
        MockTextContext(model_name=None),  # Base model
    ]
    input_row_offsets = np.array([0, 10, 25, 40], dtype=np.uint32)
    device = CPU()

    result = manager.get_lora_graph_inputs(
        context_batch, input_row_offsets, device
    )
    num_active_loras = result[3]
    lora_end_idx = result[4]
    batch_seq_len = result[5]

    assert num_active_loras.to_numpy().shape == (1,)
    assert lora_end_idx.to_numpy().shape == (1,)
    assert batch_seq_len.to_numpy().shape == (1,)

    assert num_active_loras.to_numpy()[0] == 2
    assert lora_end_idx.to_numpy()[0] == 25
    assert batch_seq_len.to_numpy()[0] == 40


def test_kv_offsets_structure() -> None:
    """Test that KV offsets correctly duplicate for K and V portions."""
    manager = create_test_lora_manager(
        max_num_loras=4, lora_configs={"lora_a": 8}
    )

    context_batch: Any = [MockTextContext(model_name="lora_a")]
    input_row_offsets = np.array([0, 100], dtype=np.uint32)
    device = CPU()

    result = manager.get_lora_graph_inputs(
        context_batch, input_row_offsets, device
    )
    lora_grouped_offsets_kv = result[7]

    offsets_kv = list(lora_grouped_offsets_kv.to_numpy())
    batch_end = 100

    assert offsets_kv == [0, 100, batch_end + 100]
