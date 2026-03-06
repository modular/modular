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
"""Test Suite for LoRA graph inputs calculation."""

from __future__ import annotations

from typing import Any
from unittest.mock import NonCallableMock, patch

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.pipelines.lib.config.lora_config import LoRAConfig
from max.pipelines.lib.lora import LoRAManager, LoRARequestProcessor


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
        mock_processor.return_value = NonCallableMock(spec=LoRARequestProcessor)
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
    # lora_grouped_offsets: token boundaries for each LoRA group in the batch.
    # Here we have 1 request with 10 tokens using lora_a, so offsets are [0, 10].
    assert list(lora_grouped_offsets.to_numpy()) == [0, 10]

    assert num_active_loras.to_numpy()[0] == 1
    assert lora_end_idx.to_numpy()[0] == 10
    assert batch_seq_len.to_numpy()[0] == 10

    ids_kv = list(lora_ids_kv.to_numpy())
    offsets_kv = list(lora_grouped_offsets_kv.to_numpy())

    # Layout is [K_ids..., V_ids...] where V_ids = K_ids + max_num_loras (4).
    # So for lora_id=0: K uses id 0, V uses id 4.
    assert ids_kv == [0, 4]
    # Layout is [K_offsets..., V_offsets...] where V offsets continue from
    # where K ends. For 10 tokens: K spans [0,10], V spans [10,20].
    # The combined offsets become [0, 10, 20].
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
    # Batch has 2 requests: lora_a (5 tokens), lora_b (10 tokens).
    # Offsets mark boundaries: [0, 5, 15] -> group 0 is [0,5), group 1 is [5,15).
    assert list(lora_grouped_offsets.to_numpy()) == [0, 5, 15]

    assert num_active_loras.to_numpy()[0] == 2
    assert lora_end_idx.to_numpy()[0] == 15
    assert batch_seq_len.to_numpy()[0] == 15

    ids_kv = list(lora_ids_kv.to_numpy())
    offsets_kv = list(lora_grouped_offsets_kv.to_numpy())

    # K projection uses ids [0, 1], V projection uses ids [4, 5].
    assert ids_kv == [0, 1, 4, 5]
    # K portion: [0, 5, 15], V portion continues: [15+5=20, 15+15=30].
    # Combined: [0, 5, 15, 20, 30].
    assert offsets_kv == [0, 5, 15, 20, 30]


def test_grouped_same_lora() -> None:
    """Test graph inputs where consecutive requests use the same LoRA."""
    manager = create_test_lora_manager(
        max_num_loras=4, lora_configs={"lora_a": 8, "lora_b": 4}
    )

    context_batch: Any = [
        MockTextContext(model_name="lora_a"),
        MockTextContext(model_name="lora_a"),
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
    # Batch has 3 requests: lora_a (5 tokens), lora_a (5 tokens), lora_b (10 tokens).
    # The two lora_a requests are grouped together: [0, 10, 20] instead of [0, 5, 10, 20].
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

    # No LoRAs so no graph-inputs
    assert list(lora_ids.to_numpy()) == []
    assert list(lora_ranks.to_numpy()) == []
    assert list(lora_grouped_offsets.to_numpy()) == [0]

    assert num_active_loras.to_numpy()[0] == 0
    assert lora_end_idx.shape[0] == 0
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
    # Batch has 2 requests: lora_a (10 tokens), base (15 tokens)
    # The lora_a is the only LoRA so we get: [0, 10] instead of [0, 10, 25].
    assert list(lora_grouped_offsets.to_numpy()) == [0, 10]

    assert num_active_loras.to_numpy()[0] == 1
    assert lora_end_idx.to_numpy()[0] == 10
    assert batch_seq_len.to_numpy()[0] == 25

    ids_kv = list(lora_ids_kv.to_numpy())
    offsets_kv = list(lora_grouped_offsets_kv.to_numpy())

    # K projection uses ids [0], V projection uses ids [4].
    assert ids_kv == [0, 4]
    # K portion: [0, 10], V portion continues: [10+10=20].
    # Combined: [0, 10, 20].
    assert offsets_kv == [0, 10, 20]


def test_lora_end_and_batch_seq_len_tensors() -> None:
    """Test lora_end_idx, and batch_seq_len tensors."""
    manager = create_test_lora_manager(
        max_num_loras=4, lora_configs={"lora_a": 8, "lora_b": 4}
    )

    context_batch: Any = [
        MockTextContext(model_name="lora_a"),
        MockTextContext(model_name="lora_b"),
        MockTextContext(model_name=None),
    ]
    input_row_offsets = np.array([0, 10, 25, 40], dtype=np.uint32)
    device = CPU()

    result = manager.get_lora_graph_inputs(
        context_batch, input_row_offsets, device
    )
    lora_end_idx = result[4]
    batch_seq_len = result[5]

    assert lora_end_idx.to_numpy().shape == (lora_end_idx.to_numpy()[0],)
    assert batch_seq_len.to_numpy().shape == (1,)

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
    lora_ids_kv = result[6]
    lora_grouped_offsets_kv = result[7]

    # K projection uses ids [0], V projection uses ids [4].
    ids_kv = list(lora_ids_kv.to_numpy())
    offsets_kv = list(lora_grouped_offsets_kv.to_numpy())
    # K portion: [0, 100], V portion continues: [100+100=200].
    # Combined: [0, 100, 200].
    assert ids_kv == [0, 4]
    assert offsets_kv == [0, 100, 200]


def test_mixed_lora_and_base_kv_offsets() -> None:
    """Test KV offsets when mixing LoRA and base model sequences."""
    manager = create_test_lora_manager(
        max_num_loras=4, lora_configs={"lora_a": 8, "lora_b": 4}
    )

    context_batch: Any = [
        MockTextContext(model_name="lora_a"),
        MockTextContext(model_name="lora_b"),
        MockTextContext(model_name=None),
    ]
    input_row_offsets = np.array([0, 10, 25, 45], dtype=np.uint32)
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
    assert list(lora_grouped_offsets.to_numpy()) == [0, 10, 25]

    assert num_active_loras.to_numpy()[0] == 2
    assert lora_end_idx.to_numpy()[0] == 25
    assert batch_seq_len.to_numpy()[0] == 45

    ids_kv = list(lora_ids_kv.to_numpy())
    offsets_kv = list(lora_grouped_offsets_kv.to_numpy())

    # K projection uses ids [0, 1], V projection uses ids [4, 5].
    assert ids_kv == [0, 1, 4, 5]
    # K portion: [0, 10, 25], V portion continues: [25 + 10, 35 + 15].
    # Combined: [0, 10, 20].
    assert offsets_kv == [0, 10, 25, 35, 50]
