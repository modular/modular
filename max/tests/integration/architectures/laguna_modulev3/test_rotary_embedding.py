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
"""Tests the ModuleV3 Laguna RoPE tables match the graph architecture's.

The graph tables are verified against the transformers reference in
``laguna/test_rotary_embedding.py``; matching them here carries that check
over. The full-attention table is where the two could drift: YaRN over a
partial rotary dim, widened with identity pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from max.driver import CPU
from max.engine import InferenceSession
from max.graph import Graph
from max.nn.rotary_embedding import RotaryEmbedding as GraphRotaryEmbedding
from max.pipelines.architectures.hf_config_shims import LagunaHFConfig
from max.pipelines.architectures.laguna import laguna as graph_laguna
from max.pipelines.architectures.laguna.model_config import LagunaConfig
from max.pipelines.architectures.laguna_modulev3.laguna import build_rope
from max.pipelines.architectures.laguna_modulev3.layers.rotary_embedding import (
    LagunaYarnPartialRotaryEmbedding,
)

TESTDATA = Path(__file__).parents[1] / "laguna" / "testdata"
HEAD_DIM = 128
MAX_SEQ_LEN = 64


def _config(layer_type: str) -> tuple[Any, dict[str, Any]]:
    with (TESTDATA / "s21_config.json").open() as f:
        hf_config = LagunaHFConfig(**json.load(f))
    schedule = LagunaConfig.read_layer_schedule(hf_config)
    params = (
        schedule.full_rope
        if layer_type == "full_attention"
        else schedule.sliding_rope
    )
    assert params is not None
    config = Mock(
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        max_seq_len=MAX_SEQ_LEN,
    )
    return config, dict(params)


def _graph_table(rope: GraphRotaryEmbedding) -> np.ndarray:
    session = InferenceSession(devices=[CPU()])
    with Graph("laguna_freqs_cis", input_types=[]) as graph:
        graph.output(rope.freqs_cis)
    (buf,) = session.load(graph, weights_registry={}).execute()
    return torch.from_dlpack(buf).float().numpy()


@pytest.mark.parametrize("layer_type", ["full_attention", "sliding_attention"])
def test_table_matches_the_graph_architecture(layer_type: str) -> None:
    config, params = _config(layer_type)
    graph_rope = graph_laguna.Laguna._build_rope(config, params, HEAD_DIM)
    v3_rope = build_rope(config, params, HEAD_DIM, CPU())

    expected = _graph_table(graph_rope)
    actual = v3_rope.freqs_cis.to_numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_full_attention_table_is_partial_rotary_yarn() -> None:
    config, params = _config("full_attention")
    rope = build_rope(config, params, HEAD_DIM, CPU())
    assert isinstance(rope, LagunaYarnPartialRotaryEmbedding)
    assert rope.rope_dim == HEAD_DIM // 2

    # The pass-through lanes hold identity pairs at every position.
    table = rope.freqs_cis.to_numpy().reshape(MAX_SEQ_LEN, HEAD_DIM // 2, 2)
    passed = table[:, rope.rope_dim // 2 :, :]
    np.testing.assert_array_equal(passed[..., 0], 1.0)
    np.testing.assert_array_equal(passed[..., 1], 0.0)
