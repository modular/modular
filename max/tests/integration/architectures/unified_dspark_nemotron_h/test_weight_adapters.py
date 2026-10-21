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
"""CPU checks of the draft checkpoint contract. No graph, no GPU."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.unified_dspark_nemotron_h.dspark_qwen3 import (
    DSparkQwen3DraftConfig,
)
from max.pipelines.architectures.unified_dspark_nemotron_h.weight_adapters import (
    MASK_EMBEDDING,
    classify_draft_weight,
    convert_draft_state_dict,
    expected_draft_weight_names,
)

VOCAB, WIDTH, MASK = 8, 4, 3

CONFIG = DSparkQwen3DraftConfig(
    hidden_size=WIDTH,
    intermediate_size=8,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=2,
    rms_norm_eps=1e-6,
    vocab_size=VOCAB,
    rope_theta=10000.0,
    sliding_window=16,
    max_position_embeddings=64,
    block_size=4,
    mask_token_id=MASK,
    target_layer_ids=(0, 1),
    markov_rank=2,
    sample_from_anchor=False,
    quantized=False,
)


def _weight(name: str, bits: np.ndarray) -> WeightData:
    bits = np.ascontiguousarray(bits, dtype=np.uint16)
    return WeightData(
        data=Buffer.from_numpy(bits.reshape(-1).view(np.uint8)).view(
            DType.bfloat16, tuple(bits.shape)
        ),
        name=name,
        dtype=DType.bfloat16,
        shape=Shape(bits.shape),
    )


def _table(offset: int = 0) -> np.ndarray:
    return (np.arange(VOCAB * WIDTH, dtype=np.uint16) + offset).reshape(
        VOCAB, WIDTH
    )


def _checkpoint(embedding: np.ndarray) -> dict[str, WeightData]:
    state = {
        name: _weight(name, np.zeros((1,), dtype=np.uint16))
        for name in expected_draft_weight_names(CONFIG)
        if name != MASK_EMBEDDING
    }
    state["embed_tokens.weight"] = _weight("embed_tokens.weight", embedding)
    return state


def _convert(
    state: dict[str, WeightData], target: np.ndarray
) -> dict[str, WeightData]:
    return convert_draft_state_dict(
        state,
        CONFIG,
        target_embedding=_weight("embed_tokens.weight", target),
    )


def test_only_the_mask_row_may_differ() -> None:
    target = _table()
    draft = target.copy()
    draft[MASK] += 100
    converted = _convert(_checkpoint(draft), target)
    assert set(converted) == expected_draft_weight_names(CONFIG)
    mask = np.from_dlpack(
        converted[MASK_EMBEDDING].to_buffer().view(DType.uint16, (WIDTH,))
    )
    np.testing.assert_array_equal(mask, draft[MASK])


def test_a_second_differing_row_is_rejected() -> None:
    target = _table()
    draft = target.copy()
    draft[MASK] += 100
    draft[MASK + 1] += 1
    with pytest.raises(ValueError, match="differs from the target's"):
        _convert(_checkpoint(draft), target)


def test_a_missing_tensor_is_rejected() -> None:
    state = _checkpoint(_table())
    del state["norm.weight"]
    with pytest.raises(ValueError, match=r"norm\.weight"):
        _convert(state, _table())


def test_an_extra_tensor_is_rejected() -> None:
    state = _checkpoint(_table())
    state["layers.0.mystery.weight"] = _weight(
        "layers.0.mystery.weight", np.zeros((1,), dtype=np.uint16)
    )
    with pytest.raises(ValueError, match="mystery"):
        _convert(state, _table())


def test_weights_are_classified() -> None:
    assert classify_draft_weight("embed_tokens.weight") == "shared"
    assert classify_draft_weight("lm_head.weight") == "shared"
    assert classify_draft_weight("confidence_head.weight") == "skipped"
    assert classify_draft_weight("fc.weight") == "loaded"
