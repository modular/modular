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
"""Laguna sizes its attention weights per layer, from the layer's own query-head
count rather than one model-wide scalar.

S-2.1 gives its sliding layers 72 query heads and its full layers 48, against 8
KV heads and a head_dim of 128 everywhere. A layer built with the wrong count
does not fail in the graph -- it fails much later, loading
``q_proj.weight`` against a checkpoint tensor of a different shape -- so the
projection shapes are asserted directly here.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.laguna.layers.attention import LagunaAttention
from max.pipelines.architectures.laguna.layers.rotary_embedding import (
    LagunaRotaryEmbedding,
)
from max.pipelines.lib import KVCacheConfig

HIDDEN_SIZE = 3072
HEAD_DIM = 128
NUM_KV_HEADS = 8


def _kv_params() -> KVCacheParams:
    return KVCacheConfig().to_params(
        dtype=DType.bfloat16,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        devices=[DeviceRef.CPU()],
    )


def _attention(
    num_attention_heads: int,
    *,
    is_sliding: bool = False,
    local_window_size: int = -1,
    gating_mode: str = "per-element",
) -> LagunaAttention:
    rope = LagunaRotaryEmbedding(
        dim=HIDDEN_SIZE,
        n_heads=num_attention_heads,
        theta=10000.0,
        max_seq_len=1024,
        head_dim=HEAD_DIM,
        interleaved=False,
        scaling_params=ProportionalScalingParams(partial_rotary_factor=1.0),
    )
    return LagunaAttention(
        rope=rope,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=NUM_KV_HEADS,
        hidden_size=HIDDEN_SIZE,
        kv_params=_kv_params(),
        layer_idx=0,
        is_sliding=is_sliding,
        local_window_size=local_window_size,
        gating_mode=gating_mode,
        dtype=DType.bfloat16,
        devices=[DeviceRef.CPU()],
        scale=HEAD_DIM**-0.5,
        norm_dtype=DType.float32,
    )


@pytest.mark.parametrize("num_attention_heads", [72, 48])
def test_projections_are_sized_by_the_layers_own_head_count(
    num_attention_heads: int,
) -> None:
    attention = _attention(num_attention_heads)
    q_dim = num_attention_heads * HEAD_DIM
    kv_dim = NUM_KV_HEADS * HEAD_DIM

    assert attention.q_proj.weight.shape == [q_dim, HIDDEN_SIZE]
    assert attention.o_proj.weight.shape == [HIDDEN_SIZE, q_dim]
    # KV heads stay uniform, so a 72-head layer is a wider GQA ratio, not a
    # wider cache.
    assert attention.k_proj.weight.shape == [kv_dim, HIDDEN_SIZE]
    assert attention.v_proj.weight.shape == [kv_dim, HIDDEN_SIZE]


@pytest.mark.parametrize("num_attention_heads", [72, 48])
def test_a_per_head_gate_is_one_weight_row_per_query_head(
    num_attention_heads: int,
) -> None:
    attention = _attention(num_attention_heads, gating_mode="per-head")
    assert attention.g_proj.weight.shape == [num_attention_heads, HIDDEN_SIZE]


@pytest.mark.parametrize("num_attention_heads", [72, 48])
def test_a_per_element_gate_is_one_weight_row_per_channel(
    num_attention_heads: int,
) -> None:
    attention = _attention(num_attention_heads, gating_mode="per-element")
    assert attention.g_proj.weight.shape == [
        num_attention_heads * HEAD_DIM,
        HIDDEN_SIZE,
    ]


def test_an_unknown_gating_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="'per-head' or 'per-element'"):
        _attention(48, gating_mode="per-layer")


def test_a_sliding_layer_needs_a_window() -> None:
    with pytest.raises(ValueError, match="needs a positive window"):
        _attention(72, is_sliding=True)
