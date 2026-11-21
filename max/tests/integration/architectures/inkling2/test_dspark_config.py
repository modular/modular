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
"""Checks the Inkling 2 DSpark drafter's config, sizing and tap selection.

The configs are thinkingmachines/Inkling-2-Small's, and the shapes are the
ones its ``mtp.safetensors`` header declares.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
from max.dtype import DType
from max.pipelines.architectures.inkling2.dspark_config import (
    Inkling2DSparkConfig,
)
from max.pipelines.architectures.inkling2.inkling2 import Inkling2
from max.pipelines.architectures.inkling2.model_config import (
    Inkling2TextConfig,
)
from max.pipelines.architectures.inkling2.state_cache import ConvStateLayout

MTP_CONFIG: dict[str, Any] = {
    "hidden_size": 4608,
    "num_hidden_layers": 5,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 18432,
    "max_position_embeddings": 1048576,
    "torch_dtype": "bfloat16",
    "q_bias": False,
    "o_bias": False,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "local_layer_ids": [0, 1, 3, 4],
    "sliding_window_size": 1024,
    "block_size": 8,
    "markov_rank": 256,
    "markov_head_type": "rnn_hidden",
    "target_layer_ids": [5, 11, 17, 23, 29, 35, 41, 48],
}

TEXT_CONFIG = Inkling2TextConfig(
    hidden_size=4608,
    num_hidden_layers=48,
    rms_norm_eps=1e-6,
    vocab_size=201024,
    unpadded_vocab_size=200058,
    num_attention_heads=48,
    head_dim=128,
    num_key_value_heads=8,
    swa_num_attention_heads=48,
    swa_head_dim=128,
    swa_num_key_value_heads=8,
    sliding_window_size=1024,
    d_rel=64,
    rel_extent=4096,
    log_scaling_alpha=0.1,
    log_scaling_n_floor=4096,
    qk_l2_clip_threshold=1.0,
    rel_l2_clip_threshold=1.0,
    sconv_kernel_size=4,
    local_layer_ids=tuple(i for i in range(48) if i % 6 != 5),
    n_layers_per_attn_res_block=6,
    attn_res_normalize_values=True,
    n_routed_experts=256,
    num_experts_per_tok=8,
    intermediate_size=1536,
    route_scale=1.0,
    mlp_gate_softcap=30.0,
    mlp_up_softcap=30.0,
    final_logit_softcapping=30.0,
    model_max_length=262144,
)


def _hf_config(**mtp_overrides: Any) -> SimpleNamespace:
    return SimpleNamespace(
        mtp_config={**MTP_CONFIG, **mtp_overrides},
        text_config=SimpleNamespace(
            hidden_size=TEXT_CONFIG.hidden_size,
            num_hidden_layers=TEXT_CONFIG.num_hidden_layers,
            vocab_size=TEXT_CONFIG.vocab_size,
        ),
    )


def test_parses_the_shipped_drafter() -> None:
    draft = Inkling2DSparkConfig.from_hf(_hf_config())
    assert draft.num_hidden_layers == 5
    assert draft.local_layer_ids == (0, 1, 3, 4)
    assert draft.target_layer_ids == (5, 11, 17, 23, 29, 35, 41, 48)
    assert draft.vocab_size == 201024


def test_sizes_the_drafter() -> None:
    draft = Inkling2DSparkConfig.from_hf(_hf_config())
    # An anchor and seven masks: seven drafted tokens per block.
    assert draft.num_speculative_tokens == 7
    assert draft.num_target_features == 8
    assert draft.feature_width == 8 * 4608
    assert draft.feature_bytes(16) == 16 * 8 * 4608 * 2
    # One fuse row per drafter layer, then the Markov head's.
    assert draft.num_fuse_rows == 6
    assert draft.attention_window(0) == 1024 + 8
    assert draft.attention_window(2) is None
    assert draft.kv_bytes_per_token() == 2 * 8 * 128 * 2
    assert draft.kv_bytes_per_token(DType.float32) == 2 * 8 * 128 * 4


def test_weight_shapes_match_the_checkpoint() -> None:
    shapes = Inkling2DSparkConfig.from_hf(_hf_config()).weight_shapes()
    assert len(shapes) == 84
    assert shapes["fuse_weight"] == (6, 8)
    assert shapes["mask_embed"] == (4608,)
    assert shapes["markov_head.markov_w1.weight"] == (201024, 256)
    assert shapes["markov_head.rnn_proj.weight"] == (768, 9728)
    assert shapes["markov_head.hidden_out.weight"] == (4608, 256)
    assert shapes["layers.4.self_attn.q_proj.weight"] == (6144, 4608)
    assert shapes["layers.4.self_attn.o_proj.weight"] == (4608, 6144)
    assert shapes["layers.4.self_attn.k_proj_ctx.weight"] == (1024, 4608)
    assert shapes["layers.4.self_attn.v_proj_ctx.weight"] == (1024, 4608)
    assert shapes["layers.4.self_attn.k_norm_ctx.weight"] == (128,)
    assert shapes["layers.4.mlp.down_proj.weight"] == (4608, 18432)


def test_taps_split_into_layer_inputs_and_the_final_norm() -> None:
    draft = Inkling2DSparkConfig.from_hf(_hf_config())
    assert draft.tapped_layer_inputs == (5, 11, 17, 23, 29, 35, 41)
    assert draft.target_layer_ids[-1] == TEXT_CONFIG.num_hidden_layers


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_layer_ids": [5, 11, 17]},
        {"target_layer_ids": [11, 5, 48]},
        {"target_layer_ids": [5, 5, 48]},
        {"target_layer_ids": [-1, 48]},
        {"target_layer_ids": [5.5, 48]},
        {"target_layer_ids": [48]},
        {"local_layer_ids": [0, 5]},
        {"markov_head_type": "linear"},
        {"q_bias": True},
        {"hidden_size": 4096},
    ],
)
def test_refuses_an_unsupported_drafter(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Inkling2DSparkConfig.from_hf(_hf_config(**overrides))


def test_refuses_a_checkpoint_without_a_drafter() -> None:
    config = _hf_config()
    config.mtp_config = None
    with pytest.raises(ValueError, match="no mtp_config"):
        Inkling2DSparkConfig.from_hf(config)


def test_conv_ring_keeps_room_for_a_rejected_block() -> None:
    plain = ConvStateLayout.from_config(TEXT_CONFIG, tp_size=4)
    assert plain.ring_len == 3
    draft = Inkling2DSparkConfig.from_hf(_hf_config())
    spec = ConvStateLayout.from_config(
        TEXT_CONFIG, tp_size=4, max_rollback=draft.num_speculative_tokens
    )
    assert spec.ring_len == 3 + 7
    assert {region.row_shape[0] for region in spec.regions()} == {10}
    # The rollback changes the ring only, never the channel widths.
    assert dataclasses.replace(spec, ring_len=3) == plain


@pytest.mark.parametrize("taps", [(11, 5), (5, 5), (49,), (-1,)], ids=str)
def test_run_layers_refuses_bad_taps(taps: tuple[int, ...]) -> None:
    # The taps are checked before the graph is touched.
    model = SimpleNamespace(text_config=TEXT_CONFIG)
    with pytest.raises(ValueError, match="taps"):
        Inkling2.run_layers(model, None, None, None, None, taps=taps)  # type: ignore[arg-type]
