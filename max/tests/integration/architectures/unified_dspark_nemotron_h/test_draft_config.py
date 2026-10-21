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
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
from max.pipelines.architectures.unified_dspark_nemotron_h.dspark_qwen3 import (
    parse_dspark_draft_config,
)
from max.pipelines.architectures.unified_dspark_nemotron_h.model_config import (
    check_draft_runs_on,
)

# nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark/config.json
DRAFT_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3DSparkModel"],
    "attention_bias": False,
    "attention_sink_bias": True,
    "block_size": 8,
    "dflash_config": {
        "attention_sink_bias": True,
        "causal": True,
        "mask_token_id": 990,
        "swa_window_size": 1024,
        "target_layer_ids": [1, 5, 19, 29, 41, 51],
        "use_swa": True,
        "sample_from_anchor": False,
    },
    "dflash_query_causal": True,
    "draft_vocab_size": None,
    "dspark_bonus_anchor": True,
    "dspark_fixup_head_type": "markov",
    "dspark_markov_rank": 512,
    "dtype": "bfloat16",
    "eagle_aux_hidden_state_layer_ids": [2, 6, 20, 30, 42, 52],
    "has_embed_tokens": True,
    "has_lm_head": False,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2688,
    "intermediate_size": 6144,
    "layer_types": ["sliding_attention"] * 6,
    "markov_head_type": "vanilla",
    "markov_rank": 512,
    "mask_token_id": 990,
    "max_position_embeddings": 1048576,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_aux_layers": 6,
    "num_hidden_layers": 6,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_parameters": None,
    "rope_theta": 10000,
    "sliding_window": 1024,
    "target_layer_ids": [1, 5, 19, 29, 41, 51],
    "vocab_size": 131072,
    "sample_from_anchor": False,
    "quantization_config": {
        "quant_method": "modelopt",
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
        "exclude_modules": ["*embed_tokens*", "*markov_w1*", "*self_attn*"],
    },
}


def _draft(**overrides: Any) -> dict[str, Any]:
    config = copy.deepcopy(DRAFT_CONFIG)
    config.update(overrides)
    return config


def _on(arch: str, api: str = "cuda") -> Any:
    device = SimpleNamespace(architecture_name=arch, api=api)
    return lambda specs: [device]


def test_parses_the_released_checkpoint() -> None:
    config = parse_dspark_draft_config(DRAFT_CONFIG)
    assert (config.hidden_size, config.num_hidden_layers) == (2688, 6)
    assert config.block_size == 8 and config.num_drafts == 7
    assert config.mask_token_id == 990
    assert config.tap_layer_inputs == (2, 6, 20, 30, 42, 52)
    assert config.quantized


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"markov_head_type": "gated"}, "vanilla"),
        ({"enable_confidence_head": True}, "confidence"),
        ({"hidden_act": "gelu"}, "SiLU"),
        ({"attention_bias": True}, "biases"),
        ({"layer_types": ["full_attention"] * 6}, "sliding-window"),
    ],
)
def test_rejects_drafts_it_does_not_build(
    override: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_dspark_draft_config(_draft(**override))


def test_rejects_a_draft_that_samples_from_the_anchor() -> None:
    # The proposer drops the anchor slot, so a draft whose anchor predicts
    # would be wired with the wrong proposal width.
    config = _draft()
    config["dflash_config"]["sample_from_anchor"] = True
    with pytest.raises(ValueError, match="sample_from_anchor"):
        parse_dspark_draft_config(config)
    top_level = _draft(sample_from_anchor=True)
    del top_level["dflash_config"]["sample_from_anchor"]
    with pytest.raises(ValueError, match="sample_from_anchor"):
        parse_dspark_draft_config(top_level)


def test_nvfp4_draft_needs_sm100() -> None:
    draft = parse_dspark_draft_config(DRAFT_CONFIG)
    check_draft_runs_on([], draft, devices_loader=_on("sm_100"))
    for arch, api in (("sm_90", "cuda"), ("gfx950", "hip")):
        with pytest.raises(ValueError, match="BF16 drafter"):
            check_draft_runs_on([], draft, devices_loader=_on(arch, api))


def test_a_bf16_draft_runs_anywhere() -> None:
    config = _draft()
    config.pop("quantization_config")
    draft = parse_dspark_draft_config(config)
    check_draft_runs_on([], draft, devices_loader=_on("sm_90"))
