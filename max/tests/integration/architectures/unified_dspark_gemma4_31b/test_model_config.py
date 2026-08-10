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
"""Speculators-format DSpark config parsing and its guards.

``testdata/redhat_speculator_config.json`` is the real (unmodified)
``config.json`` of ``RedHatAI/gemma-4-31B-it-speculator.dspark`` at revision
``0026c7d1899651ca3c45ede471712f04849723ac``. It has no top-level
``model_type``, so loading it goes through the ``_hf_config.py`` raw-JSON
fallback. Guard-negative tests mutate a copy of the real dict, one field at
a time.
"""

from __future__ import annotations

import json
import logging
import pathlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.architectures.speculators_common import (
    DSparkSpeculatorsDraftArchConfig,
    DSparkSpeculatorsDraftConfig,
)
from max.pipelines.architectures.unified_dspark_gemma4_31b.model_config import (
    UnifiedDSparkGemma4_31BConfig,
)
from max.pipelines.lib._hf_config import load_huggingface_config
from max.pipelines.lib.config import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.weights.hf_utils import HuggingFaceRepo
from transformers import PretrainedConfig

_CONFIG_PATH = (
    pathlib.Path(__file__).parent / "testdata" / "redhat_speculator_config.json"
)


def _raw() -> dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return cast(dict[str, Any], json.load(f))


def _parse(raw: dict[str, Any]) -> DSparkSpeculatorsDraftConfig:
    return DSparkSpeculatorsDraftConfig.from_huggingface_config(
        PretrainedConfig.from_dict(raw)
    )


def _assert_redhat_fields(draft: DSparkSpeculatorsDraftConfig) -> None:
    assert draft.hidden_size == 5376
    assert draft.intermediate_size == 21504
    assert draft.num_hidden_layers == 5
    assert draft.num_attention_heads == 32
    assert draft.num_key_value_heads == 16
    assert draft.head_dim == 256
    assert draft.rms_norm_eps == 1e-6
    assert draft.vocab_size == 262144
    assert draft.draft_vocab_size == 32000
    assert draft.hidden_activation == "silu"
    assert draft.rope_theta == 10000.0
    # max_position_embeddings is NESTED under transformer_layer_config.
    assert draft.max_seq_len == 262144
    assert draft.sliding_window == 2048
    assert draft.causal is True
    assert draft.block_size == 8
    assert draft.sample_from_anchor is False
    assert draft.mask_token_id == 4
    assert draft.aux_hidden_state_layer_ids == (1, 17, 29, 47, 58)
    assert draft.markov_rank == 256
    assert draft.markov_head_type == "vanilla"
    # block_size counts the anchor slot: 7 drafts per step, not 8.
    assert draft.num_speculative_tokens == 7
    # MAX captures layer OUTPUTS; vLLM aux id j is the INPUT of layer j.
    assert draft.target_layer_ids == (0, 16, 28, 46, 57)
    assert draft.num_context_features == 5


def test_parse_real_config() -> None:
    _assert_redhat_fields(_parse(_raw()))


def test_load_huggingface_config_raw_json_fallback(
    tmp_path: pathlib.Path,
) -> None:
    """End-to-end through the ``_hf_config.py`` raw-JSON fallback path."""
    raw = _raw()
    # The fallback only fires (and re-raising is suppressed) because the
    # speculators config has no top-level model_type.
    assert "model_type" not in raw
    (tmp_path / "config.json").write_text(json.dumps(raw))

    hf_config = load_huggingface_config(HuggingFaceRepo(str(tmp_path)))
    assert isinstance(hf_config, PretrainedConfig)
    _assert_redhat_fields(
        DSparkSpeculatorsDraftConfig.from_huggingface_config(hf_config)
    )


def test_wrong_speculators_model_type_rejected() -> None:
    raw = _raw()
    raw["speculators_model_type"] = "eagle3"
    with pytest.raises(ValueError, match="speculators_model_type"):
        _parse(raw)
    del raw["speculators_model_type"]
    with pytest.raises(ValueError, match="speculators_model_type"):
        _parse(raw)


def test_missing_transformer_layer_config_rejected() -> None:
    raw = _raw()
    del raw["transformer_layer_config"]
    with pytest.raises(ValueError, match="transformer_layer_config"):
        _parse(raw)


def test_block_size_lower_bound() -> None:
    raw = _raw()
    raw["block_size"] = 1
    with pytest.raises(ValueError, match="block_size must be >= 2"):
        _parse(raw)


def test_mask_token_id_bounds() -> None:
    raw = _raw()
    raw["mask_token_id"] = -1
    with pytest.raises(ValueError, match="mask_token_id"):
        _parse(raw)
    raw["mask_token_id"] = 262144
    with pytest.raises(ValueError, match="mask_token_id"):
        _parse(raw)


def test_draft_vocab_size_bounds() -> None:
    raw = _raw()
    raw["draft_vocab_size"] = 0
    with pytest.raises(ValueError, match="draft_vocab_size"):
        _parse(raw)
    raw["draft_vocab_size"] = 262145
    with pytest.raises(ValueError, match="draft_vocab_size"):
        _parse(raw)


def test_aux_layer_ids_guards() -> None:
    raw = _raw()
    raw["aux_hidden_state_layer_ids"] = []
    with pytest.raises(ValueError, match="aux_hidden_state_layer_ids"):
        _parse(raw)
    # 0 is invalid under the vLLM eagle convention (input of layer j, j >= 1).
    raw["aux_hidden_state_layer_ids"] = [0, 17, 29, 47, 58]
    with pytest.raises(ValueError, match="vLLM eagle convention"):
        _parse(raw)
    raw["aux_hidden_state_layer_ids"] = [1, 29, 17, 47, 58]
    with pytest.raises(ValueError, match="strictly"):
        _parse(raw)


def test_layer_types_guards() -> None:
    raw = _raw()
    raw["transformer_layer_config"]["layer_types"] = ["sliding_attention"] * 4
    with pytest.raises(ValueError, match="every layer"):
        _parse(raw)
    raw["transformer_layer_config"]["layer_types"] = ["chunked_attention"] * 5
    with pytest.raises(ValueError, match="unsupported layer_types"):
        _parse(raw)
    raw["transformer_layer_config"]["layer_types"] = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ]
    with pytest.raises(ValueError, match="mixed layer_types"):
        _parse(raw)


def test_causality_derivation() -> None:
    # All full_attention (the makora/GLM convention) parses as non-causal
    # and does not require a sliding window.
    raw = _raw()
    raw["transformer_layer_config"]["layer_types"] = ["full_attention"] * 5
    raw["transformer_layer_config"]["sliding_window"] = None
    draft = _parse(raw)
    assert draft.causal is False

    # An explicit top-level `causal` field overrides the layer_types rule
    # (the vLLM _dflash_layer_causal precedence).
    raw = _raw()
    raw["causal"] = False
    assert _parse(raw).causal is False


def test_sliding_layers_require_window() -> None:
    raw = _raw()
    raw["transformer_layer_config"]["sliding_window"] = 0
    with pytest.raises(ValueError, match="sliding_window > 0"):
        _parse(raw)


def test_sliding_window_non_causal_rejected() -> None:
    raw = _raw()
    raw["sliding_window_non_causal"] = True
    with pytest.raises(ValueError, match="sliding_window_non_causal"):
        _parse(raw)


def test_proposal_speculative_tokens_cross_check() -> None:
    raw = _raw()
    raw["speculators_config"]["proposal_methods"][0]["speculative_tokens"] = 6
    with pytest.raises(ValueError, match="speculative_tokens=6"):
        _parse(raw)


def test_sample_from_anchor_shifts_expected_tokens() -> None:
    # With sample_from_anchor the anchor slot also predicts: block_size 8
    # means 8 drafts (the GLM dspark convention), so the RedHat proposal
    # value of 7 must now be rejected...
    raw = _raw()
    raw["sample_from_anchor"] = True
    with pytest.raises(ValueError, match="sample_from_anchor=True"):
        _parse(raw)
    # ...and 8 accepted.
    raw["speculators_config"]["proposal_methods"][0]["speculative_tokens"] = 8
    draft = _parse(raw)
    assert draft.sample_from_anchor is True
    assert draft.num_speculative_tokens == 8


def test_markov_head_guards() -> None:
    raw = _raw()
    raw["markov_head_type"] = "dense"
    with pytest.raises(ValueError, match="markov_head_type"):
        _parse(raw)
    raw = _raw()
    raw["markov_rank"] = 0
    with pytest.raises(ValueError, match="markov_rank"):
        _parse(raw)


def test_rope_guards() -> None:
    raw = _raw()
    raw["transformer_layer_config"]["rope_parameters"]["rope_type"] = "yarn"
    with pytest.raises(ValueError, match="rope_type"):
        _parse(raw)
    raw = _raw()
    del raw["transformer_layer_config"]["rope_parameters"]["rope_theta"]
    with pytest.raises(ValueError, match="rope_theta"):
        _parse(raw)


def test_missing_nested_max_position_embeddings_rejected() -> None:
    raw = _raw()
    del raw["transformer_layer_config"]["max_position_embeddings"]
    with pytest.raises(ValueError, match="max_position_embeddings"):
        _parse(raw)


def test_draft_arch_config_reads_nested_max_position_embeddings() -> None:
    model_config = SimpleNamespace(
        huggingface_config=PretrainedConfig.from_dict(_raw())
    )
    arch_config = DSparkSpeculatorsDraftArchConfig.initialize(
        cast(PipelineConfig, None),
        model_config=cast(MAXModelConfig, model_config),
    )
    assert arch_config.get_max_seq_len() == 262144

    raw = _raw()
    del raw["transformer_layer_config"]["max_position_embeddings"]
    model_config = SimpleNamespace(
        huggingface_config=PretrainedConfig.from_dict(raw)
    )
    with pytest.raises(ValueError, match="max_position_embeddings"):
        DSparkSpeculatorsDraftArchConfig.initialize(
            cast(PipelineConfig, None),
            model_config=cast(MAXModelConfig, model_config),
        )


def _make_target(
    *,
    num_hidden_layers: int = 60,
    hidden_size: int = 5376,
    vocab_size: int = 262144,
    n_devices: int = 1,
) -> SimpleNamespace:
    """Stand-in exposing only the target attributes the config touches."""
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            return_logits=None,
            return_hidden_states=None,
            target_layer_ids=[],
        ),
        devices=[DeviceRef.CPU()] * n_devices,
    )


def _make_unified(
    draft: DSparkSpeculatorsDraftConfig,
    *,
    target: SimpleNamespace | None = None,
    num_speculative_tokens: int | None = 7,
) -> UnifiedDSparkGemma4_31BConfig:
    target = target if target is not None else _make_target()
    # None means unset: a config that never mentions num_speculative_tokens.
    speculative_config = SpeculativeConfig(
        speculative_method="dflash",
        num_speculative_tokens=num_speculative_tokens,
    )
    return UnifiedDSparkGemma4_31BConfig(
        target=cast(Gemma4ForConditionalGenerationConfig, target),
        draft=draft,
        draft_kv_params=cast(KVCacheParams, None),
        speculative_config=speculative_config,
        target_layer_ids=list(draft.target_layer_ids),
        mask_token_id=draft.mask_token_id,
        block_size=draft.block_size,
    )


def test_unified_config_wires_target_capture() -> None:
    config = _make_unified(_parse(_raw()))
    text_config = config.target.text_config
    assert text_config.return_logits == ReturnLogits.VARIABLE
    assert (
        text_config.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
    )
    assert text_config.target_layer_ids == [0, 16, 28, 46, 57]
    config.validate_dspark_fields()


def test_unified_config_single_device_only() -> None:
    with pytest.raises(ValueError, match="single device"):
        _make_unified(_parse(_raw()), target=_make_target(n_devices=2))


def test_validate_rejects_aux_ids_beyond_target_depth() -> None:
    config = _make_unified(
        _parse(_raw()), target=_make_target(num_hidden_layers=40)
    )
    with pytest.raises(ValueError, match="40-layer target"):
        config.validate_dspark_fields()


def test_validate_rejects_hidden_size_mismatch() -> None:
    config = _make_unified(
        _parse(_raw()), target=_make_target(hidden_size=3840)
    )
    with pytest.raises(ValueError, match="hidden_size"):
        config.validate_dspark_fields()


def test_validate_rejects_vocab_mismatch() -> None:
    config = _make_unified(
        _parse(_raw()), target=_make_target(vocab_size=32000)
    )
    with pytest.raises(ValueError, match="vocab"):
        config.validate_dspark_fields()


def test_validate_honors_user_num_speculative_tokens() -> None:
    # K=4 below the trained width (8 - 1 = 7) is honored: the draft block
    # is causal, so truncating trailing mask slots is prefix-stable. KV
    # headroom follows as the effective anchor+drafts width.
    config = _make_unified(_parse(_raw()), num_speculative_tokens=4)
    config.validate_dspark_fields()
    assert config.speculative_config.num_speculative_tokens == 4
    assert config.effective_block_size == 5


def test_validate_warns_num_speculative_tokens_beyond_trained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # K=9 above the trained width (8 - 1 = 7) is honored too: the block is
    # width-generic and the extra positions run as extrapolation, so a
    # warning names the trained geometry. KV headroom still follows the
    # effective anchor+drafts width.
    config = _make_unified(_parse(_raw()), num_speculative_tokens=9)
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        config.validate_dspark_fields()
    assert config.speculative_config.num_speculative_tokens == 9
    assert config.effective_block_size == 10
    assert any(
        "trained at block_size=8" in record.message for record in caplog.records
    )


def test_validate_rejects_non_positive_num_speculative_tokens() -> None:
    config = _make_unified(_parse(_raw()), num_speculative_tokens=0)
    with pytest.raises(ValueError, match="num_speculative_tokens=0"):
        config.validate_dspark_fields()


def test_validate_defaults_num_speculative_tokens_to_trained() -> None:
    # Unset keeps the drafter's trained width (block_size 8 -> 7 drafts).
    config = _make_unified(_parse(_raw()), num_speculative_tokens=None)
    config.validate_dspark_fields()
    assert config.speculative_config.num_speculative_tokens == 7
    assert config.effective_block_size == 8
