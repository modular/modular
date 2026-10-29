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

"""Loads the published Muse Spark configs through the local shims."""

import json
from pathlib import Path
from typing import Any

import pytest
from max.pipelines.architectures import hf_config_shims
from transformers import AutoConfig, PretrainedConfig

TESTDATA = Path(__file__).parent / "testdata"
CONFIGS = ["config_nvfp4.json", "config_bf16.json"]

# to_dict() stamps the installed transformers version over the checkpoint's.
_REWRITTEN = {"transformers_version"}


def _raw(name: str) -> dict[str, Any]:
    return json.loads((TESTDATA / name).read_text())


def _assert_preserved(
    raw: dict[str, Any], out: dict[str, Any], config: PretrainedConfig
) -> None:
    allowed_extra = set(PretrainedConfig().to_dict()) | set(
        getattr(config, "_defaults", {})
    )
    for key, value in raw.items():
        if key in _REWRITTEN:
            continue
        assert key in out, key
        if isinstance(value, dict) and key.endswith("_config"):
            sub = getattr(config, key)
            if isinstance(sub, PretrainedConfig):
                _assert_preserved(value, out[key], sub)
                continue
        assert out[key] == value, key
    assert set(out) - set(raw) <= allowed_extra, set(out) - set(raw)


@pytest.mark.parametrize("name", CONFIGS)
def test_published_config_round_trips(name: str) -> None:
    config = AutoConfig.from_pretrained(str(TESTDATA / name))
    assert isinstance(config, hf_config_shims.MuseSparkHFConfig)
    assert isinstance(
        config.text_config, hf_config_shims._MuseSparkTextHFConfig
    )
    assert isinstance(
        config.vision_config, hf_config_shims._MuseSparkVisionHFConfig
    )
    assert isinstance(
        config.audio_config, hf_config_shims._MuseSparkAudioHFConfig
    )
    raw = _raw(name)
    _assert_preserved(raw, config.to_dict(), config)
    if "quantization_config" in raw:
        assert config.quantization_config == raw["quantization_config"]
    else:
        assert getattr(config, "quantization_config", None) is None


@pytest.mark.parametrize("name", CONFIGS)
def test_derived_defaults_match_published(name: str) -> None:
    # The published config states every schedule, so dropping them makes
    # the shim derive what the reference's __post_init__ would.
    raw = _raw(name)
    text = raw["text_config"]
    derived = {
        k: text.pop(k)
        for k in ("layer_types", "layer_rope_theta", "mlp_layer_types")
    }
    vision_layer_types = raw["vision_config"].pop("layer_types")
    config = hf_config_shims.MuseSparkHFConfig(**raw)
    for key, value in derived.items():
        assert getattr(config.text_config, key) == value, key
    assert config.vision_config.layer_types == vision_layer_types


def test_text_defaults_without_rope_parameters() -> None:
    text = _raw("config_nvfp4.json")["text_config"]
    for key in (
        "layer_types",
        "layer_rope_theta",
        "mlp_layer_types",
        "rope_parameters",
    ):
        del text[key]
    config = hf_config_shims._MuseSparkTextHFConfig(**text)
    assert config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 500000.0,
    }
    full = [
        i for i, t in enumerate(config.layer_types) if t == "full_attention"
    ]
    assert full == list(range(1, 62, 4))
    assert [config.layer_rope_theta[i] for i in full] == [0] * 16
    assert {
        theta
        for i, theta in enumerate(config.layer_rope_theta)
        if i not in full
    } == {500000.0}


# Transcribed from transformers 5.20.0's configuration_muse_spark.py, the
# release the shims stand in for.
_TEXT_DEFAULTS = {
    "vocab_size": 202048,
    "hidden_size": 8192,
    "intermediate_size": 16384,
    "num_hidden_layers": 62,
    "num_attention_heads": 128,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "max_position_embeddings": 1048576,
    "rms_norm_eps": 1e-5,
    "use_cache": True,
    "tie_word_embeddings": False,
    "sliding_window": 2048,
    "final_logit_softcapping": 20.0,
    "qk_scale_factor": 5.4730064863875,
    "output_multiplier": 0.1767766952966369,
    "post_norm_eps": 1e-8,
    "hidden_act": "silu",
    "moe_intermediate_size": 4096,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "routed_scaling_factor": 1.0,
    "moe_hidden_size": 4096,
    "residual_gate_temperature": 0.3,
    "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
}
_VISION_DEFAULTS = {
    "patch_size": 14,
    "pos_emb_height": 32,
    "pos_emb_width": 32,
    "num_attention_heads": 16,
    "num_hidden_layers": 50,
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "hidden_act": "gelu",
    "patch_temporal": 2,
    "merge_size": 2,
    "max_position_embeddings": 1024,
    "layer_norm_eps": 1e-5,
}
_TOP_DEFAULTS = {
    "image_token_id": 200092,
    "video_token_id": 200091,
    "out_hidden_size": 6144,
    "projector_hidden_size": 4096,
    "projector_hidden_act": "gelu",
    "audio_token_id": 200038,
}


def test_defaults_match_transformers_5_20() -> None:
    config = hf_config_shims.MuseSparkHFConfig()
    for obj, expected in (
        (config.text_config, _TEXT_DEFAULTS),
        (config.vision_config, _VISION_DEFAULTS),
        (config, _TOP_DEFAULTS),
    ):
        for key, value in expected.items():
            assert getattr(obj, key) == value, key
    text = config.text_config
    # NoPE full-attention layers every 4th counted back from the last.
    assert text.layer_types == [
        "full_attention" if (61 - i) % 4 == 0 else "sliding_attention"
        for i in range(62)
    ]
    assert text.layer_rope_theta == [
        0 if (61 - i) % 4 == 0 else 500000.0 for i in range(62)
    ]
    assert text.mlp_layer_types == ["sparse"] * 62
    assert config.vision_config.layer_types == [
        "full_attention" if (i + 1) % 4 == 0 or i == 49 else "window_attention"
        for i in range(50)
    ]
