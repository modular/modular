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
"""Config normalization for Mistral Large 4, from the checkpoint's params.json.

The fixtures are the two published checkpoints' own ``params.json`` files and
the base repo's published ``config.json``, so every assertion below is about the
real model rather than a synthetic stand-in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    QUANT_ALGO_FP8_PB_WO,
    QUANT_ALGO_NVFP4,
    MistralLarge4NormalizedConfig,
    normalized_hf_config,
    read_params_json,
)

_TESTDATA = Path(__file__).parent / "testdata"

# The 44 layers that own a DSA indexer; the other 28 carry the preceding
# owner's top-k selection forward (the checkpoint states only full|shared).
_INDEXER_OWNERS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    19,
    20,
    23,
    24,
    27,
    28,
    31,
    32,
    35,
    36,
    39,
    40,
    43,
    44,
    47,
    48,
    51,
    52,
    55,
    56,
    59,
    60,
    63,
    64,
    67,
    68,
    71,
)

_NOPE = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71)

_KV_SHARERS = {
    39: 35,
    43: 35,
    47: 35,
    51: 35,
    55: 35,
    59: 35,
    63: 35,
    67: 35,
    71: 35,
}


@pytest.fixture(scope="module")
def nvfp4() -> MistralLarge4NormalizedConfig:
    params = read_params_json(_TESTDATA / "ml4_nvfp4_params.json")
    return MistralLarge4NormalizedConfig.from_params_json(params)


@pytest.fixture(scope="module")
def base() -> MistralLarge4NormalizedConfig:
    params = read_params_json(_TESTDATA / "ml4_base_params.json")
    return MistralLarge4NormalizedConfig.from_params_json(params)


def test_core_geometry(nvfp4: MistralLarge4NormalizedConfig) -> None:
    assert nvfp4.n_layers == 72
    assert nvfp4.hidden_size == 6144
    assert nvfp4.n_heads == 64
    assert nvfp4.vocab_size == 204800
    # No query LoRA: a single full-rank `wq`.
    assert nvfp4.q_lora_rank is None
    assert (nvfp4.qk_nope_head_dim, nvfp4.qk_rope_head_dim) == (64, 64)
    assert nvfp4.qk_head_dim == 128
    assert nvfp4.kv_lora_rank == 512
    assert nvfp4.rms_norm_eps == 1e-5
    assert nvfp4.final_logit_softcapping == 30.0
    assert nvfp4.tie_word_embeddings is False


def test_attention_scale_is_the_explicit_head_dim_inverse_root(
    nvfp4: MistralLarge4NormalizedConfig,
) -> None:
    # 128 ** -0.5 for every layer, NoPE included: YaRN's magnitude scaling is
    # off, so the scale is never re-derived from the padded geometry.
    assert nvfp4.attention_scale == pytest.approx(128**-0.5)


def test_nope_is_every_fourth_layer_from_the_third(
    nvfp4: MistralLarge4NormalizedConfig,
) -> None:
    assert nvfp4.layers.nope_layers == _NOPE
    assert len(nvfp4.layers.nope_layers) == 18


def test_indexer_ownership_and_carried_top_k(
    nvfp4: MistralLarge4NormalizedConfig,
) -> None:
    assert nvfp4.layers.indexer_owner_layers == _INDEXER_OWNERS
    assert len(nvfp4.layers.indexer_sharer_layers) == 28
    # A sharer carries the selection of the preceding owner: 17 and 18 both
    # read 16, and 21/22 both read 20.
    assert nvfp4.layers.indexer_source[17] == 16
    assert nvfp4.layers.indexer_source[18] == 16
    assert nvfp4.layers.indexer_source[21] == 20
    for layer in nvfp4.layers.indexer_sharer_layers:
        source = nvfp4.layers.indexer_source[layer]
        assert source < layer
        assert source in nvfp4.layers.indexer_owner_layers


def test_cross_layer_kv_sharing_points_at_layer_35(
    nvfp4: MistralLarge4NormalizedConfig,
) -> None:
    assert nvfp4.layers.kv_sharer_layers == tuple(sorted(_KV_SHARERS))
    for layer, source in _KV_SHARERS.items():
        assert nvfp4.layers.kv_source[layer] == source
    assert len(nvfp4.layers.kv_owner_layers) == 63
    # The sharers are NoPE layers and so is their source.
    assert set(nvfp4.layers.kv_sharer_layers) <= set(nvfp4.layers.nope_layers)
    assert 35 in nvfp4.layers.nope_layers


def test_moe_layers_start_after_the_dense_layer(
    nvfp4: MistralLarge4NormalizedConfig,
) -> None:
    assert nvfp4.first_k_dense_replace == 1
    assert nvfp4.layers.moe[0] is False
    assert all(nvfp4.layers.moe[1:]) is True
    assert (nvfp4.n_routed_experts, nvfp4.n_shared_experts) == (128, 1)
    assert nvfp4.num_experts_per_tok == 4
    assert nvfp4.moe_intermediate_size == 6144
    assert nvfp4.intermediate_size == 24576
    # No group-limited routing and no renormalization flag: 1 group, scale 1.0.
    assert (nvfp4.n_group, nvfp4.topk_group) == (1, 1)
    assert nvfp4.routed_scaling_factor == 1.0


def test_ssmax_is_read_and_the_two_spellings_must_agree(
    nvfp4: MistralLarge4NormalizedConfig,
) -> None:
    """`params.json` states SSMax at the top level and under `dsa`."""
    assert nvfp4.ssmax_scaling is True
    params = read_params_json(_TESTDATA / "ml4_nvfp4_params.json")
    dsa = params["dsa"]
    assert isinstance(dsa, dict)
    dsa["ssmax_scaling"] = False
    with pytest.raises(ValueError, match=r"dsa\.ssmax_scaling"):
        MistralLarge4NormalizedConfig.from_params_json(params)


def test_a_negative_kv_source_is_refused() -> None:
    """A negative source must not reach the list index it would wrap to."""
    params = read_params_json(_TESTDATA / "ml4_nvfp4_params.json")
    kv_sharing = params["kv_sharing"]
    assert isinstance(kv_sharing, dict)
    sources = kv_sharing["source_layers_str"].split(",")
    kv_sharing["source_layers_str"] = ",".join(["0", "-1"] + sources[2:])
    with pytest.raises(ValueError, match="negative"):
        MistralLarge4NormalizedConfig.from_params_json(params)


def test_yarn_betas_map_the_other_way_round(
    nvfp4: MistralLarge4NormalizedConfig,
) -> None:
    yarn = nvfp4.yarn
    # `yarn.beta` is beta_fast (32) and `yarn.alpha` is beta_slow (1).
    assert yarn.beta_fast == 32.0
    assert yarn.beta_slow == 1.0
    assert yarn.factor == 2048.0
    assert yarn.original_max_position_embeddings == 512
    assert yarn.rope_theta == 10000.0
    # `apply_scale: false` means no magnitude scaling.
    assert yarn.attention_factor == 1.0
    assert yarn.lerope is True
    assert yarn.lerope_alpha_dim == 32
    # The ramp those betas and that theta produce on the 64-wide rope slice.
    assert yarn.correction_range == (3, 16)


def test_quantization_resolves_per_module(
    nvfp4: MistralLarge4NormalizedConfig,
) -> None:
    quant = nvfp4.quantization
    assert quant.method == "modelopt"
    assert quant.algo == "MIXED_PRECISION"
    assert quant.group_size == 16
    assert quant.algo_for("layers.1.experts") == QUANT_ALGO_NVFP4
    assert quant.algo_for("layers.1.experts.0.w1") == QUANT_ALGO_NVFP4
    for module in (
        "layers.0.attention.wq",
        "layers.0.attention.wo",
        "layers.0.attention.indexer.wq_b",
        "layers.0.feed_forward.w1",
        "layers.1.shared_experts.w1",
    ):
        assert quant.algo_for(module) == QUANT_ALGO_FP8_PB_WO, module
    for module in (
        "layers.1.attention.wkv_a_with_mqa",
        "layers.1.attention.wkv_b",
        "layers.1.attention.kv_a_norm",
        "layers.1.gate",
        "layers.0.attention.indexer.wk",
        "tok_embeddings",
        "output",
        "lerope",
    ):
        assert quant.algo_for(module) == "BF16", module


def test_the_base_checkpoint_declares_no_modelopt_mix(
    base: MistralLarge4NormalizedConfig,
) -> None:
    assert base.quantization.quantized is False
    assert base.quantization.modules == {}


def test_the_drafter_geometry_is_read(
    nvfp4: MistralLarge4NormalizedConfig,
) -> None:
    mtp = nvfp4.mtp
    assert mtp is not None, "the full checkpoint ships its drafter"
    assert (mtp.n_layers, mtp.depth) == (3, 3)
    assert mtp.window == (2048, 2048, 2048)
    assert mtp.target_layers == (1, 18, 36, 54, 71)
    assert mtp.markov_rank == 512


def test_the_generator_reproduces_the_published_base_config() -> None:
    """The generated config must equal what ``transformers`` publishes.

    The base repo ships a ``config.json`` that was produced by
    ``convert_mistral_large4_to_hf.py`` from the very ``params.json`` the fixture
    holds, so equality here is the statement that this generator reproduces the
    reference converter on the fields both checkpoints share.
    """
    params = read_params_json(_TESTDATA / "ml4_base_params.json")
    generated = normalized_hf_config(
        MistralLarge4NormalizedConfig.from_params_json(params),
        params=params,
        bos_token_id=1,
        eos_token_id=2,
    )
    published = json.loads((_TESTDATA / "ml4_base_config.json").read_text())
    # MAX-side plumbing the reference converter has no field for: the indexer's
    # LayerNorm epsilon, which the checkpoint stores as ``dsa.norm_eps``.
    generated_text = generated["text_config"]
    assert isinstance(generated_text, dict)
    assert generated_text.pop("indexer_norm_eps") == 1e-6
    assert _flatten(generated) == _flatten(published)


def test_the_nvfp4_config_differs_only_by_the_quantization_block() -> None:
    params = read_params_json(_TESTDATA / "ml4_nvfp4_params.json")
    generated = normalized_hf_config(
        MistralLarge4NormalizedConfig.from_params_json(params),
        params=params,
        bos_token_id=1,
        eos_token_id=2,
    )
    published = json.loads((_TESTDATA / "ml4_base_config.json").read_text())
    flat_generated = _flatten(generated)
    flat_published = _flatten(published)
    # Deliberate difference: the NVFP4 checkpoint's ModelOpt mixture has no HF
    # `quantization_config` spelling, and the converter only synthesizes one
    # from the base checkpoint's `quantization` block.
    assert "quantization_config.quant_method" not in flat_generated
    for key in flat_published:
        if key.startswith("quantization_config."):
            continue
        assert flat_generated[key] == flat_published[key], key


def test_the_nvfp4_generator_matches_the_hf_converter_oracle() -> None:
    """The converter run on the NVFP4 checkpoint's own ``params.json``.

    The base repo's published ``config.json`` cannot cover this checkpoint: the
    NVFP4 repo ships no config at all. This oracle is
    ``convert_mistral_large4_to_hf.py --config-only`` run against the NVFP4
    ``params.json``. Two fields differ deliberately, and neither is a model
    fact: the oracle was produced by a newer ``transformers``, so its
    ``transformers_version`` differs, and it omits ``architectures``, which the
    serialization only writes for a checkpoint that declares one.
    """
    params = read_params_json(_TESTDATA / "ml4_nvfp4_params.json")
    generated = _flatten(
        normalized_hf_config(
            MistralLarge4NormalizedConfig.from_params_json(params),
            params=params,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    oracle = _flatten(
        json.loads((_TESTDATA / "ml4_nvfp4_hf_config.json").read_text())
    )
    # `indexer_norm_eps` is MAX-side plumbing for the checkpoint's own
    # ``dsa.norm_eps``; the reference converter has no field for it.
    ignored = {
        "architectures",
        "transformers_version",
        "text_config.indexer_norm_eps",
    }
    assert not set(oracle) - set(generated) - ignored
    assert not set(generated) - set(oracle) - ignored
    differing = {
        key
        for key in set(generated) & set(oracle)
        if key not in ignored and generated[key] != oracle[key]
    }
    assert not differing, differing


def test_per_layer_patterns_read_as_strings_or_lists() -> None:
    """A rewritten fixture may carry the expanded lists instead of the string.

    ``params.json`` writes each pattern as a comma-separated string; a minimized
    fixture writes the list. Both must resolve to one schedule.
    """
    params = read_params_json(_TESTDATA / "ml4_nvfp4_params.json")
    as_strings = MistralLarge4NormalizedConfig.from_params_json(params)
    as_lists = json.loads(json.dumps(params))
    as_lists["nope"] = as_lists["nope"].split(",")
    as_lists["dsa"]["sharing"] = as_lists["dsa"]["sharing"].split(",")
    as_lists["kv_sharing"]["source_layers_str"] = [
        int(layer)
        for layer in as_lists["kv_sharing"]["source_layers_str"].split(",")
    ]
    as_lists["mtp"]["ragged_attention"] = [2048, 2048, 2048]
    as_lists["mtp"]["target_activation_layers"] = [1, 18, 36, 54, 71]
    assert (
        MistralLarge4NormalizedConfig.from_params_json(as_lists) == as_strings
    )


def test_per_layer_lists_are_explicit_in_the_generated_config() -> None:
    params = read_params_json(_TESTDATA / "ml4_nvfp4_params.json")
    generated = normalized_hf_config(
        MistralLarge4NormalizedConfig.from_params_json(params),
        params=params,
        bos_token_id=1,
        eos_token_id=2,
    )
    text = generated["text_config"]
    assert isinstance(text, dict)
    assert text["no_rope_layers"] == [
        1 if i not in _NOPE else 0 for i in range(72)
    ]
    assert text["kv_source_layers"] == [
        _KV_SHARERS.get(i, i) for i in range(72)
    ]
    assert text["mlp_layer_types"] == ["dense"] + ["sparse"] * 71
    assert text["layer_types"] == ["indexed_attention"] * 72
    assert text["no_rope_layer_interval"] == 4
    assert text["head_dim"] == 64
    assert text["qk_head_dim"] == 128
    assert text["final_logit_softcapping"] == 30.0


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    """Flattens nested config dicts into ``"a.b.c" -> value`` for diffing."""
    assert isinstance(value, dict)
    out: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            out.update(_flatten(item, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = item
    return out
