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
"""``LagunaConfig.read_layer_schedule`` is where a Laguna checkpoint's shape
stops being raw HuggingFace fields and becomes the thing the graph builds from.

Two config classes can reach it, and they disagree about what is present. The
pinned transformers ships a native ``laguna`` config that normalizes
``layer_types``, ``mlp_layer_types`` and ``num_attention_heads_per_layer`` in
``__post_init__`` and defaults ``sliding_window`` to 512; the in-tree
``LagunaHFConfig`` shim leaves every absent field absent. Both must resolve to
the same schedule, so the happy-path assertions run against both -- one
exercising the populated-list path, one the defaulting path.

The failure cases run against the shim alone: they describe Laguna variants the
architecture cannot represent, and several of them the native config rejects
first, at construction, for its own reasons.

``construct_kv_params`` is exercised here too, because the schedule is what
decides between one flat KV cache and the two-leaf sliding/full tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.hf_config_shims import LagunaHFConfig
from max.pipelines.architectures.laguna.model_config import LagunaConfig
from max.pipelines.lib import KVCacheConfig
from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

TESTDATA = Path(__file__).parent / "testdata"

# S-2.1 repeats `[full, sliding, sliding, sliding]` over its 48 layers, with 48
# query heads on the full-attention layers and 72 on the sliding ones.
S21_LAYER_TYPES = ("full_attention",) + ("sliding_attention",) * 3
S21_HEADS = (48, 72, 72, 72)


def _laguna_config_classes() -> dict[str, type[PretrainedConfig]]:
    register_all_models()
    classes: dict[str, type[PretrainedConfig]] = {"shim": LagunaHFConfig}
    registered = CONFIG_MAPPING["laguna"]
    if registered is not LagunaHFConfig:
        classes["autoconfig"] = registered
    return classes


_CONFIG_CLASSES = _laguna_config_classes()
_CONFIG_IDS = list(_CONFIG_CLASSES)


@pytest.fixture(
    params=[_CONFIG_CLASSES[i] for i in _CONFIG_IDS], ids=_CONFIG_IDS
)
def config_cls(request: pytest.FixtureRequest) -> type[PretrainedConfig]:
    cls = request.param
    assert issubclass(cls, PretrainedConfig)
    return cls


def _load(config_cls: type[PretrainedConfig], name: str) -> Any:
    raw: dict[str, Any] = json.loads((TESTDATA / name).read_text())
    return config_cls(**raw)


def test_s21_schedule(config_cls: type[PretrainedConfig]) -> None:
    schedule = LagunaConfig.read_layer_schedule(
        _load(config_cls, "s21_config.json")
    )

    assert schedule.layer_types == S21_LAYER_TYPES * 12
    assert schedule.layer_types.count("full_attention") == 12
    assert schedule.layer_types.count("sliding_attention") == 36
    assert schedule.num_heads_per_layer == S21_HEADS * 12
    assert schedule.mlp_layer_types == ("dense",) + ("sparse",) * 47
    assert schedule.gating_mode == "per-head"
    assert schedule.sliding_window == 512
    assert schedule.full_rope["rope_type"] == "yarn"
    assert schedule.full_rope["factor"] == 128.0
    assert schedule.full_rope["partial_rotary_factor"] == 0.5
    assert schedule.sliding_rope is not None
    assert schedule.sliding_rope["rope_type"] == "default"
    assert schedule.sliding_rope["rope_theta"] == 10000.0


def test_m1_schedule(config_cls: type[PretrainedConfig]) -> None:
    """M.1 is the uniform shape, and its ``sliding_window: 0`` is not a window.

    The field is present and falsy on M.1 and defaults to 512 on the native
    config; neither says anything, because M.1 has no sliding layer to apply a
    window to. Resolving it to ``None`` here is what keeps the architecture on
    its single-KV-tree path.
    """
    schedule = LagunaConfig.read_layer_schedule(
        _load(config_cls, "m1_config.json")
    )

    assert schedule.layer_types == ("full_attention",) * 70
    assert schedule.num_heads_per_layer == (64,) * 70
    assert schedule.mlp_layer_types == ("dense",) * 3 + ("sparse",) * 67
    assert schedule.gating_mode == "per-element"
    assert schedule.sliding_window is None
    assert schedule.sliding_rope is None
    assert schedule.full_rope["factor"] == 64.0
    assert schedule.full_rope["partial_rotary_factor"] == 1.0


def _mutated_s21(**overrides: Any) -> LagunaHFConfig:
    """Loads S-2.1 through the shim and overwrites fields in place."""
    config = _load(LagunaHFConfig, "s21_config.json")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _flattened(name: str) -> LagunaHFConfig:
    """Rewrites a checkpoint's nested RoPE table into the flat top-level form."""
    raw: dict[str, Any] = json.loads((TESTDATA / name).read_text())
    nested = dict(raw.pop("rope_parameters")["full_attention"])
    raw["rope_theta"] = nested.pop("rope_theta")
    raw["partial_rotary_factor"] = nested.pop("partial_rotary_factor")
    raw["rope_scaling"] = nested
    return LagunaHFConfig(**raw)


def test_a_flat_rope_config_is_read_from_the_top_level_fields() -> None:
    """Without the fallback this silently yields theta 10000 and no YaRN.

    M.1 is the case the flat form can express: one table, because every layer
    is full attention.
    """
    schedule = LagunaConfig.read_layer_schedule(_flattened("m1_config.json"))

    assert schedule.full_rope["rope_theta"] == 500000.0
    assert schedule.full_rope["partial_rotary_factor"] == 1.0
    assert schedule.full_rope["rope_type"] == "yarn"
    assert schedule.full_rope["factor"] == 64.0


def test_a_flat_rope_config_is_rejected_when_layers_disagree() -> None:
    """S-2.1 needs a table per attention type, which the flat form cannot give.

    Reusing its one YaRN table on the sliding layers is what the per-type
    check already forbids, so the fallback surfaces that rather than hiding it.
    """
    with pytest.raises(ValueError, match="only default RoPE on sliding"):
        LagunaConfig.read_layer_schedule(_flattened("s21_config.json"))


def test_layer_types_length_must_match_layer_count() -> None:
    config = _mutated_s21()
    config.layer_types = config.layer_types[:-1]

    with pytest.raises(ValueError, match="layer_types has 47 entries"):
        LagunaConfig.read_layer_schedule(config)


def test_query_heads_must_be_a_multiple_of_kv_heads() -> None:
    config = _mutated_s21()
    config.num_attention_heads_per_layer[7] = 70

    with pytest.raises(ValueError, match="layer 7 has 70 query heads"):
        LagunaConfig.read_layer_schedule(config)


def test_mlp_only_layers_must_agree_with_mlp_layer_types() -> None:
    with pytest.raises(ValueError, match="mlp_only_layers"):
        LagunaConfig.read_layer_schedule(_mutated_s21(mlp_only_layers=[0, 1]))


def test_ungated_attention_is_rejected() -> None:
    with pytest.raises(ValueError, match="without an attention gate"):
        LagunaConfig.read_layer_schedule(_mutated_s21(gating=False))


def test_mixed_gating_types_are_rejected() -> None:
    """A per-layer mix would need two attention layer shapes; fail loudly."""
    gating_types = ["per_head"] * 47 + ["per_element"]

    with pytest.raises(ValueError, match="mixed gating is not supported"):
        LagunaConfig.read_layer_schedule(
            _mutated_s21(gating_types=gating_types)
        )


def test_sliding_layers_without_a_window_are_rejected() -> None:
    with pytest.raises(ValueError, match="sliding_window is 0"):
        LagunaConfig.read_layer_schedule(_mutated_s21(sliding_window=0))


def test_scaled_rope_on_sliding_layers_is_rejected() -> None:
    rope = dict(_mutated_s21().rope_parameters)
    rope["sliding_attention"] = dict(rope["sliding_attention"])
    rope["sliding_attention"]["rope_type"] = "yarn"

    with pytest.raises(ValueError, match="only default RoPE on sliding"):
        LagunaConfig.read_layer_schedule(_mutated_s21(rope_parameters=rope))


def test_yarn_attention_factor_must_match_the_hardcoded_mscale() -> None:
    """MAX computes the YaRN mscale itself and cannot be told otherwise.

    ``YarnRotaryEmbedding`` hardcodes ``0.1 * ln(factor) + 1``. Both shipped
    Laguna configs happen to carry exactly that; one that did not would be
    scaled wrongly with nothing to show for it, so it has to raise here.
    """
    rope = dict(_mutated_s21().rope_parameters)
    rope["full_attention"] = dict(rope["full_attention"])
    rope["full_attention"]["attention_factor"] = 1.0

    with pytest.raises(ValueError, match=r"attention_factor is 1\.0"):
        LagunaConfig.read_layer_schedule(_mutated_s21(rope_parameters=rope))


def _pipeline_config() -> Mock:
    """A stand-in for the fields ``construct_kv_params`` actually reads."""
    pipeline_config = Mock()
    pipeline_config.model = Mock(data_parallel_degree=1)
    pipeline_config.speculative = None
    return pipeline_config


def _kv_params(
    config_cls: type[PretrainedConfig],
    name: str,
    cache_dtype: DType = DType.bfloat16,
) -> KVCacheParams | MultiKVCacheParams:
    return LagunaConfig.construct_kv_params(
        huggingface_config=_load(config_cls, name),
        pipeline_config=_pipeline_config(),
        devices=[DeviceRef.CPU()],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=cache_dtype,
    )


def test_s21_builds_a_hybrid_kv_tree(
    config_cls: type[PretrainedConfig],
) -> None:
    """The two leaves are keyed and ordered by the tree's declaration order.

    ``unflatten_basic_kv_tree`` returns children in that order, so the graph
    build binds the sliding collections first. Swapping the two would run every
    layer against the wrong cache silently, which is why the order is asserted
    rather than assumed.
    """
    params = _kv_params(config_cls, "s21_config.json")

    assert isinstance(params, MultiKVCacheParams)
    assert list(params.children) == ["sliding_attention", "full_attention"]
    sliding, full = params.children.values()
    assert isinstance(sliding, MHAKVCacheParams)
    assert isinstance(full, MHAKVCacheParams)
    assert sliding.num_layers == 36
    assert sliding.window_size == 512
    assert full.num_layers == 12
    assert full.window_size is None
    assert sliding.n_kv_heads == full.n_kv_heads == 8
    assert sliding.head_dim == full.head_dim == 128


def test_m1_keeps_a_single_kv_cache(
    config_cls: type[PretrainedConfig],
) -> None:
    """M.1 has no sliding layer, so it must stay on the flat cache it had."""
    params = _kv_params(config_cls, "m1_config.json")

    assert isinstance(params, KVCacheParams)
    assert params.num_layers == 70
    assert params.window_size is None


def test_fp8_kv_cache_is_rejected_for_sliding_models() -> None:
    """The fused RoPE store refuses a scaled quantized cache.

    M.1 can still opt into the experimental FP8 cache; a model with sliding
    layers would fail deep inside ``rope_split_store_ragged`` instead, so it is
    refused here where the message can say why.
    """
    with pytest.raises(ValueError, match="cannot use an FP8 KV cache"):
        _kv_params(LagunaHFConfig, "s21_config.json", DType.float8_e4m3fn)


def test_unknown_layer_types_are_rejected() -> None:
    """``layer_types`` entries key the KV tree's children verbatim."""
    config = _mutated_s21()
    config.layer_types = ["linear_attention"] + list(config.layer_types[1:])

    with pytest.raises(ValueError, match="unsupported attention types"):
        LagunaConfig.read_layer_schedule(config)


def _model_config(max_length: int | None) -> Mock:
    """A stand-in for the fields ``calculate_max_seq_len`` actually reads."""
    model_config = Mock()
    model_config.max_length = max_length
    model_config.max_length_is_user_provided = max_length is not None
    return model_config


def test_default_max_seq_len_follows_the_config(
    config_cls: type[PretrainedConfig],
) -> None:
    """A default deployment resolves to the length the config declares."""
    huggingface_config = _load(config_cls, "m1_config.json")
    assert huggingface_config.max_position_embeddings == 262144

    resolved = LagunaConfig.calculate_max_seq_len(
        huggingface_config, _model_config(None)
    )

    assert resolved == 262144


def test_user_provided_max_seq_len_is_honored(
    config_cls: type[PretrainedConfig],
) -> None:
    """An explicit length wins, bounded only by ``max_position_embeddings``."""
    huggingface_config = _load(config_cls, "s21_config.json")

    assert (
        LagunaConfig.calculate_max_seq_len(
            huggingface_config, _model_config(524288)
        )
        == 524288
    )
    assert (
        LagunaConfig.calculate_max_seq_len(
            huggingface_config, _model_config(8192)
        )
        == 8192
    )
