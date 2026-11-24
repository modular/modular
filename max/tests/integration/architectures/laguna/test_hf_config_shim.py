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
"""Whatever class ``AutoConfig`` hands MAX for ``model_type: laguna`` must hand
back every field the architecture reads, unchanged.

``LagunaConfig`` (the MAX config) reads raw fields off the loaded HuggingFace
config, so a field that the config class drops, flattens or defaults away is one
the architecture silently stops seeing -- a wrong model rather than an error.
Both shipped configs are committed under ``testdata/``, so these assertions run
against the real files rather than a paraphrase of them.

Two classes can end up in that role. ``LagunaHFConfig`` is the in-tree shim; the
pinned transformers also ships its own ``laguna`` config, which wins the
``AutoConfig`` registration (see
:func:`test_laguna_model_type_resolves_through_autoconfig`). Every field
assertion below therefore runs against both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import transformers
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.hf_config_shims import LagunaHFConfig
from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

TESTDATA = Path(__file__).parent / "testdata"

# `[full, sliding, sliding, sliding]` repeated over S-2.1's 48 layers, with 48
# query heads on the full-attention layers and 72 on the sliding ones.
S21_LAYER_TYPES = ["full_attention"] + ["sliding_attention"] * 3
S21_HEADS = [48, 72, 72, 72]

# S-2.1's routed experts in layers 40-47 ship as plain BF16 rather than NVFP4,
# and this regex in the checkpoint's ignore list is the only record of it.
S21_BF16_EXPERTS_PATTERN = r"re:^model\.layers\.4[0-7]\.mlp\.experts(\..*)?$"


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


def test_s21_rope_table_stays_nested_per_attention_type(
    config_cls: type[PretrainedConfig],
) -> None:
    rope = _load(config_cls, "s21_config.json").rope_parameters
    assert rope["full_attention"]["rope_type"] == "yarn"
    assert rope["full_attention"]["factor"] == 128.0
    assert rope["full_attention"]["original_max_position_embeddings"] == 8192
    assert rope["full_attention"]["beta_fast"] == 32.0
    assert rope["full_attention"]["beta_slow"] == 1.0
    assert rope["full_attention"]["partial_rotary_factor"] == 0.5
    assert rope["full_attention"]["rope_theta"] == 500000.0
    assert rope["sliding_attention"]["rope_type"] == "default"
    assert rope["sliding_attention"]["rope_theta"] == 10000.0
    assert rope["sliding_attention"]["partial_rotary_factor"] == 1.0


def test_s21_layer_schedule_round_trips(
    config_cls: type[PretrainedConfig],
) -> None:
    config = _load(config_cls, "s21_config.json")
    assert config.num_hidden_layers == 48
    assert config.layer_types == S21_LAYER_TYPES * 12
    assert config.layer_types.count("full_attention") == 12
    assert config.sliding_window == 512
    assert config.num_attention_heads_per_layer == S21_HEADS * 12
    assert config.gating == "per-head"
    assert config.gating_types == ["per_head"] * 48
    assert config.mlp_only_layers == [0]
    assert config.mlp_layer_types == ["dense"] + ["sparse"] * 47


def test_s21_quantization_ignore_list_survives(
    config_cls: type[PretrainedConfig],
) -> None:
    ignore = _load(config_cls, "s21_config.json").quantization_config["ignore"]
    assert S21_BF16_EXPERTS_PATTERN in ignore
    assert "model.layers.0.mlp.gate_proj" in ignore
    assert r"re:.*\.mlp\.shared_expert\.down_proj$" in ignore


def test_m1_config_is_unaffected(config_cls: type[PretrainedConfig]) -> None:
    config = _load(config_cls, "m1_config.json")
    assert config.num_hidden_layers == 70
    assert config.num_attention_heads == 64
    assert config.gating == "per-element"
    assert config.rope_parameters["full_attention"]["factor"] == 64.0
    assert (
        config.rope_parameters["full_attention"]["partial_rotary_factor"] == 1.0
    )
    assert "sliding_attention" not in config.rope_parameters


def test_the_shim_invents_no_per_layer_tables() -> None:
    """M.1 ships neither ``layer_types`` nor ``num_attention_heads_per_layer``.

    The shim must not manufacture them: the architecture's uniform-attention
    defaults key off their absence, and a synthesized table would be a second,
    silently diverging source of truth.
    """
    config = _load(LagunaHFConfig, "m1_config.json")
    assert getattr(config, "layer_types", None) is None
    assert getattr(config, "num_attention_heads_per_layer", None) is None


def test_laguna_model_type_resolves_through_autoconfig() -> None:
    """``model_type: laguna`` must load without ``trust_remote_code``.

    The pinned transformers (>=5.12,<5.13) ships a native ``laguna`` config,
    which occupies the ``AutoConfig`` slot and makes the shim's registration a
    no-op; the shim is the fallback for releases that carry no native class.
    Either way the slot must be filled, because the alternative is transformers
    executing the repo's own remote ``configuration_laguna.py``.
    """
    register_all_models()
    assert CONFIG_MAPPING["laguna"] in _CONFIG_CLASSES.values()
    assert isinstance(
        transformers.AutoConfig.for_model("laguna"), PretrainedConfig
    )
