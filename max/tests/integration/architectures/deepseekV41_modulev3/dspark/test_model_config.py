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
"""CPU tests of the DSpark config fields.

Draft width is hard-pinned: ``num_speculative_tokens`` must equal the
checkpoint's ``dspark_block_size`` (the non-causal block the ``mtp.*``
weights were trained with), and the KV cache reserves exactly that many
draft slots per step.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from dspark_test_support import TESTDATA
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures import hf_config_shims
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
    dspark_draft_width,
    dspark_expert_counts,
)
from max.pipelines.lib import KVCacheConfig

BLOCK = 5


def _text_mapping(block_size: int = BLOCK) -> dict[str, Any]:
    text = json.load(open(TESTDATA / "config_full.json"))["text_config"]
    return {**text, "dspark_block_size": block_size}


def _hf(block_size: int = BLOCK) -> SimpleNamespace:
    # The pipeline-level namespace: the published checkpoint's text fields
    # under ``text_config`` (the shape ``resolve_text_fields`` reads), with
    # the draft block size overridden per test.
    return SimpleNamespace(
        text_config=SimpleNamespace(**_text_mapping(block_size))
    )


def _resolved(text_config: SimpleNamespace) -> DeepseekV41Config:
    interface = V41Config.from_hf({"text_config": vars(text_config)})
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=build_kv_params(
            interface,
            KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
            [DeviceRef.CPU()],
        ),
        devices=[DeviceRef.CPU()],
        max_seq_len=1024,
        **DeepseekV41Config.resolve_text_fields(text_config),
    )


@pytest.mark.parametrize("requested", [None, BLOCK])
def test_draft_width_resolves_to_block_size(requested: int | None) -> None:
    speculative = SimpleNamespace(num_speculative_tokens=requested)
    assert dspark_draft_width(speculative, _hf(), None) == BLOCK


@pytest.mark.parametrize("requested", [1, BLOCK - 1, BLOCK + 1])
def test_draft_width_rejects_other_widths(requested: int) -> None:
    speculative = SimpleNamespace(num_speculative_tokens=requested)
    with pytest.raises(ValueError, match="dspark_block_size"):
        dspark_draft_width(speculative, _hf(), None)


def test_stage_count_resolves_from_nextn_predict_layers() -> None:
    """``n_mtp_layers`` wins when both name the stage count; either works."""
    assert _resolved(_hf().text_config).num_dspark_layers == 3
    aliased = _hf()
    aliased.text_config = SimpleNamespace(
        **{**vars(aliased.text_config), "n_mtp_layers": 2}
    )
    assert _resolved(aliased.text_config).num_dspark_layers == 2


def test_zero_mtp_layers_declares_no_drafter() -> None:
    """Zero is a declaration, not an absence: it must not reach the default."""
    aliased = _hf()
    aliased.text_config = SimpleNamespace(
        **{**vars(aliased.text_config), "n_mtp_layers": 0}
    )
    assert _resolved(aliased.text_config).num_dspark_layers == 0


def test_checkpoint_config_resolves_every_dspark_field() -> None:
    """The full checkpoint's text_config resolves the published values."""
    full = json.load(open(TESTDATA / "config_full.json"))
    text = full["text_config"]
    config = _resolved(SimpleNamespace(**text))
    assert config.dspark_target_layer_ids == [37, 38, 39]
    assert config.num_dspark_layers == 3
    assert config.dspark_noise_token_id == 128799
    assert config.dspark_block_size == 5
    assert config.dspark_markov_rank == 256
    assert config.dspark_n_routed_experts == 128
    assert config.dspark_num_experts_per_tok == 3
    assert config.routed_scaling_factor == 1.5
    assert config.swiglu_limit == 10.0
    # The geometry the DSpark drafter reads (its stages reuse the trunk's
    # latent widths and mHC schedule).
    assert config.num_attention_heads == 64
    assert config.head_dim == 512
    assert config.q_lora_rank == 1280
    assert config.o_groups == 8
    assert config.o_lora_rank == 1024
    assert config.hc_mult == 4
    # The stage count and the tap list are declared independent; the swa
    # leaf sizes off the tap list, and only the drafter's build asserts
    # the two agree.
    assert len(config.compress_ratios) == (
        config.num_hidden_layers + len(config.dspark_target_layer_ids)
    )


def test_swiglu_limit_is_validated_when_config_is_built() -> None:
    fields = vars(_hf().text_config)
    _resolved(SimpleNamespace(**{**fields, "swiglu_limit": 10.0}))
    with pytest.raises(ValueError, match="not exact in bfloat16"):
        _resolved(SimpleNamespace(**{**fields, "swiglu_limit": 10.1}))


def _shim_hf(block_size: int = BLOCK) -> hf_config_shims.DeepseekV41HFConfig:
    """The real HF shim, built from the published checkpoint mapping."""
    return hf_config_shims.DeepseekV41HFConfig(
        text_config=_text_mapping(block_size)
    )


@pytest.mark.parametrize("requested", [None, BLOCK])
def test_draft_width_resolves_through_real_hf_shim(
    requested: int | None,
) -> None:
    """``get_text_config`` applied twice.

    A caller that passes the already-normalized text config into
    ``dspark_draft_width`` normalizes it again; the second pass must not
    unwrap (the text config has no ``text_config``).
    """
    speculative = SimpleNamespace(num_speculative_tokens=requested)
    hf = _shim_hf()
    assert dspark_draft_width(speculative, hf, None) == BLOCK
    assert (
        dspark_draft_width(
            speculative, DeepseekV41Config.get_text_config(hf), None
        )
        == BLOCK
    )


def test_draft_width_rejects_other_widths_through_real_hf_shim() -> None:
    speculative = SimpleNamespace(num_speculative_tokens=BLOCK - 1)
    with pytest.raises(ValueError, match="dspark_block_size"):
        dspark_draft_width(speculative, _shim_hf(), None)


@pytest.mark.parametrize("via_wrapper_mapping", [True, False])
def test_draft_width_resolves_mapping_forms(via_wrapper_mapping: bool) -> None:
    """A wrapper dict (``to_dict`` shape) and a bare text mapping both work."""
    speculative = SimpleNamespace(num_speculative_tokens=None)
    target: dict[str, Any] = (
        {"text_config": _text_mapping()}
        if via_wrapper_mapping
        else _text_mapping()
    )
    assert dspark_draft_width(speculative, target, None) == BLOCK


def test_resolve_text_fields_through_real_shim_text_config() -> None:
    """Model-config construction from the shim's double-normalized text."""
    text_config = DeepseekV41Config.get_text_config(
        DeepseekV41Config.get_text_config(_shim_hf())
    )
    config = _resolved(text_config)
    assert config.dspark_block_size == BLOCK
    assert config.num_dspark_layers == 3
    assert config.dspark_noise_token_id == 128799


def test_dspark_expert_counts_fall_back_to_the_trunk() -> None:
    """0 or absent uses the trunk's counts; top-k reads either key name."""
    trunk = dict(n_routed_experts=384, num_experts_per_tok=6)
    assert dspark_expert_counts(SimpleNamespace(**trunk)) == (384, 6)
    assert dspark_expert_counts(
        SimpleNamespace(
            **trunk, dspark_n_routed_experts=0, dspark_num_experts_per_tok=0
        )
    ) == (384, 6)
    assert dspark_expert_counts(
        SimpleNamespace(
            **trunk, dspark_n_routed_experts=128, dspark_n_activated_experts=4
        )
    ) == (128, 4)
    assert dspark_expert_counts(
        SimpleNamespace(
            **trunk,
            dspark_n_routed_experts=128,
            dspark_num_experts_per_tok=3,
            dspark_n_activated_experts=4,
        )
    ) == (128, 3)
