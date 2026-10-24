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


"""What a DFlash2 draft checkpoint declares, and what it is paired with.

Every value read here is one whose mismatch produces wrong drafts and no
error: the block width sets the convolution's period, the tap ids choose which
target layers ``fc`` sees, and the selector geometry sizes a walk that would
otherwise index a differently shaped tensor. The real ``z-lab/Qwen3.8-27B-
DFlash2`` config is committed under ``testdata/`` so this is parsing the
shipped file, not a paraphrase of it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from dflash2_pairing_harness import (
    TARGET_ARCH,
    Geometry,
    draft_config,
    target_config,
)
from max.graph import DeviceRef
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.architectures.dflash2_qwen3_5 import DFlash2Qwen3_5
from max.pipelines.architectures.dflash2_qwen3_5.model_config import (
    dflash2_draft_width,
    parse_dflash2_draft_hf_config,
)
from max.pipelines.lib.config.config import _construct_from_user_fields
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _check_pairing,
)
from max.pipelines.speculative import (
    SpeculativeConfig,
    proposers,
    spec_adapters,
)

TESTDATA = Path(__file__).parent / "testdata" / "zlab_dflash2_config.json"
BLOCK = 8


def _draft_hf_config(**overrides: Any) -> SimpleNamespace:
    raw = json.loads(TESTDATA.read_text())
    dflash = dict(raw["dflash_config"])
    dflash.update(overrides.pop("dflash_config", {}))
    raw.update(overrides)
    raw["dflash_config"] = dflash
    return SimpleNamespace(**raw)


def test_the_shipped_config_parses_to_the_drafters_geometry() -> None:
    parsed = parse_dflash2_draft_hf_config(_draft_hf_config())
    assert parsed.block_size == 8
    assert parsed.mask_token_id == 248070
    assert parsed.target_layer_ids == [5, 19, 33, 47, 61]
    assert parsed.num_target_layers == 64
    # The four keys v1's parser ignores. Each one sizes a tensor: the conv's
    # taps and group width, and the selector's rank and candidate count.
    assert parsed.conv_kernel_size == 2
    assert parsed.conv_group_size == 16
    assert parsed.selector_rank == 256
    assert parsed.selector_top_k == 16


@pytest.mark.parametrize(
    "missing",
    ["conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"],
)
def test_a_missing_selector_or_conv_key_is_refused(missing: str) -> None:
    raw = json.loads(TESTDATA.read_text())
    dflash = {k: v for k, v in raw["dflash_config"].items() if k != missing}
    raw["dflash_config"] = dflash
    with pytest.raises(ValueError, match=missing):
        parse_dflash2_draft_hf_config(SimpleNamespace(**raw))


def test_an_embedding_or_logit_rescale_is_refused() -> None:
    """The reference applies both; this graph applies neither.

    A checkpoint that set one would draft from a differently scaled space and
    produce fluent, low-acceptance drafts rather than an error.
    """
    with pytest.raises(ValueError, match="input_embedding_scale"):
        parse_dflash2_draft_hf_config(
            _draft_hf_config(dflash_config={"input_embedding_scale": 2.0})
        )
    with pytest.raises(ValueError, match="output_multiplier"):
        parse_dflash2_draft_hf_config(
            _draft_hf_config(dflash_config={"output_multiplier": 0.5})
        )


def test_the_draft_width_comes_from_the_checkpoints_block_size() -> None:
    unset = SpeculativeConfig(speculative_method="dflash2")
    assert dflash2_draft_width(unset, None, _draft_hf_config()) == 7
    explicit = SpeculativeConfig(
        speculative_method="dflash2", num_speculative_tokens=7
    )
    assert dflash2_draft_width(explicit, None, _draft_hf_config()) == 7


def test_a_disagreeing_draft_width_is_a_hard_error() -> None:
    """v1 warns and overrides; DFlash2 refuses.

    The export records the width beside the MEF, which the engine checks its
    own ``--num-speculative-tokens`` against at boot, so an override would
    compile a block-8 graph and declare a different width -- and the engine
    feeds the graph at the width it was told.
    """
    explicit = SpeculativeConfig(
        speculative_method="dflash2", num_speculative_tokens=3
    )
    with pytest.raises(ValueError, match="block_size=8"):
        dflash2_draft_width(explicit, None, _draft_hf_config())


def test_an_omitted_width_resolves_the_way_pipeline_config_resolves_it() -> (
    None
):
    """``from_args`` fills an omitted width only via ``checkpoint_draft_width``.

    Nothing else resolves it, so without the registered callback the pipeline
    config keeps ``None`` while the graph is built at seven, and whatever reads
    the config instead of the graph sees zero. Building a real
    ``PipelineConfig`` needs the checkpoints, so this drives the two steps
    ``from_args`` runs: the registry callback, then the override applied to the
    user-set fields.
    """
    unset = SpeculativeConfig(speculative_method="dflash2")
    assert unset.num_speculative_tokens is None

    width = dflash2_draft_width(unset, None, _draft_hf_config())
    assert width == 7
    resolved = _construct_from_user_fields(
        unset, **({"num_speculative_tokens": width} if width else {})
    )
    assert resolved.num_speculative_tokens == 7
    assert resolved.draft_width == 7


def test_the_registry_callback_refuses_a_mismatch_and_a_missing_draft() -> None:
    explicit = SpeculativeConfig(
        speculative_method="dflash2", num_speculative_tokens=3
    )
    with pytest.raises(ValueError, match="block_size=8"):
        dflash2_draft_width(explicit, None, _draft_hf_config())
    with pytest.raises(ValueError, match="requires a draft model"):
        dflash2_draft_width(
            SpeculativeConfig(speculative_method="dflash2"), None, None
        )


def _speculative() -> SpeculativeConfig:
    return SpeculativeConfig(
        speculative_method="dflash2", num_speculative_tokens=BLOCK - 1
    )


def _target_text_config(**overrides: Any) -> SimpleNamespace:
    """The target's text config, as far as a pairing reads it."""
    fields: dict[str, Any] = {
        "num_hidden_layers": 4,
        "hidden_size": 64,
        "vocab_size": 128,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_a_valid_pairing_passes_validation() -> None:
    _check_pairing(
        _speculative(), draft_config(Geometry()), _target_text_config()
    )
    assert _speculative().draft_width == 7


@pytest.mark.parametrize(
    ("dflash", "geometry", "match"),
    [
        ({"target_layer_ids": [0, 1, 2, 99]}, {}, "index the target's layers"),
        ({"num_target_layers": 8}, {}, "trained against a target of"),
        ({}, {"hidden": 32}, "hidden_size must match"),
        ({}, {"vocab": 64}, "vocab_size must match"),
        ({"mask_token_id": 999}, {}, "mask_token_id"),
    ],
)
def test_validation_refuses_a_mismatched_pairing(
    dflash: dict[str, Any], geometry: dict[str, Any], match: str
) -> None:
    config = draft_config(Geometry(**geometry))
    config = replace(config, dflash=replace(config.dflash, **dflash))
    with pytest.raises(ValueError, match=match):
        _check_pairing(_speculative(), config, _target_text_config())


def test_the_declared_target_depth_is_checked_but_stays_optional() -> None:
    """The tap ids cannot catch a depth mismatch on their own.

    They only check that each id is in range, so a drafter trained against a
    deeper target passes them against any target that holds its deepest tap
    while reading a differently trained residual stream.
    """
    config = draft_config(Geometry())
    for depth in (4, None):
        checked = replace(
            config, dflash=replace(config.dflash, num_target_layers=depth)
        )
        _check_pairing(_speculative(), checked, _target_text_config())


def test_the_target_is_put_into_tap_mode_and_out_of_subgraph_mode() -> None:
    """Two settings the fused graph cannot work without.

    Selected layers plus the tap ids is what makes the drafter's context exist
    at all; ``use_subgraphs = False`` is what lets the rollback read the verify
    pass's per-layer state-kernel inputs, which cannot cross a subgraph
    boundary.
    """
    geometry = Geometry()
    draft = draft_config(geometry)
    assert (
        proposers.target_hidden_states(_speculative(), draft)
        == ReturnHiddenStates.SELECTED_LAYERS
    )
    assert proposers.capture_layers(_speculative(), draft) == tuple(
        geometry.tap_ids
    )

    target = target_config(geometry)
    target.use_subgraphs = True
    module = cast("Module", SimpleNamespace())
    spec_adapters.verify_for(
        TARGET_ARCH, module, target, ReturnHiddenStates.SELECTED_LAYERS, draft
    )
    assert target.use_subgraphs is False
    with pytest.raises(ValueError, match="selected layers"):
        spec_adapters.verify_for(
            TARGET_ARCH,
            module,
            target,
            ReturnHiddenStates.LAST,
            draft,
        )


def test_a_draft_without_context_features_is_refused() -> None:
    """The tap ids feed ``fc``; none would build a drafter with no context,
    and the verify would return no layers for it."""
    llama = draft_config(Geometry(device=DeviceRef.CPU())).llama
    with pytest.raises(ValueError, match="must be positive"):
        DFlash2Qwen3_5(
            llama,
            num_context_features=0,
            block_size=BLOCK,
            conv_kernel_size=2,
            conv_group_size=8,
            selector_rank=4,
            selector_top_k=3,
        )


def test_a_layer_types_list_must_cover_every_draft_layer() -> None:
    config = draft_config(Geometry(device=DeviceRef.CPU(), draft_layers=2))
    with pytest.raises(ValueError, match="one entry per draft layer"):
        DFlash2Qwen3_5(
            config.llama,
            num_context_features=4,
            block_size=BLOCK,
            conv_kernel_size=2,
            conv_group_size=8,
            selector_rank=4,
            selector_top_k=3,
            layer_types=["sliding_attention"],
        )


def test_multi_device_is_refused_rather_than_silently_single_device() -> None:
    cpu = DeviceRef.CPU()
    config = draft_config(Geometry(device=cpu))
    with pytest.raises(ValueError, match="single device"):
        DFlash2Qwen3_5(
            replace(config.llama, devices=[cpu, cpu]),
            num_context_features=4,
            block_size=BLOCK,
            conv_kernel_size=2,
            conv_group_size=8,
            selector_rank=4,
            selector_top_k=3,
        )


def test_is_dflash_is_the_family_predicate_and_is_dflash2_is_exact() -> None:
    """``is_dflash()`` gates generic machinery (pipeline-class selection, the
    draft-arch rewrite) that wants both drafters; ``is_dflash2()`` gates the
    places where the two differ."""
    v1 = SpeculativeConfig(speculative_method="dflash")
    v2 = SpeculativeConfig(speculative_method="dflash2")
    assert v1.is_dflash() and v2.is_dflash()
    assert not v1.is_dflash2() and v2.is_dflash2()
    assert not SpeculativeConfig(speculative_method="mtp").is_dflash()
