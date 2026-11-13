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

"""How a target and a draft architecture resolve into one fused architecture.

Resolution decides whether a request runs through ``SpecDecodePipelineModel``
or falls back to the legacy per-pairing architectures, so a wrong answer
either serves the wrong graph or refuses a pairing that should work.
"""

from __future__ import annotations

from typing import Literal

import pytest
from max.pipelines.context.context_validators import validate_no_images
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _REFUSES_IMAGES_WITHOUT_VISION,
    SpecDecodeArchConfig,
    SpecDecodePipelineModel,
    _SpecDecodeModuleV3PipelineModel,
    _SpecDecodeModuleV3VisionCachePipelineModel,
    spec_decode_architecture,
)
from max.pipelines.modeling.types import InputModality

TARGET = "LlamaForCausalLM"
# An EAGLE checkpoint declares its target's own architecture; the target maps
# it to the draft architecture.
EAGLE_DRAFT = "LlamaForCausalLM"
DFLASH_DRAFT = "DFlashDraftModel"


def test_only_mistral_large_4_refuses_images_in_a_text_only_pairing() -> None:
    """The image refusal names one target, and every other pairing is untouched.

    A pairing refuses images exactly when its target takes them but its fused
    graph encodes none. Mistral Large 4 + DSpark is the only pair shipped that
    way, and its own target proves the validator lands on the fused graph
    (``test_spec_decode_layout``). Qwen3.5 + MTP has the same shape -- Qwen3.5
    declares image input but builds no vision graph, and its MTP draft is
    text-only -- and keeps main's behaviour, which drops the image at batch
    preparation rather than refusing the request. A pairing that encodes -- its
    target builds a vision graph and its draft declares images -- serves
    images, as Inkling + MTP, DeepseekV41 + DSpark and Glm5Next + MTP do.
    """
    # Resolving Mistral Large 4's pairing imports its architecture package,
    # which this shared target does not depend on; that target pins both ends
    # -- the name below and the validator on the fused pairing.
    assert _REFUSES_IMAGES_WITHOUT_VISION == {
        "MistralLarge4ForConditionalGeneration_ModuleV3"
    }

    qwen = spec_decode_architecture(
        "Qwen3_5ForConditionalGeneration", "mtp", None
    )
    assert qwen is not None
    assert validate_no_images not in qwen.context_validators
    assert InputModality.IMAGE not in qwen.input_modalities
    assert qwen.input_modalities == {InputModality.TEXT}

    vision_pairings: list[tuple[str, Literal["mtp", "dspark"]]] = [
        ("InklingForConditionalGeneration", "mtp"),
        ("DeepseekV41ForCausalLM", "dspark"),
        ("Glm5NextForConditionalGeneration", "mtp"),
    ]
    for target, method in vision_pairings:
        pairing = spec_decode_architecture(target, method, None)
        assert pairing is not None, (target, method)
        assert validate_no_images not in pairing.context_validators, target
        assert InputModality.IMAGE in pairing.input_modalities, target


def test_a_target_and_its_draft_resolve_to_the_fused_architecture() -> None:
    arch = spec_decode_architecture(TARGET, "eagle", EAGLE_DRAFT)
    assert arch is not None
    assert arch.name == f"{TARGET}+eagle"
    assert arch.pipeline_model is SpecDecodePipelineModel
    assert arch.config is SpecDecodeArchConfig


def test_the_fused_architecture_is_named_for_its_method() -> None:
    eagle = spec_decode_architecture(TARGET, "eagle", EAGLE_DRAFT)
    dflash = spec_decode_architecture(TARGET, "dflash", DFLASH_DRAFT)
    assert eagle is not None and dflash is not None
    assert dflash.name == f"{TARGET}+dflash"
    assert eagle.name != dflash.name


def test_only_a_block_method_serves_mixed_prefill_and_decode_batches() -> None:
    """The block driver commits per row, so a prefill row riding a decode
    batch commits its whole chunk; the sequential driver cannot."""
    eagle = spec_decode_architecture(TARGET, "eagle", EAGLE_DRAFT)
    dflash = spec_decode_architecture(TARGET, "dflash", DFLASH_DRAFT)
    assert eagle is not None and dflash is not None
    assert not eagle.supports_spec_decode_mixed_batches
    assert dflash.supports_spec_decode_mixed_batches


def test_the_fused_architecture_runs_one_text_only_device() -> None:
    arch = spec_decode_architecture(TARGET, "eagle", EAGLE_DRAFT)
    assert arch is not None
    assert not arch.multi_gpu_supported
    assert arch.cascade_pipeline_factory is None


def test_no_method_or_no_draft_does_not_pair() -> None:
    assert spec_decode_architecture(TARGET, None, EAGLE_DRAFT) is None
    assert spec_decode_architecture(TARGET, "eagle", None) is None


def test_an_unregistered_draft_does_not_pair() -> None:
    assert spec_decode_architecture(TARGET, "eagle", "NoSuchDraftModel") is None


def test_an_unregistered_target_does_not_pair() -> None:
    assert (
        spec_decode_architecture("NoSuchTarget", "eagle", EAGLE_DRAFT) is None
    )


def test_a_draft_only_architecture_is_never_the_target() -> None:
    """A draft registers without a pipeline model, so there is nothing to
    serve as the main model. Resolution must refuse it, not crash on it."""
    assert (
        spec_decode_architecture(DFLASH_DRAFT, "dflash", DFLASH_DRAFT) is None
    )


@pytest.mark.parametrize(
    ("target", "draft"),
    [
        ("Gemma4ForConditionalGeneration", "Gemma4AssistantForCausalLM"),
        # The MTP depths ship in the target's checkpoint.
        ("InklingForConditionalGeneration", None),
    ],
)
def test_an_mtp_draft_caps_the_context_its_batch_keeps_resident(
    target: str, draft: str | None
) -> None:
    """Neither target caps it alone, so an uncapped pairing lets the
    scheduler admit more resident context than the cache holds."""
    arch = spec_decode_architecture(target, "mtp", draft)
    assert arch is not None
    assert arch.requires_max_batch_context_length


def test_a_pairing_neither_side_caps_stays_uncapped() -> None:
    arch = spec_decode_architecture(TARGET, "eagle", EAGLE_DRAFT)
    assert arch is not None
    assert not arch.requires_max_batch_context_length


def test_a_modulev3_target_pairs_with_its_modulev3_draft() -> None:
    arch = spec_decode_architecture(
        "Gemma4ForConditionalGeneration",
        "mtp",
        "Gemma4AssistantForCausalLM",
        prefer_module_v3=True,
    )
    assert arch is not None
    assert arch.name == "Gemma4ForConditionalGeneration_ModuleV3+mtp"
    assert arch.pipeline_model is _SpecDecodeModuleV3VisionCachePipelineModel
    assert InputModality.IMAGE in arch.input_modalities
    assert arch.requires_max_batch_context_length


def test_a_modulev3_block_target_pairs_with_its_modulev3_drafter() -> None:
    # The checkpoint names its drafter as Llama 3's, which the target renames.
    arch = spec_decode_architecture(
        "MiMoV2ForCausalLM", "dflash", "DFlashDraftModel", prefer_module_v3=True
    )
    assert arch is not None
    assert arch.name == "MiMoV2ForCausalLM_ModuleV3+dflash"
    assert arch.pipeline_model is _SpecDecodeModuleV3PipelineModel
    assert arch.multi_gpu_supported
    assert arch.supports_spec_decode_mixed_batches


def test_a_target_that_builds_both_pairs_on_the_graph_api_by_default() -> None:
    arch = spec_decode_architecture(
        "Gemma4ForConditionalGeneration", "mtp", "Gemma4AssistantForCausalLM"
    )
    assert arch is not None
    assert arch.name == "Gemma4ForConditionalGeneration+mtp"
