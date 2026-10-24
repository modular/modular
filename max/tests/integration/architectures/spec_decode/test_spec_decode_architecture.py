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

from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodeArchConfig,
    SpecDecodePipelineModel,
    spec_decode_architecture,
)

TARGET = "LlamaForCausalLM"
# An EAGLE checkpoint declares its target's own architecture; the target maps
# it to the draft architecture.
EAGLE_DRAFT = "LlamaForCausalLM"
DFLASH_DRAFT = "DFlashDraftModel"


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
