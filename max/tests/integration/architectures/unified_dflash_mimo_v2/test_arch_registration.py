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
"""Registry wiring and verify-width resolution for MiMo-V2 with DFlash."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.mimo_v2.arch import mimo_v2_arch
from max.pipelines.architectures.unified_dflash_mimo_v2 import (
    UnifiedDflashMiMoV2Model,
)
from max.pipelines.architectures.unified_dflash_mimo_v2.batch_processor import (
    UnifiedDflashMiMoV2BatchProcessor,
)
from max.pipelines.architectures.unified_dflash_mimo_v2.model_config import (
    UnifiedDflashMiMoV2Config,
    mimo_dflash_draft_width,
)
from max.pipelines.lib import Speculator, SupportedArchitecture
from max.pipelines.lib.arch_lookup import select_speculator
from max.pipelines.lib.config.config import (
    _apply_speculative_target_architecture,
)
from max.pipelines.speculative.config import SpeculativeConfig

DRAFT_HF = {"dflash_config": {"mask_token_id": 151675, "target_layer_ids": [0]}}


def _fused(draft_arch: str) -> SupportedArchitecture:
    speculator = select_speculator("MiMoV2ForCausalLM", "dflash", draft_arch)
    assert speculator is not None
    return speculator.derive()


@pytest.mark.parametrize("draft_arch", ["LlamaForCausalLM", "DFlashDraftModel"])
def test_the_speculator_derives_the_fused_architecture(draft_arch: str) -> None:
    # Field for field what the fused architecture was when it was registered
    # by name, so selecting it runs the same graph, KV tree and planner.
    assert _fused(draft_arch) == replace(
        mimo_v2_arch,
        name="UnifiedDflashMiMoV2ForCausalLM",
        pipeline_model=UnifiedDflashMiMoV2Model,
        config=UnifiedDflashMiMoV2Config,
        batching=UnifiedDflashMiMoV2BatchProcessor,
        checkpoint_draft_width=mimo_dflash_draft_width,
        supports_spec_decode_mixed_batches=True,
    )
    assert (
        PIPELINE_REGISTRY.retrieve_architecture(
            "UnifiedDflashMiMoV2ForCausalLM"
        )
        is None
    )


def _selected(
    speculative: SpeculativeConfig | None, draft_arch: str = "LlamaForCausalLM"
) -> Speculator | None:
    models: dict[str, Any] = {
        "main": SimpleNamespace(
            huggingface_config=SimpleNamespace(
                architectures=["MiMoV2ForCausalLM"]
            )
        ),
        "draft": SimpleNamespace(
            huggingface_config=SimpleNamespace(architectures=[draft_arch])
        ),
    }
    speculator = _apply_speculative_target_architecture(speculative, models)
    assert models["main"].huggingface_config.architectures == [
        "MiMoV2ForCausalLM"
    ]
    return speculator


@pytest.mark.parametrize("draft_arch", ["LlamaForCausalLM", "DFlashDraftModel"])
def test_dflash_selects_the_fused_graph(draft_arch: str) -> None:
    speculator = _selected(
        SpeculativeConfig(speculative_method="dflash"), draft_arch
    )
    assert speculator is not None
    assert speculator.name == "UnifiedDflashMiMoV2ForCausalLM"


def test_no_speculation_keeps_the_base_graph() -> None:
    assert _selected(None) is None


def test_other_methods_are_rejected() -> None:
    with pytest.raises(ValueError, match="No speculator for MiMoV2ForCausalLM"):
        _selected(SpeculativeConfig(speculative_method="dflash2"))


def _width(k: int | None, block: int = 8) -> int:
    return mimo_dflash_draft_width(
        SpeculativeConfig(
            speculative_method="dflash", num_speculative_tokens=k
        ),
        None,
        {**DRAFT_HF, "block_size": block},
    )


def test_verify_width_is_decoupled_from_the_block() -> None:
    assert _width(None) == 7
    for k in range(1, 8):
        assert _width(k) == k
    for k in (0, 8):
        with pytest.raises(ValueError, match="verifies 1 to 7"):
            _width(k)
