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
"""Registry wiring and graph-name resolution for the ModuleV3 MiMo-V2 DFlash
speculators.

The ModuleV3 speculator is a drop-in variant of the graph-API one: the same
draft spellings, the same ``dflash`` method, and a fused graph that stages
under the graph-API build's name -- the engine's block-spec executor binds
that name by constant, so either export serves one engine. These tests pin
the wiring so a rename or a drift from the V2 lane fails here, not at
export time.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.mimo_v2_modulev3.arch import (
    mimo_v2_modulev3_arch,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3 import (
    UnifiedDflashMiMoV2ModuleV3Model,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.batch_processor import (
    UnifiedDflashMiMoV2ModuleV3BatchProcessor,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.memory_planner import (
    UnifiedDflashMiMoV2ModuleV3MemoryPlanner,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.model_config import (
    UnifiedDflashMiMoV2Config,
    mimo_dflash_draft_width,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.unified_dflash_mimo_v2 import (
    GRAPH_NAME,
)
from max.pipelines.lib import Speculator, SupportedArchitecture
from max.pipelines.lib.arch_lookup import select_speculator
from max.pipelines.lib.config.config import (
    _apply_speculative_target_architecture,
)
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
)
from max.pipelines.speculative.config import SpeculativeConfig


def _v3_speculator(draft_arch: str) -> Speculator:
    speculator = select_speculator(
        "MiMoV2ForCausalLM",
        "dflash",
        draft_arch,
        prefer_module_v3=True,
    )
    assert speculator is not None
    return speculator


@pytest.mark.parametrize("draft_arch", ["DFlashMiMoV2DraftModel"])
def test_the_flag_selects_the_modulev3_speculator(draft_arch: str) -> None:
    """``prefer_module_v3`` derives the ModuleV3 fused architecture: field
    for field the ModuleV3 base with this package's model, batching,
    config and memory planner, so a selection runs the fused graph with
    the base's tokenizer and parsers."""
    arch = _v3_speculator(draft_arch).derive()
    assert arch == replace(
        mimo_v2_modulev3_arch,
        name="UnifiedDflashMiMoV2ForCausalLM_ModuleV3",
        pipeline_model=UnifiedDflashMiMoV2ModuleV3Model,
        config=UnifiedDflashMiMoV2Config,
        memory_planner=UnifiedDflashMiMoV2ModuleV3MemoryPlanner,
        batching=UnifiedDflashMiMoV2ModuleV3BatchProcessor,
        checkpoint_draft_width=mimo_dflash_draft_width,
        supports_spec_decode_mixed_batches=True,
    )
    # The fused architecture is also a registry entry under its own name --
    # main's convention for a nameable fused graph, which gen_mef's
    # per-row-sampling claim and the engine's block-spec binding key by
    # name. The speculator stays the selection mechanism; the registered
    # arch and the derivation are the same fields.
    registered = PIPELINE_REGISTRY.retrieve_architecture(
        "UnifiedDflashMiMoV2ForCausalLM_ModuleV3"
    )
    assert registered is not None
    assert registered.name == "UnifiedDflashMiMoV2ForCausalLM_ModuleV3"


@pytest.mark.parametrize("draft_arch", ["DFlashMiMoV2DraftModel"])
def test_without_the_flag_the_generic_pairing_runs(draft_arch: str) -> None:
    """The unflagged path is the generic pairing: main composes the
    graph-API target and draft through ``SpecDecodePipelineModel`` under the
    target's own name, never the ModuleV3 speculator."""
    speculator = select_speculator("MiMoV2ForCausalLM", "dflash", draft_arch)
    assert speculator is None
    arch = _selected(SpeculativeConfig(speculative_method="dflash"), draft_arch)
    assert arch is not None
    assert arch.name == "MiMoV2ForCausalLM+dflash"
    assert arch.pipeline_model is SpecDecodePipelineModel


@pytest.mark.parametrize("draft_arch", ["DFlashMiMoV2DraftModel"])
def test_the_derived_arch_keeps_multi_gpu_support(draft_arch: str) -> None:
    """The fused graph runs the base's tensor-parallel layout, so the derived
    architecture must not lose ``multi_gpu_supported`` -- a config with
    ``device_count > 1`` is refused without it."""
    arch = _v3_speculator(draft_arch).derive()
    assert arch.multi_gpu_supported is True


def _selected(
    speculative: SpeculativeConfig | None,
    draft_arch: str = "DFlashMiMoV2DraftModel",
    *,
    prefer_module_v3: bool = False,
) -> SupportedArchitecture | None:
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
    spec_arch = _apply_speculative_target_architecture(
        speculative, models, prefer_module_v3=prefer_module_v3
    )
    assert models["main"].huggingface_config.architectures == [
        "MiMoV2ForCausalLM"
    ]
    return spec_arch


@pytest.mark.parametrize("draft_arch", ["DFlashMiMoV2DraftModel"])
def test_dflash_selects_the_modulev3_fused_graph(draft_arch: str) -> None:
    """The config rewrite picks the ModuleV3 speculator off the plain
    checkpoint name when the flag is set: the target's own architecture
    entry stays untouched, as the speculator applies to whatever the
    checkpoint resolved to."""
    speculator = _selected(
        SpeculativeConfig(speculative_method="dflash"),
        draft_arch,
        prefer_module_v3=True,
    )
    assert speculator is not None
    assert speculator.name == "UnifiedDflashMiMoV2ForCausalLM_ModuleV3"


@pytest.mark.parametrize("draft_arch", ["DFlashMiMoV2DraftModel"])
def test_the_unflagged_rewrite_composes_the_generic_pairing(
    draft_arch: str,
) -> None:
    arch = _selected(SpeculativeConfig(speculative_method="dflash"), draft_arch)
    assert arch is not None
    assert arch.name == "MiMoV2ForCausalLM+dflash"
    assert arch.pipeline_model is SpecDecodePipelineModel


def test_no_speculation_keeps_the_base_graph() -> None:
    assert _selected(None, prefer_module_v3=True) is None


def test_other_methods_do_not_reach_the_modulev3_speculator() -> None:
    """The flag's selection keeps to the speculator's declared method:
    dflash2 falls to main's generic pairing under the target's own name,
    never this package's fused graph."""
    arch = _selected(
        SpeculativeConfig(speculative_method="dflash2"),
        prefer_module_v3=True,
    )
    assert arch is not None
    assert arch.name == "MiMoV2ForCausalLM+dflash2"


def test_the_fused_graph_stages_under_the_graph_api_name() -> None:
    """The ModuleV3 fused graph must stage under the V2 build's name: Mach's
    block-spec executor binds the graph by that constant, so a rename here
    strands every deployed MEF. The V2 lane's own graph is the other half of
    the contract -- both tests must name the same string."""
    assert GRAPH_NAME == "unified_dflash_mimo_v2"


def test_the_speculators_target_the_modulev3_base() -> None:
    """The speculators derive off the ModuleV3 base architecture, so the
    fused arch inherits the ModuleV3 model, adapter and batch-processor ABI
    -- not the graph-API ones."""
    for draft_arch in ("DFlashMiMoV2DraftModel",):
        speculator = _v3_speculator(draft_arch)
        base: SupportedArchitecture = speculator.base
        assert base.name == "MiMoV2ForCausalLM_ModuleV3"
        assert speculator.method == "dflash"
        assert speculator.name == "UnifiedDflashMiMoV2ForCausalLM_ModuleV3"
