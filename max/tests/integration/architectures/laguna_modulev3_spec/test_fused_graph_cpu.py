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
"""The fused Laguna + DFlash graph's no-GPU gate.

The pairing resolves through the registry, and the fused module stages on CPU
with a tiny target and drafter, without compiling. What is pinned is what
Mach binds: the graph's name, its inputs and the drafter's weight names.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, cast

import pytest
from laguna_dflash_fixtures import (
    SAMPLEABLE,
    TESTDATA,
    build_module,
    graph_name,
)
from laguna_spec_fixtures import build_decoder, fake_weights, target_config
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import _graph_ops as kernels
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.architectures.dflash_laguna_modulev3 import (
    DFlashLagunaDraftConfig,
)
from max.pipelines.architectures.laguna_modulev3.spec_adapters import (
    DFlashLagunaProposer,
    LagunaBlockTarget,
    verify,
)
from max.pipelines.lib.arch_lookup import find_architecture
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _SpecDecodeModuleV3PipelineModel,
    spec_decode_architecture,
)
from max.pipelines.speculative import spec_adapters
from max.tensor import default_device, default_dtype
from max.tensor.compilation import StagedGraph

GRAPH_NAME = "unified_dflash_laguna"
"""The name Mach's block adapter binds Laguna's fused graph by."""


def test_the_pairing_resolves_to_the_fused_modulev3_model() -> None:
    arch = spec_decode_architecture(
        "LagunaForCausalLM",
        "dflash",
        "DFlashLagunaForCausalLM",
        prefer_module_v3=True,
    )
    assert arch is not None
    assert arch.name == "LagunaForCausalLM_ModuleV3+dflash"
    assert arch.pipeline_model is _SpecDecodeModuleV3PipelineModel
    assert arch.spec_decode_graph_names["dflash"] == GRAPH_NAME
    assert arch.supports_spec_decode_mixed_batches
    assert spec_adapters.has_adapter("LagunaForCausalLM_ModuleV3")
    # A served config checks the draft by its checkpoint's own name.
    draft = find_architecture("DFlashLagunaForCausalLM")
    assert draft is not None
    assert draft.name == "DFlashLagunaForCausalLM"
    assert not spec_adapters.relaxed_thinking_acceptance(
        "LagunaForCausalLM_ModuleV3"
    )


@pytest.fixture(scope="module")
def staged() -> tuple[StagedGraph[Any, Any], set[str]]:
    """The fused module staged on one device, and its weight names.

    Built once: the stage is the expensive step.
    """
    module = build_module()
    weights = fake_weights(module)
    with (
        pytest.MonkeyPatch.context() as mp,
        default_dtype(DType.bfloat16),
        default_device(CPU()),
        F.lazy(),
    ):
        # The stage only emits ops, so pin the SM100 paths the GPU build takes.
        mp.setattr(kernels, "_is_sm10x_gpu", lambda: True)
        graph, _ = module.stage(
            *module.driver_call().input_types,
            weights=weights,
            name=graph_name(),
            allow_subgraphs=True,
        )
    return graph, set(weights)


def test_the_graph_stages_under_the_engine_binding_name(
    staged: tuple[StagedGraph[Any, Any], set[str]],
) -> None:
    graph, _ = staged
    assert graph.graph.name == GRAPH_NAME


def test_the_inputs_are_the_block_adapters(
    staged: tuple[StagedGraph[Any, Any], set[str]],
) -> None:
    """No bitmask triple: its host-wait op is GPU-only."""
    graph, _ = staged
    inputs = graph.graph.inputs
    assert not any(
        getattr(value.type, "dtype", None) == DType.bool for value in inputs
    )


def test_the_weights_carry_the_engine_prefixes(
    staged: tuple[StagedGraph[Any, Any], set[str]],
) -> None:
    """Mach binds the target's tensors under ``target.`` and the drafter's
    under ``draft.`` plus their checkpoint names: the two layers and two tap
    norms the tiny drafter keeps, and every other tensor."""
    _, names = staged
    assert {name.split(".", 1)[0] for name in names} == {"target", "draft"}
    assert any(name.startswith("target.language_model.") for name in names)
    header = json.loads((TESTDATA / "laguna_dflash_tensors.json").read_text())
    expected = {
        f"draft.{name}"
        for name in header
        if not (m := re.match(r"(layers|aux_hidden_norms)\.(\d+)\.", name))
        or int(m[2]) < 2
    }
    assert {name for name in names if name.startswith("draft.")} == expected


def test_the_drafter_attends_causally_over_its_window(
    staged: tuple[StagedGraph[Any, Any], set[str]],
) -> None:
    """The drafter was trained causal inside its block over a 512-key window,
    which is what lets a block of ``K + 1`` rows truncate the trained one.
    The tiny target's sliding layer uses a 16-key window, so only the
    drafter's two layers match."""
    graph, _ = staged
    drafter = (
        'parameters = {local_window_size = 512 : index, mask_str = "'
        'sliding_window_causal"}'
    )
    assert str(graph).count(drafter) == 2


def test_the_adapters_refuse_a_target_they_cannot_read() -> None:
    """A tap past the target's last layer, a depth the drafter was not
    trained against, and a tied head."""
    target = build_decoder(target_config())
    with pytest.raises(ValueError, match="from its 4 layers"):
        LagunaBlockTarget(target, [1, 4], sampleable_vocab_size=SAMPLEABLE)
    draft_config = DFlashLagunaDraftConfig(
        module=cast(Any, SimpleNamespace(num_target_layers=5)),
        sampleable_vocab_size=SAMPLEABLE,
        drafter_export=cast(Any, None),
    )
    with pytest.raises(ValueError, match="trained against 5 target layers"):
        verify(target, None, ReturnHiddenStates.SELECTED_LAYERS, draft_config)
    text = target.language_model
    text.lm_head = None
    with pytest.raises(ValueError, match="untied lm_head"):
        DFlashLagunaProposer(
            cast(Any, None), text, sampleable_vocab_size=SAMPLEABLE
        )
