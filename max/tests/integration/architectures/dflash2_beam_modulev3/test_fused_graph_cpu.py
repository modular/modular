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
"""The fused Beam + DFlash2 graph's no-GPU gate.

The pairing resolves through the registry, and the fused module stages on a
1-device CPU mesh with a tiny target and drafter: the Tensor boundary, the
adapters, the verify with its taps, the drafter and its selector, without
compiling. What is pinned is what Mach binds: the graph's name, its inputs
and the drafter's weight names, and the kernels the trace runs.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from beam_dflash2_fixtures import (
    TESTDATA,
    build_module,
    build_parts,
    fake_weights,
    graph_name,
)
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import _graph_ops as kernels
from max.pipelines.architectures.dflash2_beam_modulev3.model_config import (
    dflash2_beam_draft_width,
)
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _SpecDecodeModuleV3PipelineModel,
    spec_decode_architecture,
)
from max.pipelines.speculative import spec_adapters
from max.tensor import default_device, default_dtype
from max.tensor.compilation import StagedGraph

GRAPH_NAME = "unified_dflash2_beam"
"""The name Mach's block adapter binds Beam's fused graph by."""

N_INPUTS_1DEV = 38
"""The tokens, offsets and logit count, the self-declared signal buffer, 9
inputs for each of the three KV groups (target sliding, target full, draft),
the draft tokens and the 6 sampling inputs. No ``in_thinking_phase``, which
Mach's block adapter does not bind, and no bitmask triple: its host-wait op
is GPU-only."""


def test_the_pairing_resolves_to_the_fused_modulev3_model() -> None:
    arch = spec_decode_architecture(
        "BeamForCausalLM",
        "dflash2",
        "DFlash2BeamDraftModel",
        prefer_module_v3=True,
    )
    assert arch is not None
    assert arch.name == "BeamForCausalLM_ModuleV3+dflash2"
    assert arch.pipeline_model is _SpecDecodeModuleV3PipelineModel
    assert arch.spec_decode_graph_names["dflash2"] == GRAPH_NAME
    assert arch.checkpoint_draft_width is dflash2_beam_draft_width
    assert arch.multi_gpu_supported
    assert spec_adapters.has_adapter("BeamForCausalLM_ModuleV3")


@pytest.fixture(scope="module")
def staged() -> tuple[StagedGraph[Any, Any], str, set[str]]:
    """The fused module staged at 1 device, its trace text and weights.

    Built once: the stage is the expensive step.
    """
    config, draft, kv_tree = build_parts([DeviceRef.CPU()])
    module = build_module(config, draft, kv_tree)
    weights = fake_weights(module)
    # The stage only emits ops, so pin the SM100 shape paths the GPU build
    # compiles.
    kernels._is_sm10x_gpu = lambda: True
    with default_dtype(DType.bfloat16), default_device(CPU()):
        with F.lazy():
            graph, _ = module.stage(
                *module.driver_call().input_types,
                weights=weights,
                name=graph_name(),
                allow_subgraphs=True,
            )
            text = str(graph)
    return graph, text, set(weights)


def test_the_graph_stages_under_the_engine_binding_name(
    staged: tuple[StagedGraph[Any, Any], str, set[str]],
) -> None:
    graph, _, _ = staged
    assert graph.graph.name == GRAPH_NAME


def test_the_inputs_are_the_block_adapters(
    staged: tuple[StagedGraph[Any, Any], str, set[str]],
) -> None:
    graph, _, _ = staged
    inputs = graph.graph.inputs
    assert len(inputs) == N_INPUTS_1DEV
    assert not any(
        getattr(value.type, "dtype", None) == DType.bool for value in inputs
    )


def test_the_drafter_binds_its_checkpoint_names(
    staged: tuple[StagedGraph[Any, Any], str, set[str]],
) -> None:
    """Mach binds the drafter's tensors as ``draft.`` plus their checkpoint
    names, so the fused module must name them so: the two layers the tiny
    drafter keeps, and every tensor outside the layers."""
    _, _, names = staged
    header = json.loads((TESTDATA / "beam_dflash2_tensors.json").read_text())
    expected = {
        f"draft.{name}"
        for name in header
        if not (m := re.match(r"layers\.(\d+)\.", name)) or int(m[1]) < 2
    }
    assert {name for name in names if name.startswith("draft.")} == expected


def test_the_trace_carries_the_verify_and_the_drafter(
    staged: tuple[StagedGraph[Any, Any], str, set[str]],
) -> None:
    """The target's three layer kinds share their subgraphs; the drafter's
    two layers write context and block K/V through the paged stores, rope
    through the fused QK kernel and attend inline; its head takes one top-k
    and walks seven steps."""
    _, text, _ = staged
    for op, expected in (
        # Three layer kinds, called once per layer, beside the top level.
        ("mo.graph @", 4),
        ("mo.call @", 4),
        # The full layer's K and V, and the drafter's context and block K
        # and V per layer.
        ("kv_cache.store.paged.ragged", 2 + 2 * 2 * 2),
        ("kv_cache.fused_qk_rope.ragged.paged", 2 * 2),
        # One per target subgraph body, one per drafter layer.
        ("attention.mha.ragged.paged", 3 + 2),
        ("mo.top_k", 1),
        # The verify's softcap and the draft head's.
        ("mo.tanh", 2),
    ):
        assert text.count(op) == expected, f"{op}: {text.count(op)}"
