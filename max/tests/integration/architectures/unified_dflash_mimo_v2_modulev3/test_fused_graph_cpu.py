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
"""The fused ModuleV3 MiMo-V2 DFlash graph's no-GPU gate.

The fused module stages on a 1-device CPU mesh with the synthetic
target+draft checkpoint strict-loaded: exercising the Tensor boundary, the
adapters, the verify path, the drafter and the accept without compiling.
What is pinned:

- the staged graph's name and input count, the engine's binding surface;
- the weight-name contract both directions (``Module.compile`` silently
  drops an unmatched entry, so the names are the real gate);
- the trace's ops: the shared MoE subgraph, the W4A8 chain inside it, the
  drafter's paged K/V stores and windowed attention, and the verify path's
  variable-logits range and argmax head;
- the SUBGRAPH COUNT: a refactor extracting more or fewer bodies must
  update this test consciously.

The 2-device input-tree parity check from the reference script needs a
mesh whose devices carry distinct accelerator ids (the trace inserts the
mesh's signal buffers only then); on a CPU mesh the layout equality is
declared, not traced, so the multi-device layout is covered by the V2
suite's GPU lane, not here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.compilation import StagedGraph
from max.experimental.tensor import default_device, default_dtype
from max.graph import DeviceRef
from max.nn import kernels
from max.nn.kv_cache import KVConnectorType
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.model_config import (
    UnifiedDflashMiMoV2Config,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.unified_dflash_mimo_v2 import (
    GRAPH_NAME,
)
from max.pipelines.kv_cache.config import KVCacheConfig, KVConnectorConfig
from max.pipelines.kv_cache.connectors.dkv.connector import (
    _validate_dkv_leaves,
)
from max.pipelines.speculative import SpeculativeConfig
from max.pipelines.weights import HuggingFaceRepo
from mimo_dflash_v3_fixtures import (
    DFLASH_CONFIG,
    _FakeAttrConfig,
    build_module,
    build_spec,
    fused_weights,
)

N_INPUTS_1DEV = 35
"""Inputs at 1 device without the bitmask triple (38 with it): the fused
signature's tokens, offsets, return_n_logits, the self-declared signal
buffer, the 24 KV leaves (8 per group over 3 groups), the draft tokens, the
6 sampling tails. The grammar bitmask is off: its host-wait op is GPU-only."""


def test_construct_kv_params_builds_a_tree_the_dkv_connector_serves(
    tmp_path: Any,
) -> None:
    """The dKV-configured KV tree the real export path builds is servable.

    Calls ``construct_kv_params`` with a dKV-typed connector config (the
    refusal this module used to raise is gone) and hands the built tree to
    the connector's own validation: the target's sliding window (128) beside
    the drafter's context window (1024) is the mixed-width tree, and a leaf
    kind the connector cannot serve (a scratch or recurrent leaf joining
    the tree) fails here rather than at a served boot.
    """
    # The draft repo, on disk: construct_kv_params reads its config.json
    # through repo_file, which resolves a local HuggingFaceRepo directly.
    draft_dir = tmp_path / "dflash"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps(DFLASH_CONFIG))

    devices = [DeviceRef.GPU(0)]
    pipeline_config = cast(
        Any,
        SimpleNamespace(
            draft_model=SimpleNamespace(
                huggingface_weight_repo=HuggingFaceRepo(repo_id=str(draft_dir))
            ),
            model=SimpleNamespace(data_parallel_degree=1),
            speculative=SpeculativeConfig(speculative_method="dflash"),
        ),
    )
    kv_cache_config = KVCacheConfig(
        kv_cache_page_size=128,
        kv_connector_config=KVConnectorConfig(type=KVConnectorType.dkv),
    )

    params = UnifiedDflashMiMoV2Config.construct_kv_params(
        _FakeAttrConfig(),
        pipeline_config,
        devices,
        kv_cache_config,
        DType.bfloat16,
    )
    leaves = {
        region.leaf_id: region.group_id for region in params.leaves().values()
    }
    assert len(leaves) == 3, f"the mixed tree holds three leaves: {leaves}"
    widths = {
        group.window_size
        for group in leaves.values()
        if group.is_sliding_window()
    }
    assert widths == {128, 1024}, (
        "the target's sliding window beside the drafter's context window"
    )
    # The connector's own gate: no refusal, and every leaf is a kind it
    # serves. A restored refusal, or a leaf kind dKV cannot serve, fails
    # here.
    _validate_dkv_leaves(leaves)


@pytest.fixture(scope="module")
def staged() -> tuple[StagedGraph[Any, Any], str]:
    """The fused module staged at 1 device, and its trace text.

    Built once per module: the stage is the expensive step (the trace
    realizes every adapter output), and every check below only reads it.
    """
    one = [DeviceRef.CPU()]
    spec, config, draft, kv_tree = build_spec(one)
    weights = fused_weights(config, draft)
    module = build_module(spec, kv_tree, structured_output=False)
    # The W4A8 kernels have no CPU dispatch; the stage only emits the ops,
    # so pin the SM100 shape paths the GPU build will compile, exactly as
    # the ModuleV3 MiMo CPU harness does.
    kernels._is_sm10x_gpu = lambda: True
    with default_dtype(DType.bfloat16), default_device(CPU()):
        with F.lazy():
            graph, _registry = module.stage(
                *module.input_types(),
                weights=weights,
                name=GRAPH_NAME,
                allow_subgraphs=True,
            )
            text = str(graph)
    return graph, text


def test_the_graph_stages_under_the_engine_binding_name(
    staged: tuple[StagedGraph[Any, Any], str],
) -> None:
    """Mach's block-spec executor binds the graph by the constant
    ``unified_dflash_mimo_v2`` -- the V2 build's name -- so the ModuleV3
    export must stage under it or every deployed MEF is stranded."""
    graph, _ = staged
    assert graph.graph.name == GRAPH_NAME


def test_the_input_count_matches_the_v2_signature(
    staged: tuple[StagedGraph[Any, Any], str],
) -> None:
    """The engine binds the inputs positionally; the count is the
    signature's. At 1 device the trace declares its own signal buffer at
    the engine's slot, so the count holds with the V2 graph's."""
    graph, _ = staged
    assert len(graph.graph.inputs) == N_INPUTS_1DEV


def test_the_weight_name_contract_holds_both_directions() -> None:
    """``Module.compile`` silently drops a weights entry no parameter
    matches and initializes that parameter to its default, so the fused
    module's parameters and the adapted ``target.``/``draft.``-prefixed
    checkpoint must be asserted equal as sets, and the strict load pins
    shape and backing dtype."""
    one = [DeviceRef.CPU()]
    spec, config, draft, kv_tree = build_spec(one)
    module = build_module(spec, kv_tree, structured_output=False)
    weights = fused_weights(config, draft)
    names = {name for name, _ in module.parameters}
    assert names == set(weights), (
        f"module lacks {sorted(set(weights) - names)[:8]}; no parameter "
        f"matches {sorted(names - set(weights))[:8]}"
    )
    with default_dtype(DType.bfloat16), default_device(CPU()):
        with F.lazy():
            module.load_state_dict(weights, strict=True)


def test_the_trace_carries_the_shared_moe_subgraph(
    staged: tuple[StagedGraph[Any, Any], str],
) -> None:
    """The target's 3 MoE layers share one extracted body and call it once
    each; layer 0's dense MLP stays inline. Pin the counts so an extraction
    of another body -- or a fusion folding a call away -- must update this
    test consciously, not slip past it."""
    _, text = staged
    assert text.count("mo.graph @mimo_v2_moe") == 1
    assert text.count("mo.call @mimo_v2_moe") == 3
    # The MoE body is the only shared extraction; the top-level graph makes
    # the other one. Pin the total.
    assert text.count("mo.graph @") == 2


def test_the_trace_carries_the_w4a8_chain(
    staged: tuple[StagedGraph[Any, Any], str],
) -> None:
    """The experts run the MXFP4 chain: quantize, two grouped matmuls and
    the fused SiLU, all inside the shared body; the dense MLPs run the
    W8A8 FP8 chain inline."""
    _, text = staged
    for op, expected in (
        ("mo.moe.create.indices.with.scales.offset", 1),
        ("mo.grouped.quantize.dynamic.block.scaled.with_row_indices", 1),
        # The plain name is a substring of the row-indexed one, so both
        # grouped quantizes count here.
        ("mo.grouped.quantize.dynamic.block.scaled", 2),
        ("mo.composite.grouped_matmul_block_scaled", 2),
        ("ep.fused_silu", 1),
        ("mo.moe.finalize", 1),
        ("mo.quantize_dynamic_scaled_float8", 7),
        ("mo.matmul_dynamic_scaled_fp8", 7),
        ("f8e4m3fn", None),
        ("f8e8m0fnu", None),
    ):
        n = text.count(op)
        assert n > 0, f"missing {op}"
        if expected is not None:
            assert n == expected, f"{op}: expected {expected}, got {n}"


def test_the_trace_carries_the_drafter_kernels(
    staged: tuple[StagedGraph[Any, Any], str],
) -> None:
    """The drafter's 2 layers write context and block K/V through the
    paged stores, rope through the fused QK kernel, and attend through the
    sinked windowed kernel; the target's 4 layers write their own 8
    stores."""
    _, text = staged
    for op, expected in (
        ("mo.kv_cache.store.paged.ragged", 16),
        ("mo.fused_qk_rope.ragged.paged", 8),
        ("mo.mha.ragged.paged.sink_weights", 5),
        ("mo.mha.ragged.paged", 6),
    ):
        n = text.count(op)
        assert n == expected, f"{op}: expected {expected}, got {n}"


def test_the_trace_carries_the_verify_path(
    staged: tuple[StagedGraph[Any, Any], str],
) -> None:
    """The verify path scores the trailing rows of each request through a
    host-counted range, and the drafter's proposals come out of the target
    head's argmax."""
    _, text = staged
    for op in ("mo.range", "arg_max"):
        assert text.count(op) > 0, f"missing {op}"
