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


"""The fused DFlash2 graph's signature.

Mach's spec-step executor binds this signature positionally, so a reordering
or an added slot is an ABI break that the engine can only report as an arity
mismatch (or, worse, cannot report at all when the count happens to match).
The graph is built the way a served pairing builds it, over a tiny Qwen3.5
target and DFlash2 draft, and its inputs are read back by position.

The KV group is eight slots here: ``page_stride`` and, last, the
mixed-step split record.
"""

from __future__ import annotations

import re
from typing import cast

import pytest
from dflash2_pairing_harness import (
    FULL,
    LINEAR,
    FusedGraph,
    Geometry,
    fused_graph,
    state_regions,
)
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.architectures.qwen3_5.spec_adapters import state_tail_types
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _Inputs,
    _without_tail,
)

# Three linear-attention layers and one full-attention layer: the pool tail is
# per linear layer, so a mix is what makes the count formula falsifiable.
GEOMETRY = Geometry(
    layer_types=(LINEAR,) * 3 + (FULL,),
    draft_layers=5,
    dtype=DType.float32,
)


@pytest.fixture(scope="module")
def fused() -> FusedGraph:
    return fused_graph(GEOMETRY)


@pytest.fixture(scope="module")
def types(fused: FusedGraph) -> list[TensorType | BufferType]:
    declared = [value.type for value in fused.graph.inputs]
    types = [t for t in declared if isinstance(t, (TensorType, BufferType))]
    assert len(types) == len(declared)
    return types


def test_slot_count_matches_the_declared_formula(
    types: list[TensorType | BufferType],
) -> None:
    """``15 + 25D`` at ``D = 1``.

    Five ragged/host inputs, signals, two 8-slot KV leaves,
    batch_context_lengths, the eight-entry sampling tail, the bitmask triple,
    then a pool and a row table for each of the three state leaves, the ring
    included. The layer count does not enter.
    """
    assert len(types) == 15 + 25 * 1


def test_the_prefix_and_sampling_tail_are_the_canonical_ones(
    types: list[TensorType | BufferType],
) -> None:
    prefix = [(t.dtype, tuple(str(d) for d in t.shape)) for t in types[:5]]
    assert prefix == [
        (DType.int64, ("total_seq_len",)),
        (DType.uint32, ("input_row_offsets_len",)),
        (DType.uint32, ("input_row_offsets_len",)),
        (DType.int64, ("return_n_logits",)),
        (DType.int64, ("2",)),
    ]
    # host_input_row_offsets and return_n_logits are host-side.
    cpu = DeviceRef.CPU()
    assert types[2].device == cpu and types[3].device == cpu

    # draft_tokens .. in_thinking_phase, then the bitmask triple.
    tail = [(t.dtype, tuple(str(d) for d in t.shape)) for t in types[23:34]]
    assert tail == [
        (DType.int64, ("batch_size", "num_steps")),
        (DType.uint64, ("batch_size",)),
        (DType.float32, ("batch_size",)),
        (DType.int64, ("batch_size",)),
        (DType.int64, ()),
        (DType.float32, ("batch_size",)),
        (DType.float32, ()),
        (DType.bool, ("batch_size",)),
        (
            DType.int32,
            ("batch_size", "num_bitmask_positions", "packed_vocab_size"),
        ),
        (DType.int64, ("2",)),
        (
            DType.int32,
            ("batch_size", "num_bitmask_positions", "packed_vocab_size"),
        ),
    ]


def test_the_draft_leaf_follows_the_target_leaf(
    types: list[TensorType | BufferType],
) -> None:
    """Slot 14: the draft leaf's blocks, right after the target leaf's eight.

    Five drafter layers at the drafter's own head geometry. The leaf's
    derivation is checked in ``test_pairing_cache``; this checks the graph
    declares it where Mach's registry looks for it.
    """
    # Each leaf ends with its split record, the target's just before.
    for record in (types[13], types[21]):
        assert record == TensorType(DType.uint32, [7], device=DeviceRef.CPU())
    blocks = types[14]
    assert isinstance(blocks, BufferType)
    assert blocks.dtype == GEOMETRY.dtype
    assert [str(d) for d in blocks.shape[1:]] == [
        "2",
        str(GEOMETRY.draft_layers),
        str(GEOMETRY.page_size),
        str(GEOMETRY.draft_kv_heads),
        str(GEOMETRY.draft_head_dim),
    ]


def test_the_state_pool_tail_is_the_one_the_qwen3_5_graphs_declare(
    fused: FusedGraph, types: list[TensorType | BufferType]
) -> None:
    """Checks the state tail is declared slot for slot as every speculative
    Qwen3.5 graph declares it, so one engine-side layout binds them all."""
    regions = state_regions(fused.target)
    expected = state_tail_types(regions, [GEOMETRY.device])
    tail = types[len(types) - len(expected) :]
    assert len(tail) == 2 * len(regions)
    for got, want in zip(tail, expected, strict=True):
        assert got.dtype == want.dtype
        assert [str(d) for d in got.shape] == [str(d) for d in want.shape]


def test_the_rollback_replays_the_conv_windows_per_layer(
    fused: FusedGraph,
) -> None:
    """Checks Qwen3.5's rollback still runs one conv replay per linear layer.

    GLM-5.3-Flash's KDA fold places its conv windows inside the state fold's
    launch; Qwen3.5's Gated DeltaNet keeps the per-layer path, so its graph
    holds a verify launch and a replay launch per linear layer and one fold.
    """
    linear = GEOMETRY.layer_types.count(LINEAR)
    assert _symbol_count(fused.graph, "gated_delta_conv1d_verify_fwd") == (
        2 * linear
    )
    assert _symbol_count(fused.graph, "gated_delta_state_fold") == 1


def _symbol_count(graph: Graph, symbol: str) -> int:
    return len(re.findall(rf'symbol\s*=\s*"{re.escape(symbol)}"', str(graph)))


def test_structured_output_off_is_refused() -> None:
    """The graph can't be built without the constrained-decoding bitmask.

    The checkpoint's lm_head has live padding rows, which the in-graph
    acceptance sampler excludes only through that mask.
    """
    with pytest.raises(ValueError, match="constrained-decoding bitmask"):
        fused_graph(GEOMETRY, structured_output=False)


def test_the_shared_batch_fills_every_slot_the_graph_declares(
    fused: FusedGraph,
) -> None:
    """Checks one batch packs one buffer per slot of the graph.

    MAX batches this graph with the unified processor, which declares no
    M-RoPE positions here, so the packed buffers and the signature have to
    agree on length alone.
    """
    view = _without_tail(fused.kv_tree, "state")

    def dummy() -> Buffer:
        return Buffer.zeros(shape=[1], dtype=DType.int64)

    inputs = _Inputs(
        tokens=dummy(),
        input_row_offsets=dummy(),
        return_n_logits=dummy(),
        host_input_row_offsets=dummy(),
        data_parallel_splits=dummy(),
        signal_buffers=[dummy()],
        batch_context_lengths=[dummy()],
        # Only the leaf count matters.
        kv_cache_inputs=cast(
            "KVCacheInputs[Buffer, Buffer]",
            {"kv": [dummy() for _ in view.flattened_kv_inputs()]},
        ),
        trailing=[dummy() for _ in range(2 * len(state_regions(fused.target)))],
        draft_tokens=dummy(),
        seed=dummy(),
        temperature=dummy(),
        top_k=dummy(),
        max_k=dummy(),
        top_p=dummy(),
        min_top_p=dummy(),
        in_thinking_phase=dummy(),
        pinned_bitmask=dummy(),
        wait_payload=dummy(),
        device_bitmask_scratch=dummy(),
        structured_output=True,
        spec=fused.input_spec,
    )

    assert len(inputs.buffers) == len(fused.graph.inputs)
