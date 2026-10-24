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
"""Tests for the graph groups that name a graph's outputs."""

from __future__ import annotations

from typing import Any

import pytest
from max import tree
from max._artifact.value_types import treedef_from_json, treedef_to_json
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn import graph_groups
from max.nn.graph_groups import (
    BatchContext,
    ConvStates,
    DecoderOutputs,
    EPLBProfile,
    EPPayload,
    GraphGroup,
    HiddenStates,
    Logits,
    MRoPEPositions,
    RaggedBatch,
    SparseIndices,
    TPSignals,
    VisionMerge,
)
from max.nn.transformer import ReturnHiddenStates, logits_outputs


def _values(n: int) -> list[TensorValue]:
    x = TensorType(DType.float32, [2], device=DeviceRef.CPU())
    with Graph("values", input_types=[x] * n) as graph:
        return [value.tensor for value in graph.inputs]


def _examples() -> list[GraphGroup[Any, Any]]:
    a, b = _values(2)
    logits = Logits(next_token_logits=a, logits=b, logit_offsets=a)
    t = TensorType(DType.int64, ["total_seq_len"], device=DeviceRef.CPU())
    buf = BufferType(DType.uint8, [8], device=DeviceRef.CPU())
    return [
        RaggedBatch(tokens=t, row_offsets=t),
        TPSignals(signal=(buf, buf)),
        VisionMerge(image_embeddings=(t,), image_token_indices=(t,)),
        MRoPEPositions(position_ids=t),
        BatchContext(length=(t,)),
        EPPayload(atomic_counters=(t,), send_buf_ptrs=(buf,)),
        EPLBProfile(expert_counts=(buf,)),
        logits,
        HiddenStates(last=(a, b)),
        ConvStates(states=(a,)),
        SparseIndices(rows=(b,)),
        DecoderOutputs(logits=logits, hidden_states=HiddenStates(all=(a,))),
    ]


def test_examples_cover_every_group() -> None:
    names = {type(group).__name__ for group in _examples()}
    assert names == set(graph_groups.__all__) - {"GraphGroup", "LogitsName"}


@pytest.mark.parametrize("index", range(len(_examples())))
def test_every_group_has_a_kind_and_round_trips(index: int) -> None:
    group = _examples()[index]
    assert tree.kind_of(type(group)) == group.kind
    leaves, treedef = tree.flatten(group)
    assert treedef_from_json(treedef_to_json(treedef)) == treedef
    rebuilt = tree.unflatten(treedef, leaves)
    assert type(rebuilt) is type(group)
    assert list(tree.flatten(rebuilt)[1].leaf_paths) == list(treedef.leaf_paths)


def test_a_group_needs_a_kind() -> None:
    with pytest.raises(TypeError, match="kind"):
        GraphGroup(states=1)


def test_logits_keep_graph_order() -> None:
    a, b = _values(2)
    assert list(Logits(logit_offsets=b, next_token_logits=a)) == [
        "next_token_logits",
        "logit_offsets",
    ]


def test_a_subclass_inherits_its_base_kind_without_reregistering() -> None:
    class MoreLogits(Logits):
        pass

    (a,) = _values(1)
    assert tree.node_of_kind("logits") is Logits
    assert MoreLogits(logits=a)["logits"] is a


def test_decoder_outputs_read_their_groups_by_type() -> None:
    a, b = _values(2)
    outputs = DecoderOutputs(
        logits=Logits(logits=a), hidden_states=HiddenStates(all=(b,))
    )
    assert outputs.logits["logits"] is a
    assert outputs.hidden_states.tensors() == (b,)


def test_logits_outputs_flatten_in_the_order_the_tuple_had() -> None:
    """Positional readers of a decoder graph see the same output order."""
    last, all_logits, offsets, h0, h1 = _values(5)
    outputs = logits_outputs(
        last, all_logits, offsets, ReturnHiddenStates.LAST, [h0, h1]
    )
    leaves = tree.leaves(outputs)
    assert len(leaves) == 5
    assert all(
        a is b
        for a, b in zip(
            leaves, [last, all_logits, offsets, h0, h1], strict=True
        )
    )
    assert isinstance(outputs["logits"], Logits)
    assert isinstance(outputs["hidden_states"], HiddenStates)


def test_logits_outputs_omit_absent_logits_and_hidden_states() -> None:
    (last,) = _values(1)
    outputs = logits_outputs(last, None, None, ReturnHiddenStates.NONE, [])
    assert list(outputs) == ["logits"]
    assert list(outputs["logits"]) == ["next_token_logits"]
