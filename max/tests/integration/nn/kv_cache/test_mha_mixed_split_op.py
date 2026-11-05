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
"""Which split record ``flash_attention_ragged`` hands to ragged MHA.

The graphs are only built, never run. ``attention.mha.ragged.paged`` takes the KV
leaf's declared record after its own operands, or a zeros constant when the
leaf has none (a draft forward drops it). Relative-logit calls take it too,
though the kernel doesn't split a mask that reads operands. Sink calls go to
an op that does not split, so they take no record.
"""

from __future__ import annotations

import dataclasses

import pytest
from max import mlir, tree
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import flash_attention_ragged
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kv_cache import MHAKVCacheParams
from test_common.mixed_step_split_ops import record_source

MHA_OP = "attention.mha.ragged.paged"
NUM_HEADS = 8
HEAD_DIM = 128
OP_NAMES = {
    "plain": MHA_OP,
    "split_head_dim": MHA_OP,
    "sink": "attention.mha.ragged.paged.sink_weights",
    "rel_logits": MHA_OP,
}


def _params(v_head_dim: int | None = None) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=2,
        head_dim=HEAD_DIM,
        v_head_dim=v_head_dim,
        num_layers=1,
        devices=[DeviceRef.GPU()],
        page_size=128,
    )


def _attention_op(
    call: str, *, with_record: bool, mask: MHAMaskVariant | None = None
) -> mlir.Operation:
    split_head_dims = call == "split_head_dim"
    params = _params(v_head_dim=64 if split_head_dims else None)
    gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
    cache_inputs = tree.leaves(params.get_symbolic_inputs())
    extra_types: dict[str, TensorType] = {
        "sink": TensorType(DType.bfloat16, [NUM_HEADS], gpu),
        "rel_logits": TensorType(DType.bfloat16, ["total", NUM_HEADS, 16], gpu),
    }
    input_types = [
        *cache_inputs,
        TensorType(DType.bfloat16, ["total", NUM_HEADS, HEAD_DIM], gpu),
        TensorType(DType.uint32, ["offsets"], gpu),
        TensorType(DType.uint32, [], cpu),
        *([extra_types[call]] if call in extra_types else []),
    ]
    with Graph("mha_mixed_split_op", input_types=input_types) as graph:
        n = len(cache_inputs)
        cache = params.unflatten_kv_inputs(iter(graph.inputs[:n]))[0]
        if not with_record:
            cache = dataclasses.replace(cache, mixed_step_split=None)
        optional = graph.inputs[n + 3].tensor if call in extra_types else None
        out = flash_attention_ragged(
            params,
            graph.inputs[n].tensor,
            graph.inputs[n + 1].tensor,
            cache,
            graph.inputs[n + 2].tensor,
            mask or MHAMaskVariant.CAUSAL_MASK,
            scale=HEAD_DIM**-0.5,
            sink_weights=optional if call == "sink" else None,
            rel_logits=optional if call == "rel_logits" else None,
        )
        graph.output(out)
    ops = [
        op
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if op.name == "mo.custom"
        and mlir.StringAttr(op.attributes["symbol"]).value.startswith(MHA_OP)
    ]
    assert len(ops) == 1
    assert mlir.StringAttr(ops[0].attributes["symbol"]).value == OP_NAMES[call]
    return ops[0]


@pytest.mark.parametrize(
    "mask",
    [MHAMaskVariant.CAUSAL_MASK, MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK],
)
@pytest.mark.parametrize("call", ["plain", "split_head_dim"])
def test_the_leaf_record_is_the_last_mha_operand(
    call: str, mask: MHAMaskVariant
) -> None:
    op = _attention_op(call, with_record=True, mask=mask)
    assert record_source(op) == "argument"
    last = [v for v in op.operands if "chain" not in str(v.type)][-1]
    assert "[7], ui32" in str(last.type)


@pytest.mark.parametrize("call", ["plain", "split_head_dim", "rel_logits"])
def test_a_leaf_without_a_record_sends_zeros(call: str) -> None:
    op = _attention_op(call, with_record=False)
    assert record_source(op) == "mo.constant"


def test_relative_logits_take_the_leaf_record() -> None:
    op = _attention_op("rel_logits", with_record=True)
    assert record_source(op) == "argument"


@pytest.mark.parametrize("with_record", [False, True])
def test_a_call_that_cannot_split_takes_no_record(with_record: bool) -> None:
    op = _attention_op("sink", with_record=with_record)
    assert record_source(op) is None
