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
"""Which ragged MHA op ``flash_attention_ragged`` emits for a split record.

The graphs are only built, never run. A record on the KV collection selects
``mo.mha.ragged.paged.mixed_step_split`` for a plain call. Sink, relative-logit
and split-head-dim calls have no split variant, so they keep the unsplit op.
"""

from __future__ import annotations

import dataclasses

import pytest
from max import tree
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.legacy.kernels import flash_attention_ragged
from max.nn.legacy.kv_cache import MHAKVCacheParams

SPLIT_OP = "mo.mha.ragged.paged.mixed_step_split"
NUM_HEADS = 8
HEAD_DIM = 128


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


def _graph_text(
    call: str, *, with_record: bool, mask: MHAMaskVariant | None = None
) -> str:
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
        TensorType(DType.uint32, [7], cpu),
        *([extra_types[call]] if call in extra_types else []),
    ]
    with Graph("mha_mixed_split_op", input_types=input_types) as graph:
        n = len(cache_inputs)
        cache = params.unflatten_kv_inputs(iter(graph.inputs[:n]))[0]
        if with_record:
            cache = dataclasses.replace(
                cache, mixed_step_split=graph.inputs[n + 3].tensor
            )
        optional = graph.inputs[n + 4].tensor if call in extra_types else None
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
    return str(graph)


@pytest.mark.parametrize(
    "mask",
    [MHAMaskVariant.CAUSAL_MASK, MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK],
)
def test_a_plain_call_with_a_record_picks_the_split_op(
    mask: MHAMaskVariant,
) -> None:
    text = _graph_text("plain", with_record=True, mask=mask)
    assert SPLIT_OP in text


@pytest.mark.parametrize(
    "call", ["plain", "sink", "rel_logits", "split_head_dim"]
)
def test_a_call_without_a_record_keeps_the_unsplit_op(call: str) -> None:
    assert SPLIT_OP not in _graph_text(call, with_record=False)


@pytest.mark.parametrize("call", ["sink", "rel_logits", "split_head_dim"])
def test_a_call_without_a_split_variant_ignores_the_record(call: str) -> None:
    text = _graph_text(call, with_record=True)
    assert SPLIT_OP not in text
    assert "mo.mha.ragged.paged" in text, "the unsplit op is emitted"
