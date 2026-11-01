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
"""Checks cut-query gathering and absolute tail cache coordinates on CPU."""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from functools import cache
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, StaticDim, TensorType, ops
from max.nn.legacy.kv_cache import PagedCacheValues
from max.pipelines.architectures.deepseekV41.deepseekV41 import (
    HCStreams,
    _attend,
    _check_replay,
    _gather_streams,
)
from max.pipelines.architectures.deepseekV41.layer_resolver import LayerInfo
from max.pipelines.architectures.deepseekV41.layers.attention import (
    CSA2Attention,
    CSA2Cache,
    PendingCSA2Attention,
    gather_pending_attention,
    tail_cache_view,
)
from max.pipelines.architectures.deepseekV41.layers.ragged import (
    ReplayRows,
    V41RaggedRows,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
)
from numpy.typing import NDArray


def _layer() -> LayerInfo:
    return LayerInfo(
        layer_id=20,
        dspark_stage=None,
        compress_ratio=1,
        attention_mode="Full",
        kv_source_layer_id=20,
        index_source_layer_id=20,
        candidate_role="source",
        side="decoder",
        has_engram=False,
        swa_only=False,
        rope_regime="decoder",
        rope_theta=10000.0,
        rope_yarn_factor=1.0,
        has_indexer=True,
        indexer_topk=2,
    )


@cache
def _graph() -> Model:
    cpu = DeviceRef.CPU()
    with Graph(
        "replay_tail_coordinates",
        input_types=[
            TensorType(DType.int64, ["K"], cpu),
            TensorType(DType.uint32, [3], cpu),
            TensorType(DType.int32, [2], cpu),
            TensorType(DType.uint32, [2], cpu),
            TensorType(DType.uint32, [1], cpu),
            TensorType(DType.uint32, [3], cpu),
            BufferType(DType.uint8, [1, 1, 40, 128, 1, 1], cpu),
        ],
    ) as graph:
        indices, offsets, skip, cache_lengths, longest, original = (
            x.tensor for x in graph.inputs[:6]
        )
        replay = ReplayRows.from_inputs(
            indices, offsets, skip, cache_lengths, 128, longest
        )
        leaf = PagedCacheValues(
            kv_blocks=graph.inputs[6].buffer,
            cache_lengths=cache_lengths,
            lookup_table=ops.constant([[0, 1], [2, 3]], DType.uint32, cpu),
            max_prompt_length=ops.constant(5, DType.uint32, cpu),
            max_cache_length=ops.constant(205, DType.uint32, cpu),
            page_stride=ops.constant([1], DType.int64, cpu),
        )
        cache_values = CSA2Cache(leaf, {20: leaf}, {20: leaf}, {20: 0})
        tail = tail_cache_view(cache_values, replay.rows, replay.longest)
        q = ops.constant(
            np.arange(18, dtype=np.int32).reshape(9, 2), DType.int32, cpu
        )
        state = PendingCSA2Attention(
            q=q,
            out_shape=[StaticDim(1), StaticDim(9), StaticDim(2)],
            freqs_cis=q + 100,
            swa_collection=leaf,
            comp_collection=leaf,
            comp_index=0,
            layer_comp=None,
            index_q=(q + 200, q + 300, q + 400),
            index_leaf=leaf,
        )
        gathered = gather_pending_attention(
            state, indices, replay.rows, tail, _layer()
        )
        assert gathered.index_q is not None and gathered.index_leaf is not None
        stream = HCStreams(
            ops.reshape(q, [1, 9, 1, 2]),
            pre=ops.reshape(q, [1, 9, 2]),
            out=ops.reshape(q + 10, [1, 9, 2]),
            post=ops.reshape(q + 20, [1, 9, 2]),
            comb=ops.reshape(q + 30, [1, 9, 1, 2]),
        )
        stream = _gather_streams(stream, indices)
        assert stream.pre is not None and stream.out is not None
        assert stream.post is not None and stream.comb is not None
        original_rows = V41RaggedRows.from_offsets(
            original,
            9,
            ops.cast(cache_lengths, DType.int32),
        )
        attn = object.__new__(CSA2Attention)
        attn.layer_info = _layer()
        with (
            patch.object(CSA2Attention, "begin", return_value=state) as begin,
            patch.object(CSA2Attention, "end", return_value=gathered.q) as end,
        ):
            result = _attend(
                attn,
                ops.reshape(q, [1, 9, 2]),
                original_rows,
                cache_values,
                None,
                replay=replay,
            )
            assert begin.call_args.args[0].shape[1] == 9
            assert end.call_args.args[0].out_shape[1] == indices.shape[0]
            assert end.call_args.kwargs["swa_valid_from"] is None
        with (
            patch.object(
                CSA2Attention, "begin", return_value=gathered
            ) as begin,
            patch.object(CSA2Attention, "end", return_value=gathered.q) as end,
        ):
            _attend(
                attn,
                ops.reshape(gathered.q, [1, indices.shape[0], 2]),
                replay.rows,
                tail,
                None,
                swa_valid_from=replay.swa_valid_from,
            )
            assert begin.call_args.args[0].shape[1] == indices.shape[0]
            assert (
                end.call_args.kwargs["swa_valid_from"] is replay.swa_valid_from
            )
        graph.output(
            replay.rows.positions,
            replay.rows.bid,
            replay.rows.local,
            gathered.q,
            gathered.freqs_cis,
            *gathered.index_q,
            tail.swa_collection.cache_lengths,
            gathered.index_leaf.cache_lengths,
            tail.swa_collection.max_prompt_length,
            tail.swa_collection.max_cache_length,
            stream.residual,
            stream.pre,
            stream.out,
            stream.post,
            stream.comb,
            tail.swa_collection.lookup_table,
            result,
            replay.swa_valid_from,
        )
    return InferenceSession(devices=[CPU()]).load(graph)


@pytest.mark.parametrize(
    (
        "kept",
        "offsets",
        "cache_lengths",
        "skip",
        "starts",
        "floors",
        "positions",
        "bids",
        "local",
        "longest",
        "original",
    ),
    [
        (
            [2, 3, 6, 7, 8],
            [0, 2, 5],
            [100, 200],
            [2, 2],
            [102, 202],
            [102, 202],
            [102, 103, 202, 203, 204],
            [0, 0, 1, 1, 1],
            [0, 1, 0, 1, 2],
            3,
            [0, 4, 9],
        ),
        (
            [3, 8],
            [0, 1, 2],
            [100, 200],
            [3, 4],
            [103, 204],
            [103, 204],
            [103, 204],
            [0, 1],
            [0, 0],
            1,
            [0, 4, 9],
        ),
        (
            [7, 8],
            [0, 0, 2],
            [100, 200],
            [4, 3],
            [104, 203],
            [104, 203],
            [203, 204],
            [1, 1],
            [0, 1],
            2,
            [0, 4, 9],
        ),
        # A prompt that drops six of eight rows beside a decode row kept whole.
        (
            [6, 7, 8],
            [0, 2, 3],
            [100, 200],
            [6, 0],
            [106, 200],
            [106, 72],
            [106, 107, 200],
            [0, 0, 1],
            [0, 1, 0],
            2,
            [0, 8, 9],
        ),
        (
            [0, 7, 8],
            [0, 1, 3],
            [200, 100],
            [0, 6],
            [200, 106],
            [72, 106],
            [200, 106, 107],
            [0, 1, 1],
            [0, 0, 1],
            2,
            [0, 1, 9],
        ),
        # A row whose device length ran ahead of its host length by accepted
        # drafts is kept whole and starts at the device position.
        (
            [0, 1, 2, 3, 4],
            [0, 4, 5],
            [203, 100],
            [0, 0],
            [203, 100],
            [75, 0],
            [203, 204, 205, 206, 100],
            [0, 0, 0, 0, 1],
            [0, 1, 2, 3, 0],
            4,
            [0, 4, 9],
        ),
    ],
)
def test_query_gather_preserves_positions_and_all_pending_state(
    kept: list[int],
    offsets: list[int],
    cache_lengths: list[int],
    skip: list[int],
    starts: list[int],
    floors: list[int],
    positions: list[int],
    bids: list[int],
    local: list[int],
    longest: int,
    original: list[int],
) -> None:
    indices = np.array(kept, dtype=np.int64)
    inputs: list[NDArray[np.generic]] = [
        indices,
        np.array(offsets, dtype=np.uint32),
        np.array(skip, dtype=np.int32),
        np.array(cache_lengths, dtype=np.uint32),
        np.array([longest], dtype=np.uint32),
        np.array(original, dtype=np.uint32),
        np.zeros((1, 1, 40, 128, 1, 1), dtype=np.uint8),
    ]
    outputs = []
    for value in _graph().execute(*[Buffer.from_numpy(x) for x in inputs]):
        assert isinstance(value, Buffer)
        outputs.append(value.to_numpy())
    np.testing.assert_array_equal(outputs[0], positions)
    np.testing.assert_array_equal(outputs[1], bids)
    np.testing.assert_array_equal(outputs[2], local)
    expected = np.arange(18).reshape(9, 2)[indices]
    for actual, delta in zip(
        outputs[3:8], (0, 100, 200, 300, 400), strict=True
    ):
        np.testing.assert_array_equal(actual, expected + delta)
    np.testing.assert_array_equal(outputs[8], starts)
    np.testing.assert_array_equal(outputs[9], starts)
    assert outputs[10] == longest
    assert outputs[11] == 205
    for actual, delta in zip(outputs[12:17], (0, 0, 10, 20, 30), strict=True):
        np.testing.assert_array_equal(
            actual.reshape(len(kept), 2), expected + delta
        )
    np.testing.assert_array_equal(outputs[17], [[0, 1], [2, 3]])
    np.testing.assert_array_equal(outputs[18], expected)
    np.testing.assert_array_equal(outputs[19], floors)


def _trunk(engram: list[int], cut: int = 20) -> list[LayerInfo]:
    return [
        dataclass_replace(
            _layer(), layer_id=layer_id, has_engram=layer_id in engram
        )
        for layer_id in range(cut + 3)
    ]


_CONFIG = cast(DeepseekV41Config, SimpleNamespace(kv_source_layer_ids=[0, 20]))


def test_replay_accepts_engram_layers_at_or_before_the_cut() -> None:
    _check_replay(_CONFIG, _trunk([1, 14, 20]), [37, 38, 39])


def test_replay_rejects_an_engram_layer_after_the_cut() -> None:
    with pytest.raises(ValueError, match=r"Engram layers \[21\] come after"):
        _check_replay(_CONFIG, _trunk([1, 14, 21]), [37, 38, 39])


def test_replay_rejects_a_tap_before_the_cut() -> None:
    with pytest.raises(ValueError, match="taps must be at or after"):
        _check_replay(_CONFIG, _trunk([1, 14]), [19, 38])


def test_replay_rejects_a_cut_outside_the_trunk() -> None:
    with pytest.raises(ValueError, match="not a trunk layer"):
        _check_replay(_CONFIG, _trunk([1, 14], cut=10), [37])
