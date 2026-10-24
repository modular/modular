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
"""Checks decoder SWA validity against the real absolute-slot index list."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    StaticDim,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.kv_cache import PagedCacheValues
from max.pipelines.architectures.deepseekV41.layer_resolver import LayerInfo
from max.pipelines.architectures.deepseekV41.layers.attention import (
    CandidateBlocks,
    CSA2Attention,
    PendingCSA2Attention,
    _ForwardSelectionBox,
)
from max.pipelines.architectures.deepseekV41.layers.ragged import V41RaggedRows
from max.pipelines.architectures.deepseekV41.layers.reuse import (
    ForwardSelection,
)

LAYERS = 40
LAYER = 21
PAGE = 128
TABLE = np.array([[5, 1, 7, 2], [3, 6, 0, 4]], dtype=np.uint32)


def _attention() -> CSA2Attention:
    attention = object.__new__(CSA2Attention)
    attention.layer_info = LayerInfo(
        layer_id=LAYER,
        dspark_stage=None,
        compress_ratio=1,
        attention_mode="Reuse",
        kv_source_layer_id=20,
        index_source_layer_id=20,
        candidate_role="none",
        side="decoder",
        has_engram=False,
        swa_only=False,
        rope_regime="decoder",
        rope_theta=10000.0,
        rope_yarn_factor=1.0,
        has_indexer=False,
        indexer_topk=None,
    )
    return attention


@cache
def _index_graph(
    with_floor: bool, route: str = "index", mode: str = "Reuse"
) -> Model:
    cpu = DeviceRef.CPU()
    with Graph(
        "swa_floor" if with_floor else "swa_no_floor",
        input_types=[
            TensorType(DType.int32, ["T"], cpu),
            TensorType(DType.int32, ["T"], cpu),
            TensorType(DType.int64, [2], cpu),
            TensorType(DType.int32, ["T", 2], cpu),
            TensorType(DType.uint32, [2, 4], cpu),
            BufferType(DType.uint8, [8, 1, LAYERS, PAGE, 1, 1], cpu),
        ],
    ) as graph:
        positions, bid, floor, selected, table = (
            arg.tensor for arg in graph.inputs[:5]
        )
        zero = ops.constant(0, DType.int32, cpu)
        leaf = PagedCacheValues(
            kv_blocks=graph.inputs[5].buffer,
            cache_lengths=ops.constant([0, 0], DType.uint32, cpu),
            lookup_table=table,
            max_prompt_length=zero,
            max_cache_length=zero,
            page_stride=ops.constant([1], DType.int64, cpu),
        )
        rows = V41RaggedRows(
            offsets=ops.constant([0, 0, 0], DType.uint32, cpu),
            starts=ops.constant([0, 0], DType.int32, cpu),
            bid=bid,
            index=positions,
            local=positions,
            positions=positions,
            total=positions.shape[0],
            device=cpu,
        )
        state = PendingCSA2Attention(
            q=ops.unsqueeze(positions, -1),
            out_shape=[positions.shape[0], StaticDim(640)],
            freqs_cis=positions,
            swa_collection=leaf,
            comp_collection=leaf,
            comp_index=0,
            layer_comp=None,
        )
        attention = _attention()
        attention.layer_info = replace(
            attention.layer_info,
            attention_mode=mode,
            index_source_layer_id=(
                LAYER if mode in ("Full", "Reindex") else 20
            ),
        )
        boundary = floor if with_floor else None
        indices = None if mode == "SWA-only" else selected
        if route == "index":
            result = attention._index_list(state, rows, indices, boundary)
        else:
            attention.selection_box = _ForwardSelectionBox()
            attention._selection_trace = None
            if mode == "Reuse":
                attention.selection_box.current = ForwardSelection[
                    TensorValue, CandidateBlocks
                ](source_layer_id=20, indices=selected)
            # Keep end's real selection and index-list paths, replacing only
            # GPU attention and the output projection with an index-list probe.
            with (
                patch.object(
                    CSA2Attention, "begin", return_value=state
                ) as begin,
                patch.object(
                    CSA2Attention, "_select", return_value=(selected, None)
                ),
                patch.object(
                    CSA2Attention,
                    "_attend",
                    side_effect=lambda state, rows, idx: idx,
                ),
                patch.object(
                    CSA2Attention,
                    "_output_projection",
                    side_effect=lambda out, freqs: out,
                ),
            ):
                if route == "call":
                    result = attention(
                        positions, rows, None, swa_valid_from=boundary
                    )
                    begin.assert_called_once_with(positions, rows, None)
                else:
                    result = attention.end(state, rows, swa_valid_from=boundary)
                    begin.assert_not_called()
        graph.output(result)
    return InferenceSession(devices=[CPU()]).load(graph)


def _run(
    with_floor: bool,
    positions: list[int],
    bids: list[int],
    floors: list[int],
    route: str = "index",
    mode: str = "Reuse",
) -> npt.NDArray[np.int32]:
    selected = np.tile(np.array([[0, -1]], dtype=np.int32), (len(positions), 1))
    outputs = _index_graph(with_floor, route, mode).execute(
        *(
            Buffer.from_numpy(array)
            for array in (
                np.array(positions, dtype=np.int32),
                np.array(bids, dtype=np.int32),
                np.array(floors, dtype=np.int64),
                selected,
                TABLE,
                np.zeros((8, 1, LAYERS, PAGE, 1, 1), dtype=np.uint8),
            )
        )
    )
    assert isinstance(outputs[0], Buffer)
    return outputs[0].to_numpy()


@pytest.mark.parametrize(
    "positions,bids,floors",
    [
        ([130, 130, 0, 3], [1, 0, 1, 0], [128, 129]),
        ([256, 255, 260, 257], [0, 1, 0, 1], [250, 128]),
        ([300, 300], [1, 0], [400, 299]),
        ([0, 1], [0, 1], [-4, -1]),
    ],
)
def test_floor_masks_only_swa_slots(
    positions: list[int], bids: list[int], floors: list[int]
) -> None:
    actual = _run(True, positions, bids, floors)
    unrestricted = _run(False, positions, bids, floors)
    for row, (position, bid) in enumerate(zip(positions, bids, strict=True)):
        for age in range(128):
            entry = position - age
            expected = -1
            if entry >= 0 and entry >= floors[bid]:
                block = int(TABLE[bid, entry // PAGE])
                expected = (block * LAYERS + LAYER) * PAGE + entry % PAGE
            assert actual[row, age] == expected
        assert actual[row, 128] == int(TABLE[bid, 0]) * LAYERS * PAGE
        np.testing.assert_array_equal(actual[row, 129:], -1)
    np.testing.assert_array_equal(actual[:, 128:], unrestricted[:, 128:])


def test_zero_floor_matches_unrestricted_window() -> None:
    positions, bids = [0, 127, 128, 260], [1, 0, 1, 0]
    np.testing.assert_array_equal(
        _run(True, positions, bids, [0, 0]),
        _run(False, positions, bids, [0, 0]),
    )


def test_short_chunk_retains_previous_valid_tail() -> None:
    # The new chunk starts at 256, but the preceding contiguous tail is valid.
    actual = _run(True, [256, 257], [0, 0], [128, 0])
    np.testing.assert_array_equal(
        actual, _run(False, [256, 257], [0, 0], [128, 0])
    )
    assert np.all(actual[:, :128] >= 0)


@pytest.mark.parametrize("route", ["call", "end"])
@pytest.mark.parametrize("mode", ["SWA-only", "Reuse", "Full", "Reindex"])
@pytest.mark.parametrize("with_floor", [False, True])
def test_attention_paths_preserve_floor(
    route: str, mode: str, with_floor: bool
) -> None:
    positions, bids, floors = [257, 130, 260, 0], [1, 0, 1, 0], [128, 250]
    np.testing.assert_array_equal(
        _run(with_floor, positions, bids, floors, route, mode),
        _run(with_floor, positions, bids, floors, "index", mode),
    )
