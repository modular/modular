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
"""Tests the data-parallel vision merge of one replica's requests.

Every device holds the whole batch's image rows and merges only those of its
own requests (``VisionMerge.for_rows``). Image rows of other requests must
leave every local token row unchanged; the last row is the one a wrapped
negative index lands on.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.deepseekV41.deepseekV41 import (
    VisionMerge,
    merge_vision,
)

D = 8
# Three requests of 4, 5 and 3 token rows, each with image rows.
OFFSETS = np.array([0, 4, 9, 12], dtype=np.uint32)
IMAGE_ROWS = np.array([1, 2, 6, 7, 10], dtype=np.int32)


@cache
def merge_graph() -> Any:
    cpu = DeviceRef.CPU()
    with Graph(
        "vision_merge_rows",
        input_types=[
            TensorType(DType.float32, [1, "s", D], cpu),
            TensorType(DType.float32, ["n", D], cpu),
            TensorType(DType.int32, ["n"], cpu),
            TensorType(DType.uint32, ["b"], cpu),
            TensorType(DType.int64, [2], cpu),
        ],
    ) as graph:
        x, embeddings, indices, offsets, bounds = (
            v.tensor for v in graph.inputs
        )
        vision = VisionMerge(embeddings, indices).for_rows(
            offsets, bounds[0], bounds[1]
        )
        graph.output(merge_vision(x, vision))
    return InferenceSession(devices=[CPU()]).load(graph)


@pytest.mark.parametrize(
    "first, past",
    [(0, 1), (1, 2), (2, 3), (1, 3), (0, 3)],
)
def test_merges_only_own_image_rows(first: int, past: int) -> None:
    start, stop = int(OFFSETS[first]), int(OFFSETS[past])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, stop - start, D)).astype(np.float32)
    embeddings = rng.standard_normal((len(IMAGE_ROWS), D)).astype(np.float32)

    expected = x[0].copy()
    for row, embedding in zip(IMAGE_ROWS, embeddings, strict=True):
        if start <= row < stop:
            expected[row - start] = embedding

    (out,) = merge_graph().execute(
        *(
            Buffer.from_numpy(a)
            for a in (
                x,
                embeddings,
                IMAGE_ROWS,
                OFFSETS,
                np.array([first, past], dtype=np.int64),
            )
        )
    )
    actual: npt.NDArray[np.float32] = out.to_numpy()[0]
    for row in range(stop - start):
        np.testing.assert_array_equal(
            actual[row], expected[row], err_msg=f"local row {row}"
        )
