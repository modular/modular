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
"""Graph construction tests for the ``spec.top_token_pieces`` custom op."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.mlir import DictAttr, IntegerAttr
from max.nn._graph_ops import top_token_pieces
from top_token_helpers import pieces_graph

from .norm_test_utils import assert_single_op


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
@pytest.mark.parametrize("rows", ["rows", 2])
def test_top_token_pieces_output_shape(dtype: DType, rows: str | int) -> None:
    """The op preserves the row dimension and vocabulary offset."""
    device = DeviceRef.CPU()
    start = 101
    graph = pieces_graph(TensorType(dtype, [rows, "vocab"], device), start)
    op = assert_single_op(graph, "mo.custom", "spec.top_token_pieces")
    parameters = DictAttr(op.attributes["parameters"])
    assert IntegerAttr(parameters["start"]).value == start
    assert graph.output_types == [TensorType(DType.float32, [rows, 3], device)]


def test_top_token_pieces_rejects_bad_rank() -> None:
    """Rank-3 logits are rejected at wrapper time."""
    with Graph(
        "bad_rank",
        input_types=[
            TensorType(DType.float32, [2, 3, 4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        with pytest.raises(ValueError, match="rank-2"):
            top_token_pieces(graph.inputs[0].tensor, 0)
        graph.output(graph.inputs[0].tensor)
