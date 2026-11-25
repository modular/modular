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
"""Graph construction tests for the ``spec.top_token_logit`` custom op."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import top_token_logit
from top_token_helpers import logit_graph

from .norm_test_utils import assert_single_op


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
@pytest.mark.parametrize("rows", ["rows", 0, 2])
def test_top_token_logit_output_shape(dtype: DType, rows: str | int) -> None:
    """The op emits one float32 score per row, symbolic rows included."""
    device = DeviceRef.CPU()
    graph = logit_graph(TensorType(dtype, [rows, "vocab"], device))
    assert_single_op(graph, "mo.custom", "spec.top_token_logit")
    assert graph.output_types == [TensorType(DType.float32, [rows, 1], device)]


def test_top_token_logit_rejects_bad_rank() -> None:
    """Rank-3 logits are rejected at wrapper time."""
    with Graph(
        "bad_rank",
        input_types=[
            TensorType(DType.float32, [2, 3, 4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        with pytest.raises(ValueError, match="rank-2"):
            top_token_logit(graph.inputs[0].tensor)
        graph.output(graph.inputs[0].tensor)
