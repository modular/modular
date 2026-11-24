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
"""Tests what a graph reads when it is built and what it holds after."""

import gc
import weakref

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def _negate_graph() -> Graph:
    input_type = TensorType(DType.float32, [4], device=DeviceRef.CPU())
    with Graph("negate", input_types=[input_type]) as graph:
        graph.output(ops.negate(graph.inputs[0].tensor))
    return graph


def test_dropped_graph_is_freed_without_gc() -> None:
    """Tests that a graph holds no reference cycle, so dropping it frees it."""
    gc.disable()
    try:
        ref = weakref.ref(_negate_graph())
        assert ref() is None
    finally:
        gc.enable()


def test_dropped_graph_with_subgraph_is_freed_without_gc() -> None:
    """Tests that a graph and its subgraph hold no reference cycle."""
    input_type = TensorType(DType.float32, [4], device=DeviceRef.CPU())
    gc.disable()
    try:
        with Graph("outer", input_types=[input_type]) as graph:
            with graph.add_subgraph(
                "negate", input_types=[input_type]
            ) as subgraph:
                subgraph.output(ops.negate(subgraph.inputs[0].tensor))
            graph.output(*ops.call(subgraph, graph.inputs[0]))
        graph_ref = weakref.ref(graph)
        subgraph_ref = weakref.ref(subgraph)
        del graph, subgraph
        assert graph_ref() is None
        assert subgraph_ref() is None
    finally:
        gc.enable()
