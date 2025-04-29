# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""DEPRECATED:
The Mojo graph and engine APIs are being deprecated. Internally we build graphs
using the Python APIs, and our engineering efforts have been focused on that. As
a result, the Mojo version has not kept pace with new features and language
improvements. These APIs will be open sourced for the community in a future
patch prior to being removed.

The MAX Graph API provides a low-level programming interface for
high-performance inference graphs written in Mojo. It's an API for
graph-building only, and it does not implement support for training.

To get started, you need to instantiate a
[`Graph`](/max/api/mojo/graph/graph/Graph) and specify its input
and output shapes. Then build a sequence of ops, using ops provided in the
[`graph.ops`](/max/api/mojo/graph/ops/) package or using your own
custom ops, and add them to the graph by setting the output op(s) with
[`Graph.output()`](/max/api/mojo/graph/graph/Graph#output).

For example:

```mojo
from max.graph import Graph, TensorType, ops
from max.tensor import Tensor, TensorShape

def build_model() -> Graph:
    var graph = Graph(
        in_types=TensorType(DType.float32, 2, 6),
        out_types=TensorType(DType.float32, 2, 1),
    )

    var matmul_constant_value = Tensor[DType.float32](TensorShape(6, 1), 0.15)
    var matmul_constant = graph.constant(matmul_constant_value)

    var matmul = graph[0] @ matmul_constant
    var relu = ops.elementwise.relu(matmul)
    var softmax = ops.softmax(relu)
    graph.output(softmax)

    return graph
```

You can then load the `Graph` into MAX Engine with
[`InferenceSession.load()`](/max/api/mojo/engine/session/InferenceSession#load).

For more detail, see the tutorial about how to [build a graph with MAX
Graph](/max/tutorials/get-started-with-max-graph).

"""

from .error import error, format_error
from .graph import Graph
from .symbol import Symbol
from .type import Dim, ListType, StaticDim, TensorType, Type, _OpaqueType
