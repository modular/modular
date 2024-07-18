# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements ops used when staging a graph.

Although the following modules provide a lot of the ops you want when building
a graph, you can also use functions in
[`Graph`](/max/api/python/graph/graph/Graph) to add constant values,
such as [`constant()`](/max/api/python/graph/graph/Graph#constant),
[`vector()`](/max/api/python/graph/graph/Graph#vector), and
[`scalar()`](/max/api/python/graph/graph/Graph#scalar).

The [`GraphValue`](/max/api/python/graph/graph_value/GraphValue) type (returned
by all ops) also implements various dunder methods to support operations
between GraphValues, such as `+` add, `*` multiply, and `@` matmul, plus
convenience methods such as
[`reshape()`](/max/api/python/graph/graph_value/GraphValue#reshape) and
[`swapaxes()`](/max/api/python/graph/graph_value/GraphValue#swapaxes).
"""
from .elementwise import add
