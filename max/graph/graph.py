# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""

import contextlib
from typing import Iterable

import max.graph.core as _c
from max.graph import mlir
from max.graph.graph_value import GraphValue
from max.graph.type import Type


class Graph:
    """Represents a single MAX graph.

    A `Graph` is a callable routine in [MAX Engine](/max/engine). Like
    functions, graphs have a name and signature. Unlike a function, which
    follows an imperative programming model, a `Graph` follows a
    [dataflow](https://en.wikipedia.org/wiki/Dataflow_programming) programming
    model, using lazily-executed, parallel operations instead of sequential
    instructions.

    When you instantiate a graph, you must specify the input shapes
    as one or more [`TensorType`](/max/api/python/graph/type/TensorType) or
    [`ListType`](/max/api/python/graph/type/ListType) values. Then, build a
    sequence of ops and set the graph output with [`output()`](#output). For
    example:

    ```python
    from max.graph import Type, Graph, TensorType, ops
    from max.tensor import Tensor, TensorShape

    def build_model() -> Graph:
        graph = Graph(TensorType(DType.float32, (2, 6)))

        matmul_constant_value = Tensor(TensorShape(6, 1), 0.15)
        matmul_constant = graph.constant(matmul_constant_value)

        matmul = graph[0] @ matmul_constant
        relu = ops.elementwise.relu(matmul)
        softmax = ops.softmax(relu)
        graph.output(softmax)

        return graph
    ```

    You can't call a `Graph` directly from Python. You must compile it and
    execute it with MAX Engine. For more detail, see the tutorial about how to
    [build a graph with MAX Graph](/max/graph/get-started).
    """

    def __init__(
        self,
        name: str,
        input_types: Iterable[Type] = None,
        output_types: Iterable[Type] = None,
    ) -> None:
        self._context = mlir.ir.Context()
        self._location = mlir.ir.Location.unknown(context=self._context)
        self.ctx_stack = contextlib.ExitStack()

        registry = mlir.ir.DialectRegistry()
        _c.load_modular_dialects(registry._CAPIPtr)

    def __enter__(self):
        # XXX: easy location decorator that works with python location info
        self.ctx_stack.enter_context(self._context)
        self.ctx_stack.enter_context(self._location)
        self._module = mlir.ir.Module.create()
        return self

    def __exit__(self, *exc):
        self.ctx_stack.__exit__(*exc)

    def _add_variadic_result_op(
        self,
        name: str,
        operands: Iterable[GraphValue] = (),
        # XXX: dict[str, Attribute]
        attrs: Iterable[mlir.ir.NamedAttribute] = (),
        # result_types: Iterable[Type] = (),
        enable_result_type_inference: bool = True,
    ) -> list[GraphValue]:
        # XXX: location info from stack inspection or stack trace
        assert all(o.graph == self for o in operands)

        op = mlir.ir.Operation(
            name=name,
            operands=[o._mlir_value for o in operands],
            attributes=attrs,
            enable_result_type_inference=enable_result_type_inference,
        )

    @property
    def inputs(self):
        return [42]

    def output(self, *args):
        pass

    def __repr__(self) -> str:
        return (
            f"Graph(name='{self.name}', input_types={self.input_types},"
            f" output_types={self.output_types})"
        )
