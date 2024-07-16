# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""

import contextlib
from typing import Iterable

from . import core as _c
from . import mlir
from .graph_value import GraphValue
from .type import Type


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

    _mlir_op: mlir.Operation
    inputs: tuple[GraphValue, ...]

    def __init__(
        self,
        name: str,
        input_types: Iterable[Type] = (),
        output_types: Iterable[Type] = (),
    ) -> None:
        self.name = name
        self._mlir_context = mlir.Context()
        self._context_stack = contextlib.ExitStack()
        self._input_types = list(input_types)
        self._output_types = list(output_types)

        registry = mlir.DialectRegistry()
        _c.load_modular_dialects(registry._CAPIPtr)

    def __enter__(self):
        # XXX: easy location decorator that works with python location info
        self._context_stack.enter_context(self._mlir_context)
        self._context_stack.enter_context(
            mlir.Location.unknown(context=self._mlir_context)
        )

        self._module = mlir.Module.create()
        with mlir.InsertionPoint(self._module.body):
            # Parse an empty graph.
            # - Quick cludge should update this to use a real op builder ASAP
            # - We have a simplified builder in C++, we can either call this or
            #   create a similar wrapper using MLIR python builder registration.
            argstring = ", ".join(
                f"%{i}: {type.to_mlir()}"
                for i, type in enumerate(self._input_types)
            )
            opstring = f"mo.graph @{self.name}({argstring})"
            self._mlir_op = mlir.Operation.parse(opstring)

        self.inputs = tuple(
            GraphValue(self, arg) for arg in self._body.arguments
        )

        return self

    def __exit__(self, *exc):
        self._context_stack.__exit__(*exc)

    @property
    def _body(self) -> mlir.Block:
        return self._mlir_op.regions[0].blocks[0]

    def _add_variadic_result_op(
        self,
        name: str,
        operands: Iterable[GraphValue] = (),
        attrs: dict[str, mlir.Attribute] = {},  # ugh
    ) -> list[GraphValue]:
        # XXX: location info from stack inspection or stack trace
        assert all(o.graph == self for o in operands)

        with mlir.InsertionPoint(self._body):
            op = mlir.Operation.create(
                name=name,
                operands=[o._mlir_value for o in operands],
                attributes=attrs,
                infer_type=True,
            )
        return [GraphValue(self, result) for result in op.results]

    def _add_op(
        self,
        name: str,
        operands: Iterable[GraphValue] = (),
        attrs: dict[str, mlir.Attribute] = {},  # ugh
    ) -> GraphValue:
        return self._add_variadic_result_op(name, operands, attrs)[0]

    def output(self, *outputs: GraphValue):
        self._add_variadic_result_op("mo.output", outputs)

    def __repr__(self) -> str:
        return (
            f"Graph(name='{self.name}', input_types={self._input_types},"
            f" output_types={self._output_types})"
        )
