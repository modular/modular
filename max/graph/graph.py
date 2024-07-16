# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""

from __future__ import annotations

import contextlib
import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, Optional

from . import core as _c
from . import mlir
from .graph_value import GraphValue
from .type import Type

CURRENT_GRAPH: ContextVar[Graph] = ContextVar("CURRENT_GRAPH")


def _frame_function_qualname(frame):
    """Get the qualified name of a Python stack frame.

    If not available (Python < 3.11), approximate it instead.
    """
    code = frame.f_code
    try:
        # Available in python >= 3.11
        return code.co_qualname
    except AttributeError:
        module = inspect.getmodule(frame)
        function = code.co_name
        return f"{module.__name__ if module else '<unknown>'}.{function}"


def _frame_location(frame):
    """Create an MLIR location corresponding to a single stack frame.

    An MLIR file location has a filename, a line, and a column.
    - Stack frames and function definitions don't store column info,
        so always set it to 0.
    - Encode the module, function name, and filename into the filename
        as "{qualname}:{filename}".
    """
    qualname = _frame_function_qualname(frame)
    code = frame.f_code
    if not mlir.Context.current:
        raise RuntimeError("Can't create location: No MLIR context active")
    return mlir.Location.file(
        f"{qualname}:{code.co_filename}", code.co_firstlineno, 0
    )


def location():
    """Creates an MLIR Location with the current Python call stack."""
    frame = inspect.currentframe()
    assert frame is not None and frame.f_back is not None
    # don't use this function's frame
    frame = frame.f_back
    location = _frame_location(frame)
    stack = []
    while frame := frame.f_back:
        stack.append(_frame_location(frame))
    return mlir.Location.callsite(location, stack)


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
        self._input_types = list(input_types)
        self._output_types = list(output_types)

        registry = mlir.DialectRegistry()
        _c.load_modular_dialects(registry._CAPIPtr)

        self._context = mlir.Context()
        self._module = mlir.Module.create()
        with self._context, location():
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

    # This is really awkward, I just want this to be the generator :(
    def __enter__(self) -> Graph:
        self._context_state = self._enter()
        return self._context_state.__enter__()

    def __exit__(self, *exc):
        self._context_state.__exit__(*exc)

    @contextlib.contextmanager
    def _enter(self):
        token = CURRENT_GRAPH.set(self)
        try:
            with self._context:
                yield self
        finally:
            CURRENT_GRAPH.reset(token)

    @classmethod
    @property
    def current(cls) -> Graph:
        current = CURRENT_GRAPH.get()
        assert current
        return current

    @property
    def _body(self) -> mlir.Block:
        return self._mlir_op.regions[0].blocks[0]

    def _add_variadic_result_op(
        self,
        name: str,
        operands: Iterable[GraphValue] = (),
        attrs: Optional[dict[str, mlir.Attribute]] = None,
    ) -> list[GraphValue]:
        assert all(o.graph == self for o in operands)

        with mlir.InsertionPoint(self._body), location():
            op = mlir.Operation.create(
                name=name,
                operands=[o._mlir_value for o in operands],
                attributes=attrs or {},
                infer_type=True,
            )
        return [GraphValue(self, result) for result in op.results]

    def _add_op(
        self,
        name: str,
        operands: Iterable[GraphValue] = (),
        attrs: Optional[dict[str, mlir.Attribute]] = None,
    ) -> GraphValue:
        return self._add_variadic_result_op(name, operands, attrs)[0]

    def output(self, *outputs: GraphValue):
        self._add_variadic_result_op("mo.output", outputs)

    def __repr__(self) -> str:
        return (
            f"Graph(name='{self.name}', input_types={self._input_types},"
            f" output_types={self._output_types})"
        )
