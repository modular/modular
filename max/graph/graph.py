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
from typing import Iterable

from . import core as _c
from . import mlir
from .graph_value import GraphValue
from .mlir.dialects import mo
from .type import TensorType, Type

CURRENT_GRAPH: ContextVar[Graph] = ContextVar("CURRENT_GRAPH")


def _frame_function_qualname(frame):
    """Gets the qualified name of a Python stack frame.

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
    """Creates an MLIR location corresponding to a single stack frame.

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
        self._params = {
            dim.name
            for t in input_types
            for dim in t.dims
            if isinstance(t, TensorType) and dim.is_symbolic()
        }

        registry = mlir.DialectRegistry()
        _c.load_modular_dialects(registry._CAPIPtr)

        self._context = mlir.Context()
        self._context.append_dialect_registry(registry)
        self._context.load_all_available_dialects()

        with self._context, location() as loc:
            # Create the top level module op.
            self._module = mlir.Module.create()

            with mlir.InsertionPoint(self._module.body):
                function_type = mlir.FunctionType.get(
                    [t.to_mlir() for t in input_types],
                    [t.to_mlir() for t in output_types],
                )
                # Call the C++ builder to build the MO graph op.
                self._mlir_op = _c.graph(
                    self._module._CAPIPtr,
                    loc._CAPIPtr,
                    name,
                    function_type._CAPIPtr,
                )

        self.inputs = tuple(GraphValue(arg) for arg in self._body.arguments)

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

    def _add_op(self, op, *args, **kwargs) -> list[GraphValue]:
        def unwrap(arg):
            return arg._mlir_value if isinstance(arg, GraphValue) else arg

        args = [unwrap(arg) for arg in args]
        kwargs = {k: unwrap(arg) for k, arg in kwargs.items()}

        with mlir.InsertionPoint(self._body), location():
            results = op(*args, **kwargs)

        if isinstance(results, mlir.Value):
            return [GraphValue(results)]
        elif isinstance(results, mlir.Operation):
            return []
        return [GraphValue(result) for result in results]

    def output(self, *outputs: GraphValue) -> None:
        # mo.output doesn't support infer_type
        self._add_op(mo.output, [o._mlir_value for o in outputs])
        # We have a type mismatch now, these are MLIR types
        self._output_types = [o._mlir_value.type for o in outputs]
        # Need to set some more stuff.
        function_type = mlir.FunctionType.get(
            [t.to_mlir() for t in self._input_types],
            self._output_types,
        )
        signature = mlir.Type.parse(f"!kgen.signature<{function_type}>")
        self._mlir_op.attributes["signature"] = mlir.TypeAttr.get(signature)
        self._mlir_op.attributes["functionType"] = mlir.TypeAttr.get(
            function_type
        )
        params = mlir.Attribute.parse(
            f"#kgen<param.decls[{', '.join(f'{param}: index' for param in self._params)}]>"
        )
        self._mlir_op.attributes["inputParams"] = params

        # Set the result_names metadata on the staged op, which is needed by
        # the engine for execution.
        # Note that result_names here needs to match kMgpModelResultNames.
        output_names = [f'"output{i}"' for i in range(len(self._output_types))]
        self._mlir_op.attributes["result_names"] = mlir.Attribute.parse(
            f"[{', '.join(output_names)}]"
        )

    def build(self, *args, **kwargs) -> Iterable[GraphValue]:
        """Core op staging logic to build the graph."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Graph:
        """Dispatches to the overridden `.build()` method."""
        with self:
            result = self.build(*self.inputs, *args, **kwargs)
            self.output(result)
            return self

    def __repr__(self) -> str:
        return (
            f"Graph(name='{self.name}', input_types={self._input_types},"
            f" output_types={self._output_types})"
        )
