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
from os import PathLike
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from max import _graph, mlir
from max.mlir.dialects import mo

from .dtype import DType
from .graph_value import GraphValue
from .type import ShapeLike, SymbolicDim, TensorType, Type
from .weight import Weight

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
    from dataclasses import dataclass
    import numpy as np
    from max.graph import DType, Graph, GraphValue, TensorType

    @dataclass
    class Linear:
      weight: np.ndarray
      bias: np.ndarray

      def __call__(self, x: GraphValue) -> GraphValue:
          return x @ self.weight + self.bias

    linear_graph = Graph(
        "linear",
        Linear(np.ones((2, 2)), np.ones((2,))),
        input_types=[TensorType(DType.float32, (2,))],
    )
    ```

    You can't call a `Graph` directly from Python. You must compile it and
    execute it with MAX Engine. For more detail, see the tutorial about how to
    [build a graph with MAX Graph](/max/graph/get-started).
    """

    _mlir_op: mlir.Operation
    _context: mlir.Context
    _module: mlir.Module
    inputs: tuple[GraphValue, ...]
    weights: dict[str, Weight]

    def __init__(
        self,
        name: str,
        forward: Optional[Callable] = None,
        input_types: Iterable[Type] = (),
        output_types: Iterable[Type] = (),
        *args,
        **kwargs,
    ) -> None:
        self.name = name
        self._input_types = list(input_types)
        self._output_types = list(output_types)
        self._params = {
            dim.name
            for t in input_types
            if isinstance(t, TensorType)
            for dim in t.shape
            if isinstance(dim, SymbolicDim)
        }

        registry = mlir.DialectRegistry()
        _graph.load_modular_dialects(registry)

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
                self._mlir_op = _graph.graph(
                    self._module, loc, name, function_type
                )

        self.inputs = tuple(GraphValue(arg) for arg in self._body.arguments)
        self.weights = {}

        if forward is not None:
            # If the forward method was passed stage the graph directly in the
            # constructor.
            with self:
                result = forward(*self.inputs, *args, **kwargs)
                self.output(result)

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
        try:
            current = CURRENT_GRAPH.get()
        except LookupError:
            raise ValueError("No graph found.")
        assert current
        return current

    @property
    def _body(self) -> mlir.Block:
        return self._mlir_op.regions[0].blocks[0]

    def _add_op(self, op, *args, **kwargs) -> list[GraphValue]:
        def unwrap(arg):
            if isinstance(arg, GraphValue):
                return arg._mlir_value
            if isinstance(arg, list):
                return [unwrap(elem) for elem in arg]
            else:
                return arg

        args = [unwrap(arg) for arg in args]
        kwargs = {k: unwrap(arg) for k, arg in kwargs.items()}

        diagnostics = []

        def handler(d):
            diagnostics.append(str(d))
            return True

        with mlir.InsertionPoint(self._body), location():
            # Temporarily hookup a handler to record diagnostics from the mlir op builder.
            # These are used to generate a better error message on failure.
            handle = self._context.attach_diagnostic_handler(handler)
            try:
                results = op(*args, **kwargs)
            except Exception as e:
                diags = "\n\t".join(diagnostics)
                raise ValueError(f"Mlir Diagnostics:\n\t{diags}\n") from e
            finally:
                handle.detach()

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

    def __repr__(self) -> str:
        return (
            f"Graph(name='{self.name}', input_types={self._input_types},"
            f" output_types={self._output_types})"
        )

    def add_weight(
        self,
        name: str,
        dtype: Optional[DType] = None,
        shape: Optional[ShapeLike] = None,
        filepath: Union[PathLike, str, None] = None,
        offset: Optional[int] = None,
    ):
        """Initializes a new weight in the current graph.

        Args:
            name: The name of this weight. All weights in a graph must have
              unique names.
            dtype: The DType of the weight. Defaults to Float32.
            shape: The shape of the weight. Defaults to a scalar (`shape=[1]`).
            filepath: File pointing to file containing weight value.
            offset: Offset to weight in the file (defaults to 0).

        Returns:
            The added `Weight` object.

        Raises:
            ValueError if a weight with the same name already exists in the
            graph.
        """
        if name in self.weights:
            raise ValueError(f"Weight '{name}' already exists in Graph {self}")
        shape = [1] if shape is None else shape
        tensor_type = TensorType(dtype or DType.float32, shape)

        # TODO: Allow file path to be set later.
        if filepath is None:
            raise ValueError("Filepath must be defined.")
        weights_attr = _graph.weights_attr(
            Path(filepath or ""),
            offset or 0,
            tensor_type.to_mlir(),
            name,
        )
        weights_tensor = Graph.current._add_op(
            mo.constant, result=tensor_type.to_mlir(), value=weights_attr
        )[0]
        weight = Weight(
            weights_tensor,
            name=name,
            filepath=filepath,
            offset=offset,
        )
        self.weights[name] = weight

        return weight
