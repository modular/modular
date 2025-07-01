# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import asyncio
import functools
import weakref
from contextvars import ContextVar
from dataclasses import dataclass, field

import max.driver
import max.engine
import max.graph
from max import _core as mlir
from max._core.dialects import builtin, mo
from max.graph import TensorType, TensorValueLike, ops

_SESSION: ContextVar[max.engine.api.InferenceSession] = ContextVar("_SESSION")


def _session() -> max.engine.api.InferenceSession:
    """A single global inference session for compiling and running kernels on tensors."""
    device_specs = max.driver.scan_available_devices()
    if (cpu := max.driver.DeviceSpec.cpu()) not in device_specs:
        device_specs.append(cpu)
    devices = max.driver.load_devices(device_specs)
    if not (session := _SESSION.get(None)):
        _SESSION.set(
            session := max.engine.api.InferenceSession(devices=devices)
        )
    return session


class Tensor(max.graph.TensorValue):
    """A Tensor object with numerics.

    A Tensor type that can do the kinds of things people expect
    tensors to do.

    Tensor operations should always meet the following criteria:
    - Any illegal operation on a tensor must fail immediately with
        a python exception with a clear error message
    - All operations on tensors that read or write Tensor memory
        values use our high-performance compiler and Mojo kernel library.

    The out of the box experience should be the best one available
    for working with Tensors and numerics, and give seemless access
    to direct low-level programmability in Mojo.

    Notably Tensor does _not_ require that it is backed by memory.
    If no side-effecting operation has been done on a Tensor object,
    then there is no guarantee it has been computed yet. Critically
    a user _should never know or care_ whether the tensor is backed
    by data: the behavior should be exactly as if it were.

    For discussion purposes, a "realized" tensor is a tensor which
    references concrete memory, and an "unrealized" one does not.
    """

    #: Underlying memory for a realized tensor.
    _storage: max.driver.Tensor | None
    #: - For a realized tensor this is a graph input value
    #: - For an unrealized tensor this is a node in the graph
    _mlir_value: mlir.Value[mo.TensorType]

    def __init__(
        self,
        *,
        storage: max.driver.Tensor | None = None,
        mlir_value: mlir.Value[mo.TensorType] | None = None,
    ):
        self._storage = storage
        if mlir_value is not None:
            self._mlir_value = mlir_value
            GRAPH.unrealized.add(self)
        else:
            GRAPH.add_source(self)

    @classmethod
    def from_tensor_value(cls, value: max.graph.TensorValue) -> Tensor:
        return cls(mlir_value=value._mlir_value)

    @property
    def real(self) -> bool:
        """Whether the tensor is realized or not."""
        return self._storage is not None

    @property
    def driver_tensor(self) -> max.driver.Tensor:
        """A pointer to the underlying memory.

        Raises if the tensor is unrealized.
        """
        if not self.real:
            raise TypeError("Can't get driver tensor for symbolic tensor")
        assert self._storage is not None
        return self._storage

    def __await__(self):
        """Force the tensor to realize if it is not already."""
        yield from asyncio.create_task(GRAPH.evaluate(self))
        assert self.real
        return self

    @property
    async def realize(self):
        """Force the tensor to realize if it is not already."""
        return await self

    # Hmm yuck. Would rather type everything as `TensorLike` and `Tensor`
    def __add__(self, other: TensorValueLike) -> Tensor | max.graph.TensorValue:
        return functional(ops.add)(self, other)

    def __hash__(self):
        return id(self)


@dataclass
class ComputeGraph:
    """Compute graph storage for unrealized tensors.

    The compute graph is a directed acyclic graph.

    There is a single global compute graph we use for Tensor operations.
    New tensors are added as nodes to this graph by tensor operations.
    Once they are realized the graph is simplified and the newly realized
    tensors become sources of the graph.

    Terminology:
    - A "source" of the graph is a realized tensor that some unrealized
      tensor depends on.
    - "unrealized" refers to a node in the graph which is not a source,
      or to the tensor object that it backs. There is a 1:1 relationship
      between the node and the tensor object.

    It is not obvious a priori which unrealized nodes to evaluate at
    what time. The `evaluate` method of the graph is at its heart a
    heuristic choosing among various tradeoffs of what to compute.

    The current implementation first prunes the graph of all dead
    nodes (nodes which no longer have live python references to them)
    and then realizes all remaining nodes. This is an implementation
    detail and is subject to change.
    """

    graph: max.graph.Graph
    #: Keeps a strong reference to tensor data that we need to compute graph values
    sources: dict[mlir.Value, Tensor] = field(default_factory=dict)
    #: Keeps weak references to intermediate unrealized tensor values, which may
    #: never need to be realized.
    unrealized: weakref.WeakSet[Tensor] = field(default_factory=weakref.WeakSet)

    async def evaluate(self, tensor: Tensor):
        """Realize the input tensor object.

        It is currently undefined to operate on tensors during evaluation.

        After execution
        - The compute graph object and all tensors realized or otherwise
          will be in valid states.
        - The input tensor is guaranteed to be realized.
        - Some other previously unrealized tensors may be realized
        - Any realized tensors with live references will not be unrealized.
        """
        # create strong references during execution
        unrealized = list(self.unrealized)
        with self.graph:
            # We need to remove sources that no longer exist from the graph
            self.graph.output(*unrealized)
        # Remove dead values and inputs
        module: builtin.ModuleOp = mlir.Operation._from_cmlir(
            self.graph._module.operation
        )  # type: ignore
        mlir.lower(module, [builtin.passes.RemoveDeadValues()])
        inputs = [
            self.sources[input._mlir_value] for input in self.graph.inputs
        ]
        weak_sources = weakref.WeakSet(self.sources.values())
        # Drop dead, unused sources
        self.sources = {}

        try:
            model = _session().load(self.graph)
            # This will become an await when `model` supports it
            results = model(*(input.driver_tensor for input in inputs))
        except:
            # Reset graph to a sane state.
            self.sources = {
                input._mlir_value: tensor
                for input, tensor in zip(self.graph.inputs, inputs)
            }
            # If we've tried and failed to compile the graph, remove its
            # terminator so future ops can modify the graph.
            #  - Can failed lowerings leave the module in a partially lowered state?
            self.graph._erase_output_if_present()
            raise

        # Drop dead used sources
        del inputs

        self.unrealized = weakref.WeakSet()
        self.graph = max.graph.Graph("main", input_types=[])

        for live_source in weak_sources:
            self.add_source(live_source)

        # all evaluated unrealized become realized, sources of a new graph
        for tensor, storage in zip(unrealized, results):
            # This will eventually support Mojo values also.
            assert isinstance(storage, max.driver.Tensor)
            tensor._storage = storage
            self.add_source(tensor)

    def add_source(self, tensor: Tensor):
        op: mo.GraphOp = mlir.Operation._from_cmlir(self.graph._mlir_op)  # type: ignore
        block = op.regions[0].front
        with self.graph:
            type = driver_tensor_type(tensor.driver_tensor).to_mlir()
            inputs = op.function_type.inputs
            op.function_type = builtin.FunctionType([*inputs, type])
            tensor._mlir_value = block.add_argument(
                type, self.graph._location()
            )
        self.sources[tensor._mlir_value] = tensor


GRAPH = ComputeGraph(max.graph.Graph("main", input_types=[]))


def functional(op):
    """Convert an op on symbolic tensor values to one which
    additionally supports max.Tensor inputs."""

    @functools.wraps(op)
    def wrapped(*args, **kwargs):
        with GRAPH.graph:
            results = op(*args, **kwargs)
        if isinstance(results, max.graph.TensorValue):
            return Tensor.from_tensor_value(results)
        return [Tensor.from_tensor_value(result) for result in results]

    return wrapped


def driver_tensor_type(t: max.driver.Tensor) -> TensorType:
    return TensorType(
        t.dtype, t.shape, max.graph.DeviceRef.from_device(t.device)
    )
