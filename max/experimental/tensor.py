# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import asyncio
import functools
import weakref
from collections.abc import Iterable
from contextvars import ContextVar
from itertools import chain

import numpy as np

# For clarity, since there are several Tensor and Device types
# around, only directly import types which are concretely used
# by max.Tensor.
from .. import _core, driver, engine, graph, mlir
from .._core.dialects import builtin, mo
from ..driver import CPU, Device
from ..dtype import DType
from ..graph import TensorType, TensorValueLike, ops
from ..graph.graph import _location

_SESSION: ContextVar[engine.api.InferenceSession] = ContextVar("_SESSION")


def _session() -> engine.api.InferenceSession:
    """A single global inference session for compiling and running kernels on tensors."""
    device_specs = driver.scan_available_devices()
    if (cpu := driver.DeviceSpec.cpu()) not in device_specs:
        device_specs.append(cpu)
    devices = driver.load_devices(device_specs)
    if not (session := _SESSION.get(None)):
        _SESSION.set(session := engine.api.InferenceSession(devices=devices))
    return session


class Tensor(graph.TensorValue):
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
    _storage: driver.Tensor | None
    #: - For a realized tensor this is a graph input value
    #: - For an unrealized tensor this is a node in the graph
    _mlir_value: _core.Value[mo.TensorType]

    def __init__(
        self,
        *,
        storage: driver.Tensor | None = None,
        mlir_value: _core.Value[mo.TensorType] | None = None,
    ):
        self._storage = storage
        if mlir_value is not None:
            self._mlir_value = mlir_value
            GRAPH.unrealized.add(self)
        else:
            GRAPH.add_source(self)

    @classmethod
    def from_tensor_value(cls, value: graph.TensorValue) -> Tensor:
        return cls(mlir_value=value._mlir_value)

    @classmethod
    def constant(cls, value, dtype, device) -> Tensor:
        if hasattr(value, "__iter__") and not isinstance(value, np.ndarray):
            value = np.array(value, dtype.to_numpy())

        return functional(ops.constant)(
            value, dtype, graph.DeviceRef.from_device(device)
        )

    @property
    def device_(self) -> Device:
        """The tensor's device."""
        return super().device.to_device()

    @property
    def real(self) -> bool:
        """Whether the tensor is realized or not."""
        return self._storage is not None

    @property
    def driver_tensor(self) -> driver.Tensor:
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

    def _sync_realize(self):
        if not self.real:
            asyncio.run(self.realize)

    def __add__(self, other: TensorValueLike) -> Tensor:
        return functional(ops.add)(self, other)

    def __sub__(self, other: TensorValueLike) -> Tensor:
        return functional(ops.sub)(self, other)

    def __truediv__(self, other: TensorValueLike) -> Tensor:
        return functional(ops.div)(self, other)

    def __eq__(self, other: TensorValueLike) -> Tensor:  # type: ignore
        return functional(ops.equal)(self, other)

    def __gt__(self, other: TensorValueLike) -> Tensor:
        return functional(ops.greater)(self, other)

    def __abs__(self) -> Tensor:
        return functional(ops.abs)(self)

    def __bool__(self) -> bool:
        return bool(self.item())

    def __getitem__(self, idx):
        return functional(graph.TensorValue.__getitem__)(self, idx)

    def _values(self):
        self._sync_realize()
        dt = self.driver_tensor.to(CPU())
        for idx in dt._iterate_indices():
            yield dt[idx].item()

    def __repr__(self):
        # Janky repr for bootstrapping, we can do much better.
        return f"{self.type}: [{', '.join(str(v) for v in self._values())}]"

    def item(self):
        if self.num_elements() != 1:
            raise TypeError()
        self._sync_realize()
        return self.driver_tensor.to(CPU()).item()

    def num_elements(self) -> int:
        elts = 1
        for dim in self.shape:
            elts *= int(dim)
        return elts

    def argmax(self) -> Tensor:
        return functional(ops.argmax)(self)

    def max(self) -> Tensor:
        return functional(ops.max)(self)

    def __hash__(self):
        return id(self)

    @classmethod
    def arange(
        cls,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        dtype: DType = DType.float32,
        device: Device = CPU(),
    ):
        if stop is None:
            start, stop = 0, start
        return functional(ops.range)(
            start,
            stop,
            step,
            dtype=dtype,
            device=graph.DeviceRef.from_device(device),
        )


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

    graph: graph.Graph
    #: Keeps a strong reference to tensor data that we need to compute graph values
    sources: dict[_core.Value, Tensor]
    #: Keeps weak references to intermediate unrealized tensor values, which may
    #: never need to be realized.
    unrealized: weakref.WeakSet[Tensor]

    def __init__(
        self,
        context: mlir.Context | None = None,
        sources: Iterable[Tensor] = (),
        seed: int = 0,
    ):
        self.context = context or mlir.Context()
        self.sources = {}

        self.unrealized = weakref.WeakSet()
        self.graph = graph.Graph("main", input_types=[], context=self.context)

        with self.graph:
            ops.random.set_seed(seed)

        for source in sources:
            self.add_source(source)

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
            self.graph.output(ops.random._next_seed(), *unrealized)
        # Remove dead values and inputs
        module: builtin.ModuleOp = _core.Operation._from_cmlir(
            self.graph._module.operation
        )  # type: ignore
        # Remove sources that no longer exist from the graph
        _core.lower(module, [builtin.passes.RemoveDeadValues()])
        # Handling seeds is really awkward, and might have some catastrophic failures
        # mixing calls to random between a hand-constructed graph and usage of max.Tensor.
        # First output is always seed:
        inputs = [
            self.sources[input._mlir_value] for input in self.graph.inputs
        ]

        try:
            model = _session().load(self.graph)
            # This will become an await when `model` supports it
            seed, *results = model(*(input.driver_tensor for input in inputs))
            assert isinstance(seed, driver.Tensor)
        except:
            # If we've tried and failed to compile the graph, remove its
            # terminator so future ops can modify the graph.
            #  - Can failed lowerings leave the module in a partially lowered state?
            self.graph._erase_output_if_present()
            raise

        for tensor, storage in zip(unrealized, results):
            # This will eventually support Mojo values also.
            assert isinstance(storage, driver.Tensor)
            tensor._storage = storage

        # Reset the graph to a new empty graph with only inputs
        ComputeGraph.__init__(
            self,
            context=self.graph._context,
            # - Re-add any sources that still have live references
            # - All evaluated tensors become realized sources of a new graph
            sources=chain(inputs, unrealized),
            seed=seed.item(),
        )

    def add_source(self, tensor: Tensor):
        op = _core.Operation._from_cmlir(self.graph._mlir_op)
        assert isinstance(op, mo.GraphOp)
        block = op.regions[0].front
        with self.graph:
            type = driver_tensor_type(tensor.driver_tensor).to_mlir()
            inputs = op.function_type.inputs
            op.function_type = builtin.FunctionType([*inputs, type])
            tensor._mlir_value = block.add_argument(type, _location())
        self.sources[tensor._mlir_value] = tensor


def functional(op):
    """Convert an op on symbolic tensor values to one which
    additionally supports max.Tensor inputs."""

    @functools.wraps(op)
    def wrapped(*args, **kwargs):
        with GRAPH.graph:
            results = op(*args, **kwargs)
        if isinstance(results, graph.TensorValue):
            return Tensor.from_tensor_value(results)
        return [Tensor.from_tensor_value(result) for result in results]

    return wrapped


def driver_tensor_type(t: driver.Tensor) -> TensorType:
    return TensorType(t.dtype, t.shape, graph.DeviceRef.from_device(t.device))


GRAPH = ComputeGraph()
