# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import asyncio
import contextlib
import gc
import sys
import warnings
import weakref
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from itertools import chain
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from .. import _core, driver, engine, graph, mlir
from .._core.dialects import builtin, kgen, mo
from ..driver import CPU, Accelerator, Device, DLPackArray, accelerator_count
from ..dtype import DType
from ..graph import (
    ShapeLike,
    TensorType,
    TensorValueLike,
    Value,
    ops,
)
from ..graph.graph import _location
from . import functional as F

_SESSION: ContextVar[engine.api.InferenceSession] = ContextVar("_SESSION")
_DEFAULT_DEVICE: ContextVar[Device] = ContextVar("_DEFAULT_DEVICE")
_DEFAULT_DTYPE: ContextVar[DType] = ContextVar("_DEFAULT_DTYPE")


T = TypeVar("T")


@contextlib.contextmanager
def contextvar_context(var: ContextVar[T], value: T):
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def _default_dtype(device: Device) -> DType:
    if dtype := _DEFAULT_DTYPE.get(None):
        return dtype
    return DType.float32 if isinstance(device, CPU) else DType.bfloat16


def _default_device() -> Device:
    if device := _DEFAULT_DEVICE.get(None):
        return device
    return Accelerator() if accelerator_count() else CPU()


def defaults(
    dtype: DType | None, device: Device | None
) -> tuple[DType, Device]:
    device = device or _default_device()
    return (dtype or _default_dtype(device)), device


def default_device(device: Device):
    """Context manager for setting the default device for tensors."""
    return contextvar_context(_DEFAULT_DEVICE, device)


def default_dtype(dtype: DType):
    """Context manager for setting the default dtype for tensors."""
    return contextvar_context(_DEFAULT_DTYPE, dtype)


def _session() -> engine.api.InferenceSession:
    """A single global inference session for compiling and running kernels on tensors."""
    device_specs = driver.scan_available_devices()
    if (cpu := driver.DeviceSpec.cpu()) not in device_specs:
        device_specs.append(cpu)
    devices = driver.load_devices(device_specs)
    if not (session := _SESSION.get(None)):
        _SESSION.set(session := engine.api.InferenceSession(devices=devices))
    return session


class Tensor:
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
    _value: graph.TensorValue

    def __init__(
        self,
        *,
        storage: driver.Tensor | None = None,
        value: graph.TensorValue | None = None,
    ):
        self._storage = storage
        if value is not None:
            self._value = value
            GRAPH.unrealized.add(self)
        else:
            GRAPH.add_source(self)

    @classmethod
    def from_tensor_value(cls, value: graph.TensorValue) -> Tensor:
        return cls(value=value)

    @classmethod
    def from_dlpack(cls, array: DLPackArray) -> Tensor:
        return Tensor(storage=driver.Tensor.from_dlpack(array))

    @classmethod
    def constant(
        cls,
        value: npt.NDArray[np.number[Any]]
        | int
        | float
        | Sequence[int | float],
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        dtype, device = defaults(dtype, device)
        if hasattr(value, "__iter__") and not isinstance(value, np.ndarray):
            value = np.array(value, dtype.to_numpy())

        return F.constant(value, dtype, graph.DeviceRef.from_device(device))

    @classmethod
    def full(
        cls,
        shape: ShapeLike,
        value: int | float,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        return F.broadcast_to(
            cls.constant(value, dtype=dtype, device=device), shape
        )

    @classmethod
    def full_like(cls, type: TensorType, value: int | float) -> Tensor:
        return cls.full(
            type.shape,
            value=value,
            dtype=type.dtype,
            device=type.device.to_device(),
        )

    @classmethod
    def zeros(
        cls,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        return cls.full(shape, value=0, dtype=dtype, device=device)

    @classmethod
    def zeros_like(cls, type: TensorType) -> Tensor:
        return cls.zeros(
            type.shape, dtype=type.dtype, device=type.device.to_device()
        )

    @classmethod
    def ones(
        cls,
        shape: ShapeLike,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        return cls.full(shape, value=1, dtype=dtype, device=device)

    @classmethod
    def ones_like(cls, type: TensorType) -> Tensor:
        return cls.ones(
            type.shape, dtype=type.dtype, device=type.device.to_device()
        )

    @classmethod
    def arange(
        cls,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
    ) -> Tensor:
        dtype, device = defaults(dtype, device)
        if stop is None:
            start, stop = 0, start
        return F.range(
            start,
            stop,
            step,
            dtype=dtype,
            device=device,
        )

    @property
    def type(self) -> TensorType:
        return self._value.type

    @property
    def rank(self) -> int:
        return self._value.rank

    @property
    def shape(self) -> graph.Shape:
        return self._value.shape

    @property
    def dtype(self) -> DType:
        return self._value.dtype

    @property
    def device(self) -> Device:
        """The tensor's device."""
        return self._value.device.to_device()

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

    def __tensorvalue__(self) -> graph.TensorValue:
        return self._value

    def __await__(self):
        """Force the tensor to realize if it is not already."""
        yield from asyncio.create_task(GRAPH.evaluate(self))
        assert self.real
        return self

    @property
    async def realize(self):
        """Force the tensor to realize if it is not already."""
        return await self

    def _sync_realize(self) -> None:
        if self.real:
            return

        # If there's no running loop, just use asyncio.run
        try:
            # - asyncio.get_event_loop().is_running() works in most scenarios
            # - asyncio.get_event_loop() raises in some environments
            # - use asyncio.get_running_loop() and check if it fails instead
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop
            asyncio.run(self.realize)
            return

        # If there is a running loop, execute using a ThreadPoolExecutor
        # - This is a common case inside a Jupyter notebook, eg.
        #   printing a tensor.
        # - Otherwise, this is probably accidental. The code is running
        #   inside an async environment, but for some reason is trying
        #   to synchronously await. Check for this case explicitly and warn.
        def is_interactive():
            import __main__ as main

            return not hasattr(main, "__file__")

        if not is_interactive():
            warnings.warn(
                "Use of synchronous tensor method inside another event loop. "
                "Use `await tensor`."
            )

        # Run self.realize in another thread
        loop = asyncio.new_event_loop()
        with ThreadPoolExecutor() as pool:
            pool.submit(loop.run_until_complete, self.realize)

    def __bool__(self) -> bool:
        return bool(self.item())

    def _values(self):
        self._sync_realize()
        dt = self.driver_tensor.to(CPU())
        for idx in dt._iterate_indices():
            yield dt[idx].item()

    def __hash__(self):
        return id(self)

    def __dlpack__(self, stream: int | None = None):
        self._sync_realize()
        assert self._storage is not None
        return self._storage.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        self._sync_realize()
        assert self._storage is not None
        return self._storage.__dlpack_device__()

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
        return F.argmax(self)

    def max(self) -> Tensor:
        return F.max(self)

    def reshape(self, shape: ShapeLike) -> Tensor:
        return F.reshape(self, shape)

    def cast(self, dtype: DType) -> Tensor:
        return F.cast(self, dtype)

    def permute(self, dims: list[int]) -> Tensor:
        return F.permute(self, dims)

    def transpose(self, dim1: int, dim2: int) -> Tensor:
        return F.transpose(self, dim1, dim2)

    @property
    def T(self) -> Tensor:
        return self.transpose(-1, -2)

    def __getitem__(self, idx):  # noqa: ANN001
        return F.functional(graph.TensorValue.__getitem__)(self, idx)

    def __abs__(self) -> Tensor:
        return F.abs(self)

    def __neg__(self) -> Tensor:
        return F.negate(self)

    def __eq__(self, rhs: Any) -> Tensor:  # type: ignore[override]
        return F.equal(self, rhs)

    def __ne__(self, rhs: Any) -> Tensor:  # type: ignore[override]
        return F.not_equal(self, rhs)

    def __ge__(self, rhs: Any) -> Tensor:
        return F.greater_equal(self, rhs)

    def __gt__(self, rhs: Any) -> Tensor:
        return F.greater(self, rhs)

    def __lt__(self, rhs: Any) -> Tensor:
        return ~(self >= rhs)

    def __le__(self, rhs: Any) -> Tensor:
        return ~(self > rhs)

    def __add__(self, rhs: TensorValueLike) -> Tensor:
        return F.add(self, rhs)

    def __radd__(self, lhs: TensorValueLike) -> Tensor:
        return F.add(lhs, self)

    def __sub__(self, rhs: TensorValueLike) -> Tensor:
        return F.sub(self, rhs)

    def __rsub__(self, lhs: TensorValueLike) -> Tensor:
        return F.sub(lhs, self)

    def __mul__(self, rhs: TensorValueLike) -> Tensor:
        return F.mul(self, rhs)

    def __rmul__(self, lhs: TensorValueLike) -> Tensor:
        return F.mul(lhs, self)

    def __truediv__(self, rhs: TensorValueLike) -> Tensor:
        return F.div(self, rhs)

    def __rtruediv__(self, lhs: TensorValueLike) -> Tensor:
        return F.div(lhs, self)

    def __floordiv__(self, rhs: TensorValueLike) -> Tensor:
        return F.floor(F.div(self, rhs))

    def __rfloordiv__(self, lhs: TensorValueLike) -> Tensor:
        return F.floor(F.div(lhs, self))

    def __mod__(self, rhs: TensorValueLike) -> Tensor:
        return F.mod(self, rhs)

    def __rmod__(self, lhs: TensorValueLike) -> Tensor:
        return F.mod(lhs, self)

    def __divmod__(self, rhs: TensorValueLike) -> tuple[Tensor, Tensor]:
        return (self // rhs, self % rhs)

    def __rdivmod__(self, lhs: TensorValueLike) -> tuple[Tensor, Tensor]:
        return (self.__rfloordiv__(lhs), self.__rmod__(lhs))

    def __matmul__(self, rhs: TensorValueLike) -> Tensor:
        return F.matmul(self, rhs)

    def __rmatmul__(self, lhs: TensorValueLike) -> Tensor:
        return F.matmul(lhs, self)

    def __pow__(self, rhs: TensorValueLike) -> Tensor:
        return F.pow(self, rhs)

    def __rpow__(self, lhs: TensorValueLike) -> Tensor:
        return F.pow(lhs, self)

    def __and__(self, rhs: TensorValueLike) -> Tensor:
        return F.logical_and(self, rhs)

    def __rand__(self, lhs: TensorValueLike) -> Tensor:
        return F.logical_and(lhs, self)

    def __or__(self, rhs: TensorValueLike) -> Tensor:
        return F.logical_or(self, rhs)

    def __ror__(self, lhs: TensorValueLike) -> Tensor:
        return F.logical_or(lhs, self)

    def __xor__(self, rhs: TensorValueLike) -> Tensor:
        return F.logical_xor(self, rhs)

    def __rxor__(self, lhs: TensorValueLike) -> Tensor:
        return F.logical_xor(lhs, self)

    def __invert__(self) -> Tensor:
        return F.logical_not(self)


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
    sources: dict[_core.Value[Any], Tensor]
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

    async def evaluate(self, tensor: Tensor) -> None:
        """Realize the input tensor object.

        It is currently undefined to operate on tensors during evaluation.

        After execution
        - The compute graph object and all tensors realized or otherwise
          will be in valid states.
        - The input tensor is guaranteed to be realized.
        - Some other previously unrealized tensors may be realized
        - Any realized tensors with live references will not be unrealized.
        """
        # Single-global-graph is (unsurprisingly) causing the spooky action at a distance :(
        # - Specifically, bad things give a nice compiler error!
        #   ... but then the exception hangs around, which holds a traceback
        #   which holds the call frame which holds the tensors that caused the error.
        # - Since these tensors don't get garbage collected they survive as unrealized
        # - Then the next evaluate call tries to realize again and necessarily
        #   will have the same error.
        # - HACK/workaround: running tensor evaluation clears the last exception.
        # This also means a user needs to explicitly drop any references to the
        # offending tensors. Ultimately this feels like a strong case for only
        # running partial graphs.
        sys.last_value = None
        sys.last_traceback = None
        gc.collect()

        # create strong references during execution
        unrealized = list(self.unrealized)
        with self.graph:
            # peek rather than next! If the seed is rotated but compilation
            # or execute fails, we need the seed to not rotate.
            self.graph.output(
                ops.random._peek_seed(), *map(graph.TensorValue, unrealized)
            )
        # Remove dead values and inputs
        module: builtin.ModuleOp = _core.Operation._from_cmlir(
            self.graph._module.operation
        )  # type: ignore
        # Remove sources that no longer exist from the graph
        _core.lower(module, [builtin.passes.RemoveDeadValues()])
        # The graph symbol is public, so RemoveDeadValues won't remove
        # unused arguments. Do that explicitly.
        _remove_unused_arguments(self.graph)
        inputs = [
            self.sources[input._mlir_value] for input in self.graph.inputs
        ]

        try:
            model = _session().load(self.graph)
            # This will become an await when `model` supports it
            seed, *results = model(*(input.driver_tensor for input in inputs))
            assert isinstance(seed, driver.Tensor)
        except BaseException as e:
            # If we've tried and failed to compile the graph, remove its
            # terminator so future ops can modify the graph.
            #  - Can failed lowerings leave the module in a partially lowered state?
            self.graph._erase_output_if_present()
            raise RuntimeError(
                "Failed to compile and execute graph! Please file an issue. "
                "This error should have been caught at op creation time."
            ) from e

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
            sources=chain(self.sources.values(), unrealized),
            seed=seed.item(),
        )

    def add_source(self, tensor: Tensor) -> None:
        op = _core.Operation._from_cmlir(self.graph._mlir_op)
        assert isinstance(op, mo.GraphOp)
        block = op.regions[0].front
        # Update the GraphOP to reflect the new argument
        with self.graph:
            type = driver_tensor_type(tensor.driver_tensor).to_mlir()
            inputs = op.function_type.inputs
            op.function_type = builtin.FunctionType([*inputs, type])  # type: ignore
            tensor._value = graph.TensorValue.from_mlir(
                block.add_argument(type, _location())
            )
        self.sources[tensor._value._mlir_value] = tensor


def _remove_unused_arguments(graph: graph.Graph):
    # Obviously this is deeply tied to the implementation of Graph.
    #  - GraphOp should be simplified to have a single API for managing arguments
    #  - Graph should expose this behavior
    op = _core.Operation._from_cmlir(graph._mlir_op)
    assert isinstance(op, mo.GraphOp)

    block = op.regions[0].front
    # reverse so indices don't during iteration+mutation
    for i, input in reversed(list(enumerate(graph.inputs))):
        if not input._mlir_value.num_uses:
            block.erase_argument(i)

    # graph.inputs is a cached_property, so reset it
    graph.inputs = [Value.from_mlir(arg) for arg in block.arguments]

    # update the graph op to correctly reflect the input changes
    with graph:
        op.function_type = builtin.FunctionType(  # type: ignore
            [input.type.to_mlir() for input in graph.inputs],
            op.function_type.results,
        )
        op.signature = kgen.FuncTypeGeneratorType([], op.function_type)  # type: ignore
        op.discardable_attributes["argument_names"] = builtin.ArrayAttr(
            [builtin.StringAttr(f"input{i}") for i in range(len(graph.inputs))]
        )


def driver_tensor_type(t: driver.Tensor) -> TensorType:
    return TensorType(t.dtype, t.shape, graph.DeviceRef.from_device(t.device))


GRAPH = ComputeGraph()
