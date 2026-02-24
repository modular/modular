# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Implementations of various realization strategies.

**"Eager" execution**: tensors are realized as soon as the realization context
exits. This is the default behavior.

This has a huge concrete advantage over eagerly executing one operation
at a time: by controlling the boundary of where the eager context starts
and ends, we can give advanced users a tool to _enable fine-grained
bounds for automatic fusion_!

In practice the easiest way to do this is to mark a function as
`F.functional`. This function is then assumed to be "atomic" for the
purposes of eager execution. All ops within the function execute as
part of the same graph, meaning the compiler is free to fuse operations
and generate fused kernels within this region.

**"Lazy" execution**: tensors are realized only when code later tries to use
them.

This enables a class of interface design common in the ML world, in
which layers are constructed with randomized weights which are never
used. Lazy execution neatly allows constructing entire models,
only performing the weight initialization and allocating memory for
them if and when those weights are actually used.

**Graph compilation**: tensors must never be realized.

This allows tensor operations to be composed with direct usage of
the Graph API, for instance `Module.compile`, or using `F.*` operations
in another Graph API usage.
"""

from __future__ import annotations

import os
import weakref
from contextvars import ContextVar
from types import TracebackType
from typing import Any, TypeVar

from max import _core, driver, engine
from max._core.dialects import builtin, rmo
from max.experimental import _passes
from max.experimental import functional as F
from max.experimental.support import driver_tensor_type
from max.experimental.tensor import (
    GraphValue,
    RealizationContext,
    RealizationState,
    Tensor,
    current_realization_context,
    realization_context,
)
from max.graph import BufferValue, Graph, TensorValue, Value, ops

Ex = TypeVar("Ex", bound=BaseException)

_SESSION: ContextVar[engine.api.InferenceSession] = ContextVar("_SESSION")
_SEED: Tensor | None = None

# Environment variable to control interpreter usage
_USE_INTERPRETER_ENV_VAR = "MAX_USE_EAGER_INTERPRETER"


def _default_use_interpreter() -> bool:
    """Get the default value for use_interpreter from environment.

    Checks the MAX_USE_EAGER_INTERPRETER environment variable. Set to "1" or "true"
    (case-insensitive) to enable the interpreter by default.

    Returns:
        True if interpreter should be used by default, False otherwise.
    """
    env_value = os.environ.get(_USE_INTERPRETER_ENV_VAR, "").lower()
    return env_value in ("1", "true")


def seed() -> Tensor:
    """Gets the global random seed tensor.

    Returns the global seed tensor used for random number generation in eager
    execution mode. Creates the seed tensor on first access, initialized with
    the dtype, shape, and device specified by :obj:`ops.random.SeedType`.

    Returns:
        Tensor: The global seed tensor for random number generation.
    """
    global _SEED
    if _SEED is None:
        SeedType = ops.random.SeedType
        shape = [int(d) for d in SeedType.shape]
        seed_data = driver.Buffer(
            SeedType.dtype, shape, SeedType.device.to_device()
        )
        _SEED = Tensor(storage=seed_data)
    return _SEED


def set_seed(value: int) -> None:
    """Sets the global random seed value.

    Updates the global random seed to the specified value. This affects all
    subsequent random number generation in eager execution mode.

    Args:
        value: The integer seed value to set.
    """
    seed().driver_tensor[0] = value


def _session() -> engine.api.InferenceSession:
    """A single global inference session for compiling and running kernels on tensors."""
    device_specs = driver.scan_available_devices()
    if (cpu := driver.DeviceSpec.cpu()) not in device_specs:
        device_specs.append(cpu)
    devices = driver.load_devices(device_specs)
    if not (session := _SESSION.get(None)):
        _SESSION.set(session := engine.api.InferenceSession(devices=devices))
    return session


class EagerRealizationContext(RealizationContext):
    """Computation graph for managing tensor operations.

    This class manages the directed acyclic graph (DAG) of tensor operations
    for lazy evaluation and optimization. It tracks both realized tensors
    (with concrete data in memory) and unrealized tensors (pending computations)
    to enable efficient batch compilation and execution.
    """

    graph: Graph
    #: Keeps a strong reference to tensor data that we need to compute graph values
    sources: dict[_core.Value[Any], Tensor]
    #: Reverse map of sources
    source_values: dict[int, BufferValue]
    #: Unrealized values
    unrealized: list[weakref.ref[Tensor]]

    def __init__(self, use_interpreter: bool | None = None):
        if use_interpreter is None:
            use_interpreter = _default_use_interpreter()
        self._use_interpreter = use_interpreter
        self.sources = {}
        self.source_values = {}
        self.unrealized = []

        self.graph = Graph("main", input_types=[])

        with realization_context(self), self.graph:
            ops.random.set_seed(seed())

    def finalize_graph(self) -> tuple[list[Tensor], Graph]:
        """Finalizes the computation graph for execution.

        Prepares the graph for compilation by setting outputs, removing dead
        code and unused arguments, and replacing static shapes with symbolic
        parameters. This method is called internally before graph execution.

        Returns:
            tuple[list[Tensor], Graph]: A tuple containing the list of output
                tensors (including the seed) and the finalized graph.
        """
        with realization_context(self), self.graph:
            # peek rather than next! If compilation or execute fails
            # the seed should remain the same.
            outputs = [
                Tensor.from_graph_value(ops.random._peek_seed()),
                *(
                    tensor
                    for ref in self.unrealized
                    if (tensor := ref()) is not None
                ),
            ]
            self.graph.output(*outputs)
        # Remove dead values and inputs
        module: builtin.ModuleOp = _core.Operation._from_cmlir(
            self.graph._module.operation
        )  # type: ignore
        # Remove sources that no longer exist from the graph
        _core.lower(
            module,
            [
                builtin.passes.RemoveDeadValues(),
                rmo.passes.LegalizeRMOOps(),
            ],
        )
        # The graph symbol is public, so RemoveDeadValues won't remove
        # unused arguments. Do that explicitly.
        _passes.remove_unused_arguments(self.graph)
        return outputs, self.graph

    async def realize_all(self) -> list[Tensor]:
        """Compiles and executes the computation graph, realizing all tensors.

        Finalizes the computation graph, compiles it using the inference
        session, and executes it to produce concrete values for all pending
        (unrealized) tensors. After execution, all tensors tracked by this
        context will have their data in memory.

        Returns:
            list[Tensor]: The list of realized output tensors (excluding the
                internal seed tensor).

        Raises:
            TypeError: If called while still inside this realization context.
        """
        if current_realization_context(None) is self:
            raise TypeError(
                "Can't realize tensor before realization context is completed."
            )

        outputs, graph = self.finalize_graph()

        # Execute graph via interpreter or compilation
        use_interpreter = self._use_interpreter
        if use_interpreter:
            # Lazy import to avoid circular dependency
            from max._interpreter import MOInterpreter

            interp = MOInterpreter()
            if not interp.can_execute(graph):
                use_interpreter = False

        if use_interpreter:
            inputs = [self.sources[input._mlir_value] for input in graph.inputs]
            results = interp.execute(
                graph,
                [inp.driver_tensor for inp in inputs],
            )
        else:
            # Compile and execute graph
            model = _session().load(graph)
            # Inputs may have been removed by optimization
            inputs = [self.sources[input._mlir_value] for input in graph.inputs]
            # This will become an await when `model` supports it
            results = model(*(input.driver_tensor for input in inputs))

        # Update tensors to realized
        for tensor, storage in zip(outputs, results, strict=True):
            # This will eventually support Mojo values also.
            assert isinstance(storage, driver.Buffer)
            tensor.storage = storage
            tensor.state = None

        # Update mutated buffer inputs to realized
        for source in self.sources.values():
            # This was set by calling `__buffervalue__` on the source.
            # Mark the tensor as realized again.
            if source.state and source.state.ctx is self:
                source.state = None

        new_seed, *outputs = outputs
        set_seed(new_seed.item())

        return outputs

    def add_source(self, tensor: Tensor) -> RealizationState:
        """Adds a realized tensor as an input source to the computation graph.

        Registers a realized tensor as a graph input, allowing it to be used
        in subsequent graph operations. The tensor's data will be passed to
        the compiled graph during execution. This operation is idempotent;
        adding the same tensor multiple times returns the same state.

        Args:
            tensor: A realized tensor to add as a graph input source.

        Returns:
            RealizationState: The state associating the tensor with its graph
                value and this context.

        Raises:
            TypeError: If the tensor is not realized (has no concrete data).
        """
        if not tensor.real:
            raise TypeError("Only realized tensors may be graph sources.")

        value: Value[Any] | None

        # Safe to use IDs because self.sources keeps references alive
        if (value := self.source_values.get(id(tensor))) is not None:
            return RealizationState(value, self)

        assert tensor.storage
        type = driver_tensor_type(tensor.storage).as_buffer()
        value = _passes.add_input(self.graph, type)
        assert isinstance(value, BufferValue)
        self.sources[value._mlir_value] = tensor
        self.source_values[id(tensor)] = value
        return RealizationState(value, self)

    def create_unrealized(self, value: BufferValue | TensorValue) -> Tensor:
        """Creates an unrealized tensor backed by a graph value.

        Wraps a graph value (TensorValue or BufferValue) in a Tensor object
        and tracks it for later realization. The tensor will not contain
        concrete data until the context's ``realize_all()`` is called.

        Args:
            value: The graph value to wrap in a tensor.

        Returns:
            Tensor: An unrealized tensor that will be computed when the
                context realizes all pending operations.
        """
        state = RealizationState(value, self)
        tensor = Tensor(state=state)
        self.unrealized.append(weakref.ref(tensor))
        return tensor

    def __enter__(self):
        self.graph.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[Ex] | None,
        exception: Ex | None,
        traceback: TracebackType | None,
    ):
        self.graph.__exit__(exception_type, exception, traceback)
        if not exception:
            F._run(self.realize_all())


class LazyRealizationContext(EagerRealizationContext):
    """A realization context that defers execution until explicitly requested.

    Unlike :class:`EagerRealizationContext`, this context does not automatically
    execute the computation graph when the context exits. Tensors remain
    unrealized until explicitly awaited via ``await tensor.realize``.

    This is useful for batching many operations together before execution,
    improving performance by reducing compilation overhead.

    Example::

        with F.lazy():
            a = Tensor.zeros([5, 5])
            b = a + 1
            c = b * 2
        # No execution yet - all tensors are unrealized
        assert not c.real

        await c.realize  # Now compile and execute
        assert c.real
    """

    def __exit__(
        self,
        exception_type: type[Ex] | None,
        exception: Ex | None,
        traceback: TracebackType | None,
    ):
        self.graph.__exit__(exception_type, exception, traceback)


class GraphRealizationContext(RealizationContext):
    """A realization context for ahead-of-time graph compilation.

    This context is used when building computation graphs that will be compiled
    and executed later (e.g., during :meth:`Module.compile`). Tensors in this
    context remain as symbolic graph values and cannot be realized.

    Unlike eager contexts, this context does not support executing operations
    immediately. Attempting to realize tensors will raise a TypeError.

    Attributes:
        graph: The graph being constructed in this context.

    Example::

        graph = Graph("my_model", input_types=[TensorType(...)])
        with GraphRealizationContext(graph) as ctx:
            x = Tensor.from_graph_value(graph.inputs[0])
            y = x + 1  # Creates graph operation, not computation
            graph.output(y)
        # Graph can now be compiled and executed separately
    """

    graph: Graph

    def __init__(self, graph: Graph):
        """Initializes the graph realization context.

        Args:
            graph: The graph to construct operations in.
        """
        self.graph = graph

    async def realize_all(self) -> list[Tensor]:
        """Raises TypeError - graph contexts cannot realize tensors.

        Raises:
            TypeError: Always raised, as graph contexts are for symbolic
                graph construction only.
        """
        raise TypeError("Can't realize from a graph context.")

    def add_source(self, tensor: Tensor) -> RealizationState:
        """Adds a tensor as a constant in the graph.

        In graph context, source tensors become constant values embedded
        in the graph rather than graph inputs.

        Args:
            tensor: The tensor to embed as a constant.

        Returns:
            RealizationState: The state with the constant graph value.
        """
        return RealizationState(ops.constant(tensor), self)

    def create_unrealized(self, value: GraphValue) -> Tensor:
        """Creates a tensor backed by a graph value.

        Args:
            value: The graph value to wrap.

        Returns:
            Tensor: A tensor backed by the graph value.
        """
        return Tensor(state=RealizationState(value, self))

    def __enter__(self):
        self.graph.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[Ex] | None,
        exception: Ex | None,
        traceback: TracebackType | None,
    ):
        self.graph.__exit__(exception_type, exception, traceback)
