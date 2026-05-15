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
"""Base classes and decorators for building neural network modules in MAX."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Generic

from max.driver import CPU, Buffer, Device, DLPackArray
from max.engine import Model
from max.experimental.realization_context import (
    GraphRealizationContext,
    _session,
)
from max.experimental.sharding import DeviceMapping, DeviceMesh
from max.experimental.tensor import Tensor, realization_context
from max.graph import DeviceRef, Graph
from rich.pretty import pretty_repr
from typing_extensions import ParamSpec, Self, TypeVar, dataclass_transform

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# Type variables for Module's forward signature.
_P = ParamSpec("_P")
_R = TypeVar("_R")

from max.experimental.nn._compilation_timer import CompilationTimer
from max.experimental.nn._compile_utils import (
    InputType,
    _detect_signals,
    _flatten_input_types,
    _flatten_named_buffers,
    _flatten_outputs,
    _InputSlot,
    _OutputSlot,
    _prepare_weight_for_parameter,
    _process_provided_weights,
    _reconstruct_outputs,
    _wrap_graph_inputs,
    engine_call_error,
    flatten_input_buffers,
)
from max.nn.comm.allreduce import Signals
from max.profiler import Tracer


class CompiledModel:
    """Compiled model returned by :meth:`Module.compile`.

    Provides two execution paths:

    * **Tensor path** — ``compiled(tensor_a, tensor_b)`` handles distributed
      Tensors transparently (unflatten shards, append signals, reconstruct).
      Used by tests and the high-level API.
    * **Buffer path** — ``compiled.execute_raw(*buffers)`` passes flat Buffers
      straight to the engine, auto-appending signal buffers. Returns
      ``list[Buffer]`` with zero Tensor overhead.  Used by pipeline
      ``execute()`` methods.

    For CUDA graph capture/replay, access ``compiled.engine_model`` directly::

        compiled.engine_model.capture(key, *all_buffers)
        compiled.engine_model.replay(key, *all_buffers)

    For multi-GPU capture, append ``compiled.signal_buffers`` to the buffer
    list passed to capture/replay.
    """

    def __init__(
        self,
        engine_model: Model,
        input_slots: list[_InputSlot],
        output_slots: list[_OutputSlot],
        signal_buffers: list[Buffer],
        unary: bool,
    ) -> None:
        self._engine_model = engine_model
        self._input_slots = input_slots
        self._output_slots = output_slots
        self._signal_buffers = signal_buffers
        self._unary = unary

    @property
    def engine_model(self) -> Model:
        """The underlying :class:`~max.engine.Model` for capture/replay."""
        return self._engine_model

    @property
    def signal_buffers(self) -> list[Buffer]:
        """Signal buffers for multi-GPU collectives (empty for single-GPU)."""
        return self._signal_buffers

    def __call__(self, *args: Any) -> Any:
        """Tensor-in, Tensor-out execution (distributed-aware)."""
        flat_args = flatten_input_buffers(args, self._input_slots)
        flat_args.extend(self._signal_buffers)
        try:
            raw_results = list(self._engine_model(*flat_args))
        except (TypeError, ValueError) as e:
            raise engine_call_error(
                e,
                self._engine_model,
                user_args=args,
                flat_args=flat_args,
                input_slots=self._input_slots,
                signal_buffer_count=len(self._signal_buffers),
            ) from e
        return _reconstruct_outputs(
            raw_results, self._output_slots, self._unary
        )

    def execute_raw(self, *buffers: Buffer) -> list[Buffer]:
        """Buffer-in, Buffer-out execution (no Tensor wrapping).

        Auto-appends signal buffers for multi-GPU collectives.
        """
        all_bufs: list[Any] = list(buffers)
        all_bufs.extend(self._signal_buffers)
        try:
            return list(self._engine_model(*all_bufs))
        except (TypeError, ValueError) as e:
            raise engine_call_error(
                e,
                self._engine_model,
                user_args=None,
                flat_args=all_bufs,
                input_slots=self._input_slots,
                signal_buffer_count=len(self._signal_buffers),
            ) from e


class _DevicePinned:
    """Sentinel marker for parameters whose device should not be changed.

    Used as annotation metadata in `PinnedDeviceTensor`. Do not use directly;
    annotate fields with `PinnedDeviceTensor` instead.
    """


PinnedDeviceTensor = Annotated[Tensor, _DevicePinned]
"""Type alias for a `Tensor` parameter that `Module.to` will leave on its
current device.

Use this for parameters that must stay on a specific device regardless of where
the rest of the module is moved. For example, scalar quantization scale factors
that GPU kernels consume as host-side launch arguments should remain on CPU;
moving them to the accelerator would force an expensive device sync every
forward pass.
"""


def _get_pinned_device_fields(cls: type) -> frozenset[str]:
    """Returns field names annotated as `PinnedDeviceTensor` on `cls`.

    Walks the MRO so that inherited annotations are included.
    """
    pinned: set[str] = set()
    for base in cls.__mro__:
        for name, ann in getattr(base, "__annotations__", {}).items():
            if ann is PinnedDeviceTensor or ann == "PinnedDeviceTensor":
                pinned.add(name)
    return frozenset(pinned)


class Module(Generic[_P, _R]):
    """The core unit of composition for modeling in MAX.

    Informally, a ``Module`` is a container class. It can contain
    other ``Module`` instances, tensors (the ``Module``'s "local parameters")
    or other arbitrary Python data.

    A ``Module`` also has a ``forward()`` method which defines how the ``Module``
    computes its output. In the simplest case this is a function from one tensor
    to another tensor. Users call the module using ``__call__()`` which internally
    invokes ``forward()``.

    Formally modules form a tree, and subtrees of modules can be manipulated
    directly. A ``Module`` may also be thought of as a closure, where the parameters
    form the data of the closure and ``forward()`` is the application of the closure.

    Users who do not use a Python type checker, or use lax settings for their
    type checker, may inherit from ``Module`` without parameters. Users who use
    a type checker with stricter settings (including MAX internal code) should
    specify explicit types for full type checking::

        class Linear(Module[[Tensor], Tensor]):
            def forward(self, x: Tensor) -> Tensor:
                return x @ self.weight.T + self.bias

    **Terminology:**

    - A "child" of a ``Module`` is a sub-``Module`` stored directly on that ``Module``.
    - A "descendant" of a ``Module`` is one of its children, or one of their
      descendants.
    - A "parameter" is a tensor storing data on the ``Module`` or one of its
      descendants.
    - The "qualified path" of a descendant is a period-separated string
      of the names of the child module attributes which lead to that
      descendant module, for instance ``child.sub.last``.
    - The "qualified path" of a parameter is the qualified path of the
      descendant directly holding that parameter, followed by a final
      path component for the attribute name of the tensor.
      For instance ``weight`` for a local parameter, or
      ``child.sub.last.weight`` for a descendant's parameter.

    .. code-block:: python

        from max.experimental.tensor import Tensor
        from max.experimental.nn import Module, module_dataclass

        @module_dataclass
        class Linear(Module):
            weight: Tensor
            bias: Tensor | int = 0

            def forward(self, x: Tensor) -> Tensor:
                return x @ self.weight.T + self.bias

        linear = Linear(Tensor.zeros([5, 4]))
        print(linear)
        print(linear(Tensor([1, 2, 3, 4])))

    **Device placement:**

    MAX uses a compiled graph model that separates *weight storage* from
    *computation placement*. Understanding this distinction is essential for
    running models on GPU.

    :meth:`to` is the single pre-compilation entry point for device placement.
    It moves all weight tensors to the target device and records it on the
    module via the :attr:`device` property. :meth:`input_types` implementations
    should reference ``self.device`` when constructing
    :obj:`~max.graph.TensorType` objects, so a single ``to()`` call drives
    both weight placement and computation placement:

    .. code-block:: python

        from max.driver import Accelerator
        from max.experimental.nn import Linear

        model = Linear(10, 5)
        model.to(Accelerator())                       # sets device, moves weights
        compiled = model.compile(*model.input_types())  # computation runs on GPU

    For CPU (the default), calling ``to()`` is optional. The :attr:`device`
    property defaults to :obj:`~max.driver.CPU`:

    .. code-block:: python

        model = Linear(10, 5)
        compiled = model.compile(*model.input_types())  # runs on CPU

    Because :attr:`device` is tracked per-module instance, sub-modules can be
    placed on different devices independently:

    .. code-block:: python

        encoder.to(Accelerator(0))
        decoder.to(Accelerator(1))

    For graph-level tensor routing *inside* ``forward()`` (e.g., pulling an
    activation back to CPU at the end of the graph), use
    :func:`~max.graph.ops.transfer_to` or :meth:`~max.graph.TensorValue.to`
    instead; those insert transfer nodes into the compiled graph and are
    unrelated to pre-compilation weight placement.

    .. list-table::
       :header-rows: 1
       :widths: 30 25 45

       * - API
         - When it runs
         - What it moves
       * - ``Module.to(device)``
         - Python host, before ``compile()``
         - Stored weight tensors; records ``module.device``
       * - ``ops.transfer_to(x, d)`` / ``TensorValue.to(d)``
         - Graph execution time (inside ``forward()``)
         - Activation tensors within the compiled graph
       * - ``Tensor.to(device)``
         - Eager runtime (outside a graph)
         - Concrete eager tensors (e.g., staging inputs)
    """

    def forward(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Defines the computation performed by the module.

        Users must override this method in their subclass to define the
        module's computation.

        Args:
            *args: Positional arguments for the computation.
            **kwargs: Keyword arguments for the computation.

        Returns:
            The result of applying the module to the input.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Applies the module to the input by calling ``forward``.

        This method wraps the user-defined ``forward`` method, following
        PyTorch's convention. Users should override ``forward`` to define
        their module's computation.

        Args:
            *args: The arguments to pass to ``forward``.
            **kwargs: The keyword arguments to pass to ``forward``.

        Returns:
            The result of applying the module to the input.
        """
        return self.forward(*args, **kwargs)

    @property
    def local_parameters(self) -> Iterable[tuple[str, Tensor]]:
        """Iterates over the local parameters of the ``Module``.

        Yields:
            ``(name, tensor)`` pairs, where ``name`` is the attribute name of
            the tensor on the module.
        """
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                yield name, value

    @property
    def parameters(self) -> Iterable[tuple[str, Tensor]]:
        """Iterates over all parameters in this module and its sub-modules.

        This property performs a depth-first traversal of the module hierarchy,
        yielding each parameter tensor with its qualified name. The qualified name
        uses dot-notation to represent the module tree structure (e.g.,
        ``encoder.layer1.weight``).

        Parameters are yielded in depth-first order: first the current module's
        direct parameters, then recursively each sub-module's parameters.

        Counting total parameters:

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.experimental.nn import Module, module_dataclass
            from max.experimental.nn import Linear

            @module_dataclass
            class MLP(Module):
                fc1: Linear
                fc2: Linear

                def forward(self, x: Tensor) -> Tensor:
                    return self.fc2(self.fc1(x))

            model = MLP(
                fc1=Linear(10, 20),
                fc2=Linear(20, 5)
            )

            # Count parameters
            total_params = sum(
                param.num_elements()
                for name, param in model.parameters
            )
            print(f"Total parameters: {total_params}")

        Yields:
            ``(name, parameter)`` tuples where ``name`` is the
            dot-separated qualified path of the parameter and ``parameter``
            is the :class:`~max.experimental.tensor.Tensor`.
        """
        yield from self.local_parameters
        for prefix, descendant in self.descendants:
            for name, parameter in descendant.local_parameters:
                yield f"{prefix}.{name}", parameter

    @property
    def children(self) -> Iterable[tuple[str, Module[..., Any]]]:
        """Iterates over the direct child modules of the ``Module``.

        Yields:
            ``(name, module)`` pairs, where ``name`` is the attribute name of
            the child on the module.
        """
        for name, value in vars(self).items():
            if isinstance(value, Module):
                yield name, value

    @property
    def descendants(self) -> Iterable[tuple[str, Module[..., Any]]]:
        """Iterates over the ``Module``'s descendant modules.

        Yields:
            ``(name, module)`` pairs, where ``name`` is the qualified path
            of the descendant with respect to the module.
        """
        for prefix, child in self.children:
            yield prefix, child
            for name, descendant in child.descendants:
                yield f"{prefix}.{name}", descendant

    def apply_to_local_parameters(
        self, f: Callable[[str, Tensor], Tensor]
    ) -> None:
        """Applies a transformation to each local parameter tensor on the ``Module``.

        The transformation is applied in-place, updating the module's values.
        It will not be applied to descendant's parameters.

        For example:

        .. code-block:: python

            from max.driver import Accelerator
            from max.experimental.nn import Linear

            model = Linear(2, 3)
            model.apply_to_parameters(lambda _, t: t.to(Accelerator()))

        Args:
            f: The transformation to apply to each local parameter.
                The transformation takes two arguments, a name and a tensor:

                - The name is the attribute name of the parameter on the module.
                - The tensor is the current value of that parameter.

                The return value of this function is the new value that will
                replace the value at that name.
        """
        for name, attr in self.local_parameters:
            setattr(self, name, f(name, attr))

    def apply_to_parameters(self, f: Callable[[str, Tensor], Tensor]) -> None:
        """Applies a transformation to all parameters in the module hierarchy.

        This method traverses the module tree and applies the transformation function
        to each parameter in-place, updating both the current module's parameters
        and all nested sub-module parameters. The transformation receives the
        parameter's qualified name (dot-separated path) and current tensor value.

        Transfer all parameters to accelerator:

        .. code-block:: python

            from max.driver import Accelerator
            from max.experimental.tensor import Tensor
            from max.experimental.nn import Module, module_dataclass, Linear

            @module_dataclass
            class MLP(Module):
                fc1: Linear
                fc2: Linear

                def forward(self, x: Tensor) -> Tensor:
                    return self.fc2(self.fc1(x))

            model = MLP(
                fc1=Linear(10, 20),
                fc2=Linear(20, 5)
            )

            # Move all parameters to accelerator
            model.apply_to_parameters(lambda name, t: t.to(Accelerator()))

        Args:
            f: Transformation function taking ``(name, tensor)`` and returning
                the transformed tensor. Parameters:

                - ``name`` (:obj:`str`): Qualified dot-separated path of the parameter
                  (e.g., ``fc1.weight``, ``encoder.layer2.bias``)
                - ``tensor`` (:class:`~max.experimental.tensor.Tensor`): Current value of the parameter

                Returns the new tensor value to replace the parameter.
        """
        self.apply_to_local_parameters(f)
        for prefix, child in self.children:
            # Bind an explicit reference to `prefix` into the closure
            # See https://stackoverflow.com/a/54289183
            child.apply_to_parameters(
                functools.partial(
                    (lambda prefix, name, t: f(f"{prefix}.{name}", t)),
                    prefix,
                )
            )

    def load_state(  # noqa: ANN201
        self, lookup: Callable[[str, Tensor], DLPackArray]
    ):
        """Replaces each parameter in the module and its descendants.

        The transformation is applied in-place, updating the module's values
        and those of its descendants.

        For example, if we have a model with two parameters, ``weight`` and
        ``bias``, we can load the state of the model from a dictionary with the
        following code:

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.experimental.nn import Linear

            model = Linear(2, 3)
            weights = {
                "weight": Tensor.zeros([3, 2]),
                "bias": Tensor.zeros([3]),
            }
            model.load_state(lambda name, _: weights[name])

        The lookup is defined as a function rather than a dictionary, allowing
        for functional remapping of names during this process to account
        for differences in common weight naming and storage conventions.

        For instance, certain representations may not store weights as
        transposed, or may need to be quantized, or split out from a shared
        qkv block, or may just have slightly different names or paths.

        This can also be used for instance to provide a default value for
        initializing LoRA weights.

        Args:
            lookup: The lookup function for each parameter:

                - The first argument is the qualified name of the parameter
                  with respect to the module on which ``load_state()`` was
                  called.
                - The second argument is the existing tensor value.
                - The return value of this function is the new value that will
                  replace the value at that name in the module tree.
        """
        return self.apply_to_parameters(
            lambda name, existing: Tensor.from_dlpack(lookup(name, existing))
        )

    def load_state_dict(
        self, state: Mapping[str, DLPackArray], strict: bool = True
    ) -> None:
        """Loads parameter values from a dictionary into the module hierarchy.

        This method updates all module parameters in-place by loading values from
        the provided state dictionary. The dictionary maps qualified parameter names
        (dot-separated paths like ``fc1.weight``) to tensor values.

        The ``strict`` mode (default) ensures all weights in the dictionary are
        actually used, catching errors from mismatched architectures or incorrect
        weight names.

        For example, the following loads weights from a dictionary into a model:

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.experimental.nn import Module, module_dataclass

            @module_dataclass
            class Linear(Module):
                weight: Tensor
                bias: Tensor

                def forward(self, x: Tensor) -> Tensor:
                    return x @ self.weight.T + self.bias

            model = Linear(
                weight=Tensor.zeros([10, 5]),
                bias=Tensor.zeros([10])
            )

            # Load weights from dictionary
            weights = {
                "weight": Tensor.zeros([10, 5]),
                "bias": Tensor.zeros([10]),
            }
            model.load_state(lambda name, _: weights[name])

        Args:
            state: Dictionary mapping qualified parameter names to tensor values.
                Keys should match the names from :attr:`Module.parameters` property.
                Values should be DLPack-compatible arrays or :class:`~max.experimental.tensor.Tensor` objects.
                Their shapes and dtypes must match the existing parameters with the
                corresponding name, but they may be on a different device. In the
                case that the new value has a different device, it will be copied to
                the same device as the existing value, and the parameter will be set
                to the new copy.
            strict: If :obj:`True` (default), verify that all keys in ``state``
                are used (i.e., match actual parameters). If :obj:`False`, silently
                ignore extra keys that don't match any parameters.

        Raises:
            ValueError: If ``strict=True`` and some weights in ``state`` don't
                match any model parameters (indicates architecture mismatch or
                incorrect weight names).
            ValueError: If a loaded tensor has a different dtype or shape than
                the existing parameter.
            KeyError: If a required parameter name in the model is missing from
                ``state`` (regardless of ``strict`` setting).
        """
        loaded = set()

        def lookup(name: str, existing: Tensor) -> Tensor:
            loaded.add(name)
            return _prepare_weight_for_parameter(name, state[name], existing)

        self.apply_to_parameters(lookup)

        if strict and (unloaded := state.keys() - loaded):
            raise ValueError(
                f"load_state_dict did not use some weights: {unloaded}"
            )

    def map_parameters(self, f: Callable[[str, Tensor], Tensor]) -> Self:
        """Creates a new ``Module`` with its parameters transformed by the function.

        The transformation is functional rather than in-place. The module is
        deep-copied; its descendants are also replaced via the same transform
        without affecting the original module.

        For example:

        .. code-block:: python

            from max.driver import Accelerator
            from max.experimental.nn import Linear

            model = Linear(2, 3)
            model_on_gpu = model.map_parameters(lambda _, t: t.to(Accelerator()))

        Args:
            f: The transformation to apply to each parameter.
                The transformation takes two arguments, a name and a tensor:

                - The name is the qualified name of the parameter
                  with respect to the module on which ``map_parameters()``
                  was called.
                - The tensor is the current value of that parameter.

                The return value of this function is the new value that will
                replace the value at that name in the module tree.

        Returns:
            A new module tree of the same type resulting from mapping the
            transformation over all model parameters.
        """
        new = copy.deepcopy(self)
        new.apply_to_parameters(f)
        return new

    @property
    def device(self) -> Device:
        """The canonical device for this module's weights and computation.

        Set by calling :meth:`to` or by assigning ``self.device`` in a
        subclass ``__init__``. When neither has been called the property
        returns :obj:`~max.driver.CPU` as a safe default so that modules
        without an explicit device placement still compile and run on CPU.
        :meth:`input_types` implementations should reference ``self.device``
        when constructing :obj:`~max.graph.TensorType` objects so that a
        single :meth:`to` call drives both weight placement and computation
        placement.

        .. code-block:: python

            from max.driver import Accelerator
            from max.experimental.nn import Linear

            model = Linear(2, 3)
            print(model.device)     # CPU()  - CPU default
            model.to(Accelerator())
            print(model.device)     # Accelerator(id=0)

        Returns:
            The device this module is placed on, defaulting to
            :obj:`~max.driver.CPU` if :meth:`to` has not been called and
            ``self.device`` has not been set in a subclass ``__init__``.
        """
        device = getattr(self, "_module_target_device", None)
        return device if device is not None else CPU()

    @device.setter
    def device(self, value: Device | DeviceRef | None) -> None:
        """Sets the device for this module.

        Args:
            value: The device to assign to this module. Accepts a
                :class:`~max.driver.Device`, a :class:`~max.graph.DeviceRef`,
                or ``None`` to clear the explicit device assignment.
        """
        if isinstance(value, DeviceRef):
            value = value.to_device()
        object.__setattr__(self, "_module_target_device", value)

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Transfers all module parameters to a device, mesh, or mapping.

        See :meth:`~max.experimental.Tensor.to` for details about using
        ``Device`` vs ``DeviceMesh`` vs ``DeviceMapping``.

        This is the single entry point for device placement. After calling
        ``to(device)``, both weight storage and :meth:`input_types` reflect the
        target device, so ``compile(*self.input_types())`` works correctly
        without any additional device configuration:

        .. code-block:: python

            from max.driver import Accelerator
            from max.experimental.nn import Linear
            from max.graph import TensorType
            from max.dtype import DType

            model = Linear(2, 3)
            model.to(Accelerator())

            # input_types() uses self.device, so computation runs on GPU:
            compiled = model.compile(*model.input_types())

        Unlike PyTorch's eager mode where weights and computation are
        inseparable, MAX uses a compiled graph model. ``to()`` handles the
        weight side; :meth:`input_types` implementations use ``self.device``
        to handle the computation side. Together they form one coherent
        mechanism.

        For graph-level tensor routing at execution time (inside
        :meth:`forward`), use :func:`~max.graph.ops.transfer_to` or
        :meth:`~max.graph.TensorValue.to` instead; those insert transfer ops
        into the compiled graph and are unrelated to pre-compilation device
        placement.

        Args:
            target: The target for all module parameters. Can be:

                - :class:`~max.driver.Device`: Target device for transfer.
                - :class:`~max.experimental.sharding.DeviceMesh`: New mesh,
                  keeping existing placements (or fully replicated for
                  unsharded parameters).
                - :class:`~max.experimental.sharding.DeviceMapping`: New mesh
                  and placements; triggers shard collective for multi-device.

        Returns:
            A reference to the model. The transfer is applied mutably; the
            module's :attr:`device` property and all internal parameters are
            updated in place.
        """
        # Determine the primary device for the module's device property
        if isinstance(target, Device):
            primary_device = target
        elif isinstance(target, DeviceMapping):
            primary_device = target.mesh.devices[0]
        elif isinstance(target, DeviceMesh):
            primary_device = target.devices[0]
        else:
            raise TypeError(
                f"to() expects Device, DeviceMesh, or DeviceMapping, "
                f"got {type(target).__name__}"
            )

        object.__setattr__(self, "_module_target_device", primary_device)
        pinned = _get_pinned_device_fields(type(self))
        for name, attr in self.local_parameters:
            if name not in pinned:
                setattr(self, name, attr.to(target))
        for _, child in self.children:
            child.to(target)
        return self

    @contextlib.contextmanager
    def _mapped_parameters(self, f: Callable[[str, Tensor], Tensor]):  # noqa: ANN202
        parameters = dict(self.parameters)
        try:
            self.apply_to_parameters(f)
            yield parameters
        finally:
            self.load_state_dict(parameters)

    def _trace(
        self,
        input_types: Sequence[InputType],
        *,
        custom_extensions: Iterable[Path] = (),
    ) -> tuple[
        Graph,
        list[_InputSlot],
        list[_OutputSlot],
        bool,
        Signals | None,
    ]:
        """Shared tracing core used by :meth:`trace` and :meth:`compile`.

        Builds a :class:`~max.graph.Graph` by symbolically executing
        ``forward`` with the given input types.  Returns the graph plus
        bookkeeping needed by :meth:`compile`.
        """
        # Expand distributed input types into per-device local types.
        # For pure single-device usage this is a no-op pass-through.
        graph_types, input_slots = _flatten_input_types(input_types)

        # Detect multi-GPU distributed inputs and create signal buffers
        # so that collective ops (all_reduce_sum, all_gather, etc.) use
        # hardware-accelerated multi-device communication.
        signals = _detect_signals(input_types, parameters=self.parameters)
        if signals is not None:
            graph_types.extend(signals.input_types())

        graph = Graph(
            type(self).__qualname__,
            input_types=graph_types,
            custom_extensions=custom_extensions,
        )

        # Extract signal BufferValues from the graph inputs (at the end).
        sig_buf_values = None
        if signals is not None:
            n_sig = len(signals.devices)
            sig_buf_values = [
                graph.inputs[len(graph_types) - n_sig + i].buffer
                for i in range(n_sig)
            ]

        ctx = GraphRealizationContext(graph, signal_buffers=sig_buf_values)
        with realization_context(ctx), ctx:
            # Only wrap tensor inputs, not signal buffer inputs.
            n_tensor_inputs = len(graph_types) - (
                len(sig_buf_values) if sig_buf_values else 0
            )
            inputs = _wrap_graph_inputs(
                list(graph.inputs[:n_tensor_inputs]), input_slots
            )

            def as_weight(name: str, tensor: Tensor):  # noqa: ANN202
                return tensor._as_constant_external(name)

            # Temporarily replace the parameters with external constants
            # while building the graph.
            with self._mapped_parameters(as_weight):
                outputs: Tensor | Sequence[Tensor] = self(*inputs)  # type: ignore[call-arg,assignment,arg-type]

            # Flatten sharded outputs into per-shard graph values.
            flat_values, output_slots, unary = _flatten_outputs(outputs)
            graph.output(*flat_values)

        return graph, input_slots, output_slots, unary, signals

    def trace(
        self,
        *input_types: InputType,
        custom_extensions: Iterable[Path] = (),
    ) -> Graph:
        """Traces the module's forward pass into a :class:`~max.graph.Graph`.

        Like :meth:`compile`, but returns the raw graph without compiling
        it.  Useful for inspecting the IR, debugging sharding propagation,
        or feeding into a custom compilation pipeline.

        Args:
            *input_types: Type specifications for each input to ``forward``.
            custom_extensions: Paths to custom Mojo kernel libraries.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """
        graph, *_ = self._trace(
            input_types, custom_extensions=custom_extensions
        )
        return graph

    def compile(
        self,
        *input_types: InputType,
        weights: Mapping[str, DLPackArray] | None = None,
        custom_extensions: Iterable[Path] = (),
    ) -> CompiledModel:
        """Compiles the module to an optimized executable through graph tracing.

        This method performs symbolic tracing of the module's ``forward`` method
        to construct a MAX :class:`~max.graph.Graph`, which is then compiled and optimized for
        efficient execution on CPU, GPU, or other accelerators.

        The compilation process:

        1. Creates symbolic :class:`~max.experimental.tensor.Tensor` instances based on provided type specifications
        2. Executes ``forward`` with symbolic tensors to record operations
        3. Constructs a :class:`~max.graph.Graph` representing the computation
        4. Includes all module parameters as weights in the graph
        5. Compiles and optimizes the graph for target hardware
        6. Returns an executable function with the same signature as ``forward``

        The input type specifications must match the signature of ``forward``.
        Use positional arguments for positional parameters.

        **Device placement:** The canonical pattern is to call :meth:`to`
        before ``compile``. :meth:`to` sets :attr:`device`, moves all weights
        to that device, and causes :meth:`input_types` to return
        :obj:`~max.graph.TensorType` objects annotated with that device. This
        means a single :meth:`to` call drives both weight placement and
        computation placement:

        .. code-block:: python

            from max.driver import Accelerator
            from max.experimental.nn import Linear

            model = Linear(10, 5)
            model.to(Accelerator())  # sets device, moves weights to GPU

            # input_types() uses self.device — computation runs on GPU:
            compiled = model.compile(*model.input_types())

        Basic compilation with fixed shapes:

        .. code-block:: python

            from max.dtype import DType
            from max.experimental.tensor import Tensor, TensorType, defaults
            from max.experimental.nn import Module, module_dataclass

            @module_dataclass
            class Linear(Module):
                weight: Tensor
                bias: Tensor

                def forward(self, x: Tensor) -> Tensor:
                    return x @ self.weight.T + self.bias

            linear = Linear(
                weight=Tensor.zeros([10, 5]),
                bias=Tensor.zeros([10])
            )

            # Compile with fixed input shape
            _, device = defaults()
            input_type = TensorType(DType.float32, [3, 5], device=device)
            model = linear.compile(input_type)

            # Execute compiled model
            input_data = Tensor.ones([3, 5], dtype=DType.float32)
            result = model(input_data)
            print(result)

        Compilation with custom Mojo kernel extensions:

        .. code-block:: python

            from pathlib import Path
            from max.experimental import functional as F

            @module_dataclass
            class CustomModule(Module):
                def forward(self, x: Tensor) -> Tensor:
                    return F.custom(
                        "my_op", device=x.device,
                        values=[x], out_types=[x.type],
                    )[0]

            module = CustomModule()
            compiled = module.compile(
                input_type,
                custom_extensions=[Path("my_ops.mojoc")],
            )

        Args:
            *input_types: Type specifications for each positional argument to
                ``forward``. Must match the number and order of arguments.
                Each should be a :class:`~max.graph.Type` (typically
                :class:`~max.graph.TensorType`) describing the shape and dtype. The
                ``device`` field on each :obj:`~max.graph.TensorType`
                determines where activations are computed; use :meth:`to` to
                set this consistently across weights and inputs.
            weights: Mapping of parameter names to weight data. Weights should
                be on CPU and will be transferred to the target device as part
                of model initialization. If not passed, the model's parameters
                will be used as the weights.
            custom_extensions: Paths to custom Mojo kernel libraries
                (``.mojoc`` files or Mojo source directories) to load into
                the graph before tracing. Required when ``forward`` uses
                :func:`~max.experimental.functional.custom` or
                :func:`~max.experimental.functional.inplace_custom` with
                custom kernels, so that kernel signatures are available for
                validation during graph construction.

        Returns:
            Callable[..., Any]
                A compiled executable function with the same signature as
                ``forward``. This function runs the optimized graph and
                returns results with the same structure as ``forward``
                (single :class:`~max.experimental.tensor.Tensor` or tuple of tensors).

        Raises:
            TypeError: If input types don't match ``forward`` signature or if
                operations in ``forward`` cannot be traced.
            RuntimeError: If graph construction fails due to incompatible
                operations or parameter access issues.
        """
        compile_name = type(self).__name__
        with (
            Tracer(f"Module.compile({compile_name})"),
            CompilationTimer(compile_name) as timer,
        ):
            with Tracer("Module.compile.trace"):
                graph, input_slots, output_slots, unary, signals = self._trace(
                    input_types, custom_extensions=custom_extensions
                )

            with Tracer("Module.compile.weights_registry"):
                # Compile the graph with module parameters as weights
                session = _session()

                # Build weights registry from parameters.
                if weights is None:
                    weights_registry = _flatten_named_buffers(self.parameters)
                else:
                    weights_registry = _process_provided_weights(
                        weights, self.parameters
                    )

            timer.mark_build_complete()
            with Tracer("Module.compile.session_load"):
                session_model = session.load(
                    graph, weights_registry=weights_registry
                )

            with Tracer("Module.compile.finalize"):
                # Allocate signal buffers once for all future invocations.
                cached_sig_bufs = (
                    signals.buffers() if signals is not None else []
                )

                return CompiledModel(
                    engine_model=session_model,
                    input_slots=input_slots,
                    output_slots=output_slots,
                    signal_buffers=cached_sig_bufs,
                    unary=unary,
                )

    def __rich_repr__(self):
        yield from self.children

    def __repr__(self):
        """Returns a string representation of the module's structure.

        The representation displays the module's class name, all sub-modules with
        their types (nested with indentation), and parameter information (name,
        shape). The format mirrors the module's hierarchical composition, making it
        easy to understand the model architecture at a glance.

        The following is an example output for a module:

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.experimental.nn import Module, module_dataclass

            @module_dataclass
            class Linear(Module):
                weight: Tensor
                bias: Tensor

                def forward(self, x: Tensor) -> Tensor:
                    return x @ self.weight.T + self.bias

            layer = Linear(
                weight=Tensor.zeros([128, 64]),
                bias=Tensor.zeros([128])
            )

        Returns:
            str
                Multi-line string representation showing the module's class name,
                all sub-modules (recursively indented), and parameters with their
                specifications.
        """
        return pretty_repr(self)


def _module_dataclass_rich_repr(self: DataclassInstance):  # noqa: ANN202
    for field in dataclasses.fields(self):
        value = getattr(self, field.name)
        if isinstance(value, Tensor):
            # Rich will try to == compare the value with the default.
            # Avoid this by never passing a default value for tensors.
            yield field.name, value
        else:
            yield field.name, value, field.default


@dataclass_transform()
def module_dataclass(  # noqa: ANN201
    cls: type[Module[..., Any]] | None = None,
    /,
    *,
    repr: bool = False,
    **kwargs,
):
    """Converts a class into a MAX module with automatic parameter tracking.

    This decorator enables a regular Python class to function as a :class:`Module`,
    providing automatic discovery and registration of parameters (Tensor fields)
    and nested modules. The decorated class gains all capabilities of :class:`Module`,
    including parameter iteration, graph compilation via :meth:`Module.compile`,
    and hierarchical module composition.

    The decorator applies Python's ``@dataclass`` decorator internally while
    preserving :class:`Module`'s specialized ``__repr__`` method for better
    debugging experience when printing module structures.

    .. code-block:: python

        from max.experimental.nn import Module, Linear, module_dataclass
        from max.experimental.tensor import Tensor
        from max.experimental import functional as F

        @module_dataclass
        class MLP(Module):
            fc1: Linear
            fc2: Linear

            def forward(self, x: Tensor) -> Tensor:
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                return x

        # Create module with automatic parameter tracking
        mlp = MLP(
            fc1=Linear(128, 256),
            fc2=Linear(256, 128)
        )

        # All parameters are automatically tracked
        print(dict(mlp.parameters).keys())
        # {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}

        # Use the module
        x = Tensor.randn([4, 128])
        output = mlp(x)
        print(output.shape)  # (4, 128)

    Args:
        cls: The class to decorate. Must define a ``forward`` method.
            When :obj:`None`, returns a decorator function (supports
            using ``@module_dataclass`` with or without parentheses).
        repr: If :obj:`True`, use dataclass's default ``__repr__`` instead of
            :class:`Module`'s rich representation. Defaults to :obj:`False`.
        **kwargs: Additional keyword arguments forwarded to Python's
            ``@dataclass`` decorator (e.g., ``frozen``, ``eq``).

    Returns:
        The decorated class as a :class:`Module` subclass with automatic parameter
        tracking and graph compilation capabilities. When ``cls`` is :obj:`None`,
        returns a decorator function.
    """
    dataclass_decorator = dataclasses.dataclass(repr=repr, **kwargs)

    def decorator(cls: type[Module[..., Any]]) -> type[Module[..., Any]]:
        decorated = dataclass_decorator(cls)
        if cls.__rich_repr__ is Module.__rich_repr__:
            decorated.__rich_repr__ = _module_dataclass_rich_repr  # type: ignore
        return decorated

    return decorator(cls) if cls else decorator
