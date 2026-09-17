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

import dataclasses
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Generic

from max import tree
from max.driver import CPU, Device, DLPackArray
from max.experimental import compilation
from max.experimental.nn._compilation_timer import CompilationTimer
from max.experimental.nn._weight_utils import (
    _ExternalWeight,
    load_checkpoint,
    name_weights,
    shard_checkpoint,
)
from max.experimental.sharding import DeviceMapping, DeviceMesh, TensorLayout
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Type
from rich.pretty import pretty_repr
from typing_extensions import ParamSpec, Self, TypeVar, dataclass_transform

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# Type variables for Module's forward signature.
_P = ParamSpec("_P")
_R = TypeVar("_R")

#: The type of one tensor argument of ``forward``.
InputType = Type[Any] | TensorLayout

__all__ = [
    "Module",
    "module_dataclass",
]
from max.profiler import Tracer


class _DevicePinned:
    """Sentinel marker for parameters whose device should not be changed.

    Used as annotation metadata in `PinnedDeviceTensor`. Do not use directly;
    annotate fields with `PinnedDeviceTensor` instead.
    """


@dataclasses.dataclass(frozen=True)
class _TransparentKey:
    """The tree key of a child with :attr:`~Module.name_transparent` set.

    It renders as ``""`` so that :func:`~max.tree.extend_path` adds no
    segment to the paths of the child's descendants, but it compares by name
    so that two transparent children remain distinct keys.
    """

    name: str

    def __str__(self) -> str:
        return ""


PinnedDeviceTensor = Annotated[Tensor, _DevicePinned]
"""Type alias for a `Tensor` parameter that `Module.to` will leave on its
current device.

Use this for parameters that must stay on a specific device regardless of where
the rest of the module is moved. For example, scalar quantization scale factors
that GPU kernels consume as host-side launch arguments should remain on CPU;
moving them to the accelerator would force an expensive device sync every
forward pass.
"""


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
    module via the :attr:`device` property. When you construct the
    :obj:`~max.graph.TensorType` objects you pass to :meth:`compile`, reference
    ``model.device`` for their ``device`` field, so a single ``to()`` call
    drives both weight placement and computation placement:

    .. code-block:: python

        from max.dtype import DType
        from max.experimental.nn import Linear
        from max.experimental.tensor import Tensor, defaults
        from max.graph import TensorType

        model = Linear(5, 10)
        _, device = defaults()
        model.to(device)

        input_type = TensorType(DType.float32, ["batch", 5], device=model.device)
        compiled = model.compile(input_type)
        result = compiled(Tensor.ones([3, 5], dtype=DType.float32))

    .. invisible-code-block: python

        assert list(result.shape) == [3, 10]

    For CPU (the default), calling ``to()`` is optional. The :attr:`device`
    property defaults to :obj:`~max.driver.CPU`:

    .. code-block:: python

        from max.driver import CPU
        from max.experimental.nn import Linear

        model = Linear(5, 10)
        print(model.device)

    .. invisible-code-block: python

        from max.driver import CPU

        assert isinstance(model.device, CPU)

    Because :attr:`device` is tracked per-module instance, sub-modules can be
    placed on different devices independently. Here two :class:`Linear`
    sub-modules are placed on separate CPU device references (use distinct
    :obj:`~max.driver.Accelerator` instances when accelerators are available):

    .. code-block:: python

        from max.driver import CPU
        from max.experimental.nn import Linear

        encoder = Linear(5, 8)
        decoder = Linear(8, 4)

        encoder.to(CPU(0))
        decoder.to(CPU(0))

    .. invisible-code-block: python

        assert isinstance(encoder.device, CPU)
        assert isinstance(decoder.device, CPU)

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

    #: Whether this module's own name is dropped from its descendants' paths.
    name_transparent: bool = False

    def __tree_flatten__(self) -> tuple[dict[Any, Any], None]:
        """Returns the attributes of this module as its children.

        Children are keyed by attribute name. A child with
        :attr:`name_transparent` set is keyed by a ``_TransparentKey``
        instead, which omits it from the paths of its descendants.
        """
        return {
            _TransparentKey(name)
            if isinstance(value, Module) and value.name_transparent
            else name: value
            for name, value in vars(self).items()
        }, None

    @classmethod
    def __tree_empty__(cls, meta: None) -> Module[..., Any]:
        """Creates an empty module whose children are set afterwards."""
        del meta
        return object.__new__(cls)

    def __tree_setattr__(self, key: Any, value: Any) -> None:
        """Sets the child ``key`` to ``value``."""
        # ``object.__setattr__`` also works for a frozen ``module_dataclass``.
        object.__setattr__(
            self, key.name if isinstance(key, _TransparentKey) else key, value
        )

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
            f"{type(self).__name__} must implement forward() "
            "(or, for Modules with multiple entry points, expose "
            "explicit methods called from a parent Module's forward())."
        )

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Calls :meth:`forward`.

        To share one subgraph between several modules, wrap the call with
        :func:`~max.experimental.compilation.as_subgraph`.

        Args:
            *args: The arguments to pass to ``forward``.
            **kwargs: The keyword arguments to pass to ``forward``.

        Returns:
            The result of applying the module to the input.
        """
        return self.forward(*args, **kwargs)

    @property
    def local_parameters(self) -> Iterable[tuple[str, Tensor]]:
        """Iterates over the tensors held directly on this ``Module``.

        Yields:
            ``(name, tensor)`` pairs, where ``name`` is the attribute name of
            the tensor on the module.
        """
        children, keys, _ = tree.flatten_one_level(self)
        for key, child in zip(keys, children, strict=True):
            if isinstance(child, Tensor):
                yield (
                    key.name if isinstance(key, _TransparentKey) else key,
                    child,
                )

    @property
    def parameters(self) -> Iterable[tuple[str, Tensor]]:
        """Iterates over all parameters in this module and its sub-modules.

        This property performs a depth-first traversal of the module hierarchy,
        yielding each parameter tensor with its qualified name. The qualified name
        uses dot-notation to represent the module tree structure (e.g.,
        ``encoder.layer1.weight``).

        Parameters are yielded in depth-first order: first the current module's
        direct parameters, then recursively each sub-module's parameters.

        Yields:
            ``(name, parameter)`` tuples where ``name`` is the
            dot-separated qualified path of the parameter and ``parameter``
            is the :class:`~max.experimental.tensor.Tensor`.
        """
        return iter(tree.paths(self, leaf=Tensor).items())

    @property
    def children(self) -> Iterable[tuple[str, Module[..., Any]]]:
        """Iterates over the direct child modules of the ``Module``.

        Yields:
            ``(name, module)`` pairs, where ``name`` is the attribute name of
            the child on the module.
        """
        children, keys, _ = tree.flatten_one_level(self)
        for key, child in zip(keys, children, strict=True):
            if isinstance(child, Module):
                yield (
                    key.name if isinstance(key, _TransparentKey) else key,
                    child,
                )

    @property
    def descendants(self) -> Iterable[tuple[str, Module[..., Any]]]:
        """Iterates over the ``Module``'s descendant modules.

        Yields:
            ``(name, module)`` pairs, where ``name`` is the qualified path
            of the descendant with respect to the module.
        """
        # A name-transparent child is still yielded under its own name. Only
        # the paths of its descendants omit it.
        children, keys, _ = tree.flatten_one_level(self)
        for key, child in zip(keys, children, strict=True):
            if not isinstance(child, Module):
                continue
            yield key.name if isinstance(key, _TransparentKey) else key, child
            base = tree.extend_path("", key)
            for name, descendant in child.descendants:
                yield tree.extend_path(base, name), descendant

    def load_state_dict(
        self,
        state: Mapping[str, DLPackArray],
        strict: bool = True,
        *,
        auto_cast: bool = False,
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

            weights = {
                "weight": Tensor.zeros([10, 5]),
                "bias": Tensor.zeros([10]),
            }
            model.load_state_dict(weights)

        By default (``auto_cast=False``) any dtype mismatch between the loaded
        tensor and the parameter raises. When ``auto_cast=True``, loaded
        weights whose dtype is in the safe-cast set (currently ``float32`` and
        ``bfloat16``) are automatically cast to the parameter's dtype when
        shapes match, and a single summary message is logged at ``WARNING``
        level per call describing how many parameters were cast. Narrowing
        casts (e.g. ``float32`` -> ``bfloat16``) are flagged in the log
        message as ``(precision loss)``. Dtype mismatches outside the
        safe-cast set still raise regardless of ``auto_cast``.

        Args:
            state: Dictionary mapping qualified parameter names to tensor values.
                Keys should match the names from :attr:`Module.parameters` property.
                Values should be DLPack-compatible arrays or :class:`~max.experimental.tensor.Tensor` objects.
                Shapes must match the existing parameters with the corresponding
                name. Dtypes must match exactly *or* both lie in the safe-cast
                set above. Values may be on a different device; in that case the
                tensor is copied to the existing parameter's device.
            strict: If :obj:`True` (default), verify that all keys in ``state``
                are used (i.e., match actual parameters). If :obj:`False`, silently
                ignore extra keys that don't match any parameters.
            auto_cast: If :obj:`True`, permit safe dtype auto-casting between
                ``float32`` and ``bfloat16`` when shapes match. Defaults to
                :obj:`False` — dtype mismatches always raise. Pipelines that
                want to opt in via ``MODULAR_AUTO_CAST_WEIGHTS`` should pass
                ``auto_cast=max.pipelines.lib.weight_loading.auto_cast_weights_from_env()``.

        Raises:
            ValueError: If ``strict=True`` and some weights in ``state`` don't
                match any model parameters (indicates architecture mismatch or
                incorrect weight names).
            ValueError: If a loaded tensor has a different shape than the
                existing parameter, or a dtype mismatch that is not covered by
                the safe-cast set (or ``auto_cast=False``).
            KeyError: If a required parameter name in the model is missing from
                ``state`` (regardless of ``strict`` setting).
        """
        load_checkpoint(self, state, strict=strict, auto_cast=auto_cast)

    @property
    def device(self) -> Device:
        """The canonical device for this module's weights and computation.

        Set by calling :meth:`to` or by assigning ``self.device`` in a
        subclass ``__init__``. When neither has been called the property
        returns :obj:`~max.driver.CPU` as a safe default so that modules
        without an explicit device placement still compile and run on CPU.
        When constructing the :obj:`~max.graph.TensorType` objects you pass to
        :meth:`compile`, reference ``self.device`` for their ``device`` field
        so that a single :meth:`to` call drives both weight placement and
        computation placement.

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

        Records the target as :attr:`device`. Build the input types you pass
        to :meth:`compile` from it, so that the computation runs where the
        parameters are.

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
        if isinstance(target, Device):
            device = target
        elif isinstance(target, DeviceMesh):
            device = target.devices[0]
        elif isinstance(target, DeviceMapping):
            device = target.mesh.devices[0]
        else:
            raise TypeError(
                "to() expects Device, DeviceMesh, or DeviceMapping, "
                f"got {type(target).__name__}"
            )
        # Call ``to`` on each child rather than moving its parameters here,
        # because quantized layers override ``to``.
        object.__setattr__(self, "_module_target_device", device)
        pinned = {
            name
            for base in type(self).__mro__
            for name, annotation in getattr(base, "__annotations__", {}).items()
            if annotation is PinnedDeviceTensor
            or annotation == "PinnedDeviceTensor"
        }
        for name, weight in self.local_parameters:
            if name not in pinned:
                setattr(self, name, weight.to(target))
        for _, child in self.children:
            child.to(target)
        return self

    def trace(
        self,
        *input_types: InputType,
        custom_extensions: Iterable[Path] = (),
        allow_subgraphs: bool = True,
        is_device_graph: bool = False,
    ) -> compilation.StagedGraph[..., Any]:
        """Traces ``forward`` into a graph without compiling it.

        The graph refers to each parameter by name, as an external constant,
        rather than embedding its value. :meth:`compile` supplies the values
        from the weight registry. Use this method to inspect the MLIR that
        :meth:`compile` would compile.

        The following example traces a linear layer and prints its MLIR:

        .. code-block:: python

            from max.driver import CPU
            from max.dtype import DType
            from max.experimental.nn import Linear
            from max.experimental.tensor import TensorType, default_device

            with default_device(CPU()):
                linear = Linear(4, 8)
                x_type = TensorType(DType.float32, [2, 4], device=linear.device)
                staged = linear.trace(x_type)
                print(staged)

        .. invisible-code-block: python

            assert "mo.constant.external" in str(staged)

        Args:
            *input_types: The type of each tensor argument of ``forward``, as
                a :class:`~max.graph.TensorType` or a
                :class:`~max.experimental.sharding.TensorLayout`.
            custom_extensions: Paths to custom Mojo kernel libraries.
            allow_subgraphs: Whether calls made through
                :func:`~max.experimental.compilation.as_subgraph` are traced
                as subgraphs. If ``False``, they are inlined.
            is_device_graph: Whether to build a device graph.

        Returns:
            The :class:`~max.experimental.compilation.StagedGraph`. Print it
            to see the MLIR.
        """
        declared = tree.map(
            _ExternalWeight.like, self, leaf=Tensor, shared=True
        )
        name_weights(declared)
        return compilation.stage(
            declared,
            name=type(self).__qualname__,
            custom_extensions=custom_extensions,
            allow_subgraphs=allow_subgraphs,
            is_device_graph=is_device_graph,
        )(*input_types)

    def compile(
        self,
        *input_types: InputType,
        weights: Mapping[str, DLPackArray] | None = None,
        custom_extensions: Iterable[Path] = (),
        auto_cast: bool = False,
        allow_subgraphs: bool = True,
        is_device_graph: bool = False,
    ) -> compilation.CompiledCallable[_P, _R]:
        """Traces ``forward`` and compiles it into a callable.

        The graph refers to each parameter by name, and ``weights`` supplies
        the values. Call :meth:`to` before compiling and build the input types
        from :attr:`device`, so that the parameters and the computation are
        on the same device.

        The following example compiles a linear layer and runs it:

        .. code-block:: python

            from max.dtype import DType
            from max.experimental.nn import Module, module_dataclass
            from max.experimental.tensor import Tensor, TensorType, defaults

            @module_dataclass
            class Linear(Module):
                weight: Tensor
                bias: Tensor

                def forward(self, x: Tensor) -> Tensor:
                    return x @ self.weight.T + self.bias

            linear = Linear(Tensor.zeros([10, 5]), Tensor.zeros([10]))
            _, device = defaults()
            run = linear.compile(TensorType(DType.float32, [3, 5], device))
            result = run(Tensor.ones([3, 5], dtype=DType.float32))

        .. invisible-code-block: python

            assert list(result.shape) == [3, 10]

        Args:
            *input_types: The type of each tensor argument of ``forward``, as
                a :class:`~max.graph.TensorType` or a
                :class:`~max.experimental.sharding.TensorLayout`. The device
                of each type determines where the computation runs.
            weights: The values of the parameters, keyed by the names that
                :attr:`parameters` yields. Defaults to the values currently
                held by this module.
            custom_extensions: Paths to custom Mojo kernel libraries, required
                when ``forward`` calls
                :func:`~max.experimental.functional.custom`.
            auto_cast: Whether to permit a safe ``float32``/``bfloat16`` cast
                when shapes match. See :meth:`load_state_dict`.
            allow_subgraphs: Whether calls made through
                :func:`~max.experimental.compilation.as_subgraph` are traced
                as subgraphs. If ``False``, they are inlined.
            is_device_graph: Whether to build a device graph.

        Returns:
            A :class:`~max.experimental.compilation.CompiledCallable` with
            the signature of ``forward``.

        Raises:
            TypeError: If an input type does not match ``forward``'s
                signature, or an operation in ``forward`` cannot be traced.
            ValueError: If a weight does not fit the parameter it loads
                into.
            KeyError: If ``weights`` has no entry for a parameter.
        """
        name = type(self).__name__
        with Tracer(f"Module.compile({name})"), CompilationTimer(name) as timer:
            declared: Module[_P, _R] = tree.map(
                _ExternalWeight.like, self, leaf=Tensor, shared=True
            )
            name_weights(declared)
            # Build the registry against the declared parameters, which record
            # the layout each checkpoint entry arrives in.
            registry = shard_checkpoint(
                declared,
                dict(self.parameters) if weights is None else weights,
                auto_cast=auto_cast,
            )
            staged = compilation.stage(
                declared,
                name=type(self).__qualname__,
                custom_extensions=custom_extensions,
                allow_subgraphs=allow_subgraphs,
                is_device_graph=is_device_graph,
            )(*input_types)
            timer.mark_build_complete()
            return compilation.CompiledCallable(staged, registry)

    def __rich_repr__(self):
        yield from self.children

    def __repr__(self):
        """Returns a string representation of the module's structure.

        The representation displays the module's class name, all sub-modules with
        their types (nested with indentation), and parameter information (name,
        shape). The format mirrors the module's hierarchical composition, making it
        easy to understand the model architecture at a glance.

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
    """Applies :func:`~dataclasses.dataclass` to a :class:`Module` subclass.

    The decorated class gets an ``__init__`` generated from its field
    annotations and keeps the :class:`Module` repr. Parameter tracking does
    not depend on this decorator.

    .. code-block:: python

        from max.driver import CPU
        from max.dtype import DType
        from max.experimental import functional as F
        from max.experimental import random
        from max.experimental.nn import Module, Linear, module_dataclass
        from max.experimental.tensor import Tensor, default_device
        from max.graph import TensorType

        @module_dataclass
        class MLP(Module):
            fc1: Linear
            fc2: Linear

            def forward(self, x: Tensor) -> Tensor:
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                return x

        with default_device(CPU()):
            mlp = MLP(
                fc1=Linear(128, 256),
                fc2=Linear(256, 128)
            )

            print(dict(mlp.parameters).keys())

            random.set_seed(0)
            x = random.normal([4, 128], dtype=DType.float32)
            output = mlp(x)

            input_type = TensorType(DType.float32, ["batch", 128], device=mlp.device)
            compiled = mlp.compile(input_type)
            compiled_output = compiled(x)

    .. invisible-code-block: python

        import numpy as np

        assert set(dict(mlp.parameters).keys()) == {
            "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"
        }
        assert list(output.shape) == [4, 128]
        assert list(compiled_output.shape) == [4, 128]
        assert np.allclose(output.to_numpy(), compiled_output.to_numpy(), atol=1e-4)

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
