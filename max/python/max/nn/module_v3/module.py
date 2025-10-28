# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from typing import TYPE_CHECKING, Any

from rich.pretty import pretty_repr
from typing_extensions import Self, dataclass_transform

from ... import graph
from ...driver import Device, DLPackArray
from ...experimental import functional as F
from ...experimental.tensor import Tensor, _session
from ...graph import Graph

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class Module:
    """The core unit of composition for modeling in MAX.

    Informally, a ``Module`` is a container class. It can contain
    other ``Module`` instances, tensors (the ``Module``'s "local parameters")
    or other arbitrary Python data.

    A ``Module`` also has a ``__call__()`` which applies that ``Module`` to
    some input. In the simplest case this is a function from one tensor
    to another tensor.

    Formally modules form a tree, and subtrees of modules can be manipulated
    directly. A ``Module`` may also be thought of as a closure, where the parameters
    form the data of the closure and ``__call__()`` is the application of the closure.

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
        from max.nn.module_v3 import Module, module_dataclass

        @module_dataclass
        class Linear(Module):
            weight: Tensor
            bias: Tensor | int = 0

            def __call__(self, x: Tensor) -> Tensor:
                return x @ self.weight.T + self.bias

        linear = Linear(Tensor.zeros([5, 4]))
        print(linear)
        print(linear(Tensor.constant([1, 2, 3, 4])))
    """

    __call__: Callable[..., Any]

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
        ``"encoder.layer1.weight"``).

        Parameters are yielded in depth-first order: first the current module's
        direct parameters, then recursively each sub-module's parameters.

        Counting total parameters:

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.nn.module_v3 import Module, module_dataclass
            from max.nn.module_v3 import Linear

            @module_dataclass
            class MLP(Module):
                fc1: Linear
                fc2: Linear

                def __call__(self, x: Tensor) -> Tensor:
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
            is the :obj:`Tensor`.
        """
        yield from self.local_parameters
        for prefix, descendant in self.descendants:
            for name, parameter in descendant.local_parameters:
                yield f"{prefix}.{name}", parameter

    @property
    def children(self) -> Iterable[tuple[str, Module]]:
        """Iterates over the direct child modules of the ``Module``.

        Yields:
            ``(name, module)`` pairs, where ``name`` is the attribute name of
            the child on the module.
        """
        for name, value in vars(self).items():
            if isinstance(value, Module):
                yield name, value

    @property
    def descendants(self) -> Iterable[tuple[str, Module]]:
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
            from max.nn.module_v3 import Linear

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
            from max.nn.module_v3 import Module, module_dataclass, Linear

            @module_dataclass
            class MLP(Module):
                fc1: Linear
                fc2: Linear

                def __call__(self, x: Tensor) -> Tensor:
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
                  (e.g., ``"fc1.weight"``, ``"encoder.layer2.bias"``)
                - ``tensor`` (:obj:`Tensor`): Current value of the parameter

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

    def load_state(self, lookup: Callable[[str], DLPackArray]):  # noqa: ANN201
        """Replaces each parameter in the module and its descendants.

        The transformation is applied in-place, updating the module's values
        and those of its descendants.

        For example, if we have a model with two parameters, ``weight`` and
        ``bias``, we can load the state of the model from a dictionary with the
        following code:

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.nn.module_v3 import Linear

            model = Linear(2, 3)
            weights = {
                "weight": Tensor.zeros([3, 2]),
                "bias": Tensor.zeros([3]),
            }
            model.load_state(weights.__getitem__)

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

                - The argument to the lookup function is the qualified name
                  of the parameter with respect to the module on which
                  ``load_state()`` was called.
                - The return value of this function is the new value that will
                  replace the value at that name in the module tree.
        """
        return self.apply_to_parameters(
            lambda name, _: Tensor.from_dlpack(lookup(name))
        )

    def load_state_dict(
        self, state: Mapping[str, DLPackArray], strict: bool = True
    ) -> None:
        """Loads parameter values from a dictionary into the module hierarchy.

        This method updates all module parameters in-place by loading values from
        the provided state dictionary. The dictionary maps qualified parameter names
        (dot-separated paths like ``"fc1.weight"``) to tensor values.

        The ``strict`` mode (default) ensures all weights in the dictionary are
        actually used, catching errors from mismatched architectures or incorrect
        weight names.

        For example, the following loads weights from a dictionary into a model:

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.nn.module_v3 import Module, module_dataclass

            @module_dataclass
            class Linear(Module):
                weight: Tensor
                bias: Tensor

                def __call__(self, x: Tensor) -> Tensor:
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
            model.load_state(weights.__getitem__)

        Args:
            state: Dictionary mapping qualified parameter names to tensor values.
                Keys should match the names from :attr:`Module.parameters` property.
                Values should be DLPack-compatible arrays or :obj:`Tensor` objects.
            strict: If :obj:`True` (default), verify that all keys in ``state``
                are used (i.e., match actual parameters). If :obj:`False`, silently
                ignore extra keys that don't match any parameters.

        Raises:
            ValueError: If ``strict=True`` and some weights in ``state`` don't
                match any model parameters (indicates architecture mismatch or
                incorrect weight names).
            KeyError: If a required parameter name in the model is missing from
                ``state`` (regardless of ``strict`` setting).
        """
        loaded = set()

        def lookup(name: str) -> DLPackArray:
            loaded.add(name)
            return state[name]

        self.load_state(lookup)

        if strict and (unloaded := state.keys() - loaded):
            raise ValueError(
                f"load_state_dict did not read some weights: {unloaded}"
            )

    def map_parameters(self, f: Callable[[str, Tensor], Tensor]) -> Self:
        """Creates a new ``Module`` with its parameters transformed by the function.

        The transformation is functional rather than in-place. The module is
        deep-copied; its descendants are also replaced via the same transform
        without affecting the original module.

        For example:

        .. code-block:: python

            from max.driver import Accelerator
            from max.nn.module_v3 import Linear

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

    def to(self, device: Device) -> Self:
        """Updates the module's parameters, transferring them to the specified device.

        .. code-block:: python

            from max.driver import CPU
            from max.nn.module_v3 import Linear

            model = Linear(2, 3)
            model.to(CPU())

        Args:
            device: The device to which all model parameters will be transferred.

        Returns:
            A reference to the model. The transfer is applied mutably; internal
            parameters are updated to be transferred to the specified device.
        """
        self.apply_to_parameters(lambda _, t: t.to(device))
        return self

    @contextlib.contextmanager
    def _mapped_parameters(self, f: Callable[[str, Tensor], Tensor]):  # noqa: ANN202
        parameters = dict(self.parameters)
        try:
            self.apply_to_parameters(f)
            yield parameters
        finally:
            self.load_state_dict(parameters)

    def compile(self, *input_types: graph.Type[Any]) -> Callable[..., Any]:
        """Compiles the module to an optimized executable through graph tracing.

        This method performs symbolic tracing of the module's ``__call__`` method
        to construct a MAX :obj:`Graph`, which is then compiled and optimized for
        efficient execution on CPU, GPU, or other accelerators.

        The compilation process:

        1. Creates symbolic :obj:`Tensor` instances based on provided type specifications
        2. Executes ``__call__`` with symbolic tensors to record operations
        3. Constructs a :obj:`Graph` representing the computation
        4. Includes all module parameters as weights in the graph
        5. Compiles and optimizes the graph for target hardware
        6. Returns an executable function with the same signature as ``__call__``

        The input type specifications must match the signature of ``__call__``.
        Use positional arguments for positional parameters.

        Basic compilation with fixed shapes:

        .. code-block:: python

            from max.dtype import DType
            from max.experimental.tensor import Tensor, TensorType, defaults
            from max.nn.module_v3 import Module, module_dataclass

            @module_dataclass
            class Linear(Module):
                weight: Tensor
                bias: Tensor

                def __call__(self, x: Tensor) -> Tensor:
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

        Args:
            *input_types: Type specifications for each positional argument to
                ``__call__``. Must match the number and order of arguments.
                Each should be a :obj:`max.graph.Type` (typically
                :obj:`TensorType`) describing the shape and dtype.

        Returns:
            Callable[..., Any]
                A compiled executable function with the same signature as
                ``__call__``. This function runs the optimized graph and
                returns results with the same structure as ``__call__``
                (single :obj:`Tensor` or tuple of tensors).

        Raises:
            TypeError: If input types don't match ``__call__`` signature or if
                operations in ``__call__`` cannot be traced.
            RuntimeError: If graph construction fails due to incompatible
                operations or parameter access issues.
        """

        with Graph(type(self).__qualname__, input_types=input_types) as graph:
            # Wrap the graph inputs in Tensors
            inputs = [Tensor(value=input.tensor) for input in graph.inputs]

            def as_weight(name: str, tensor: Tensor):  # noqa: ANN202
                return F.constant_external(name, tensor.type)

            # Temporarily replace the parameters with external constants
            # while building the graph.
            #  - Pure tensors as Module parameters are treated as constants
            #  - Making them external constants allows them to be compiled as
            #       weights instead.
            #  - Weights aren't constant-folded (improving compile time) but
            #       can be replaced in the compiled model and still subject
            #       to exec-invariant-code-motion optimizations.
            with self._mapped_parameters(as_weight):
                outputs: Tensor | Sequence[Tensor] = self(*inputs)

            # Set the outputs.
            # - The graph API and model assume that all graphs and models
            #   have variadic outputs
            # - Module allows returning a single Tensor or variadic return
            # - The compiled model should have the same semantics as the module
            if unary := isinstance(outputs, Tensor):
                graph.output(outputs)
            else:
                graph.output(*outputs)

        # Compile the graph with module parameters as weights
        session = _session()
        weights = dict(self.parameters)
        compiled = F.functional(session.load(graph, weights_registry=weights))

        if unary:
            # Return the single result for a unary module
            return functools.wraps(self)(lambda *inputs: compiled(*inputs)[0])

        return compiled

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
            from max.nn.module_v3 import Module, module_dataclass

            @module_dataclass
            class Linear(Module):
                weight: Tensor
                bias: Tensor

                def __call__(self, x: Tensor) -> Tensor:
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
    cls: type[Module] | None = None, /, *, repr: bool = False, **kwargs
):
    """Converts a class into a MAX module with automatic parameter tracking.

    This decorator enables a regular Python class to function as a :obj:`Module`,
    providing automatic discovery and registration of parameters (Tensor fields)
    and nested modules. The decorated class gains all capabilities of :obj:`Module`,
    including parameter iteration, graph compilation via :meth:`Module.compile`,
    and hierarchical module composition.

    The decorator applies Python's ``@dataclass`` decorator internally while
    preserving :obj:`Module`'s specialized ``__repr__`` method for better
    debugging experience when printing module structures.

    Args:
        cls: The class to decorate. Must define a ``__call__`` method.
            When :obj:`None`, returns a decorator function (supports
            using ``@module_dataclass`` with or without parentheses).
        repr: If :obj:`True`, use dataclass's default ``__repr__`` instead of
            :obj:`Module`'s rich representation. Defaults to :obj:`False`.
        **kwargs: Additional keyword arguments forwarded to Python's
            ``@dataclass`` decorator (e.g., ``frozen``, ``eq``).

    Returns:
        The decorated class as a :obj:`Module` subclass with automatic parameter
        tracking and graph compilation capabilities. When ``cls`` is :obj:`None`,
        returns a decorator function.
    """
    dataclass_decorator = dataclasses.dataclass(repr=repr, **kwargs)

    def decorator(cls: type[Module]) -> type[Module]:
        decorated = dataclass_decorator(cls)
        decorated.__rich_repr__ = _module_dataclass_rich_repr  # type: ignore
        return decorated

    return decorator(cls) if cls else decorator
