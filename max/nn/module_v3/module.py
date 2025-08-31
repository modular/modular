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
"""Module implementation using eager tensors."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Callable

from rich.pretty import pretty_repr
from typing_extensions import Self, dataclass_transform

from ... import graph
from ...driver import DLPackArray
from ...experimental import functional as F
from ...experimental.tensor import Tensor, _session
from ...graph import Graph

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class Module:
    """The core unit of composition for modeling in MAX.

    Informally, a Module is a container class. It can contain
    other Module instances, Tensors (the Module's "local parameters")
    or other arbitrary Python data.

    A Module also has a `__call__` which applies that Module to
    some input. In the simplest case this is a function from one Tensor
    to another Tensor.

    Formally Modules form a tree, and subtrees of Modules can be manipulate
    directly. A Module may also be thought of as a closure, where the parameters
    form the data of the closure and `__call__` is the application of the closure.

    Terms:
        - A "child" of a Module is sub-Module stored directly on that Module.
        - A "descendent" of a Module is one of its children, or one of their
            descendents.
        - A "parameter" is a Tensor storing data on the Module or one of its
            descendents.
        - The "qualified path" of a descendent is a period-separated string
            of the names of the child module attributes which lead to that
            descendent module, for instance `child.sub.last`.
        - The "qualified path" of a parameter is the qualified path of the
            descendent directly holding that parameter, followed by a final
            path component for the attribute name of the tensor.
            For instance `weight` for a local parameter, or
            `child.sub.last.weight` for a descendent's parameter.

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

    __call__: Callable

    @property
    def local_parameters(self) -> Iterable[tuple[str, Tensor]]:
        """Iterates over the local parameters of the Module.

        Yields:
            (name, tensor) pairs, where name is the attribute name of
                the tensor on the module.
        """
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                yield name, value

    @property
    def parameters(self) -> Iterable[tuple[str, Tensor]]:
        """Iterates over the parameters of the Module and its descendents.

        Yields:
            (name, tensor) pairs, where name is the qualified path of the
                parameter with respect to the module.
        """
        yield from self.local_parameters
        for prefix, descendent in self.descendents:
            for name, parameter in descendent.local_parameters:
                yield f"{prefix}.{name}", parameter

    @property
    def children(self) -> Iterable[tuple[str, Module]]:
        """Iterates over the direct child modules of the Module.

        Yields:
            (name, module) pairs, where name is the attribute name of
                the child on the module.
        """
        for name, value in vars(self).items():
            if isinstance(value, Module):
                yield name, value

    @property
    def descendents(self) -> Iterable[tuple[str, Module]]:
        """Iterates over the Module's descendent modules.

        Yields:
            (name, module) pairs, where name is the qualified path
                of the descendent with respect to the module.
        """
        for prefix, child in self.children:
            yield prefix, child
            for name, descendent in child.descendents:
                yield f"{prefix}.{name}", descendent

    def apply_to_local_parameters(self, f: Callable[[str, Tensor], Tensor]):
        """Applies a transformation to each local parameter tensor on the Module.

        The transformation is applied in-place, updating the module's values.
        It will not be applied to descendent's parameters.

        .. code-block:: python

            from max.driver import Accelerator
            from max.nn.module_v3 import Linear

            model = Linear(2, 3)
            model.apply_to_parameters(lambda _, t: t.to(Accelerator())

        Args:
            f: The transfomation to apply to each local parameter.
                The transformation takes two arguments, a name and a tensor.
                - The name is the attribute name of the parameter on the module
                - The tensor is the current value of that parameter
                The return value of this function is the new value that will
                replace the value at that name.
        """
        for name, attr in self.local_parameters:
            setattr(self, name, f(name, attr))

    def apply_to_parameters(self, f: Callable[[str, Tensor], Tensor]):
        """Applies a transformation to each parameter tensor on the Module
        and its descendents.

        The transformation is applied in-place, updating the module's values
        and those of its descendents.

        .. code-block:: python

            from max.driver import Accelerator
            from max.nn.module_v3 import Linear

            model = Linear(2, 3)
            model.apply_to_parameters(lambda _, t: t.to(Accelerator())

        Args:
            f: The transfomation to apply to each parameter.
                The transformation takes two arguments, a name and a tensor.
                - The name is the qualified name of the parameter
                    with respect to the module on which `apply_to_parameters`
                    was called.
                - The tensor is the current value of that parameter
                The return value of this function is the new value that will
                replace the value at that name in the module tree.
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

    def load_state(self, lookup: Callable[[str], DLPackArray]):
        """Replaces each parameter in the module and its descendents.

        The transformation is applied in-place, updating the module's values
        and those of its descendents.

        Example:

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

        For instance certain representations may not store weights as
        transposed, or may need to be quantized, or split out from a shared
        qkv block, or may just have slightly different names or paths.

        This can also be used for instance to provide a default value for
        initializing LoRA weights.

        Args:
            lookup: The lookup function for each parameter.
                - The argument to the lookup function is the qualified name
                  of the parameter with respect to the module on which
                  `load_state` was called.
                - The return value of this function is the new value that will
                  replace the value at that name in the module tree.
        """
        return self.apply_to_parameters(
            lambda name, _: Tensor.from_dlpack(lookup(name))
        )

    def load_state_dict(
        self, state: Mapping[str, DLPackArray], strict: bool = True
    ):
        """Replaces each parameter in the module and its descendents.

        The transformation is applied in-place, updating the module's values
        and those of its descendents.

        Example:

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.nn.module_v3 import Linear

            model = Linear(2, 3)
            weights = {
                "weight": Tensor.zeros([3, 2]),
                "bias": Tensor.zeros([3]),
            }
            model.load_state(weights)

        Args:
            state: A mapping from qualified name to weight
            strict: If true, verify that every value in `state` is loaded
                at least once.
        Raises:
            If `strict` is set (default) and not all weights in `state` were loaded.
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
        """Creates a new Module with its parameters transformed by the function.

        The transformation is functional rather than in-place. The module is
        deep-copied; its descendents are also replaced via the same transform
        without affecting the original module.

        .. code-block:: python

            from max.driver import Accelerator
            from max.nn.module_v3 import Linear

            model = Linear(2, 3)
            model_on_gpu = model.map_parameters(lambda _, t: t.to(Accelerator())

        Args:
            f: The transfomation to apply to each parameter.
                The transformation takes two arguments, a name and a tensor.
                - The name is the qualified name of the parameter
                    with respect to the module on which `map_parameters`
                    was called.
                - The tensor is the current value of that parameter
                The return value of this function is the new value that will
                replace the value at that name in the module tree.

        Returns:
            A new module tree of the same type resulting from mapping the
            transformation over all model parameters.
        """
        new = copy.deepcopy(self)
        new.apply_to_parameters(f)
        return new

    @contextlib.contextmanager
    def _mapped_parameters(self, f: Callable[[str, Tensor], Tensor]):
        parameters = dict(self.parameters)
        try:
            self.apply_to_parameters(f)
            yield parameters
        finally:
            self.load_state_dict(parameters)

    def compile(self, *input_types: graph.Type) -> Callable:
        """Compiles the module to a model operating on the given input types.

        Example:

        .. code-block:: python

            from max.dtype import DType
            from max.experimental import random
            from max.experimental.tensor import Tensor, TensorType, defaults
            from max.nn.module_v3 import Linear

            linear = Linear(2, 3)
            _, device = defaults()
            input_type = TensorType(DType.float32, ["batch", 2], device=device)
            model = linear.compile(input_type)

            print(model(random([3, 2], dtype=DType.float32))
            print(model(random([10, 2], dtype=DType.float32))

        Args:
            input_types: The types of the inputs to the model.

        Returns:
            A compiled implementation of the module.
        """

        with Graph(type(self).__qualname__, input_types=input_types) as graph:
            # Wrap the graph inputs in Tensors
            inputs = [Tensor(value=input.tensor) for input in graph.inputs]

            def as_weight(name: str, tensor: Tensor):
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
        return pretty_repr(self)


def _module_dataclass_rich_repr(self: DataclassInstance):
    for field in dataclasses.fields(self):
        value = getattr(self, field.name)
        if isinstance(value, Tensor):
            # Rich will try to == compare the value with the default.
            # Avoid this by never passing a default value for tensors.
            yield field.name, value
        else:
            yield field.name, value, field.default


@dataclass_transform()
@functools.wraps(dataclasses.dataclass)
def module_dataclass(
    cls: type[Module] | None = None, /, *, repr: bool = False, **kwargs
):
    """Decorate a Module subclass as a dataclass.

    `module_dataclass`es are regular Python dataclasses and also Modules.
    Using the builtin `dataclass` decorator works fine, but will
    override Module's __repr__, which may lead to a degraded usage experience
    when debugging and printing modules.

    Args:
        cls: The Module class to decorate as a dataclass
        **kwargs: Forwarded to the `dataclass` decorator.
    """
    dataclass_decorator = dataclasses.dataclass(repr=repr, **kwargs)

    def decorator(cls: type[Module]) -> type[Module]:
        decorated = dataclass_decorator(cls)
        decorated.__rich_repr__ = _module_dataclass_rich_repr  # type: ignore
        return decorated

    return decorator(cls) if cls else decorator
