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

import dataclasses
import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable

from rich.pretty import pretty_repr
from typing_extensions import dataclass_transform

from ...experimental.tensor import Tensor

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
