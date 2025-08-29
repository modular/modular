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
"""A Module for a sequence of tensor transformations."""

import functools
from collections.abc import Callable, Iterable

from ...experimental.tensor import Tensor
from .module import Module


class ModuleList(list, Module):
    """A Module subclass which is locally a list container.

    ModuleLists will use the stringified integer index of their
    submodules as the name of the module for the purposes of
    qualified paths.

    For instance:

    .. code-block:: python

        from max.nn.module_v3 import Linear, Sequential

        model = Sequential(
            Linear(5, 10),
            Linear(10, 5),
        )

        assert dict(model.parameters).keys() == {
            "0.weight", "0.bias", "1.weight", "1.bias"
        }
    """

    @property
    def children(self) -> Iterable[tuple[str, Module]]:
        """Iterates over the direct child modules of the Module.

        Yields:
            (name, module) pairs, where name is the attribute name of
                the child on the module.
        """
        for i, child in enumerate(self):
            yield str(i), child

    def __rich_repr__(self):
        """Omits the path for children in the repr."""
        for _, child in self.children:
            yield child

    # C3 linearization resolves list.__repr__ before Module.__repr__.
    # This explicitly overrides and tells the class to use Module.__repr__.
    __repr__ = Module.__repr__


class Sequential(ModuleList):
    """A Module subclass which holds a sequence of unary Modules.

    A unary Module is one whose `__call__` method has the signature

    .. code-block:: python

        def __call__(self, x: Tensor) -> Tensor: ...

    `Sequential` is itself a unary Module. Its `__call__` method
    computes the result of applying each of its child modules
    in sequence to its input.

    The following example will apply a linear transformation
    up to a dimension of 10, apply a LayerNorm, and then apply a final
    linear transformation to reduce back to the input dimension of 5.

    .. code-block:: python

        from max.experimental import Tensor
        from max.nn.module_v3 import LayerNorm, Linear, Sequential

        model = Sequential(
            Linear(5, 10),
            LayerNorm(10),
            Linear(10, 5),
        )

        result = model(Tensor.ones([5]))
        assert result.shape == [5]
    """

    def __init__(self, *modules: Callable[[Tensor], Tensor]):
        """Constructs a sequential from a sequence of modules.

        Following PyTorch, Sequential takes its inputs as a variadic
        rather than an iterable. Use the splat operator (*seq) to make
        a Sequential from an iterable.

        .. code-block:: python

            from max.nn.module_v3 import Linear, Sequential

            hidden_dims = [5, 10, 15, 20]

            model = Sequential(*(
                Linear(in_dim, out_dim) for in_dim, out_dim in
                zip(hidden_dims, hidden_dims[1:])
            ))

        Args:
            modules: The sequence of contained Modules in the order
                of desired application.
        """
        super().__init__(modules)

    def __call__(self, x: Tensor) -> Tensor:
        """Applies the contained modules in order.

        For example, the following code creates a sequence of
        linear transformations which each increase the dimension
        of the input by 5.

        The input tensor must have dim 5. The intermediate applications
        will result in intermediate tensors of dim 10 and 15 respectively,
        and the final result will have dim 20.

        .. code-block:: python

            from max.experimental.tensor import Tensor
            from max.nn.module_v3 import Linear, Sequential

            hidden_dims = [5, 10, 15, 20]

            model = Sequential(*(
                Linear(in_dim, out_dim) for in_dim, out_dim in
                zip(hidden_dims, hidden_dims[1:])
            ))

            result = model(Tensor.ones([5]))
            assert result.shape == [20]

        Args:
            x: The input tensor
        Returns:
            The result of iteratively applying each contained
            Module in sequence.
        """
        return functools.reduce(lambda x, f: f(x), self, x)
