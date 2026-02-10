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
"""Library for graph dimension types."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from max._core.dialects import builtin, kgen


class Dim:
    """A tensor dimension.

    Tensor dimensions can be one of three types:

    - **Static**: Known size
    - **Symbolic**: Unknown size but named
    - **Algebraic**: Unknown size has an algebraic expression


    In most cases, you don't need to work with a ``Dim`` directly.
    Instead, use conversion constructors:


    .. code-block:: python

        from max.graph import Dim, TensorType, DeviceRef

        tensor_type = TensorType(DType.int64, ("batch", 10), device=DeviceRef.CPU())

    This creates a tensor type with two dimensions:

    - A symbolic "batch" dimension
    - A static dimension of size 10

    For explicit dimension construction, use the following helpers:

    .. code-block:: python

        from max.graph import Dim

        some_dims = [
            SymbolicDim("batch"),
            StaticDim(5),
            AlgebraicDim(Dim("batch") + 1),
        ]

    Constraining tensor dimensions is one important way to improve model
    performance. If tensors have unknown dimensions, we can't optimize them
    as aggressively. Symbolic tensors allow the compiler to learn constraints
    on a specific dimension (eg. if 2 inputs have the same `batch` dimension),
    but static dims are the easiest to optimize and therefore the easiest to
    create and work with.
    """

    def __new__(cls, value: DimLike):
        """Converts valid input values to Dim."""
        if cls is not Dim:
            # For subclass constructors, preserve identity only for same subclass.
            if isinstance(value, cls):
                return value
            return super().__new__(cls)
        if isinstance(value, Dim):
            # For base Dim constructor, pass through any existing Dim.
            return value
        if isinstance(value, int | np.integer):
            return super().__new__(StaticDim)
        if isinstance(value, str):
            return super().__new__(SymbolicDim)
        if isinstance(value, kgen.ParamOperatorAttr):
            return super().__new__(AlgebraicDim)

        raise TypeError(f"Unsupported dimension type {value} ({type(value)})")

    def __index__(self) -> int:
        """Converts this dim to an index as used by indexing and slicing.

        This raises and suggests explicitly converting to int, so that we only
        support implicit slicing operations on TensorValues.
        Types such as list and np.ndarray call __index__ on inputs to their
        __getitem__ special methods to convert those inputs to int.

        This also prevents a MyPy false positive error: Slice index must be an
        integer or None.
        Related MyPy discussion: https://github.com/python/mypy/issues/2410
        """
        raise TypeError(
            "when using dims to index into a list or NumPy array, explicitly "
            "convert to int with int(dim)"
        )

    def __int__(self) -> int:
        """Conversion to an int only supported for static dims."""
        raise TypeError(
            f"int({self!r}): Int conversions only supported for static dims"
        )

    def __eq__(self, other: Any) -> bool:
        """Checks whether two dimensions are equal.

        Dimensions are equal if they are the same dimension type
        (symbolic, static). Additionally, static dimensions
        are only equal if their dimension is the same size, and symbolic
        dimensions are only equal if they have the same name.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if the dimensions are equal, false otherwise.
        """
        raise NotImplementedError

    def __ne__(self, other: Any) -> bool:
        """Checks whether two dimensions are not equal.

        The inverse of __eq__.

        Args:
            other: The other dimension to check inequality against.

        Returns:
            False if the dimensions are equal, true otherwise.
        """
        return not self == other

    def __add__(self, rhs: DimLike) -> Dim:
        return AlgebraicDim.apply(kgen.POC.add, self, rhs)

    # hitting https://github.com/python/mypy/issues/11595 which causes mypy to fail to typecheck.
    def __radd__(self, lhs: DimLike) -> Dim:
        return Dim(lhs) + self

    def __mul__(self, rhs: DimLike) -> Dim:
        return AlgebraicDim.apply(kgen.POC.mul_no_wrap, self, rhs)

    # hitting https://github.com/python/mypy/issues/11595 which causes mypy to fail to typecheck.
    def __rmul__(self, lhs: DimLike) -> Dim:  # type: ignore
        return Dim(lhs) * self

    def __neg__(self) -> Dim:
        return -1 * self

    def __sub__(self, rhs: DimLike) -> Dim:
        return self + -Dim(rhs)

    def __rsub__(self, lhs: DimLike) -> Dim:
        return lhs + -self

    def __floordiv__(self, rhs: DimLike) -> Dim:
        if isinstance(rhs, int | StaticDim) and int(rhs) == 0:
            raise ZeroDivisionError
        return AlgebraicDim.apply(kgen.POC.div, self, rhs)

    def __rfloordiv__(self, lhs: DimLike) -> Dim:
        return lhs // self

    def to_mlir(self) -> builtin.TypedAttr:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        raise NotImplementedError

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute``.

        Args:
            attr: The MLIR Attribute to parse into a dimension.

        Returns:
            Dim: The dimension represented by the MLIR Attr value.
        """
        if isinstance(attr, builtin.IntegerAttr):
            return StaticDim.from_mlir(attr)
        elif isinstance(attr, kgen.ParamDeclRefAttr):
            return SymbolicDim.from_mlir(attr)
        elif isinstance(attr, kgen.ParamOperatorAttr):
            return AlgebraicDim.from_mlir(attr)
        else:
            raise ValueError("graph api does not support unknown dimensions")

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Lists the symbolic dimension names on which this dim depends."""
        raise NotImplementedError


@dataclass(frozen=True)
class SymbolicDim(Dim):
    """A symbolic tensor dimension.

    Symbolic dimensions represent named dimensions in MO tensor types.

    Symbolic dimensions don't have a static value, but they allow a readable
    name to understand what's going on in the model IR better, and they also
    allow users to hint to the compiler that two dimensions will have the same
    value, which can often allow important speedups.

    In tensor type notation:

    .. code-block::

        !mo.tensor<[batch, x, 10], si32]>

    The first and second dimensions are named ``batch`` and ``x`` respectively.

    Creating a ``SymbolicDim``:

    .. code-block:: python

        dim = SymbolicDim("name")

    Using ``SymbolicDim`` in a :obj:`TensorType`:

    .. code-block:: python

        tensor_type = TensorType(DType.bool, (SymbolicDim("batch"), SymbolicDim("x"), 10))
    """

    name: str
    """The name of the dimension."""

    def __init__(self, name: str | SymbolicDim) -> None:
        if not isinstance(name, str | SymbolicDim):
            raise TypeError(
                f"SymbolicDim.__init__ only accepts str or SymbolicDim, got {type(name).__name__}"
            )
        # Can't assign directly to frozen dataclasses.
        super().__setattr__("name", str(name))
        # TODO(MSDK-695): less restrictive names
        if not re.match(r"^[a-zA-Z_]\w*$", self.name):
            raise ValueError("Invalid name for symbolic dimension")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Dim({self.name!r})"

    def __eq__(self, other: Any) -> bool:
        """Whether the dimension is the same as another symbolic dimension.

        Symbolic dimensions with the same name are interpreted as the same
        dimensionality! If you use Symbolic dimensions, make sure you're naming
        them consistently, your model will likely fail to compile if you name
        two actually different dimensions the same name.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if the dimensions have the same name, false otherwise.
        """
        return self.name == other or (
            isinstance(other, SymbolicDim) and self.name == other.name
        )

    def to_mlir(self) -> kgen.ParamDeclRefAttr:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        si64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
        return kgen.ParamDeclRefAttr(self.name, si64)

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute``.

        Args:
            attr: The MLIR Attribute to parse into a dimension.

        Returns:
            Dim: The dimension represented by the MLIR Attr value.
        """
        if not isinstance(attr, kgen.ParamDeclRefAttr):
            raise TypeError(f"Attr is not a symbolic dim: {attr}")
        return SymbolicDim(attr.name.value)

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Lists the symbolic dimension names on which this dim depends."""
        yield self


@dataclass(frozen=True)
class AlgebraicDim(Dim):
    """An algebraic tensor dimension to enable expressions over symbolic dimensions.

    That is, any expression over a symbolic dimension returns ``AlgebraicDim``.
    Furthermore, algebraic dimensions automatically simplify into a canonical
    form.

    The following example demonstrates how to create and use algebraic dimensions with symbolic values:

    .. code-block:: python

        from max.graph import AlgebraicDim, Dim
        isinstance(Dim("batch") * 5, AlgebraicDim)  # Returns True
        print(Dim("batch") * 5)  # Outputs: batch * 5
        -Dim("x") - 4 == -(Dim("x") + 4)  # Returns True
    """

    attr: kgen.ParamOperatorAttr

    def __init__(self, attr: kgen.ParamOperatorAttr | AlgebraicDim) -> None:
        if not isinstance(attr, kgen.ParamOperatorAttr | AlgebraicDim):
            raise TypeError(
                f"AlgebraicDim.__init__ only accepts kgen.ParamOperatorAttr or AlgebraicDim, got {type(attr).__name__}"
            )
        super().__setattr__(
            "attr", attr.attr if isinstance(attr, AlgebraicDim) else attr
        )

    @classmethod
    def apply(cls, op: kgen.POC, *operands: DimLike):  # noqa: ANN206
        """Applies a parametric operator to operands and returns the resulting dimension."""
        # kgen.ParamOperatorAttr eagerly folds on construction!
        #  - this can return static or symbolic dims
        #  - let Dim decide what type to return
        attr = kgen.ParamOperatorAttr(
            op, [Dim(operand).to_mlir() for operand in operands]
        )
        return Dim.from_mlir(attr)

    def __format__(self, format_spec: str) -> str:
        formatters: Mapping[str, Callable[[Any], str]] = {
            "str": str,
            "repr": repr,
        }
        formatter = formatters[format_spec or "str"]

        def format(dim: Dim):  # noqa: ANN202
            formatted = formatter(dim)
            return (
                f"({formatted})" if isinstance(dim, AlgebraicDim) else formatted
            )

        # For the opcodes we support in the graph api, print with python math.
        opcodes = {
            kgen.POC.add: "+",
            kgen.POC.mul_no_wrap: "*",
            kgen.POC.div: "//",
        }
        opcode = self.attr.opcode
        dims = [Dim.from_mlir(operand) for operand in self.attr.operands]
        if opcode in opcodes:
            # Wrap algebraic sub-expressions in parens
            return f" {opcodes[opcode]} ".join(map(format, dims))
        return formatter(self.attr)

    def __str__(self) -> str:
        return f"{self:str}"

    def __repr__(self) -> str:
        return f"{self:repr}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AlgebraicDim) and self.attr == other.attr

    def to_mlir(self) -> kgen.ParamOperatorAttr:
        """Creates an mlir.Attribute representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An mlir.Attribute in the context representing the dimension.
        """
        return self.attr

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute`` (algebraic)."""
        if not isinstance(attr, kgen.ParamOperatorAttr):
            raise TypeError(f"Attribute is not an algebraic dimension: {attr}")
        return AlgebraicDim(attr)

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Lists the symbolic dimension names on which this dim depends."""
        for operand in self.attr.operands:
            yield from Dim.from_mlir(operand).parameters


@functools.total_ordering
@dataclass(frozen=True)
class StaticDim(Dim):
    """A static tensor dimension.

    Static tensor dimensions will always have exactly the same value,
    and are key to good model performance.

    The following example shows how static dimensions can be created implicitly:

    .. code-block:: python

        from max.graph import TensorType
        from max.dtype import DType
        tensor = TensorType(DType.int64, (4, 5))
        # This creates a tensor with 2 static dimensions: 4 and 5 respectively
    """

    dim: int
    """The size of the static dimension."""

    def __init__(self, dim: int | StaticDim) -> None:
        if not isinstance(dim, int | StaticDim):
            raise TypeError(
                f"StaticDim.__init__ only accepts int or StaticDim, got {type(dim).__name__}"
            )
        # Can't assign directly to frozen dataclasses.
        super().__setattr__("dim", int(dim))
        if not -(2**63) <= self.dim < 2**63:
            raise ValueError("Dim value must be -2**63 <= dim < 2**63")

    def __str__(self) -> str:
        return str(self.dim)

    def __repr__(self) -> str:
        return f"Dim({repr(self.dim)})"

    def __int__(self) -> int:
        return self.dim

    def __eq__(self, other: Any) -> bool:
        """Whether the dimension has the same size as another dimension.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if both dimensions have the same static size, false otherwise.
        """
        return self.dim == other or (
            isinstance(other, StaticDim) and self.dim == other.dim
        )

    def __lt__(self, other: int | StaticDim):
        return self.dim < (other.dim if isinstance(other, StaticDim) else other)

    def __hash__(self):
        return hash(self.dim)

    def to_mlir(self) -> builtin.IntegerAttr:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        si64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
        return builtin.IntegerAttr(si64, self.dim)

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute``.

        Args:
            attr: The MLIR Attribute to parse into a dimension.

        Returns:
            The dimension represented by the MLIR Attr value.
        """
        if not isinstance(attr, builtin.IntegerAttr):
            raise TypeError(f"Attribute is not a static dimension: {attr}")
        return StaticDim(attr.value)

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Lists the symbolic dimension names on which this dim depends."""
        return ()


DimLike = int | str | Dim | np.integer[Any]
