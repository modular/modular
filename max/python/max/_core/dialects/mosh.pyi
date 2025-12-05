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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Callable, Sequence
from typing import overload

import max._core
import max._core.dialects.builtin
import max._core.dialects.kgen
import max._core.dialects.m
from max.mlir import Location

# Many of the generated overloads for constructors are more specialized in
# C++ than they are in Python. For example, `int32_t` and `int64_t` and `size_t`
# all map to `int` in Python typing. It may not always be clear which of these
# overloads will be run for a given set of inputs (though in most cases it's the first one)
# but we disable mypy errors for shadowed overloads.
#
# mypy: disable-error-code="overload-cannot-match"

# DiagnosticHandlers aren't a thing that Python can reasonably provided. In most cases
# these are automatically provided, but there are a few custom verifiers not covered yet.
# This binding prevents errors in those cases.
DiagnosticHandler = Callable

class BroadcastOp(max._core.Operation):
    """
    Given two tensor shapes, return the shape that both broadcast to under the
    numpy rules. The semantics of this op are nearly identical to those of
    `mo.broadcast_shape` (see the documentation of that op for details), with
    the only difference that `mosh.broadcast` operates on `!mosh.ape` values
    instead of shape-like tensors.

    Example:

    ```mlir
    %sh1: !mosh.ape
    %sh2: !mosh.ape
    %sh = mosh.broadcast(%sh1, %sh2) : !mosh.ape
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        shape: ShapeType,
        input_x: max._core.Value[ShapeType],
        input_y: max._core.Value[ShapeType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[ShapeType]: ...
    @property
    def input_y(self) -> max._core.Value[ShapeType]: ...

class ConcatOp(max._core.Operation):
    """
    Return a new shape by concatenating individual dimensions in the input
    shapes in order.

    Example:

    ```mlir
    %sh0 : !mosh.ape
    %sh1 : !mosh.ape
    %sh2 : !mosh.ape
    %sh3 = mosh.concat(%sh0, %sh1, %sh2, %sh1)
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: ShapeType,
        inputs: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input1: max._core.Value[ShapeType],
        input2: max._core.Value[ShapeType],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class EqOp(max._core.Operation):
    """
    Compare individual dimensions of the input shapes, and return `true` if they
    are all equal, or `false` otherwise.

    Example:

    ```mlir
    %sh0 : !mosh.ape
    %sh1 : !mosh.ape
    %b = mosh.equal(%sh0, %sh1)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.builtin.IntegerType,
        input1: max._core.Value[ShapeType],
        input2: max._core.Value[ShapeType],
    ) -> None: ...
    @property
    def input1(self) -> max._core.Value[ShapeType]: ...
    @property
    def input2(self) -> max._core.Value[ShapeType]: ...

class GetDimOp(max._core.Operation):
    """
    Return the size of the specified dimension in the given shape.

    This operation supports dimensions specified by compile time constants only.
    The dimension can be negative in which case Python-like index semantics are
    used (e.g. -2 refers to the second to last dimension).

    Example:

    ```mlir
    %sh : !mosh.ape
    %dim : si64
    %firstDim = mosh.get_dim(%sh)[%dim]
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.builtin.IntegerType,
        input: max._core.Value[ShapeType],
        dim: max._core.Value[max._core.dialects.builtin.IntegerType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[ShapeType]: ...
    @property
    def dim(
        self,
    ) -> max._core.Value[max._core.dialects.builtin.IntegerType]: ...

class GetRankOp(max._core.Operation):
    """
    Return the rank of the given shape.

    Example:

    ```mlir
    %sh : !mosh.ape
    %r = mosh.get_rank(%sh)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.builtin.IntegerType,
        input: max._core.Value[ShapeType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[ShapeType]: ...

class NewOp(max._core.Operation):
    """
    Return a new shape whose dimensions equal to the given values in order.

    Example:

    ```mlir
    %d0 : si64
    %d1 : si64
    %sh0 = mosh.new(%d0, %d1)

    %d2 : si64
    %sh1 = mosh.new(%d0, %d1, %d2)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: ShapeType,
        dims: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def dims(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class NumElementsOp(max._core.Operation):
    """
    The op computes the product of the dimensions in a shape.

    Example:

    ```mlir
    %sh : !mosh.ape
    %numEl = mosh.num_elements(%sh)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.builtin.IntegerType,
        input: max._core.Value[ShapeType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[ShapeType]: ...

class ParamFromValueOp(max._core.Operation):
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        param_decl: max._core.dialects.kgen.ParamDeclAttr,
        value: max._core.Value,
    ) -> None: ...
    @property
    def param_decl(self) -> max._core.dialects.kgen.ParamDeclAttr: ...
    @param_decl.setter
    def param_decl(
        self, arg: max._core.dialects.kgen.ParamDeclAttr, /
    ) -> None: ...
    @property
    def value(self) -> max._core.Value: ...

class ParamToValueOp(max._core.Operation):
    """
    The `mo.param.to_value` operation materializes the value of a parameter
    expression as an SSA value that may be used by other operations.
    Conceptually, it bridges the parameter value domain to the SSA value domain.

    Example:

    ```mlir
    %idx   = mo.param.to_value = <D0>
    %shape = mo.param.to_value: !mosh.ape = <[10, D1, 20]>
    ```

    Note that in the assembly format, we allow omitting output type annotation
    if it's `index`, for historical reasons.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.Type,
        value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @value.setter
    def value(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class ShapeAttr(max._core.Attribute):
    """
    The `#mosh.ape` attribute contains a variable number of `TypedAttr`s, each
    of which have index type. The type of this attribute is always `!mosh.ape`.

    Each dimension `TypedAttr` inside `values` is one of:
    1. `KGEN::UnknownAttr` for unparameterized dimension, e.g., `?`.
    2. `KGEN::ParamDeclRefAttr` for a dimension parameter, e.g., `D0`.
    3. `KGEN::ParamOperatorAttr` for a dimension parameter expression, e.g.,
    `add(D1, 2)`.
    4. `IntegerAttr` for a concrete integer dimension, e.g., `42`

    Note that -1 can be used as a special dimension value that denotes a
    dimension to be inferred from other dimensions and total number of elements
    in the tensor. At most 1 dimension can be -1.

    Example:

    ```mlir
    kgen.param.declare N = <3>
    #mosh<ape[1, ?, N]> : !mosh.ape
    #mosh<ape[3, -1, 42]> : !mosh.ape
    ```
    """

    @overload
    def __init__(
        self,
        values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: ShapeType,
    ) -> None: ...
    @overload
    def __init__(self, values: Sequence[int], type: ShapeType) -> None: ...
    @overload
    def __init__(
        self,
        int_dims: max._core.dialects.m.IntArrayElementsAttr,
        type: ShapeType,
    ) -> None: ...
    @overload
    def __init__(
        self,
        values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: ShapeType,
    ) -> None: ...
    @property
    def values(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def type(self) -> ShapeType: ...

class ShapeType(max._core.Type):
    def __init__(self) -> None: ...

class SliceOp(max._core.Operation):
    """
    The slice supports start and end indices, and they must obey relationships
    that one would expect from Python-like slices. Note that if `end` is none,
    we emulate the pythonic `shape[start:]` semantics.

    Example:

    ```mlir
    %sh0 : !mosh.ape
    %zero = mosh.param.to_value = <0>
    %two = mosh.param.to_value = <-2>
    %negOne = mosh.param.to_value = <-2>
    %negTwo = mosh.param.to_value = <-2>

    %sh1 = mosh.slice(%sh0)[%zero, %two]      // first two dims
    %sh2 = mosh.slice(%sh0)[%negTwo]          // last two dims
    %sh3 = mosh.slice(%sh0)[%zero, %negTwo]   // all but last two dims
    %sh4 = mosh.slice(%sh0)[%negTwo, %negOne] // second to last dim
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: ShapeType,
        input: max._core.Value[ShapeType],
        start: max._core.Value[max._core.dialects.builtin.IntegerType],
        end: max._core.Value[max._core.dialects.builtin.IntegerType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[ShapeType]: ...
    @property
    def start(
        self,
    ) -> max._core.Value[max._core.dialects.builtin.IntegerType]: ...
    @property
    def end(
        self,
    ) -> max._core.Value[max._core.dialects.builtin.IntegerType]: ...
