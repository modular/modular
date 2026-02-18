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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Callable, Sequence
from typing import Protocol, overload

import max._core
import max._core.dialects.builtin
import max._core.dialects.kgen
import max._core.dialects.m
import max._core.dtype
from max.mlir import Context, Location

from . import passes as passes

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

class BufferType(max._core.Type):
    """
    This is a close analogue of the existing mo.tensor type but is meant
    to represent tensors that can be mutated.

    In conjunction with the operations mo.mutable.load and mo.mutable.store
    this type can be used to model in-place operations in the MO dialect.

    The `shapeAttr` is less permisive than the equivalent for `!mo.tensor`
    values and must be a `MOSH::ShapeAttr` (i.e. statically ranked).

    The element type is an M::DType, with `invalid` denoting an unknown type.

    Examples:
    ```mlir
    !mo.buffer<[4, 16], f32>    // static shape
    !mo.buffer<[N, N, 6], i32>  // parameterized shape
    !mo.tensor<Sh, invalid>     // shape parameter reference
    ```
    """

    @overload
    def __init__(self, tensor_type: TensorType) -> None: ...
    @overload
    def __init__(
        self,
        shape_attr: max._core.dialects.builtin.TypedAttr,
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape_attr: max._core.dialects.builtin.TypedAttr,
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[max._core.dialects.builtin.TypedAttr],
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: max._core.dialects.builtin.TypedAttr,
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr,
        metadata: max._core.dialects.builtin.DictionaryAttr,
    ) -> None: ...
    @property
    def shape_attr(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...
    @property
    def device_ref(self) -> max._core.dialects.m.DeviceRefAttr: ...
    @property
    def metadata(self) -> max._core.dialects.builtin.DictionaryAttr: ...

class ChainType(max._core.Type):
    """
    This type is used to sequence side-effecting operations. Any operation in
    the MO dialect that has side-effects should both consume and produce a
    chain type.
    """

    def __init__(self) -> None: ...

class ListType(max._core.Type):
    """
    This type represents an immutable list of elements (currently restricted to
    `!mo.tensor`).
    """

    def __init__(self, element_type: max._core.Type) -> None: ...
    @property
    def element_type(self) -> max._core.Type | None: ...

class OpaqueType(max._core.Type):
    """
    This is a custom user-defined type.
      Example:
      ```mlir
        !mo.opaque<"my_list">
        !mo.opaque<"my_list", {foo = 42}>
      ```
    """

    @overload
    def __init__(
        self, symbol: max._core.dialects.builtin.StringAttr
    ) -> None: ...
    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.StringAttr,
        parameters: max._core.dialects.builtin.DictionaryAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.StringAttr,
        parameters: max._core.dialects.builtin.DictionaryAttr,
    ) -> None: ...
    @property
    def symbol(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def parameters(self) -> max._core.dialects.builtin.DictionaryAttr: ...

class ScalarType(max._core.Type):
    """This type represents scalars."""

    def __init__(self, dtype: max._core.dtype.DType) -> None: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...

class TensorType(max._core.Type):
    """
    This type represents the shape and element type of a tensor, an optional
    device ref, and an optional dictionary of metadata (e.g., layout, etc.).

    The `shapeAttr` is one of:
    1. `KGEN::ParamDeclRefAttr` for a a shape parameter, e.g., `Sh0`.
    2. `MOSH::ShapeAttr` for a shape of known rank, e.g., `[D0, 42, ?]`.

    The element type is an M::DType, with `invalid` denoting an unknown type.
    The type implements a subset of the methods in ShapedTypeInterface.

    The `deviceRef` optional field denotes the device the tensor lives on.

    Examples:
    ```mlir
    !mo.tensor<[4, 16], f32>           // static shape
    !mo.tensor<[N, N, 6], i32>         // parameterized shape
    !mo.tensor<[?, ?], i32>            // unknown shape of known rank
    !mo.tensor<[1, ?, N], i32>         // partially known and parameterized shape
    !mo.tensor<?, invalid>             // unknown shape of unknown rank
    !mo.tensor<Sh, invalid>            // shape parameter reference
    !mo.tensor<[4, 16], f32>    // optional device
    ```
    """

    @overload
    def __init__(
        self,
        shape_attr: max._core.dialects.builtin.TypedAttr,
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape_attr: max._core.dialects.builtin.TypedAttr,
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[int],
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[max._core.dialects.builtin.TypedAttr],
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: max._core.dialects.builtin.TypedAttr,
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr,
        metadata: max._core.dialects.builtin.DictionaryAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[int],
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[max._core.dialects.builtin.TypedAttr],
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def shape_attr(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...
    @property
    def device_ref(self) -> max._core.dialects.m.DeviceRefAttr: ...
    @property
    def metadata(self) -> max._core.dialects.builtin.DictionaryAttr: ...

class ChainAttr(max._core.Attribute):
    """
    Represents non-error chain values. The type of this attribute is always
    `!mo.chain`.

    Example:

    ```mlir
    #mo<chain> : !mo.chain
    ```
    """

    @overload
    def __init__(self, type: ChainType) -> None: ...
    @overload
    def __init__(self, type: ChainType) -> None: ...
    @property
    def type(self) -> ChainType: ...

class DTypeAttr(max._core.Attribute):
    """This attribute holds the data type of a tensor."""

    def __init__(self, dtype: max._core.dtype.DType) -> None: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...

class LayoutAttr(max._core.Attribute):
    """
    This attribute holds the memory layout information for some tensor value.
    """

    @overload
    def __init__(
        self, format: max._core.dialects.builtin.StringAttr
    ) -> None: ...
    @overload
    def __init__(self, format_str: str) -> None: ...
    @property
    def format(self) -> max._core.dialects.builtin.StringAttr: ...

class CoordinateTransformMode(enum.Enum):
    half_pixel = 0

    align_corners = 1

    asymmetric = 2

    half_pixel_1D = 3

class CoordinateTransformModeAttr(max._core.Attribute):
    """This attribute is used by `mo.resize`."""

    def __init__(self, value: CoordinateTransformMode) -> None: ...
    @property
    def value(self) -> CoordinateTransformMode: ...

class IOKind(enum.Enum):
    _unknown = 32

    _output = 0

    _input = 1

    _fused_input = 2

    _fused_output = 3

    _fused_compute_output = 31

class IOKindAttr(max._core.Attribute):
    def __init__(self, arg0: Context, arg1: IOKind, /) -> None: ...
    @property
    def value(self) -> IOKind: ...

class MOConditionallyInPlaceInterface(Protocol):
    """
    Interface that ops that can conditionally represent an in-place computation
    (e.g. a custom op that directly operates on a mo.buffer value or a
     mogg.kernel after it has been load and store fused).

    Should be used in conjunction with MOMutableOpInterface.
    """

    @property
    def in_place(self) -> bool: ...

class ConstantLike(Protocol):
    """Interface for modeling constant operations."""

    @property
    def type(self) -> TensorType: ...
    @property
    def result(self) -> max._core.Value[TensorType]: ...

class MOControlOpInterface(Protocol):
    """Interface marking ops that are control flow"""

class ElementWiseBinary(Protocol):
    """Interface for modeling binary element-wise operations."""

    @property
    def lhs_input(self) -> max._core.Value[TensorType]: ...
    @lhs_input.setter
    def lhs_input(self, arg: max._core.Value, /) -> None: ...
    @property
    def rhs_input(self) -> max._core.Value[TensorType]: ...
    @rhs_input.setter
    def rhs_input(self, arg: max._core.Value, /) -> None: ...
    @property
    def result(self) -> max._core.Value[TensorType]: ...

class ElementWiseLike(Protocol):
    """Represents an generic element-wise op."""

class ElementWiseUnary(Protocol):
    """Interface for modeling unary element-wise operations."""

    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @input.setter
    def input(self, arg: max._core.Value, /) -> None: ...
    @property
    def result(self) -> max._core.Value[TensorType]: ...

class MOHasDeviceInterface(Protocol):
    """
    Interface for ops with an optional `#M.device_ref` attribute describing the
    intended execution device. The reference must resolve to an `#M.device_spec`
    in the op's containing `mo.graph` 'device_specs' attribute.
    """

    @property
    def execution_devices(self) -> list[max._core.dialects.m.DeviceRefAttr]: ...
    @property
    def execution_device(self) -> max._core.dialects.m.DeviceRefAttr: ...

class MatmulLike(Protocol):
    """
    Interface for modeling operations that implement matmul-like behavior,
    including vanilla matmul and batchmatmul.

    Dynamically ranked/shaped inputs and results are permitted, but if the shape
    is known, the last 2 dimensions of each input are assumed to be the matrix
    dimension. This interface refers to the left and right matrices as A and B,
    respectively.
    """

    @property
    def tensor_a(self) -> max._core.Value[TensorType]: ...
    @property
    def tensor_b(self) -> max._core.Value[TensorType]: ...
    @property
    def tensor_result(self) -> max._core.Value[TensorType]: ...

class MOMutableOpInterface(Protocol):
    """
    Interface that all mutable ops under rmo/mo implement and any ops that
    could represent in-place compute should implement as well.

    In the case where an op can be conditionally in-place (e.g. mo.custom) it
    should also implement the MOConditionallyInPlaceInterface as well.
    """

    @property
    def in_chains(self) -> list[max._core.Value[ChainType]]: ...
    @property
    def out_chains(self) -> list[max._core.Value[ChainType]]: ...
    @property
    def in_chains_mutable(self) -> list[max._core.OpOperand]: ...

class PadLike(Protocol):
    """Interface for modeling pad operations."""

    @property
    def implicitly_parametric(self) -> bool: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def type(self) -> TensorType: ...
    def get_effects(
        self, arg: Sequence[max._core._MemoryEffect], /
    ) -> None: ...
    def walk_declarations(
        self, arg: Callable[[max._core.dialects.kgen.ParamDeclAttr], None], /
    ) -> None: ...
    def walk_definitions(
        self,
        arg: Callable[
            [
                max._core.dialects.kgen.ParamDeclAttr,
                max._core.dialects.kgen.ParamDefValue,
            ],
            None,
        ],
        /,
    ) -> None: ...
    def rename_declarations(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    def collect_parameter_uses(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...
    def collect_parameter_uses_below(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...

class ParamDeclarationInterface(Protocol):
    """
    Interface to be implemented by ops that declare shape or dimension
    parameters.
    """

    @property
    def implicitly_parametric(self) -> bool: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    def get_effects(
        self, arg: Sequence[max._core._MemoryEffect], /
    ) -> None: ...
    def walk_declarations(
        self, arg: Callable[[max._core.dialects.kgen.ParamDeclAttr], None], /
    ) -> None: ...
    def walk_definitions(
        self,
        arg: Callable[
            [
                max._core.dialects.kgen.ParamDeclAttr,
                max._core.dialects.kgen.ParamDefValue,
            ],
            None,
        ],
        /,
    ) -> None: ...
    def rename_declarations(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    def collect_parameter_uses(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...
    def collect_parameter_uses_below(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...

class PreservedDuringKernelLowering(Protocol):
    """
    Represents a MO operation that must have must lowered using a kernel
    implementation.
    """

class Reduction(Protocol):
    """Interface for modeling reduction operations."""

    @property
    def implicitly_parametric(self) -> bool: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def input_mutable(self) -> max._core.OpOperand: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def axis_mutable(self) -> max._core.OpOperand: ...
    @property
    def result(self) -> max._core.Value[TensorType]: ...
    def get_effects(
        self, arg: Sequence[max._core._MemoryEffect], /
    ) -> None: ...
    def walk_declarations(
        self, arg: Callable[[max._core.dialects.kgen.ParamDeclAttr], None], /
    ) -> None: ...
    def walk_definitions(
        self,
        arg: Callable[
            [
                max._core.dialects.kgen.ParamDeclAttr,
                max._core.dialects.kgen.ParamDefValue,
            ],
            None,
        ],
        /,
    ) -> None: ...
    def rename_declarations(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    def collect_parameter_uses(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...
    def collect_parameter_uses_below(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...

class SameVariadicOperandSizeInterface(Protocol):
    """
    Interface that represent MO Ops that take multiple variadics, all with the same size.
    Wrapper around the builtin `SameVariadicOperandSize` that can't be checked in C++.
    """

class ScatterLike(Protocol):
    """
    Interface for modeling Scatter-like operations (i.e., regular Scatter
    and Scatter with reductions).
    """

    @property
    def implicitly_parametric(self) -> bool: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def input_mutable(self) -> max._core.OpOperand: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def updates_mutable(self) -> max._core.OpOperand: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def indices_mutable(self) -> max._core.OpOperand: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def axis_mutable(self) -> max._core.OpOperand: ...
    @property
    def result(self) -> max._core.Value[TensorType]: ...
    def get_effects(
        self, arg: Sequence[max._core._MemoryEffect], /
    ) -> None: ...
    def walk_declarations(
        self, arg: Callable[[max._core.dialects.kgen.ParamDeclAttr], None], /
    ) -> None: ...
    def walk_definitions(
        self,
        arg: Callable[
            [
                max._core.dialects.kgen.ParamDeclAttr,
                max._core.dialects.kgen.ParamDefValue,
            ],
            None,
        ],
        /,
    ) -> None: ...
    def rename_declarations(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    def collect_parameter_uses(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...
    def collect_parameter_uses_below(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...

class ScatterNdLike(Protocol):
    """
    Interface for modeling ScatterND-like operations (i.e., regular ScatterND
    and ScatterND with reductions).
    """

    @property
    def implicitly_parametric(self) -> bool: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def input_mutable(self) -> max._core.OpOperand: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def updates_mutable(self) -> max._core.OpOperand: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def indices_mutable(self) -> max._core.OpOperand: ...
    @property
    def result(self) -> max._core.Value[TensorType]: ...
    def get_effects(
        self, arg: Sequence[max._core._MemoryEffect], /
    ) -> None: ...
    def walk_declarations(
        self, arg: Callable[[max._core.dialects.kgen.ParamDeclAttr], None], /
    ) -> None: ...
    def walk_definitions(
        self,
        arg: Callable[
            [
                max._core.dialects.kgen.ParamDeclAttr,
                max._core.dialects.kgen.ParamDefValue,
            ],
            None,
        ],
        /,
    ) -> None: ...
    def rename_declarations(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    def collect_parameter_uses(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...
    def collect_parameter_uses_below(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...

class ShapeMaterialization(Protocol):
    """
    Interface that models ops which
    1. declare parameters whose values depend on the op's input shape or data.
    2. might know how to define the declared parameters in terms of new ops
       (a best effort process that depends on the op type and its inputs).
    """

    @property
    def implicitly_parametric(self) -> bool: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    @property
    def data_dependent_input_indices(self) -> list[int]: ...
    def get_effects(
        self, arg: Sequence[max._core._MemoryEffect], /
    ) -> None: ...
    def walk_declarations(
        self, arg: Callable[[max._core.dialects.kgen.ParamDeclAttr], None], /
    ) -> None: ...
    def walk_definitions(
        self,
        arg: Callable[
            [
                max._core.dialects.kgen.ParamDeclAttr,
                max._core.dialects.kgen.ParamDefValue,
            ],
            None,
        ],
        /,
    ) -> None: ...
    def rename_declarations(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    def collect_parameter_uses(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...
    def collect_parameter_uses_below(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...
    def materialize_shape_defs(
        self, arg: Sequence[max._core.dialects.builtin.TypedAttr], /
    ) -> ShapeMaterializeResult: ...

class SlidingWindow(Protocol):
    """Interface for modeling operations that have sliding window semantics."""

    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...

class Staticization(Protocol):
    """
    Interface that models op which, if given staticized versions of its inputs,
    might be able to get staticized as well, where "being staticized" means
    having the op's output value represented as a parameter expression, i.e.,
    one of:
    - `IntegerAttr` for a integer or boolean constant, e.g., `42`.
    - `BoolAttr` for a boolean constant, e.g., `true`.
    - `KGEN::ParamDeclRefAttr` for a parameter reference, e.g., `D0`.
    - `KGEN::ParamOperatorAttr` for a parameter expression, e.g., `add(D1, 2)`.
    """

    def try_staticize(
        self,
        arg0: Sequence[max._core.dialects.builtin.TypedAttr],
        arg1: ParamExprBuilder,
        /,
    ) -> max._core.dialects.builtin.TypedAttr: ...

class MOTensorOpInterface(Protocol):
    """
    Interface that all MO ops should implement, mainly providing typed accessors
    to inputs, outputs, and their shapes.
    """

    def get_input_tensor(self, arg: int, /) -> max._core.Value[TensorType]: ...
    def get_output_tensor(self, arg: int, /) -> max._core.Value[TensorType]: ...

class ViewLike(Protocol):
    """Represents a view op."""

class IfOp(max._core.Operation):
    """
    The `mo.if` op takes an `i1` condition, a 'then' block and an 'else'
    block. If the condition is true, the 'then' block is run and the op
    returns the results of that block, otherwise the "else" block is
    run and its values are returned.

    The blocks have access to all outer values. The blocks must return
    values using the `mo.yield' op, and the returned values must match the
    types given in the `mo.if` result signature.

    Example:

    ```mlir
      %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<?, f32> {
        %v1 = mo.add(%x, %y) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>
                                ) -> !mo.tensor<?, f32>
        mo.yield %v1 : !mo.tensor<?, f32>
      } else {
        %v2 = mo.sub(%x, %y) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>
                                  ) -> !mo.tensor<?, f32>
        mo.yield %v2 : !mo.tensor<?, f32>
      }
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        cond: max._core.Value,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def cond(self) -> max._core.Value: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ShapeFromTensorOp(max._core.Operation):
    """
    Casts the input shape value to a shape-like tensor.

    Example:

    ```mlir
      %sh: !mosh.ape
      %sht = mo.shape.to_tensor(%sh) -> !mo.tensor<[2], si64>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mosh.ShapeType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ShapeToTensorOp(max._core.Operation):
    """
    Casts the input shape value to a shape-like tensor.

    Example:

    ```mlir
      %sh: !mosh.ape
      %sht = mo.shape.to_tensor(%sh) -> !mo.tensor<[2], si64>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[max._core.dialects.mosh.ShapeType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mosh.ShapeType]: ...

class StaticBroadcastToOp(max._core.Operation):
    """
    Broadcasts the input tensor to the result tensor. The shape of the input and
    result tensors must not be unknown or contain unknown dimensions, but can be
    parametric.

    This op only has limited compile-time check on the validity of the target
    shape (we expect the user to add any necessary runtime checks); therefore it
    is not recommended for frontend conversion code to rely on this op (use
    `mo.broadcast_to` with a constant shape-like tensor instead).

    The broadcasting follows numpy semantics.

    Example:

    ```mlir
      %from: !mo.tensor<[3], f32>
      %res1 = mo.static.broadcast_to(%from)
        : !mo.tensor<[3], f32> -> !mo.tensor<[2, 3], f32>
      kgen.param.declare N = <...>
      %res2 = mo.static.broadcast_to(%from)
        : !mo.tensor<[3], f32> -> !mo.tensor<[N, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class StaticReshapeOp(max._core.Operation):
    """
    Returns a tensor with the same underlying data, but different shape. The
    shape of the input and result tensors must not be unknown or contain unknown
    dimensions, but can be parametric. We do not allow inferred dimensions
    (e.g. -1 in MO_ReshapeOp).

    The op has no compile-time or runtime checks on the validity of the target
    shape (we expect the user to add any necessary runtime checks); therefore it
    is not recommended for frontend conversion code to rely on this op (use
    `mo.reshape` with a constant shape-like tensor instead).

    Example:

    ```mlir
      %from: !mo.tensor<[2, 3], f32>
      %res = mo.static.reshape(%from)
        : !mo.tensor<[2, 3], f32> -> !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class AbsOp(max._core.Operation):
    """
    Returns `abs(x)`, where `x` is the input tensors.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.abs(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class AddOp(max._core.Operation):
    """
    Returns `x + y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class AddSingletonDimOp(max._core.Operation):
    """
    Adds a dimension of `1` to a shape at the given axis.

    Example:
    ```mlir
      mo.add_singleton_dim[1](%res): (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class DistributedAllgatherOp(max._core.Operation):
    """
    AllGather takes in inputs each coming from a different device and collects
    the data into an output tensor along the 0th dimension. The output is
    replicated across the same devices.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        outputs: Sequence[max._core.Type],
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class DistributedAllreduceSumOp(max._core.Operation):
    """
    Allreduce takes in inputs each coming from a different device with
    the same shape as the final output and performs a sum reduction
    across the devices. This op instance executes on a specific device
    (specified by the device attribute) and produces the output for that device.

    Multiple instances of this op are created (one per device) to enable
    multi-threaded execution.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: TensorType,
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        device: max._core.dialects.m.DeviceRefAttr,
        has_device_barrier: max._core.dialects.builtin.UnitAttr,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def device(self) -> max._core.dialects.m.DeviceRefAttr: ...
    @device.setter
    def device(self, arg: max._core.dialects.m.DeviceRefAttr, /) -> None: ...
    @property
    def has_device_barrier(self) -> bool: ...
    @has_device_barrier.setter
    def has_device_barrier(
        self, arg: max._core.dialects.builtin.UnitAttr, /
    ) -> None: ...

class AndOp(max._core.Operation):
    """
    Returns `x and y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.and(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class ArgMaxOp(max._core.Operation):
    """
    This op is equivalent to reduce_max, but returns indices instead of values.

    The axis attribute specifies the reduction axis.

    Like reductions, the output shape is the same as the input shape, except for
    the reduced axis which is set to 1. Moreover, the value of `axis` follows
    numpy semantics, e.g., -1 represents the last axis.

    For identical maximum values, the lowest index is returned.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<si32>
      } : !mo.tensor<[], si32>
      %1 = mo.arg_max(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_ty: TensorType = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ArgMinOp(max._core.Operation):
    """
    This op is equivalent to reduce_min, but returns indices instead of values.

    The axis attribute specifies the reduction axis.

    Like reductions, the output shape is the same as the input shape, except for
    the reduced axis which is set to 1. Moreover, the value of `axis` follows
    numpy semantics, e.g., -1 represents the last axis.

    For identical minimum values, the lowest index is returned.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<si32>
      } : !mo.tensor<[], si32>
      %1 = mo.arg_min(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_ty: TensorType = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ArgNonzeroOp(max._core.Operation):
    """
    Returns a tensor of coordinates of the nonzero values in the given tensor.
    The return value is a 2D tensor of shape [nnz x rank_in], where nnz is the
    number of nonzero elements in the input tensor, and rank_in is the rank of
    the input tensor. Coordinates are generated in row-major order.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8> : tensor<3x3xsi32>
      } : !mo.tensor<[3, 3], si32>
      %1 = mo.arg_nonzero(%0) : (!mo.tensor<[3, 3], si32>) -> !mo.tensor<[?, 2], si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class AssertOp(max._core.Operation):
    """
    Asserts that a single boolean value is true.

    Currently bare assert operations are *not* used in MO graphs,
    rather they are wrapped first in a KGEN MLIR op attr, a KGEN apply
    attr and finally materialized as an actual operation via the KGENParamDeclare op.

    In order to properly materialize assert logic for a particular op and ensure
    the execution dependencies are properly tracked you must:

    1. Create each individual assert via calls to AssertOp's emitStaticCall method
    2. Supply all the asserts (really KGENParamDeclare ops) to the materializeAllAssertLogic
       function which will handle linking the individual assertions together via !mo.chains
       and materialize the remaining structures needed to properly track the execution
       dependencies from the asserts to their compute op.

    Example:

    ```mlir
    // When using emitStaticCall and materializeAllAssertLogic
    %X: mo.tensor<[D1, D2]>,
    %Y: mo.tensor<[D3, D4]>

    ...

    kgen.param.declare CH0:
      <apply(:(i1) -> !mo.chain "mo.assert"{message = "Error 1" : !kgen.string}, eq(D1, D3))>

    kgen.param.declare CH1:
      <apply(:(!mo.chain, i1) -> !mo.chain "mo.assert"{message = "Error 2" : !kgen.string}, CH0, eq(D2, D4))>

    %chain1 = mosh.param.to_value = <CH0>
    %chain2 = mosh.param.to_value = <CH1>

    %guard_chain = mo.chain.create(%chain1, %chain2)

    %Xp, Yp = mo.guard[%guard_chain](%X, %Y)

    ...

    %Z = mo.op(%Xp, %Yp)
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        chain: ChainType,
        in_chain: max._core.Value[ChainType],
        cond: max._core.Value[max._core.dialects.builtin.IntegerType],
        message: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        cond: max._core.Value,
        message: str,
    ) -> None: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def cond(
        self,
    ) -> max._core.Value[max._core.dialects.builtin.IntegerType]: ...
    @property
    def message(self) -> max._core.dialects.builtin.TypedAttr: ...
    @message.setter
    def message(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class AtanhOp(max._core.Operation):
    """
    Returns `atanh(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.atanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class AvgPoolCeilModeTrueOp(max._core.Operation):
    """
    Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2d avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.avg_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter_shape: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        count_boundary: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def count_boundary(self) -> bool: ...
    @count_boundary.setter
    def count_boundary(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class AvgPoolOp(max._core.Operation):
    """
    Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2D avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. `strides`, `dilations`, `padding`) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.avg_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter_shape: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        count_boundary: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def count_boundary(self) -> bool: ...
    @count_boundary.setter
    def count_boundary(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class LinalgBandPartOp(max._core.Operation):
    """
    Copies a tensor setting everything outside central (diagonal) band of the
    matrices to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matrices.

    Assumes the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values values are given by

    out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

    With the indicator function

    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))

    If `exclude` is set, the selection is reverted: The elements in band are set
    to zero while the elements outside the band are copied to the output tensor.

    Please explicitly note that with negative values, this kernel returns the
    entire lower or upper triangle of the matrix, and otherwise returns
    a diagonal band around the main diagonal of the matrix.

    Example:

    ```mlir
      %arg: !mo.tensor<[3, 2, 3], f32>
      %num_lower = mo.constant {
        value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %num_upper = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %exclude = mo.constant {
        value = #M.dense_array<0> : tensor<1xui8>} : !mo.tensor<[], bool>
      %res = mo.linalg.band_part(%arg, %num_lower, %num_upper, %exclude) : (
        !mo.tensor<[3, 2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>,
        !mo.tensor<[], bool>
        ) -> !mo.tensor<[3, 2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        num_lower: max._core.Value[TensorType],
        num_upper: max._core.Value[TensorType],
        exclude: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def num_lower(self) -> max._core.Value[TensorType]: ...
    @property
    def num_upper(self) -> max._core.Value[TensorType]: ...
    @property
    def exclude(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BatchMatmulOp(max._core.Operation):
    """
    Performs matrix multiplication on two batches of matrices, represented by
    two N-dimensional tensors.

    The last two dimensions of each input are the matrix dimensions.

    Example:

    ```mlir
      %lhs: ... !mo.tensor<[3, 4, 5], f32>
      %rhs: ... !mo.tensor<[3, 5, 6], f32>
      %res = mo.batch_matmul(%lhs, %rhs) :
        mo.tensor<[3, 4, 5], f32>, !mo.tensor<[3, 5, 6], f32>
      ) -> !mo.tensor<[3, 4, 6], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_a: max._core.Value[TensorType],
        input_b: max._core.Value[TensorType],
        transpose_b: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        type: max._core.Type,
        input_a: max._core.Value,
        input_b: max._core.Value,
        decls: Sequence[max._core.dialects.kgen.ParamDeclAttr],
    ) -> None: ...
    @property
    def input_a(self) -> max._core.Value[TensorType]: ...
    @property
    def input_b(self) -> max._core.Value[TensorType]: ...
    @property
    def transpose_b(self) -> bool: ...
    @transpose_b.setter
    def transpose_b(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BottomKOp(max._core.Operation):
    """
    Computes the bottom (lowest) values and their corresponding indices in a
    tensor along a specified axis. Returned values along the axis are always
    sorted (stable).

    Example:
    ```mlir
      %in = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>
      } : !mo.tensor<[2, 6], si64>
      %k = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<3> : tensor<si64> } : !mo.tensor<[], si64>
      %axis = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64> } : !mo.tensor<[], si64>
      %sorted = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<1xi1> } : !mo.tensor<[], bool>
      %values, %indices = mo.bottom_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: TensorType,
        indices: TensorType,
        input: max._core.Value[TensorType],
        _k: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        sorted: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def _k(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def sorted(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BroadcastShapeOp(max._core.Operation):
    """
    Returns the shape that two shapes would broadcast to under the numpy rules:

    1. Make the shapes the same rank by adding 1 to the front
       of the shorter shape
    2. Ensure each dimension is equal, either by matching them or promoting a 1
       to a larger number from the other shape

    Examples:

    ```mlir
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %arg2: !mo.tensor<[4, 2, 1], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[4, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    In this example, `shape1` will compute to [4, 2, 3]

    ```mlir
    %arg1: !mo.tensor<[10, 2, 1], f32>
    %arg2: !mo.tensor<[5], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[10, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[5], f32>) -> !mo.tensor<[1], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    In this example, `shape1` will compute to [10, 2, 5]
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        shape: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BroadcastToOp(max._core.Operation):
    """
    Broadcasts the input tensor to the specified shape.

    The broadcasting follows numpy semantics.

    Example:

    ```mlir
      %from: !mo.tensor<[3], f32>
      %to = mo.constant {
        value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %res = mo.broadcast_to(%from, %to) : (
        !mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        new_shape: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        broadcast_like: max._core.Value[TensorType],
        shape_of_input: max._core.Value[TensorType] = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def new_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BufferCreateOp(max._core.Operation):
    """
    This operation creates an uninitialized buffer with the specified shape and data type on a given device.
    The buffer is not initialized with any values, and the operation is intended for use cases where the buffer
    will be filled with data later in the computation.

    Example:
    ```mlir
    %buf = mo.buffer.create : !mo.buffer<[20, 20], f32, gpu:0>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: BufferType,
    ) -> None: ...

class BufferTransferOp(max._core.Operation):
    """
    This operation transfers data from a source buffer to a destination buffer.
    Both buffers must have the same shape and data type. The operation takes an input
    chain and produces an output chain to sequence the transfer with other operations.

    Example:
    ```mlir
    %outChain = mo.buffer.transfer[%inChain](%src, %dst) : !mo.buffer<[2,3], f32, gpu:0>, !mo.buffer<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: ChainType,
        src: max._core.Value[BufferType],
        dst: max._core.Value[BufferType],
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @property
    def src(self) -> max._core.Value[BufferType]: ...
    @property
    def dst(self) -> max._core.Value[BufferType]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class CallOp(max._core.Operation):
    """
    This op calls a computation graph.

    Example:

    ```mlir
      %res = mo.call @gelu(%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        operands: Sequence[max._core.Value[max._core.Type]],
        callee: max._core.dialects.builtin.FlatSymbolRefAttr,
        prefix: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
        arg_attrs: max._core.dialects.builtin.ArrayAttr,
        res_attrs: max._core.dialects.builtin.ArrayAttr,
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def callee(self) -> str: ...
    @callee.setter
    def callee(
        self, arg: max._core.dialects.builtin.FlatSymbolRefAttr, /
    ) -> None: ...
    @property
    def prefix(self) -> str: ...
    @prefix.setter
    def prefix(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def arg_attrs(self) -> max._core.dialects.builtin.ArrayAttr | None: ...
    @arg_attrs.setter
    def arg_attrs(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...
    @property
    def res_attrs(self) -> max._core.dialects.builtin.ArrayAttr | None: ...
    @res_attrs.setter
    def res_attrs(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...

class CastOp(max._core.Operation):
    """
    Returns the input tensor, cast to the specified element type.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], i32>
      %res = mo.cast(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        dtype: max._core.dtype.DType,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class CeilOp(max._core.Operation):
    """
    Returns the smallest largest integer greater than `x`, where `x` is input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.ceil(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ChainCreateOp(max._core.Operation):
    """
    This operation consumes an arbitrary number of values and produces a chain.
    Can be used for the sequencing for side-effecting ops when they might
    depend on multiple other ops producing chains.

    ```mlir

    %ch0 = mo.assert ...
    %x = mo.matmul ....

    %ch1 = mo.chain.create(%ch0: !mo.chain, %x: mo.tensor)
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class ConcatOp(max._core.Operation):
    """
    Concatenates the input tensors along a given dimension.

    `mo.concat` concatenates the `inputs` tensors into an output tensor. There
    must be at least 1 input tensor.

    The following constraints apply to the inputs/outputs:

    - The input tensors and output tensors all has the same shape except along
      the concatenation dimension `axis`.
    - The size of the concatenation dimension in output tensor have be the sum
      of sizes of the concatenation dimension in input tensors.
    - The element type of the input and output tensors must match.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg0: !mo.tensor<[2, 3], f32>
      %arg1: !mo.tensor<[2, 5], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = mo.concat[%axis: !mo.tensor<[], si64>](%arg0, %arg1) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>
      ) -> !mo.tensor<[2, 8], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        axis: max._core.Value[TensorType],
        inputs: Sequence[max._core.Value[max._core.Type]],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ConstantExternalOp(max._core.Operation):
    """
    Represents an undefined reference to an "external" constant.
    This constant's backing data is undefined at graph compile time, but all
    other attributes of the tensor are statically known.
    In a given mo.graph, mo.constant.external ops should be uniquely named.

    Currently, the `device` attribute on the op determines which device the
    weights registry pointers reside on.
    This should be checked at runtime.
    The device attribute and result type can mismatch only when the `device`
    attribute is on the host.
    This is to support mmap'ed weights, a common use case for external
    constants.

    TODO(MSDK-1060): implement the runtime check by storing device alongside
    runtime pointer in the weights registry.
    And emit an explicit mo.transfer in the graph API instead of implicitly
    inserting an HtoD copy in converting mo.constant.external to
    mgp.buffer.constant.external.

    If the `hasAlias` attribute is set to true, the external constant might be
    updated through external alias between graph invocation, and it will not
    be lifted to the init phase.

    Example:

    ```mlir
    %weight = mo.constant.external {
      align = 16 : ui64, device = #M.device_ref<"gpu", 0>, name = "foo"
    } : !mo.tensor<[4096, 4096], bf16>
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        name: max._core.dialects.builtin.StringAttr,
        align: max._core.dialects.builtin.IntegerAttr,
        device: max._core.dialects.m.DeviceRefAttr,
        has_alias: max._core.dialects.builtin.BoolAttr,
        is_placeholder: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @name.setter
    def name(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def align(self) -> int: ...
    @align.setter
    def align(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def device(self) -> max._core.dialects.m.DeviceRefAttr: ...
    @device.setter
    def device(self, arg: max._core.dialects.m.DeviceRefAttr, /) -> None: ...
    @property
    def has_alias(self) -> bool: ...
    @has_alias.setter
    def has_alias(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def is_placeholder(self) -> bool: ...
    @is_placeholder.setter
    def is_placeholder(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class ConstantOp(max._core.Operation):
    """
    This op allows storing literal values inside the graph.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result_type: max._core.Type,
        value: max._core.dialects.builtin.ElementsAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: Sequence[int],
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: int,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: Sequence[int],
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: int,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: Sequence[float],
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: float,
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: Sequence[int],
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: int,
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: float,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: float,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: Sequence[bool],
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: bool,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @property
    def value(self) -> max._core.dialects.builtin.ElementsAttr: ...
    @value.setter
    def value(
        self, arg: max._core.dialects.builtin.ElementsAttr, /
    ) -> None: ...

class ConstantScalarOp(max._core.Operation):
    """
    Same as `mo.constant`, but specialized to scalar types.

    Example:

    ```mlir
      %0 = mo.constant.scalar { value = 3 : si64 } : !mo.scalar<si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: ScalarType,
        value: max._core.Attribute,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def value(self) -> max._core.Attribute | None: ...
    @value.setter
    def value(self, arg: max._core.Attribute, /) -> None: ...

class ConvOp(max._core.Operation):
    """
    Computes the convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports 1D-3D convolution, with the following layout assumptions:
    - input has channel last layout. For 2D, that's NHWC, i.e.,
      (batch_size, height, width, in_channels)
    - filter has layout RSCF, i.e.,
      (height, width, in_channels / num_groups, out_channels)

    `strides`, `dilations`, and `padding` must be of rank 1, or unranked.
    If the input has static rank, all hyperparameters with static shape must
    have sizes of `input_rank - 2`, except padding, which must have
    size `2 * (input_rank - 2)`. Individual elements in the hyperparameters
    apply to corresponding dimensions of the input (after ignoring the batch
    and channel dimensions), with padding representing a before/after pair for
    each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    Convolution, dim1 here represents H and dim2 represents W. In python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    The input, output and filter tensors' ranks must match if statically known.

    `num_groups` must be a ranked scalar. The number of input and output
    channels must both be divisible by the number of groups `num_groups`.

    This op currently only supports strides and padding on the input.

    Example:

    ```mlir
      %st = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %di = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
        : !mo.tensor<[4], si64>
      %ng = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64>}
        : %!mo.tensor<[], si64>
      %res = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        num_groups: max._core.Value[TensorType],
        input_layout: max._core.dialects.builtin.StringAttr,
        filter_layout: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def num_groups(self) -> max._core.Value[TensorType]: ...
    @property
    def input_layout(self) -> str: ...
    @input_layout.setter
    def input_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def filter_layout(self) -> str: ...
    @filter_layout.setter
    def filter_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ConvTransposeOp(max._core.Operation):
    """
    This op effectively computes the gradient of a convolution with
    respect to its input (as if the original convolution operation had the same
    filter and hyperparameters as this op). A visualization of the computation
    can be found in https://d2l.ai/chapter_computer-vision/transposed-conv.html.

    The op supports 1D-3D spatial dimensions, with the following layout
    assumptions (note the `out_channel` is w.r.t. the original convolution):
    - input has channel last layout.For 2D, that's NHWC, i.e.,
      (batch_size, height, width, out_channels)
    - filter has layout RSFC, i.e., (height, width, out_channels, in_channels)

    All hyperparameters (i.e. strides, dilations, padding, output_paddings) must
    be of rank 1, or unranked. If the input has static rank, all hyperparameters
    with static shape must have sizes of `input_rank - 2`, except padding, which
    must have size `2 * (input_rank - 2)`. Individual elements in the
    hyperparameters applies to corresponding dimensions of the input (after
    ignoring the batch and channel dimensions), with padding representing a
    before/after pair for each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    ConvTranspose, dim1 here represents H_out and dim2 represents W_out. In
    python like syntax, padding a 2x4 spatial `output` with [0, 1, 2, 1] would
    yield:

    ```python
    output = [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
    # Shape is 2x4

    padded_input = [
      [3],
    ]
    # Shape is 1x1
    ```

    The `output_paddings` argument is meant to resolve the ambiguity of multiple
    potential output shapes when any stride is greater than 1. Basically,
    we'll add `output_paddings[i]` number of zeros at the end of output's ith
    axis. We only support output_paddings = 0.

    The input, output and filter tensors' ranks must match if statically known.

    Example:

    ```mlir
      %st = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %di = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
        : !mo.tensor<[4], si64>
      %op = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %res = mo.conv_transpose(%input, %filter)
        [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
        !mo.tensor<[10, 4, 4, 64], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[10, 5, 5, 32], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        output_paddings: max._core.Value[TensorType],
        input_layout: max._core.dialects.builtin.StringAttr,
        filter_layout: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def output_paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def input_layout(self) -> str: ...
    @input_layout.setter
    def input_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def filter_layout(self) -> str: ...
    @filter_layout.setter
    def filter_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class CosOp(max._core.Operation):
    """
    Returns `cos(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.cos(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class CumsumOp(max._core.Operation):
    """
    Returns the cumulative summation of input tensors along an axis. By default,
    it copies the first element as is. If the `exclusive` attribute is set to 1,
    then the first element is excluded. The `reverse` attribute causes the
    summation to be done in the opposite direction of the axis.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example of outputs:

    ```
      input_x = [1, 2, 3]
      axis=0
      output = [1, 3, 6]
      exclusive=1
      output = [0, 1, 3]
      exclusive=0
      reverse=1
      output = [6, 5, 3]
      exclusive=1
      reverse=1
      output = [5, 3, 0]
    ```

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis: !mo.tensor<[], i64>
    %res = mo.cumsum(%arg, %axis) {exclusive = 1 : index, reverse = 0 : index} : (
      !mo.tensor<[2, 3], f32>., !mo.tensor<[], i64>) -> !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        exclusive: max._core.dialects.builtin.IntegerAttr,
        reverse: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def exclusive(self) -> int: ...
    @exclusive.setter
    def exclusive(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def reverse(self) -> int: ...
    @reverse.setter
    def reverse(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...

class CustomOp(max._core.Operation):
    """
    The `symbol` attribute specifies which underlying Mojo kernel implements
    this operation. This kernel must be decorated with the appropriate decorator
    with the exact same symbol string.

    The `function` attribute specifies which labeled function the operation
    refers to. Examples: `mogg.shape`.

    Example:

    ```mlir
      %0 = mo.custom {symbol = "test_custom_op"}(%arg0)
        (!mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
    ```

    Corresponding kernel definition:

    ```mojo
      @register_internal("test_custom_op")
      fn foo(...):
        pass
    ```

    Also allows the definition of custom kernels that have side-effects on
    mo.buffer values via the use of chains.
    (Currently limited to NDBuffer based kernels).

    Nothing needs to change about the kernel definition, but now the custom op
    must take a `mo.buffer` operand for the value it wants to mutate and
    produce an output chain as its final result and take an input chain as its
    final operand.

    Example:

    ```mlir
    %ch1 = mo.custom {symbol = "test_mutable_op"}(%arg0, %ch0) :
      (!mo.buffer<[D1, D2], f32>, !mo.chain) -> !mo.chain
    ```
    ```mojo
      @register_internal("test_custom_op")
      fn foo(...):
        pass
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        operands: Sequence[max._core.Value[max._core.Type]],
        symbol: max._core.dialects.builtin.StringAttr,
        device: max._core.dialects.m.DeviceRefAttr,
        parameters: max._core.dialects.builtin.DictionaryAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def symbol(self) -> str: ...
    @symbol.setter
    def symbol(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def device(self) -> max._core.dialects.m.DeviceRefAttr: ...
    @device.setter
    def device(self, arg: max._core.dialects.m.DeviceRefAttr, /) -> None: ...
    @property
    def parameters(self) -> max._core.dialects.builtin.DictionaryAttr: ...
    @parameters.setter
    def parameters(
        self, arg: max._core.dialects.builtin.DictionaryAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class DebugPrintOp(max._core.Operation):
    """
    Prints a debug string. If a label attribute is supplied the string is printed with that label.
    Otherwise just the string is printed. For debugging and testing only.

    Example:
    ```mlir
      %ch0: !mo.chain
      %ch1 = mo.debug.print(%ch0) {value = "message", label = "label"}
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: ChainType,
        in_chain: max._core.Value[ChainType],
        value: max._core.dialects.builtin.StringAttr,
        label: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def value(self) -> str: ...
    @value.setter
    def value(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def label(self) -> str: ...
    @label.setter
    def label(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...

class DebugTensorPrintOp(max._core.Operation):
    """
    Prints a debug representation of argument input. If a label attribute
    is supplied the tensor contents is printed with that label. Otherwise
    just the tensor metadata is printed. For debugging and testing only.

    Example:
    ```mlir
      %arg: !mo.tensor<[5], f32>
      %ch0: !mo.chain
      %ch1 = mo.debug.tensor.print(%ch0, %arg) {label = "label"} : !mo.tensor<[5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: ChainType,
        in_chain: max._core.Value[ChainType],
        input: max._core.Value[TensorType],
        label: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def label(self) -> str: ...
    @label.setter
    def label(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...

class DebugTensorUnsafePrintOp(max._core.Operation):
    """
    Deprecated use mo.debug.tensor.print instead to ensure proper sequencing
    of print operations.

    This is just kept around due to the Mojo Graph API not supporting chains at
    the time of writing.

    Prints a debug representation of argument input. If a label attribute
    is supplied the tensor contents is printed with that label. Otherwise
    just the tensor metadata is printed. For debugging and testing only.

    Example:
    ```mlir
      %arg: !mo.tensor<[5], f32>
      mo.debug.tensor.print(%arg) {label = "label"} : !mo.tensor<[5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        label: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def label(self) -> str: ...
    @label.setter
    def label(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...

class DivOp(max._core.Operation):
    """
    Returns `x / y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.div(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class EqualOp(max._core.Operation):
    """
    Returns `x == y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                    !mo.tensor<[2, 3], f32>
                                    ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class ErfOp(max._core.Operation):
    """
    Computes the Gauss error function of the input tensor elements.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.erf(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ExpOp(max._core.Operation):
    """
    Returns `exp(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.exp(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class FloorOp(max._core.Operation):
    """
    Returns the elementwise largest integer not greater than `x`, where `x` is
    input tensor.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.floor(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class GatherNdOp(max._core.Operation):
    """
    Variant of `mo.gather` that accepts multi-dimensional indices.

    The last dimension stores the index whereas
    the outer dimensions act like batch dimensions. The size of the last
    dimension is at most the rank of the input. When the dimension size is less
    than the rank of the input, slices of the input are gathered, starting from
    the leftmost dimension.

    ```
    output_shape = (
          input.shape[:batch_dims]
        + indices.shape[batch_dims:-1]
        + data.shape[batch_dims + indices.shape[-1]:]
    )
    ```

    ```mlir
      %input = mo.constant {device = #M.device_ref<"cpu", 0>, value =
        #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> :
        tensor<2x2x4xsi64>} : !mo.tensor<[2, 2, 4], si64>
      %indices = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0> : tensor<3xsi64>} :
        !mo.tensor<[3], si64>

      %result = mo.gather_nd(%input, %indices) {batchDims = 0} :
        (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[3], si64>) ->
        !mo.tensor<[], si64>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        batch_dims: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def batch_dims(self) -> int: ...
    @batch_dims.setter
    def batch_dims(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class GatherOp(max._core.Operation):
    """
    Gathers slices from input's axis according to indices.

    If input and indices are statically ranked, the output rank must be
    `inputRank - 1 + indicesRank`. In general, the output satisfies the
    following:
    ```
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      input[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
    ```
    where `indices` appears at given axis of input.

    The values of `axis` and `indices` follows numpy semantics, e.g., -1
    represents the last axis.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 2], f32>
      %indices: !mo.tensor<[2], si64>
      %axis = mo.constant {
        value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = mo.gather(%input, %indices, %axis) : (
        !mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[2, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class GatherSumOp(max._core.Operation):
    """
    A temporary composite op that composes a gather with sum reduction.
    The gather axis is 0 and reduction axis is 1.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...

class GraphOp(max._core.Operation):
    """
    This op represents a computation graph that consists of:
    - input data and their types
    - output types
    - other ops representing computations on input and intermediate data
    - a terminating output op that returns the outputs

    Example:

    ```mlir
      mo.graph @example<D1 -> D2>(%arg: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
        // ... intermediate computations ...
        %res : !mo.tensor<[D3], f32>
        mo.output<D3> %arg : !mo.tensor<[D3], f32>
      }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        signature: max._core.dialects.builtin.TypeAttr,
        function_type: max._core.dialects.builtin.TypeAttr,
        input_parameters: max._core.dialects.kgen.ParamDeclArrayAttr,
        result_parameters: max._core.dialects.kgen.ParamDeclArrayAttr,
        counter: int,
        is_subgraph: max._core.dialects.builtin.UnitAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        name: str,
        input_types: Sequence[max._core.Type],
        result_types: Sequence[max._core.Type],
        is_subgraph: bool = False,
    ) -> None: ...
    @property
    def sym_name(self) -> str: ...
    @sym_name.setter
    def sym_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def signature(self) -> max._core.dialects.kgen.FuncTypeGeneratorType: ...
    @signature.setter
    def signature(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def function_type(self) -> max._core.dialects.builtin.FunctionType: ...
    @function_type.setter
    def function_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def input_parameters(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @input_parameters.setter
    def input_parameters(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def result_parameters(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @result_parameters.setter
    def result_parameters(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def is_subgraph(self) -> bool: ...
    @is_subgraph.setter
    def is_subgraph(
        self, arg: max._core.dialects.builtin.UnitAttr, /
    ) -> None: ...

class GreaterEqualOp(max._core.Operation):
    """
    Returns `x >= y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                            !mo.tensor<[2, 3], f32>
                                            ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class GreaterOp(max._core.Operation):
    """
    Returns `x > y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                      !mo.tensor<[2, 3], f32>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class GuardOp(max._core.Operation):
    """
    Consumes a mo.chain operation and a variadic number of inputs and is
    semantically equivalent to a copy of the variadic inputs.

    This can be used in conjunction with the !mo.chain returning ops
    (currently only mo.assert) in order to set up execution dependencies
    between the chain producing op and any appropriate downstream operations.

    Example:

    ```mlir
      %X: !mo.tensor<[D1, D2], f32>
      %Y: !mo.tensor<[D2, D3], f32>

      %chain : !mo.chain

      %guarded:2 = mo.guard[%chain](%X, %Y)

      %z = mo.matmul(%guarded#0, %guarded#1)
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        chain: max._core.Value[ChainType],
        inputs: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        chain: max._core.Value[ChainType],
        inputs: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def chain(self) -> max._core.Value[ChainType]: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class IndexToTensorOp(max._core.Operation):
    """
    Example:

    ```mlir
      %c: index
      %scalarT = mo.index.to_tensor(%c) -> !mo.tensor<[], si64>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[max._core.dialects.builtin.IntegerType],
    ) -> None: ...
    @property
    def input(
        self,
    ) -> max._core.Value[max._core.dialects.builtin.IntegerType]: ...

class InvokeShapeFuncOp(max._core.Operation):
    """
    Invokes a shape function with given name and bind its return values to the
    output parameters. Each argument must be a shape-like or integer scalar
    tensor. The `kgenParams` will be passed along to the shape function as is.

    Example:

    ```mlir
      %input : !mo.tensor<[D1, D2], f32>
      %axis  : !mo.tensor<[], si64>
      %input_shape = mo.shape_of(%input)
      : (!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[2], si64>

      mo.invoke_shape_func["shape_func_name"]<() -> D1, D2>(%input_shape, %axis)
      : (!mo.tensor<[2], si64>, !mo.tensor<[], si64>)
    ```

    `dataDeptTensors` Is an instruction to the lowering that the inputs at
    these indices should be treated as data dependent tensors. These tensors
    include things like dynamic "axis" in reductions or "steps" in slice. Any
    tensor which needs to have its values read to compute the shape for the op
    falls into this category.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        inputs: Sequence[max._core.Value[max._core.Type]],
        shape_func_name: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
        kgen_params: max._core.dialects.builtin.DictionaryAttr,
        data_dept_tensors: max._core.dialects.builtin.ArrayAttr,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def shape_func_name(self) -> str: ...
    @shape_func_name.setter
    def shape_func_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def kgen_params(
        self,
    ) -> max._core.dialects.builtin.DictionaryAttr | None: ...
    @kgen_params.setter
    def kgen_params(
        self, arg: max._core.dialects.builtin.DictionaryAttr, /
    ) -> None: ...
    @property
    def data_dept_tensors(
        self,
    ) -> max._core.dialects.builtin.ArrayAttr | None: ...
    @data_dept_tensors.setter
    def data_dept_tensors(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...

class IsInfOp(max._core.Operation):
    """
    Returns true if `x` represents a floating point Inf, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = mo.is_inf(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...

class IsNanOp(max._core.Operation):
    """
    Returns true if `x` represents a floating point NaN, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = mo.is_nan(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...

class LayerNormOp(max._core.Operation):
    """
    Layer normalization operation which operates on the last dimension of
    `input`:

      meanInput = mean(input)
      varInput = var(input)
      result = (input - meanInput) / sqrt(varInput + epsilon) * gamma + beta.

    We expect gamma and beta to be shape [channels], where channels is the size
    of the last input dimension.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        gamma: max._core.Value[TensorType],
        beta: max._core.Value[TensorType],
        epsilon: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def gamma(self) -> max._core.Value[TensorType]: ...
    @property
    def beta(self) -> max._core.Value[TensorType]: ...
    @property
    def epsilon(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class LayoutTransformOp(max._core.Operation):
    """
    This op transforms the layout of input tensor to the layout specified in
    the output type. The `kgenParams` will be passed along as named parameters
    to the underlying mojo kernel.

    It requires both the input and output tensor types to contain layout
    annotations.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        kgen_params: max._core.dialects.builtin.DictionaryAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def kgen_params(
        self,
    ) -> max._core.dialects.builtin.DictionaryAttr | None: ...
    @kgen_params.setter
    def kgen_params(
        self, arg: max._core.dialects.builtin.DictionaryAttr, /
    ) -> None: ...

class Log1pOp(max._core.Operation):
    """
    Returns `log(1 + x)`, maintaining accuracy for small `x` that could
    otherwise lead to floating-point roundings of the kind `1 + x = 1`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.log1p(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class LogOp(max._core.Operation):
    """
    Returns the natural logarithm, `log(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.log(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class LogsoftmaxOp(max._core.Operation):
    """
    Returns `log(softmax(x, axis))`, where `x` is input tensor, and `axis` is
    the axis along which `softmax` is applied.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis: !mo.tensor<[], si64>
      %res = mo.logsoftmax(%arg, %axis) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...

class DistributedMatmulAllreduceOp(max._core.Operation):
    """
    Execute a multi-gpu Matmul + AllReduce.
    This op encode multiple GPU kernels.
    This needs to be a single op to handle kernel overlap / PDL correctly.
    Mega kernel that execute a multi-gpu Matmul + AllReduce.
    The computation is potentially split on the row / col axis
    (For H100+ and large enough dimensions).
    This way we can overlap some of the operations that are independent.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        outputs: Sequence[max._core.Type],
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        weights: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def weights(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class MatmulOp(max._core.Operation):
    """
    Performs matrix multiplication on two 2D tensors.

    Example:

    ```mlir
      %lhs: ... !mo.tensor<[10, 20], f32>
      %rhs: ... !mo.tensor<[20, 5], f32>
      %res = mo.matmul(%lhs, %rhs) : (
        !mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>
      ) -> !mo.tensor<[10, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_a: max._core.Value[TensorType],
        input_b: max._core.Value[TensorType],
        transpose_b: max._core.dialects.builtin.BoolAttr,
        packed_b: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def input_a(self) -> max._core.Value[TensorType]: ...
    @property
    def input_b(self) -> max._core.Value[TensorType]: ...
    @property
    def transpose_b(self) -> bool: ...
    @transpose_b.setter
    def transpose_b(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def packed_b(self) -> bool: ...
    @packed_b.setter
    def packed_b(self, arg: max._core.dialects.builtin.BoolAttr, /) -> None: ...

class MaxOp(max._core.Operation):
    """
    Returns `max(x, y)`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.max(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class MaxPoolCeilModeTrueOp(max._core.Operation):
    """
    Computes max pooling with the given filter shape, strides, and dilations.

    The op supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.max_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter_shape: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MaxPoolOp(max._core.Operation):
    """
    Computes max pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.max_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter_shape: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MeanOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their mean value, changng that
    axis's dimension to 1.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = mo.mean(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_ty: TensorType = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MergeDimOp(max._core.Operation):
    """
    Merges two adjacent dimensions of a tensor into one. Example:
    Input=[A, B, C, D], Axis=1

    Output=[A, B*C, D].

    We merge axis i and i+1 into one dimension.

    Example:
    ```mlir
      %out = mo.merge_dim[1](%res): (!mo.tensor<[1, 2, 3, 4], f32>) -> !mo.tensor<[1, 6, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class MinOp(max._core.Operation):
    """
    Returns `min(x, y)`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.min(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class ModOp(max._core.Operation):
    """
    Returns `x mod y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], si32>
      %rhs: !mo.tensor<[2, 3], si32>
      %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class MulOp(max._core.Operation):
    """
    Returns `x * y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.mul(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class MutableLoadOp(max._core.Operation):
    """
    Allows modelling of in-place operations in MO in conjunction with mo.mutable.store.

    This is semantically equivalent to a copy from `inBuffer` to `outTensor`

    The output chain of this operation is only allowed to have at most one use.

    If the value semantic output of this operation has more than one use the
    operation becomes ineligibile for fusion with compute operations.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_tensor: TensorType,
        out_chain: ChainType,
        in_buffer: max._core.Value[BufferType],
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @property
    def in_buffer(self) -> max._core.Value[BufferType]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class MutableStoreOp(max._core.Operation):
    """
    Allows modelling of in-place operations in MO in conjunction with mo.mutable.load.

    This is semantically equivalent to a copy from `inTensor` to `inBuffer`

    The output chain of this operation is only allowed to have at most one use.

    If the value semantic tensor input of this operation has more than one use
    the operation becomes ineligibile for fusion with compute operations.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: ChainType,
        in_buffer: max._core.Value[BufferType],
        in_tensor: max._core.Value[TensorType],
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @property
    def in_buffer(self) -> max._core.Value[BufferType]: ...
    @property
    def in_tensor(self) -> max._core.Value[TensorType]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class MutableStoreSliceOp(max._core.Operation):
    """
    Stores the  tensor `slice` to a subset of the elements in `inBuffer`.

    The subset is chosen using the `start`, `stop`, and `step`
    1D index tensors; each index tensor has N elements, one for each dimension
    of the `input` tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
    ```mlir
      %buffer: !mo.buffer<[20, 20], f32>
      %slice: !mo.tensor<[D0, D1], f32>
      %ch: !mo.chain

      %start: !mo.tensor<[2], si64> // [1,  -6]
      %stop: !mo.tensor<[2], si32>  // [-3, -3]
      %step: !mo.tensor<[2], si64>  // [5,   1]

      // equivalent to this in numpy: `buffer[1:-3:5, -6:-3:1] = slice`
      %ch' = mo.mutable.store.slice(%ch, %buffer, %slice, %start, %stop, %step)
    ```

    Both consumes and produces a chain. The output chain is allowed to have at
    most one use.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: ChainType,
        in_buffer: max._core.Value[BufferType],
        slice: max._core.Value[TensorType],
        start: max._core.Value[TensorType],
        stop: max._core.Value[TensorType],
        step: max._core.Value[TensorType],
        in_chain: max._core.Value[ChainType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def in_buffer(self) -> max._core.Value[BufferType]: ...
    @property
    def slice(self) -> max._core.Value[TensorType]: ...
    @property
    def start(self) -> max._core.Value[TensorType]: ...
    @property
    def stop(self) -> max._core.Value[TensorType]: ...
    @property
    def step(self) -> max._core.Value[TensorType]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class NegativeOp(max._core.Operation):
    """
    Returns `-x`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.negative(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class NonMaximumSuppressionOp(max._core.Operation):
    """
    Filters out boxes that have high intersection-over-union (IOU).

    `boxes` is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the
    coordinates of any diagonal pair of box corners and the coordinates can be
    provided as normalized (i.e., lying in the interval [0, 1]) or absolute.

     Example:

     ```mlir
       %boxes : !mo.tensor<[1, 6, 4], f32>
       %scores : !mo.tensor<[1, 1, 6], f32>
       %maxOutputBoxesPerClass : !mo.tensor<[], si64>
       %iouThreshold : !mo.tensor<[], si64>
       %scoreThreshold : !mo.tensor<[], si64>
       %res = mo.non_maximum_suppression(%boxes, %scores, %maxOutputBoxesPerClass, %iouThreshold, %scoreThreshold) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64>
     ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: TensorType,
        boxes: max._core.Value[TensorType],
        scores: max._core.Value[TensorType],
        max_output_boxes_per_class: max._core.Value[TensorType],
        iou_threshold: max._core.Value[TensorType],
        score_threshold: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def boxes(self) -> max._core.Value[TensorType]: ...
    @property
    def scores(self) -> max._core.Value[TensorType]: ...
    @property
    def max_output_boxes_per_class(self) -> max._core.Value[TensorType]: ...
    @property
    def iou_threshold(self) -> max._core.Value[TensorType]: ...
    @property
    def score_threshold(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class NotEqualOp(max._core.Operation):
    """
    Returns elementwise `x != y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                        !mo.tensor<[2, 3], f32>
                                        ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class NotOp(max._core.Operation):
    """
    Returns `not x` on given input, where input is a boolean tensor.

    Example:

    ```mlir
      %in: !mo.tensor<[2, 3], bool>
      %res = mo.not(%in) : (!mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class OrOp(max._core.Operation):
    """
    Returns `x or y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.or(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class OutputOp(max._core.Operation):
    """
    This op specifies the output parameters and values for a `mo.graph`. The
    op takes variable number of operands and produces no results. The operand
    number and types must match the signature of the `mo.graph` that contains
    the op (after substituting the bindings to signature output type).

    Examples:

    ```mlir
      mo.graph @no_params(%arg0: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
        mo.output %arg0 : !mo.tensor<?, f32>
      }

      mo.graph @with_params<D1 -> D2>(
          %arg0: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
        mo.output<D1> %arg0 : !mo.tensor<[D1], f32>
      }
    ```

    Note that in the parameterized example, the output operands type annotation
    and the graph's output type don't necessarily need to match textually, but
    the compiler will eventually verify that they do.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        parameters: max._core.dialects.kgen.ParameterExprArrayAttr,
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def parameters(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @parameters.setter
    def parameters(
        self, arg: max._core.dialects.kgen.ParameterExprArrayAttr, /
    ) -> None: ...

class PadConstantOp(max._core.Operation):
    """
    Pads the `input` tensor with a scalar tensor `constant` according to the
    `paddings`. Assumes input has rank `N`, the `paddings` tensor should have
    shape `(2 * N)`, where each consecutive pair of elements has the form
    `[before, after]`, indicating how many of `constant` to add before and after
    the contents of `input` in that dimension. The size of each dimension D of
    the padded output is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %constant = mo.constant {
        value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output =   mo.pad.constant(%input, %paddings, %constant) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[3, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        constant: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def constant(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class PadReflectOp(max._core.Operation):
    """
    Pads the `input` tensor by reflecting it according to the `paddings`.
    Assumes input has rank `N`, the `paddings` tensor should have shape `(2 *
    N)`, where each consecutive pair of elements has the form `[before, after]`,
    indicating how many of `constant` to add before and after the contents of
    `input` in that dimension. The size of each dimension D of the padded output
    is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output   = mo.pad.reflect(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class PadRepeatOp(max._core.Operation):
    """
    Pads the `input` tensor by repeating border values according to `paddings`.
    Assumes input has rank `N`, the `paddings` tensor should have shape `(2 *
    N)`, where each consecutive pair of elements has the form `[before, after]`,
    indicating how many of `constant` to add before and after the contents of
    `input` in that dimension. The size of each dimension D of the padded output
    is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output   = mo.pad.repeat(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ParallelOp(max._core.Operation):
    """
    The `mo.parallel` operation takes a single "body" block, which is executed
    in parallel for each set of inputs.

    The results of the `mo.parallel` op are the operands of the
    `mo.yield` op from each iteration, which must all have the same type.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        inputs: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class PowOp(max._core.Operation):
    """
    Computes `x ** y`, where `x` and `y` are input tensors.

    Examples:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = mo.pow(
          %x: !mo.tensor<[2, 3], f32>,
          %y: !mo.tensor<[2, 3], f32>
      ) : !mo.tensor<[2, 3], f32>

      %x: !mo.tensor<[2, 3], f32>
      %y_int: !mo.tensor<[2, 3], si32>
      %res = mo.pow(
          %x: !mo.tensor<[2, 3], f32>,
          %y: !mo.tensor<[2, 3], si32>
      ) : !mo.tensor<[2, 3], f32>
      %res = mo.pow(%x, %y_int) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class RandomNormalOp(max._core.Operation):
    """
    Returns a tensor with shape `shape` populated with random
      values from a normal distribution, with the mean of the distribution equal
      to `mean` and the standard deviation equal to `variance`.

    Example:
      ```mlir
        %size = mo.constant {
          value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
        %mean = mo.constant {
          value = #M.dense_array<2.0> : tensor<1xf32> } : !mo.tensor<[], f32>
        %variance = mo.constant {
          value = #M.dense_array<0.5> : tensor<1xf32> } : !mo.tensor<[], f32>
        %seed = mo.constant {
          value = #M.dense_array<1> : tensor<1xsi64> } : !mo.tensor<[], si64>
        %res = mo.random.normal(%size, %mean, %variance, %seed) :
              (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
              !mo.tensor<[], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
      ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        shape: max._core.Value[TensorType],
        mean: max._core.Value[TensorType],
        variance: max._core.Value[TensorType],
        seed: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def shape(self) -> max._core.Value[TensorType]: ...
    @property
    def mean(self) -> max._core.Value[TensorType]: ...
    @property
    def variance(self) -> max._core.Value[TensorType]: ...
    @property
    def seed(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RandomUniformOp(max._core.Operation):
    """
    Returns a tensor with shape `shape` populated with random values from a
    uniform distribution on the half-open interval [lowerBound, upperBound).

    Example:
    ```mlir
    %size = mo.constant {
      value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    %lowerBound = mo.constant {
      value = #M.dense_array<2.0> : tensor<1xf32> } : !mo.tensor<[], f32>
    %upperBound = mo.constant {
      value = #M.dense_array<0.5> : tensor<1xf32> } : !mo.tensor<[], f32>
    %seed = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64> } : !mo.tensor<[], si64>
    %res = mo.random.uniform(%size, %lowerBound, %upperBound, %seed) :
          (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
          !mo.tensor<[], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        shape: max._core.Value[TensorType],
        lower_bound: max._core.Value[TensorType],
        upper_bound: max._core.Value[TensorType],
        seed: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def shape(self) -> max._core.Value[TensorType]: ...
    @property
    def lower_bound(self) -> max._core.Value[TensorType]: ...
    @property
    def upper_bound(self) -> max._core.Value[TensorType]: ...
    @property
    def seed(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RangeOp(max._core.Operation):
    """
    Creates a sequence of numbers. The sequence goes from `start` with
    increments of size `step` up to (but not including) `limit`. All arguments
    are mandatory and must have the same element type.

    Note the following restrictions on input values:
    1. `step` must be non-zero
    2. `limit - start` must be zero or have the same sign as `step`

    Example:

    ```mlir
      %limit : !mo.tensor<[], f32>
      %start = mo.constant {
        value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
      %step = mo.constant {
        value = #M.dense_array<1.5> : tensor<f32>} : !mo.tensor<[], f32>
      %res = mo.range(%start, %limit, %step) : (
        !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[?], f32>

      %startInt = mo.constant {
        value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
      %stepInt = mo.constant {
        value = #M.dense_array<2> : tensor<si32>} : !mo.tensor<[], si32>
      %limitInt = mo.constant {
        value = #M.dense_array<11> : tensor<si32>} : !mo.tensor<[], si32>
      %oddNumbersBelowTen = mo.range(%startInt, %limitInt, %stepInt) : (
        !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
      ) -> !mo.tensor<[5], si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        start: max._core.Value[TensorType],
        limit: max._core.Value[TensorType],
        step: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def start(self) -> max._core.Value[TensorType]: ...
    @property
    def limit(self) -> max._core.Value[TensorType]: ...
    @property
    def step(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RebindOp(max._core.Operation):
    """
    This op represents the "rebinding" of a type to another type. This is
    typically used for rebinding the shape of !mo.tensor types to other
    !mo.tensor types to get things to type-check due to strict shape-related
    verifiers.

    Rebinding is similar to casting, except no data conversion takes place.
    It is assumed that the two types ultimately prove to be the same at runtime.

    Therefore, in cases where it is not statically provable, the left and right
    runtime types are the same, it is expected we will insert runtime assertions
    of some sort. Note, this operation does not do this, and any assertions
    must be inserted separately.

    Note, rebinds which use parameters they declare essentially "rename"
    the associated dimensions which is a useful tool. These types of rebinds
    are guaranteed to be removed by shape inference unless they are used to
    name unknown dims (denoted by `?`) from 3P dialects like those in PT.

    Examples:
    ```mlir
      // something like ASSERT N == 3 AND Sh == {3, 1} if we aren't statically
      // sure of this fact.
      %1 = mo.rebind(%0) : !mo.tensor<[N, 1], f32> -> !mo.tensor<Sh, f32>
      %2 = mo.rebind(%1) : !mo.tensor<Sh, f32> -> !mo.tensor<[3, 1], f32>

      // Renames dims. K == 3, M == 1
      %3 = mo.rebind<() -> K, M>(%0) :
        !mo.tensor<[3, 1], f32> -> !mo.tensor<[K, M], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceAddOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their sum, changing that axis's
    dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = mo.reduce.add(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_ty: TensorType = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceMaxOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their maximum, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = mo.reduce.max(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_ty: TensorType = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceMinOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their minimum, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = mo.reduce.min(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_ty: TensorType = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceMulOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their product, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = mo.reduce.mul(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_ty: TensorType = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class DistributedReducescatterSumOp(max._core.Operation):
    """
    ReduceScatter takes in inputs each coming from a different device, and
    partitions the reduction such that each device receives a disjoint subset
    of the result. This op instance executes on a specific device (specified by
    the device attribute) and produces the output for that device.

    Multiple instances of this op are created (one per device) to enable
    multi-threaded execution.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: TensorType,
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        device: max._core.dialects.m.DeviceRefAttr,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def device(self) -> max._core.dialects.m.DeviceRefAttr: ...
    @device.setter
    def device(self, arg: max._core.dialects.m.DeviceRefAttr, /) -> None: ...

class ReluOp(max._core.Operation):
    """
    Returns `max(0, x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.relu(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ReshapeOp(max._core.Operation):
    """
    Returns a tensor with the same underlying data, but different shape.

    The first argument is the tensor to reshape.  The second tensor is the
    shape to reshape the first tensor to.  The second tensor may contain a
    single "-1" element, which signifies that that dimension should be
    automatically computed.

    Example:

    ```mlir
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape = mo.constant {
        value = #M.dense_array<3, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```

    Auto-sizing example:

    ```mlir
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape = mo.constant {
        value = #M.dense_array<3, -1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        new_shape: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def new_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ResizeBicubicOp(max._core.Operation):
    """
    Resizes a tensor to a new shape using the bicubic interpolation algorithm.

    Bicubic interpolation uses a 4x4 pixel neighborhood and cubic polynomials
    to produce smoother results than linear interpolation. This implementation
    uses Keys' cubic convolution with a = -0.5.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 3, 224, 224], f32>
      %size = mo.constant {
        value = #M.dense_array<1, 3, 448, 448> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.resize.bicubic(%input, %size) :
        (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 3, 448, 448], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        size: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def size(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ResizeLinearOp(max._core.Operation):
    """
    Resizes a tensor to a new shape using the linear algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    When set to true, the antialias attribute causes an antialiasing filter to be applied
    when downscaling.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        size: max._core.Value[TensorType],
        coordinate_transform_mode: CoordinateTransformModeAttr,
        antialias: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def size(self) -> max._core.Value[TensorType]: ...
    @property
    def coordinate_transform_mode(self) -> CoordinateTransformMode: ...
    @coordinate_transform_mode.setter
    def coordinate_transform_mode(
        self, arg: CoordinateTransformModeAttr, /
    ) -> None: ...
    @property
    def antialias(self) -> bool: ...
    @antialias.setter
    def antialias(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ResizeNearestOp(max._core.Operation):
    """
    Resizes a tensor to a new shape using the nearest-neighbor algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    The values for round mode are:
      - 0: HalfDown
      - 1: HalfUp
      - 2: Floor
      - 3: Ceil

    Round mode is HalfDown (0) by default.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 1, 2, 2], f32>
      %size = mo.constant {
        value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.resize.nearest(%input, %size) {
        coordinate_transform_mode = 0,
        round_mode = 2}:
        (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 1, 7, 8], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        size: max._core.Value[TensorType],
        coordinate_transform_mode: CoordinateTransformModeAttr,
        round_mode: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def size(self) -> max._core.Value[TensorType]: ...
    @property
    def coordinate_transform_mode(self) -> CoordinateTransformMode: ...
    @coordinate_transform_mode.setter
    def coordinate_transform_mode(
        self, arg: CoordinateTransformModeAttr, /
    ) -> None: ...
    @property
    def round_mode(self) -> int: ...
    @round_mode.setter
    def round_mode(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RoiAlignOp(max._core.Operation):
    """
    ROI align consumes an input tensor and regions of interest in which to apply pooling.

    Example:
    ```mlir
      %inp: !mo.tensor<[1, 10, 10, 1], f32>
      %rois: !mo.tensor<[1, 5], f32>
      %output_height = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<5> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %spatial_scale = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[], f32>
      %sampling_ratio = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.0> : tensor<1xf32>} : !mo.tensor<[], f32>

      %res = mo.roi_align(%inp, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio)
        {aligned = false,  mode = "AVG"}
        : (!mo.tensor<[1, 10, 10, 1], f32>,
          !mo.tensor<[1, 5], f32>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], f32>,
          !mo.tensor<[], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        rois: max._core.Value[TensorType],
        output_height: max._core.Value[TensorType],
        output_width: max._core.Value[TensorType],
        spatial_scale: max._core.Value[TensorType],
        sampling_ratio: max._core.Value[TensorType],
        aligned: max._core.dialects.builtin.BoolAttr,
        mode: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def rois(self) -> max._core.Value[TensorType]: ...
    @property
    def output_height(self) -> max._core.Value[TensorType]: ...
    @property
    def output_width(self) -> max._core.Value[TensorType]: ...
    @property
    def spatial_scale(self) -> max._core.Value[TensorType]: ...
    @property
    def sampling_ratio(self) -> max._core.Value[TensorType]: ...
    @property
    def aligned(self) -> bool: ...
    @aligned.setter
    def aligned(self, arg: max._core.dialects.builtin.BoolAttr, /) -> None: ...
    @property
    def mode(self) -> max._core.dialects.builtin.StringAttr: ...
    @mode.setter
    def mode(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RoundOp(max._core.Operation):
    """
    Returns the elementwise nearest integer, with ties going towards the
    nearest even number.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.round(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class RsqrtOp(max._core.Operation):
    """
    Returns `1/sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.rsqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ScatterAddOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the sum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via addition.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] += updates[i][j] if axis = 0,
      output[i][indices[i][j]] += updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterMaxOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the maximum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via maximum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = max(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = max(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterMinOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the minimum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via minimum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = min(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = min(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterMulOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the product of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via multiplication.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] *= updates[i][j] if axis = 0,
      output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterNdAddOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the sum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterNdMaxOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the maximum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterNdMinOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the minimum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterNdMulOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the product of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterNdOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is updated to the corresponding entry in `updates`.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = updates[i][j] if axis = 0,
      output[i][indices[i][j]] = updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class SelectOp(max._core.Operation):
    """
    Returns `cond ? x : y` (element-wise), where `cond`, `x` and `y` are input
    tensors.

    Example:

    ```mlir
      %cond: !mo.tensor<[2, 3], bool>
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = mo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        cond: max._core.Value[TensorType],
        x: max._core.Value[TensorType],
        y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def cond(self) -> max._core.Value[TensorType]: ...
    @property
    def x(self) -> max._core.Value[TensorType]: ...
    @property
    def y(self) -> max._core.Value[TensorType]: ...

class ShapeOfOp(max._core.Operation):
    """
    Returns the shape of a tensor.

    Examples:

    ```mlir
      // statically ranked
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape1 = mo.shape_of(%arg1) : !mo.tensor<[3], si64>

      // dynamically ranked
      %arg2: !mo.tensor<?, f32>
      %shape2 = mo.shape_of(%arg2) : !mo.tensor<[?], si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        shape: TensorType,
        input: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        width: int,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class SinOp(max._core.Operation):
    """
    Returns `sin(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.sin(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class SliceDimOp(max._core.Operation):
    """
    Returns a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The slice will be taken using `start` / `stop` / `step`
    along the `dim` dimension.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
      %input: !mo.tensor<[?, ?, ?], f32>
      %start: !mo.tensor<[1], si64> // [1]
      %stop: !mo.tensor<[1], si32>  // [-3]
      %step: !mo.tensor<[1], si64>  // [5]
      // equivalent to this in numpy: `input[1:-3:5, :, :]`
      %res = mo.slice_dim[0](%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10, 10], f32>,
        !mo.tensor<[1], si64>,
        !mo.tensor<[1], si32>,
        !mo.tensor<[1], si64>
      ) -> !mo.tensor<[?, ?, ?], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        start: max._core.Value[TensorType],
        stop: max._core.Value[TensorType],
        step: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def start(self) -> max._core.Value[TensorType]: ...
    @property
    def stop(self) -> max._core.Value[TensorType]: ...
    @property
    def step(self) -> max._core.Value[TensorType]: ...

class SliceOp(max._core.Operation):
    """
    Returns a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The subset is chosen using the `start`, `stop`, and `step`
    1D index tensors; each index tensor has N elements, one for each dimension
    of the `input` tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
      %input: !mo.tensor<[?, ?], f32>
      %start: !mo.tensor<[2], si64> // [1, -6]
      %stop: !mo.tensor<[2], si32>  // [-3, 6]
      %step: !mo.tensor<[2], si64>  // [5, 1]
      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = mo.slice(%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10], f32>,
        !mo.tensor<[2], si64>,
        !mo.tensor<[2], si32>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[?, ?], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        start: max._core.Value[TensorType],
        stop: max._core.Value[TensorType],
        step: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def start(self) -> max._core.Value[TensorType]: ...
    @property
    def stop(self) -> max._core.Value[TensorType]: ...
    @property
    def step(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class SoftmaxOp(max._core.Operation):
    """
    Returns `exp(input) / sum(exp(input))`, where `x` is input tensor.

    The `sum` reduction is applied along `axis`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis: !mo.tensor<[], si64>
      %res = mo.softmax(%arg, %axis) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...

class SplitDimOp(max._core.Operation):
    """
    Splits tensor at `axis` into two dimensions. Example:
    Input=[N, K], Axis=0

    Output=[S1, S2, K], where S1 = N / S2.

    Value of S2 is taken from the output shape.

    Example:
    ```mlir
      %out = mo.split_dim[0](%res): (!mo.tensor<[4, 9], f32>) -> !mo.tensor<[2, 2, 9], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class SplitOp(max._core.Operation):
    """
    Splits the input tensor into multiple tensors along a given dimension.

    `mo.split` splits the tensor `input` into multiple output tensors.
    The number of output tensors is equal to the number of elements in
    `splitSizes`, which is a rank-1 tensor of integers.
    Each of the output tensors has the same shape as `input` except along the
    split dimension `axis`, where the size is given by the corresponding
    element in `splitSizes`.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 8], f32>
      %splitSizes = mo.constant {
        value = #M.dense_array<3, 5> : tensor<2xsi64>
      } : !mo.tensor<[2], si64>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>
      } : !mo.tensor<[], si64>
      %res:2 = mo.split[%axis: !mo.tensor<[], si64>](%input, %splitSizes) : (
        !mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>
      ) -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        input: max._core.Value[TensorType],
        split_sizes: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def split_sizes(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class SqrtOp(max._core.Operation):
    """
    Returns `sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.sqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class SqueezeShapeOp(max._core.Operation):
    """
    Calculates the shape from squeeze like operators. Given an input shape
    vector representing a tensor of rank `N`, and a list of indices of length
    `M`, returns a new shape vector representing a tensor of rank `N - M`. The
    indices represent the 0-based index of dimensions in the original rank `N`
    tensor.

    The indicated indices must represent dimensions of size 1. If
    an index does not point to a dimension to size 1, an error is thrown
    instead.

    This operator supports negative indexing with python-like semantics.
    That is all indices must be in [-N, N), if an index is < 0, it is as if
    we added `N` to it.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[8], si32>
      %indices : !mo.tensor<[4], si32>
      %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[8], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_shape: max._core.Value[TensorType],
        remove_indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def remove_indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class SubOp(max._core.Operation):
    """
    Returns `x - y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.sub(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class TanhOp(max._core.Operation):
    """
    Computes `tanh(x)`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class TileOp(max._core.Operation):
    """
    Returns a new Tensor as the result of copying the input tensor N_i times
    on each dimension, where N_i = tiles[i].

    The i-th dimension of output shape will be the ith dimension of input shape
    multiplied by N_i.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %repeats : !mo.tensor<[2], si64>
      %res = mo.tile(%input, %repeats) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        repeats: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def repeats(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class TopKOp(max._core.Operation):
    """
    Computes the largest values and their corresponding indices in a tensor
    along a specified axis. Returned values along the axis are always sorted
    (stable).

    axis: The axis to compute the largest values over.
      The axis must be in [-rank, rank).
    k: The number of values to compute.
    sorted: Whether to return the values and indices sorted or not.

    Example:
    ```mlir
      %in = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>
      } : !mo.tensor<[2, 6], si64>
      %k = mo.constant() { value = #M.dense_array<3> : tensor<si64> } : !mo.tensor<[], si64>
      %axis = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64> } : !mo.tensor<[], si64>
      %sorted = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<1xi1> } : !mo.tensor<[], bool>
      %values, %indices = mo.top_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: TensorType,
        indices: TensorType,
        input: max._core.Value[TensorType],
        _k: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        sorted: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def _k(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def sorted(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class TransferOp(max._core.Operation):
    """
    This op represents a possible copy or aliasing operation to make the
    contents of the operand tensor available on the (virtual) device of the
    result tensor.

    It is valid for the source and destination devices to be identical. If the
    `alwaysElideSameDeviceCopy` flag is not set, it is implementation defined as
    to whether the result tensor is a copy or alias of the operand tensor; if
    this flag is true, transfers to the same device never result in a copy.

    Example:

    ```mlir
      %arg : !mo.tensor<[N, 8], f32, gpu:3>
      %onOtherDevice = mo.transfer %arg : !mo.tensor<[N, 8], f32, gpu:3> to <"gpu", 1>
      %onHost = mo.transfer %arg : !mo.tensor<[N, 8], f32, gpu:0> to <"cpu", 1>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        out_chain: ChainType,
        input: max._core.Value[TensorType],
        always_elide_same_device_copy: max._core.dialects.builtin.BoolAttr,
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        dest_device: max._core.dialects.m.DeviceRefAttr,
        in_chain: max._core.Value[ChainType],
        always_elide_same_device_copy: bool = True,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def always_elide_same_device_copy(self) -> bool: ...
    @always_elide_same_device_copy.setter
    def always_elide_same_device_copy(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class TransposeOp(max._core.Operation):
    """
    Returns a new Tensor as the result of permuting the dimensions of the input
    tensor according to the value of perm.

    Note that `perm` must contain unique values from `[0, input_rank)`.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %perm : !mo.tensor<[2], si64>
      %res = mo.transpose(%input, %perm) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>

      %input : !mo.tensor<[?, 5, ?], f32>
      %perm : !mo.tensor<[3], si32>
      %res = mo.transpose(%input, %perm) : (
        !mo.tensor<[?, 5, ?], f32>, !mo.tensor<[3], si32>
      ) -> !mo.tensor<[?, ?, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        perm: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def perm(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class TruncOp(max._core.Operation):
    """
    Returns the elementwise integer from truncating the decimal. Also known
    as round-toward-zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.trunc(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class UnsqueezeShapeOp(max._core.Operation):
    """
    Calculates the shape from unsqueeze like operators. Given an input shape
    vector representing a tensor of rank `N`, and a list of indices of length
    `M`, returns a new shape vector representing a tensor of rank `N + M`.

    The indices in the given list map to the new vector of length `N + M` where
    the indicated dimensions are replaced with `1`. The remaining dimension in
    the original input shape vector are copied over in the non-1 dimensions.

    This operator supports negative indexing.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[3], si32>
      %indices : !mo.tensor<[4], si32>
      %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[7], si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_shape: max._core.Value[TensorType],
        padding_indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def padding_indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class WhileConditionOp(max._core.Operation):
    """
    This op takes the continuation condition of the parent `mo.while`. The op
    takes variable number of operands which must match the operands of the
    parent `mo.while`.

    See the mo.while operation for an example.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        condition: max._core.Value[TensorType],
        args: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def condition(self) -> max._core.Value[TensorType]: ...
    @property
    def args(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class XorOp(max._core.Operation):
    """
    Returns `x xor y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.xor(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class YieldOp(max._core.Operation):
    """
    This op specifies the output values for control flow blocks. The op
    takes variable number of operands and produces no results.

    Example:

    ```mlir
      mo.if $cond : !mo.tensor<[], bool> (!mo.tensor<?, f32>) {
        mo.yield %arg0 : !mo.tensor<?, f32>
      } else {
        mo.yield %arg1 : !mo.tensor<?, f32>
      }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        parameters: max._core.dialects.kgen.ParameterExprArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self, builder: max._core.OpBuilder, location: Location
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def parameters(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @parameters.setter
    def parameters(
        self, arg: max._core.dialects.kgen.ParameterExprArrayAttr, /
    ) -> None: ...

class WhileOp(max._core.Operation):
    """
    The `mo.while` operation takes "cond" and "body" blocks. While the "cond"
    block evaluates to the true condition, the "body" block is executed.

    The "cond" and "body" blocks have access to both the values listed in the
    `mo.while` signature and any other outer values.

    - The "cond" block must return a `!mo.tensor<[], bool>` result
    followed by values whose types match the inputs of the `mo.while` signature.

    - The "body" block must return results with the same types as the `mo.while`
    signature, again using a `mo.yield`. The signature must include an "as"
    clause for every input so as to rename it to a fresh symbol.

    The results of the `mo.while` op are the operands of the
    `mo.while.condition` op when the condition operand evaluates to false.

    Example:

    ```mlir
      %x: !mo.tensor<[2], f32>
      %y: !mo.tensor<[], f32>
      %res = mo.while (%x as %inner_x: !mo.tensor<[2], f32>) {
        %zero = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
        %shape = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<> : tensor<0xsi64>} : !mo.tensor<[0], si64>
        %mean0 = mo.mean (%inner_x, %zero) : (!mo.tensor<[2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1], f32>
        %mean1 = mo.reshape (%mean0, %shape) : (!mo.tensor<[1], f32>, !mo.tensor<[0], si64>) -> !mo.tensor<[], f32>
        %cond = mo.greater (%y, %mean1) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
        mo.while.condition(%cond) %inner_x : !mo.tensor<[2], f32>
      } do {
        %new_x = mo.add(%inner_x, %inner_x) : !mo.tensor<[2], f32>
        mo.yield %new_x : !mo.tensor<[2], f32>
      }
      %res: !mo.tensor<[2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        inputs: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class ParamExprBuilder:
    pass

class ShapeMaterializeResult:
    pass
