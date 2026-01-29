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

"""Mojo kernel wrappers for the MO interpreter."""

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import simd_width_of

from algorithm.functional import elementwise, IndexList
from reflection import get_base_type_name
from tensor import (
    ElementwiseBinaryOp,
    ElementwiseBinaryComparisonOp,
    ElementwiseUnaryOp,
)
from MOGGKernelAPI.MOGGKernelAPI import (
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Max,
    Min,
    And,
    Or,
    Xor,
    Equal,
    Greater,
    GreaterEqual,
    NotEqual,
    Negative,
    Abs,
    ReLU,
    Ceil,
    Floor,
    Round,
    Exp,
    Log,
    Log1p,
    Sqrt,
    Rsqrt,
    Tanh,
    ATanh,
    Sin,
    Cos,
    Erf,
    Trunc,
    Not,
    Select,
)


# TODO(EMF-96): add support for remaining float dtypes


comptime BINARY_ARITHMETIC_OPS = Variadic.types[
    T=ElementwiseBinaryOp, Add, Sub, Mul, Div, Mod, Max, Min
]

# Binary boolean operations
comptime BINARY_BOOLEAN_OPS = Variadic.types[
    T=ElementwiseBinaryOp, And, Or, Xor
]

# Binary comparison operations
comptime BINARY_COMPARISON_OPS = Variadic.types[
    T=ElementwiseBinaryComparisonOp, Equal, Greater, GreaterEqual, NotEqual
]

# Unary elementwise operations (all dtypes)
comptime UNARY_ELEMENTWISE_OPS = Variadic.types[
    T=ElementwiseUnaryOp, Negative, Abs, ReLU, Ceil, Floor, Round
]

# Unary elementwise operations (float only)
comptime UNARY_FLOAT_ONLY_OPS = Variadic.types[
    T=ElementwiseUnaryOp,
    Exp,
    Log,
    Log1p,
    Sqrt,
    Rsqrt,
    Tanh,
    ATanh,
    Sin,
    Cos,
    Erf,
    Trunc,
]


@export
fn PyInit_mojo_ops() -> PythonObject:
    """Create a Python module with kernel function bindings."""
    try:
        var b = PythonModuleBuilder("mojo_ops")

        # Register dtype-dispatching functions

        # Binary arithmetic operations
        comptime bin_arithmetic_ops = Variadic.types[
            T=ElementwiseBinaryOp, Add, Sub, Mul, Div, Mod, Max, Min
        ]

        @parameter
        for i in range(Variadic.size(BINARY_ARITHMETIC_OPS)):
            comptime op = BINARY_ARITHMETIC_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString(
                "Elementwise " + name + " with dtype dispatch"
            )
            b.def_function[bin_elementwise_dispatcher[op]](
                name, docstring=docstring
            )

        @parameter
        for i in range(Variadic.size(BINARY_BOOLEAN_OPS)):
            comptime op = BINARY_BOOLEAN_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString(
                "Elementwise " + name + " (boolean only)"
            )
            b.def_function[bin_bool_dispatcher[op]](name, docstring=docstring)

        @parameter
        for i in range(Variadic.size(BINARY_COMPARISON_OPS)):
            comptime op = BINARY_COMPARISON_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString(
                "Elementwise " + name + " comparison"
            )
            b.def_function[bin_comparison_dispatcher[op]](
                name, docstring=docstring
            )

        @parameter
        for i in range(Variadic.size(UNARY_ELEMENTWISE_OPS)):
            comptime op = UNARY_ELEMENTWISE_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString("Elementwise " + name)
            b.def_function[unary_elementwise_dispatcher[op]](
                name, docstring=docstring
            )

        @parameter
        for i in range(Variadic.size(UNARY_FLOAT_ONLY_OPS)):
            comptime op = UNARY_FLOAT_ONLY_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString(
                "Elementwise " + name + " (float only)"
            )
            b.def_function[unary_elementwise_dispatcher[op, float_only=True]](
                name, docstring=docstring
            )

        # Unary boolean operation
        b.def_function[unary_bool_dispatcher[Not]](
            "Not", docstring="Elementwise Not (bool only)"
        )

        return b.finalize()
    except e:
        abort(String("failed to create interpreter op bindings module: ", e))


fn _get_dtype(buffer: PythonObject) raises -> DType:
    return DType._from_ui8(UInt8(py=buffer.dtype.value)._mlir_value)


# Dtype dispatch wrappers - extract dtype value from buffer and dispatch
fn bin_elementwise_dispatcher[
    op: ElementwiseBinaryOp
](
    out_buffer: PythonObject, lhs_buffer: PythonObject, rhs_buffer: PythonObject
) raises:
    """Binary elementwise operation dispatcher that handles dtype dispatch in Mojo.
    """
    var dtype = _get_dtype(lhs_buffer)
    var rhs_dtype = _get_dtype(rhs_buffer)
    if dtype != rhs_dtype:
        raise Error(
            "Mismatched input dtypes for binary elementwise operation: "
            + String(dtype)
            + " and "
            + String(rhs_dtype)
        )

    if dtype == DType.float16:
        bin_elementwise_op[op, DType.float16](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.float32:
        bin_elementwise_op[op, DType.float32](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.float64:
        bin_elementwise_op[op, DType.float64](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.bfloat16:
        bin_elementwise_op[op, DType.bfloat16](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.int8:
        bin_elementwise_op[op, DType.int8](out_buffer, lhs_buffer, rhs_buffer)
    elif dtype == DType.int16:
        bin_elementwise_op[op, DType.int16](out_buffer, lhs_buffer, rhs_buffer)
    elif dtype == DType.int32:
        bin_elementwise_op[op, DType.int32](out_buffer, lhs_buffer, rhs_buffer)
    elif dtype == DType.int64:
        bin_elementwise_op[op, DType.int64](out_buffer, lhs_buffer, rhs_buffer)
    elif dtype == DType.uint8:
        bin_elementwise_op[op, DType.uint8](out_buffer, lhs_buffer, rhs_buffer)
    elif dtype == DType.uint16:
        bin_elementwise_op[op, DType.uint16](out_buffer, lhs_buffer, rhs_buffer)
    elif dtype == DType.uint32:
        bin_elementwise_op[op, DType.uint32](out_buffer, lhs_buffer, rhs_buffer)
    elif dtype == DType.uint64:
        bin_elementwise_op[op, DType.uint64](out_buffer, lhs_buffer, rhs_buffer)
    else:
        raise Error(
            "Unsupported dtype for binary elementwise operation: "
            + String(dtype)
        )


fn bin_bool_dispatcher[
    op: ElementwiseBinaryOp
](
    out_buffer: PythonObject, lhs_buffer: PythonObject, rhs_buffer: PythonObject
) raises:
    """Binary boolean operation dispatcher (bool only)."""
    var dtype = _get_dtype(lhs_buffer)

    if dtype == DType.bool:
        bin_elementwise_op[op, DType.bool](out_buffer, lhs_buffer, rhs_buffer)
    else:
        raise Error(
            "Boolean operation requires bool dtype, got: " + String(dtype)
        )


fn bin_comparison_dispatcher[
    op: ElementwiseBinaryComparisonOp
](
    out_buffer: PythonObject, lhs_buffer: PythonObject, rhs_buffer: PythonObject
) raises:
    """Binary comparison operation dispatcher."""
    var dtype = _get_dtype(lhs_buffer)
    var rhs_dtype = _get_dtype(rhs_buffer)
    if dtype != rhs_dtype:
        raise Error(
            "Mismatched input dtypes for binary comparison operation: "
            + String(dtype)
            + " and "
            + String(rhs_dtype)
        )

    if dtype == DType.float32:
        bin_elementwise_comparison_op[op, DType.float32](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.float64:
        bin_elementwise_comparison_op[op, DType.float64](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.float16:
        bin_elementwise_comparison_op[op, DType.float16](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.bfloat16:
        bin_elementwise_comparison_op[op, DType.bfloat16](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.int8:
        bin_elementwise_comparison_op[op, DType.int8](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.int16:
        bin_elementwise_comparison_op[op, DType.int16](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.int32:
        bin_elementwise_comparison_op[op, DType.int32](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.int64:
        bin_elementwise_comparison_op[op, DType.int64](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.uint8:
        bin_elementwise_comparison_op[op, DType.uint8](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.uint16:
        bin_elementwise_comparison_op[op, DType.uint16](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.uint32:
        bin_elementwise_comparison_op[op, DType.uint32](
            out_buffer, lhs_buffer, rhs_buffer
        )
    elif dtype == DType.uint64:
        bin_elementwise_comparison_op[op, DType.uint64](
            out_buffer, lhs_buffer, rhs_buffer
        )
    else:
        raise Error(
            "Unsupported dtype for comparison operation: " + String(dtype)
        )


fn unary_elementwise_dispatcher[
    op: ElementwiseUnaryOp, *, float_only: Bool = False
](out_buffer: PythonObject, in_buffer: PythonObject) raises:
    """Unary elementwise operation dispatcher (all dtypes)."""
    var dtype = _get_dtype(in_buffer)

    @parameter
    if float_only:
        if dtype == DType.float16:
            unary_elementwise_op[op, DType.float16](out_buffer, in_buffer)
        elif dtype == DType.float32:
            unary_elementwise_op[op, DType.float32](out_buffer, in_buffer)
        elif dtype == DType.float64:
            unary_elementwise_op[op, DType.float64](out_buffer, in_buffer)
        elif dtype == DType.bfloat16:
            unary_elementwise_op[op, DType.bfloat16](out_buffer, in_buffer)
        else:
            raise Error(
                "Unsupported dtype for unary elementwise operation: "
                + String(dtype)
            )
    else:
        if dtype == DType.int8:
            unary_elementwise_op[op, DType.int8](out_buffer, in_buffer)
        elif dtype == DType.int16:
            unary_elementwise_op[op, DType.int16](out_buffer, in_buffer)
        elif dtype == DType.int32:
            unary_elementwise_op[op, DType.int32](out_buffer, in_buffer)
        elif dtype == DType.int64:
            unary_elementwise_op[op, DType.int64](out_buffer, in_buffer)
        elif dtype == DType.uint8:
            unary_elementwise_op[op, DType.uint8](out_buffer, in_buffer)
        elif dtype == DType.uint16:
            unary_elementwise_op[op, DType.uint16](out_buffer, in_buffer)
        elif dtype == DType.uint32:
            unary_elementwise_op[op, DType.uint32](out_buffer, in_buffer)
        elif dtype == DType.uint64:
            unary_elementwise_op[op, DType.uint64](out_buffer, in_buffer)
        elif dtype == DType.float16:
            unary_elementwise_op[op, DType.float16](out_buffer, in_buffer)
        elif dtype == DType.float32:
            unary_elementwise_op[op, DType.float32](out_buffer, in_buffer)
        elif dtype == DType.float64:
            unary_elementwise_op[op, DType.float64](out_buffer, in_buffer)
        elif dtype == DType.bfloat16:
            unary_elementwise_op[op, DType.bfloat16](out_buffer, in_buffer)
        else:
            raise Error(
                "Unsupported dtype for unary elementwise operation: "
                + String(dtype)
            )


fn unary_bool_dispatcher[
    op: ElementwiseUnaryOp
](out_buffer: PythonObject, in_buffer: PythonObject) raises:
    """Unary boolean operation dispatcher (bool only)."""
    var dtype = _get_dtype(in_buffer)

    if dtype == DType.bool:
        unary_elementwise_op[op, DType.bool](out_buffer, in_buffer)
    else:
        raise Error(
            "Boolean operation requires bool dtype, got: " + String(dtype)
        )


@always_inline
fn bin_elementwise_op[
    op: ElementwiseBinaryOp, dtype: DType
](
    out_buffer: PythonObject, lhs_buffer: PythonObject, rhs_buffer: PythonObject
) raises:
    """Binary elementwise operation: out = op(lhs, rhs).

    Parameters:
        op: The binary elementwise operation to perform, expressed as a function
            of two SIMD values.
        dtype: The data type of the arrays.

    Args:
        out_buffer: The output buffer object.
        lhs_buffer: The left-hand side buffer object.
        rhs_buffer: The right-hand side buffer object.
    """

    var out_ptr = UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=Int(py=out_buffer._data_ptr())
    )
    var lhs_ptr = UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=Int(py=lhs_buffer._data_ptr())
    )
    var rhs_ptr = UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=Int(py=rhs_buffer._data_ptr())
    )

    var size = Int(py=out_buffer.num_elements)

    @always_inline
    @parameter
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]

        var res = op.elementwise(
            lhs_ptr.load[width=width](i), rhs_ptr.load[width=width](i)
        )
        out_ptr.store[width=width](i, res)

    elementwise[
        func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
    ](IndexList[1](size))


@always_inline
fn bin_elementwise_comparison_op[
    op: ElementwiseBinaryComparisonOp, dtype: DType
](
    out_buffer: PythonObject, lhs_buffer: PythonObject, rhs_buffer: PythonObject
) raises:
    """Elementwise comparison: out = lhs op rhs.

    Parameters:
        op: The binary elementwise comparison operation to perform, expressed as a function
            of two SIMD values.
        dtype: The data type of the arrays.

    Args:
        out_buffer: The output buffer object.
        lhs_buffer: The left-hand side buffer object.
        rhs_buffer: The right-hand side buffer object.
    """

    var out_ptr = UnsafePointer[Scalar[DType.uint8], MutExternalOrigin](
        unsafe_from_address=Int(py=out_buffer._data_ptr())
    )
    var lhs_ptr = UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=Int(py=lhs_buffer._data_ptr())
    )
    var rhs_ptr = UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=Int(py=rhs_buffer._data_ptr())
    )

    var size = Int(py=out_buffer.num_elements)

    @always_inline
    @parameter
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]

        var res = op.elementwise(
            lhs_ptr.load[width=width](i), rhs_ptr.load[width=width](i)
        )
        out_ptr.store[width=width](i, res.cast[DType.uint8]())

    elementwise[
        func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
    ](IndexList[1](size))


@always_inline
fn unary_elementwise_op[
    op: ElementwiseUnaryOp, dtype: DType
](out_buffer: PythonObject, in_buffer: PythonObject) raises:
    """Elementwise unary operation: out = op(input).

    Parameters:
        op: The unary elementwise operation to perform.
        dtype: The data type of the arrays.

    Args:
        out_buffer: The output buffer object.
        in_buffer: The input buffer object.
    """

    var out_ptr = UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=Int(py=out_buffer._data_ptr())
    )
    var in_ptr = UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=Int(py=in_buffer._data_ptr())
    )

    var size = Int(py=out_buffer.num_elements)

    @always_inline
    @parameter
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]

        var res = op.elementwise(in_ptr.load[width=width](i))
        out_ptr.store[width=width](i, res)

    elementwise[
        func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
    ](IndexList[1](size))
