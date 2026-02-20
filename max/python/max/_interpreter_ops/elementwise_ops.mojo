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

"""Mojo kernel wrappers for elementwise MO interpreter operations."""

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator, simd_width_of

from algorithm.functional import elementwise, IndexList
from memory import OpaquePointer
from reflection import get_base_type_name
from runtime.asyncrt import DeviceContextPtr
from tensor import (
    ElementwiseBinaryOp,
    ElementwiseBinaryComparisonOp,
    ElementwiseUnaryMixedOp,
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
    Cast,
    IsNan,
    IsInf,
    Select,
    Pow,
)

from op_utils import _get_dtype, _get_buffer_ptr, _get_size, _get_ctx


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

# Unary mixed-type predicate operations (float input -> bool output)
comptime UNARY_PREDICATE_OPS = Variadic.types[
    T=ElementwiseUnaryMixedOp, IsNan, IsInf
]

# =============================================================================
# GPU Support Configuration
# =============================================================================
# DTypes that are allowed to run on GPU for the given operation.


fn _is_gpu_allowed_binary_op[op: ElementwiseBinaryOp]() -> Bool:
    """Check if a binary op is allowed on GPU at compile time."""
    comptime name = get_base_type_name[op]()
    # Arithmetic and boolean ops that work on GPU
    return (
        name == "Add"
        or name == "Sub"
        or name == "Mul"
        or name == "Div"
        or name == "Mod"
        or name == "Max"
        or name == "Min"
        or name == "And"
        or name == "Or"
        or name == "Xor"
    )


fn _is_gpu_allowed_comparison_op[op: ElementwiseBinaryComparisonOp]() -> Bool:
    """Check if a comparison op is allowed on GPU at compile time."""
    comptime name = get_base_type_name[op]()
    return (
        name == "Equal"
        or name == "Greater"
        or name == "GreaterEqual"
        or name == "NotEqual"
    )


fn _is_gpu_allowed_unary_op[op: ElementwiseUnaryOp]() -> Bool:
    """Check if a unary op is allowed on GPU at compile time."""
    comptime name = get_base_type_name[op]()
    # Basic ops, float ops, and boolean ops that work on GPU
    # Note: ATanh, Log1p, Erf use libm and don't work on GPU
    return (
        name == "Negative"
        or name == "Abs"
        or name == "ReLU"
        or name == "Ceil"
        or name == "Floor"
        or name == "Round"
        or name == "Trunc"
        or name == "Exp"
        or name == "Log"
        or name == "Sqrt"
        or name == "Rsqrt"
        or name == "Tanh"
        or name == "Sin"
        or name == "Cos"
        or name == "Not"
    )


fn _is_gpu_allowed_mixed_unary_op[op: ElementwiseUnaryMixedOp]() -> Bool:
    """Check if a mixed-type unary op is allowed on GPU at compile time."""
    comptime name = get_base_type_name[op]()
    return name == "IsNan" or name == "IsInf" or name == "Cast"


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_elementwise_ops() -> PythonObject:
    """Create a Python module with elementwise kernel function bindings."""
    try:
        var b = PythonModuleBuilder("elementwise_ops")

        # Binary arithmetic operations
        comptime for i in range(Variadic.size(BINARY_ARITHMETIC_OPS)):
            comptime op = BINARY_ARITHMETIC_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString(
                "Elementwise " + name + " with dtype dispatch"
            )
            b.def_function[bin_elementwise_dispatcher[op]](
                name, docstring=docstring
            )

        # Binary boolean operations
        comptime for i in range(Variadic.size(BINARY_BOOLEAN_OPS)):
            comptime op = BINARY_BOOLEAN_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString(
                "Elementwise " + name + " (boolean only)"
            )
            b.def_function[bin_bool_dispatcher[op]](name, docstring=docstring)

        # Binary comparison operations
        comptime for i in range(Variadic.size(BINARY_COMPARISON_OPS)):
            comptime op = BINARY_COMPARISON_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString(
                "Elementwise " + name + " comparison"
            )
            b.def_function[bin_comparison_dispatcher[op]](
                name, docstring=docstring
            )

        # Unary elementwise operations
        comptime for i in range(Variadic.size(UNARY_ELEMENTWISE_OPS)):
            comptime op = UNARY_ELEMENTWISE_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString("Elementwise " + name)
            b.def_function[unary_elementwise_dispatcher[op]](
                name, docstring=docstring
            )

        # Unary float-only operations
        comptime for i in range(Variadic.size(UNARY_FLOAT_ONLY_OPS)):
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

        # Unary predicate operations (float -> bool)
        comptime for i in range(Variadic.size(UNARY_PREDICATE_OPS)):
            comptime op = UNARY_PREDICATE_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString(
                "Elementwise " + name + " predicate (float -> bool)"
            )
            b.def_function[unary_predicate_dispatcher[op]](
                name, docstring=docstring
            )

        # Cast operation (mixed input/output dtypes)
        b.def_function[cast_dispatcher](
            "Cast", docstring="Elementwise Cast with dtype dispatch"
        )

        # Pow operation (custom dispatch - Pow doesn't conform to
        # ElementwiseBinaryOp)
        b.def_function[pow_dispatcher]("Pow", docstring="Elementwise Pow")

        # Select operation (ternary: cond ? x : y)
        b.def_function[select_dispatcher](
            "Select", docstring="Elementwise select (cond ? x : y)"
        )

        return b.finalize()
    except e:
        abort(String("failed to create elementwise op bindings module: ", e))


# =============================================================================
# Dispatchers
# =============================================================================


# Dtype dispatch wrappers - extract dtype value from buffer and dispatch
fn bin_elementwise_dispatcher[
    op: ElementwiseBinaryOp
](
    out_buffer: PythonObject,
    lhs_buffer: PythonObject,
    rhs_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Binary elementwise operation dispatcher that handles dtype dispatch in Mojo.

    Args:
        out_buffer: The output buffer object.
        lhs_buffer: The left-hand side buffer object.
        rhs_buffer: The right-hand side buffer object.
        device_context_ptr: Device context pointer (null for CPU).
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

    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float16:
        bin_elementwise_op[op, DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](lhs_buffer),
            _get_buffer_ptr[DType.float16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float32:
        bin_elementwise_op[op, DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](lhs_buffer),
            _get_buffer_ptr[DType.float32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float64:
        bin_elementwise_op[op, DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](lhs_buffer),
            _get_buffer_ptr[DType.float64](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        bin_elementwise_op[op, DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](lhs_buffer),
            _get_buffer_ptr[DType.bfloat16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int8:
        bin_elementwise_op[op, DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.int8](lhs_buffer),
            _get_buffer_ptr[DType.int8](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int16:
        bin_elementwise_op[op, DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.int16](lhs_buffer),
            _get_buffer_ptr[DType.int16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int32:
        bin_elementwise_op[op, DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.int32](lhs_buffer),
            _get_buffer_ptr[DType.int32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int64:
        bin_elementwise_op[op, DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.int64](lhs_buffer),
            _get_buffer_ptr[DType.int64](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint8:
        bin_elementwise_op[op, DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.uint8](lhs_buffer),
            _get_buffer_ptr[DType.uint8](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint16:
        bin_elementwise_op[op, DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer),
            _get_buffer_ptr[DType.uint16](lhs_buffer),
            _get_buffer_ptr[DType.uint16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint32:
        bin_elementwise_op[op, DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer),
            _get_buffer_ptr[DType.uint32](lhs_buffer),
            _get_buffer_ptr[DType.uint32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint64:
        bin_elementwise_op[op, DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer),
            _get_buffer_ptr[DType.uint64](lhs_buffer),
            _get_buffer_ptr[DType.uint64](rhs_buffer),
            size,
            ctx,
        )
    else:
        raise Error(
            "Unsupported dtype for binary elementwise operation: "
            + String(dtype)
        )


fn pow_dispatcher(
    out_buffer: PythonObject,
    lhs_buffer: PythonObject,
    rhs_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Pow dispatcher with dtype dispatch.

    Pow has a non-standard kernel signature (separate dtype/pow_dtype params)
    so it cannot use the generic bin_elementwise_dispatcher.

    Args:
        out_buffer: The output buffer object.
        lhs_buffer: The base buffer object.
        rhs_buffer: The exponent buffer object.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(lhs_buffer)
    var rhs_dtype = _get_dtype(rhs_buffer)
    if dtype != rhs_dtype:
        raise Error(
            "Mismatched input dtypes for pow: "
            + String(dtype)
            + " and "
            + String(rhs_dtype)
        )

    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float16:
        pow_elementwise_op[DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](lhs_buffer),
            _get_buffer_ptr[DType.float16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float32:
        pow_elementwise_op[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](lhs_buffer),
            _get_buffer_ptr[DType.float32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float64:
        pow_elementwise_op[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](lhs_buffer),
            _get_buffer_ptr[DType.float64](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        pow_elementwise_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](lhs_buffer),
            _get_buffer_ptr[DType.bfloat16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int8:
        pow_elementwise_op[DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.int8](lhs_buffer),
            _get_buffer_ptr[DType.int8](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int16:
        pow_elementwise_op[DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.int16](lhs_buffer),
            _get_buffer_ptr[DType.int16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int32:
        pow_elementwise_op[DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.int32](lhs_buffer),
            _get_buffer_ptr[DType.int32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int64:
        pow_elementwise_op[DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.int64](lhs_buffer),
            _get_buffer_ptr[DType.int64](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint8:
        pow_elementwise_op[DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.uint8](lhs_buffer),
            _get_buffer_ptr[DType.uint8](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint16:
        pow_elementwise_op[DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer),
            _get_buffer_ptr[DType.uint16](lhs_buffer),
            _get_buffer_ptr[DType.uint16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint32:
        pow_elementwise_op[DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer),
            _get_buffer_ptr[DType.uint32](lhs_buffer),
            _get_buffer_ptr[DType.uint32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint64:
        pow_elementwise_op[DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer),
            _get_buffer_ptr[DType.uint64](lhs_buffer),
            _get_buffer_ptr[DType.uint64](rhs_buffer),
            size,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for pow: " + String(dtype))


fn bin_bool_dispatcher[
    op: ElementwiseBinaryOp
](
    out_buffer: PythonObject,
    lhs_buffer: PythonObject,
    rhs_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Binary boolean operation dispatcher (bool only).

    Args:
        out_buffer: The output buffer object.
        lhs_buffer: The left-hand side buffer object.
        rhs_buffer: The right-hand side buffer object.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(lhs_buffer)

    if dtype == DType.bool:
        bin_elementwise_op[op, DType.bool](
            _get_buffer_ptr[DType.bool](out_buffer),
            _get_buffer_ptr[DType.bool](lhs_buffer),
            _get_buffer_ptr[DType.bool](rhs_buffer),
            _get_size(out_buffer),
            _get_ctx(device_context_ptr),
        )
    else:
        raise Error(
            "Boolean operation requires bool dtype, got: " + String(dtype)
        )


fn bin_comparison_dispatcher[
    op: ElementwiseBinaryComparisonOp
](
    out_buffer: PythonObject,
    lhs_buffer: PythonObject,
    rhs_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Binary comparison operation dispatcher.

    Args:
        out_buffer: The output buffer object.
        lhs_buffer: The left-hand side buffer object.
        rhs_buffer: The right-hand side buffer object.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(lhs_buffer)
    var rhs_dtype = _get_dtype(rhs_buffer)
    if dtype != rhs_dtype:
        raise Error(
            "Mismatched input dtypes for binary comparison operation: "
            + String(dtype)
            + " and "
            + String(rhs_dtype)
        )

    var out_ptr = _get_buffer_ptr[DType.uint8](out_buffer)
    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        bin_elementwise_comparison_op[op, DType.float32](
            out_ptr,
            _get_buffer_ptr[DType.float32](lhs_buffer),
            _get_buffer_ptr[DType.float32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float64:
        bin_elementwise_comparison_op[op, DType.float64](
            out_ptr,
            _get_buffer_ptr[DType.float64](lhs_buffer),
            _get_buffer_ptr[DType.float64](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float16:
        bin_elementwise_comparison_op[op, DType.float16](
            out_ptr,
            _get_buffer_ptr[DType.float16](lhs_buffer),
            _get_buffer_ptr[DType.float16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        bin_elementwise_comparison_op[op, DType.bfloat16](
            out_ptr,
            _get_buffer_ptr[DType.bfloat16](lhs_buffer),
            _get_buffer_ptr[DType.bfloat16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int8:
        bin_elementwise_comparison_op[op, DType.int8](
            out_ptr,
            _get_buffer_ptr[DType.int8](lhs_buffer),
            _get_buffer_ptr[DType.int8](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int16:
        bin_elementwise_comparison_op[op, DType.int16](
            out_ptr,
            _get_buffer_ptr[DType.int16](lhs_buffer),
            _get_buffer_ptr[DType.int16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int32:
        bin_elementwise_comparison_op[op, DType.int32](
            out_ptr,
            _get_buffer_ptr[DType.int32](lhs_buffer),
            _get_buffer_ptr[DType.int32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int64:
        bin_elementwise_comparison_op[op, DType.int64](
            out_ptr,
            _get_buffer_ptr[DType.int64](lhs_buffer),
            _get_buffer_ptr[DType.int64](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint8:
        bin_elementwise_comparison_op[op, DType.uint8](
            out_ptr,
            _get_buffer_ptr[DType.uint8](lhs_buffer),
            _get_buffer_ptr[DType.uint8](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint16:
        bin_elementwise_comparison_op[op, DType.uint16](
            out_ptr,
            _get_buffer_ptr[DType.uint16](lhs_buffer),
            _get_buffer_ptr[DType.uint16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint32:
        bin_elementwise_comparison_op[op, DType.uint32](
            out_ptr,
            _get_buffer_ptr[DType.uint32](lhs_buffer),
            _get_buffer_ptr[DType.uint32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint64:
        bin_elementwise_comparison_op[op, DType.uint64](
            out_ptr,
            _get_buffer_ptr[DType.uint64](lhs_buffer),
            _get_buffer_ptr[DType.uint64](rhs_buffer),
            size,
            ctx,
        )
    else:
        raise Error(
            "Unsupported dtype for comparison operation: " + String(dtype)
        )


fn unary_elementwise_dispatcher[
    op: ElementwiseUnaryOp, *, float_only: Bool = False
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Unary elementwise operation dispatcher (all dtypes).

    Args:
        out_buffer: The output buffer object.
        in_buffer: The input buffer object.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(in_buffer)
    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    comptime if float_only:
        if dtype == DType.float16:
            unary_elementwise_op[op, DType.float16](
                _get_buffer_ptr[DType.float16](out_buffer),
                _get_buffer_ptr[DType.float16](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.float32:
            unary_elementwise_op[op, DType.float32](
                _get_buffer_ptr[DType.float32](out_buffer),
                _get_buffer_ptr[DType.float32](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.float64:
            unary_elementwise_op[op, DType.float64](
                _get_buffer_ptr[DType.float64](out_buffer),
                _get_buffer_ptr[DType.float64](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.bfloat16:
            unary_elementwise_op[op, DType.bfloat16](
                _get_buffer_ptr[DType.bfloat16](out_buffer),
                _get_buffer_ptr[DType.bfloat16](in_buffer),
                size,
                ctx,
            )
        else:
            raise Error(
                "Unsupported dtype for unary elementwise operation: "
                + String(dtype)
            )
    else:
        if dtype == DType.int8:
            unary_elementwise_op[op, DType.int8](
                _get_buffer_ptr[DType.int8](out_buffer),
                _get_buffer_ptr[DType.int8](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.int16:
            unary_elementwise_op[op, DType.int16](
                _get_buffer_ptr[DType.int16](out_buffer),
                _get_buffer_ptr[DType.int16](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.int32:
            unary_elementwise_op[op, DType.int32](
                _get_buffer_ptr[DType.int32](out_buffer),
                _get_buffer_ptr[DType.int32](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.int64:
            unary_elementwise_op[op, DType.int64](
                _get_buffer_ptr[DType.int64](out_buffer),
                _get_buffer_ptr[DType.int64](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.uint8:
            unary_elementwise_op[op, DType.uint8](
                _get_buffer_ptr[DType.uint8](out_buffer),
                _get_buffer_ptr[DType.uint8](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.uint16:
            unary_elementwise_op[op, DType.uint16](
                _get_buffer_ptr[DType.uint16](out_buffer),
                _get_buffer_ptr[DType.uint16](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.uint32:
            unary_elementwise_op[op, DType.uint32](
                _get_buffer_ptr[DType.uint32](out_buffer),
                _get_buffer_ptr[DType.uint32](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.uint64:
            unary_elementwise_op[op, DType.uint64](
                _get_buffer_ptr[DType.uint64](out_buffer),
                _get_buffer_ptr[DType.uint64](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.float16:
            unary_elementwise_op[op, DType.float16](
                _get_buffer_ptr[DType.float16](out_buffer),
                _get_buffer_ptr[DType.float16](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.float32:
            unary_elementwise_op[op, DType.float32](
                _get_buffer_ptr[DType.float32](out_buffer),
                _get_buffer_ptr[DType.float32](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.float64:
            unary_elementwise_op[op, DType.float64](
                _get_buffer_ptr[DType.float64](out_buffer),
                _get_buffer_ptr[DType.float64](in_buffer),
                size,
                ctx,
            )
        elif dtype == DType.bfloat16:
            unary_elementwise_op[op, DType.bfloat16](
                _get_buffer_ptr[DType.bfloat16](out_buffer),
                _get_buffer_ptr[DType.bfloat16](in_buffer),
                size,
                ctx,
            )
        else:
            raise Error(
                "Unsupported dtype for unary elementwise operation: "
                + String(dtype)
            )


fn unary_bool_dispatcher[
    op: ElementwiseUnaryOp
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Unary boolean operation dispatcher (bool only).

    Args:
        out_buffer: The output buffer object.
        in_buffer: The input buffer object.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(in_buffer)

    if dtype == DType.bool:
        unary_elementwise_op[op, DType.bool](
            _get_buffer_ptr[DType.bool](out_buffer),
            _get_buffer_ptr[DType.bool](in_buffer),
            _get_size(out_buffer),
            _get_ctx(device_context_ptr),
        )
    else:
        raise Error(
            "Boolean operation requires bool dtype, got: " + String(dtype)
        )


fn unary_predicate_dispatcher[
    op: ElementwiseUnaryMixedOp
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Unary predicate operation dispatcher (float input -> bool output).

    Args:
        out_buffer: The output buffer object (uint8/bool).
        in_buffer: The input buffer object (float).
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(in_buffer)
    var out_ptr = _get_buffer_ptr[DType.bool](out_buffer)
    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float16:
        unary_mixed_op[op, DType.float16, DType.bool](
            out_ptr,
            _get_buffer_ptr[DType.float16](in_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float32:
        unary_mixed_op[op, DType.float32, DType.bool](
            out_ptr,
            _get_buffer_ptr[DType.float32](in_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float64:
        unary_mixed_op[op, DType.float64, DType.bool](
            out_ptr,
            _get_buffer_ptr[DType.float64](in_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        unary_mixed_op[op, DType.bfloat16, DType.bool](
            out_ptr,
            _get_buffer_ptr[DType.bfloat16](in_buffer),
            size,
            ctx,
        )
    else:
        raise Error(
            "Unsupported dtype for unary predicate operation: " + String(dtype)
        )


fn cast_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Cast operation dispatcher that handles double dtype dispatch.

    Args:
        out_buffer: The output buffer object.
        in_buffer: The input buffer object.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var in_dtype = _get_dtype(in_buffer)
    var out_dtype = _get_dtype(out_buffer)
    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    if in_dtype == DType.float16:
        _cast_dispatch_out[DType.float16](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.float32:
        _cast_dispatch_out[DType.float32](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.float64:
        _cast_dispatch_out[DType.float64](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.bfloat16:
        _cast_dispatch_out[DType.bfloat16](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.int8:
        _cast_dispatch_out[DType.int8](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.int16:
        _cast_dispatch_out[DType.int16](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.int32:
        _cast_dispatch_out[DType.int32](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.int64:
        _cast_dispatch_out[DType.int64](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.uint8:
        _cast_dispatch_out[DType.uint8](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.uint16:
        _cast_dispatch_out[DType.uint16](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.uint32:
        _cast_dispatch_out[DType.uint32](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.uint64:
        _cast_dispatch_out[DType.uint64](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.bool:
        _cast_dispatch_out[DType.bool](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    else:
        raise Error("Unsupported input dtype for cast: " + String(in_dtype))


fn _cast_dispatch_out[
    in_dtype: DType
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    out_dtype: DType,
    size: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Second level dispatch for cast: dispatches on output dtype.

    Parameters:
        in_dtype: The input data type (already resolved).

    Args:
        out_buffer: The output buffer object.
        in_buffer: The input buffer object.
        out_dtype: The output data type to dispatch on.
        size: Number of elements.
        ctx: Device context pointer.
    """
    var in_ptr = _get_buffer_ptr[in_dtype](in_buffer)

    if out_dtype == DType.float16:
        unary_mixed_op[Cast, in_dtype, DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.float32:
        unary_mixed_op[Cast, in_dtype, DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.float64:
        unary_mixed_op[Cast, in_dtype, DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.bfloat16:
        unary_mixed_op[Cast, in_dtype, DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.int8:
        unary_mixed_op[Cast, in_dtype, DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.int16:
        unary_mixed_op[Cast, in_dtype, DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.int32:
        unary_mixed_op[Cast, in_dtype, DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.int64:
        unary_mixed_op[Cast, in_dtype, DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.uint8:
        unary_mixed_op[Cast, in_dtype, DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.uint16:
        unary_mixed_op[Cast, in_dtype, DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.uint32:
        unary_mixed_op[Cast, in_dtype, DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.uint64:
        unary_mixed_op[Cast, in_dtype, DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.bool:
        unary_mixed_op[Cast, in_dtype, DType.bool](
            _get_buffer_ptr[DType.bool](out_buffer), in_ptr, size, ctx
        )
    else:
        raise Error("Unsupported output dtype for cast: " + String(out_dtype))


fn select_dispatcher(
    out_buffer: PythonObject,
    cond_buffer: PythonObject,
    true_buffer: PythonObject,
    false_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Select dispatcher with dtype dispatch.

    Performs element-wise: out = cond ? true_val : false_val.

    Args:
        out_buffer: The output buffer object.
        cond_buffer: Boolean condition buffer.
        true_buffer: Values selected where condition is true.
        false_buffer: Values selected where condition is false.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(true_buffer)
    var false_dtype = _get_dtype(false_buffer)
    if dtype != false_dtype:
        raise Error(
            "Mismatched input dtypes for select: "
            + String(dtype)
            + " and "
            + String(false_dtype)
        )

    var cond_dtype = _get_dtype(cond_buffer)
    if cond_dtype != DType.bool:
        raise Error("Select condition must be bool, got: " + String(cond_dtype))

    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float16:
        select_elementwise_op[DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.float16](true_buffer),
            _get_buffer_ptr[DType.float16](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float32:
        select_elementwise_op[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.float32](true_buffer),
            _get_buffer_ptr[DType.float32](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float64:
        select_elementwise_op[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.float64](true_buffer),
            _get_buffer_ptr[DType.float64](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        select_elementwise_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.bfloat16](true_buffer),
            _get_buffer_ptr[DType.bfloat16](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int8:
        select_elementwise_op[DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.int8](true_buffer),
            _get_buffer_ptr[DType.int8](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int16:
        select_elementwise_op[DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.int16](true_buffer),
            _get_buffer_ptr[DType.int16](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int32:
        select_elementwise_op[DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.int32](true_buffer),
            _get_buffer_ptr[DType.int32](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int64:
        select_elementwise_op[DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.int64](true_buffer),
            _get_buffer_ptr[DType.int64](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint8:
        select_elementwise_op[DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.uint8](true_buffer),
            _get_buffer_ptr[DType.uint8](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint16:
        select_elementwise_op[DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.uint16](true_buffer),
            _get_buffer_ptr[DType.uint16](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint32:
        select_elementwise_op[DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.uint32](true_buffer),
            _get_buffer_ptr[DType.uint32](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint64:
        select_elementwise_op[DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.uint64](true_buffer),
            _get_buffer_ptr[DType.uint64](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bool:
        select_elementwise_op[DType.bool](
            _get_buffer_ptr[DType.bool](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.bool](true_buffer),
            _get_buffer_ptr[DType.bool](false_buffer),
            size,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for select operation: " + String(dtype))


# =============================================================================
# Kernel implementations
# =============================================================================


@always_inline
fn bin_elementwise_op[
    op: ElementwiseBinaryOp, dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Binary elementwise operation: out = op(lhs, rhs).

    Parameters:
        op: The binary elementwise operation to perform, expressed as a function
            of two SIMD values.
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        lhs_ptr: Pointer to the left-hand side buffer data.
        rhs_ptr: Pointer to the right-hand side buffer data.
        size: Number of elements to process.
        ctx: Device context pointer (null for CPU).
    """

    @always_inline
    @parameter
    @__copy_capture(out_ptr, lhs_ptr, rhs_ptr)
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]

        var res = op.elementwise(
            lhs_ptr.load[width=width](i), rhs_ptr.load[width=width](i)
        )
        out_ptr.store[width=width](i, res)

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl=True
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](size))
    else:
        # GPU execution - check GPU availability and op/dtype support
        comptime if has_accelerator():
            comptime if _is_gpu_allowed_binary_op[
                op
            ]() and dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for this binary elementwise"
                    " op or dtype"
                )
        else:
            raise Error("No GPU accelerator available")


@always_inline
fn pow_elementwise_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Pow elementwise operation: out = lhs ** rhs.

    Pow has a non-standard signature (separate dtype/pow_dtype params)
    so it cannot use the generic bin_elementwise_op.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        lhs_ptr: Pointer to the base buffer data.
        rhs_ptr: Pointer to the exponent buffer data.
        size: Number of elements to process.
        ctx: Device context pointer (null for CPU).
    """

    @always_inline
    @parameter
    @__copy_capture(out_ptr, lhs_ptr, rhs_ptr)
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]

        var res = Pow.elementwise[dtype, dtype, width](
            lhs_ptr.load[width=width](i), rhs_ptr.load[width=width](i)
        )
        out_ptr.store[width=width](i, res)

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl=True
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](size))
    else:
        # GPU execution - check GPU availability and dtype support
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for pow with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


@always_inline
fn bin_elementwise_comparison_op[
    op: ElementwiseBinaryComparisonOp, dtype: DType
](
    out_ptr: UnsafePointer[Scalar[DType.uint8], MutExternalOrigin],
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Elementwise comparison: out = lhs op rhs.

    Parameters:
        op: The binary elementwise comparison operation to perform, expressed as a function
            of two SIMD values.
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data (uint8 for bool result).
        lhs_ptr: Pointer to the left-hand side buffer data.
        rhs_ptr: Pointer to the right-hand side buffer data.
        size: Number of elements to process.
        ctx: Device context pointer (null for CPU).
    """

    @always_inline
    @parameter
    @__copy_capture(out_ptr, lhs_ptr, rhs_ptr)
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]

        var res = op.elementwise(
            lhs_ptr.load[width=width](i), rhs_ptr.load[width=width](i)
        )
        out_ptr.store[width=width](i, res.cast[DType.uint8]())

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl=True
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](size))
    else:
        # GPU execution - check GPU availability and op/dtype support
        comptime if has_accelerator():
            comptime if _is_gpu_allowed_comparison_op[
                op
            ]() and dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for this comparison op or"
                    " dtype"
                )
        else:
            raise Error("No GPU accelerator available")


@always_inline
fn unary_elementwise_op[
    op: ElementwiseUnaryOp, dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Elementwise unary operation: out = op(input).

    Parameters:
        op: The unary elementwise operation to perform.
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        in_ptr: Pointer to the input buffer data.
        size: Number of elements to process.
        ctx: Device context pointer (null for CPU).
    """

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]

        var res = op.elementwise(in_ptr.load[width=width](i))
        out_ptr.store[width=width](i, res)

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl=True
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](size))
    else:
        # GPU execution - check GPU availability and op/dtype support
        comptime if has_accelerator():
            comptime if _is_gpu_allowed_unary_op[
                op
            ]() and dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for this unary elementwise"
                    " op or dtype"
                )
        else:
            raise Error("No GPU accelerator available")


@always_inline
fn unary_mixed_op[
    op: ElementwiseUnaryMixedOp, dtype: DType, out_dtype: DType
](
    out_ptr: UnsafePointer[Scalar[out_dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Elementwise unary mixed-type operation: out = op(input).

    Parameters:
        op: The unary mixed-type elementwise operation to perform.
        dtype: The input data type.
        out_dtype: The output data type.

    Args:
        out_ptr: Pointer to the output buffer data.
        in_ptr: Pointer to the input buffer data.
        size: Number of elements to process.
        ctx: Device context pointer (null for CPU).
    """

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]

        var res = op.elementwise[dtype, out_dtype, width](
            in_ptr.load[width=width](i)
        )
        out_ptr.store[width=width](i, res)

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl=True
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](size))
    else:
        # GPU execution - check GPU availability and op/dtype support
        comptime if has_accelerator():
            comptime if _is_gpu_allowed_mixed_unary_op[
                op
            ]() and dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for this mixed-type unary"
                    " op or dtype"
                )
        else:
            raise Error("No GPU accelerator available")


@always_inline
fn select_elementwise_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    cond_ptr: UnsafePointer[Scalar[DType.bool], MutExternalOrigin],
    true_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    false_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Select elementwise operation: out = cond ? true_val : false_val.

    Parameters:
        dtype: The data type of the value arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        cond_ptr: Pointer to the condition buffer data (bool).
        true_ptr: Pointer to the true-case buffer data.
        false_ptr: Pointer to the false-case buffer data.
        size: Number of elements to process.
        ctx: Device context pointer (null for CPU).
    """

    @always_inline
    @parameter
    @__copy_capture(out_ptr, cond_ptr, true_ptr, false_ptr)
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]

        var cond = cond_ptr.load[width=width](i)
        var tc = true_ptr.load[width=width](i)
        var fc = false_ptr.load[width=width](i)
        var res = Select.elementwise[DType.bool, dtype, width](cond, tc, fc)
        out_ptr.store[width=width](i, res)

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl=True
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](size))
    else:
        # GPU execution
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for select with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")
