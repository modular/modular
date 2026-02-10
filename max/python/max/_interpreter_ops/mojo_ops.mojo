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
from sys.info import has_accelerator, simd_width_of

from math import exp, iota, log
from random import NormalRandom, Random
from algorithm.functional import elementwise, IndexList
from algorithm import max as reduce_max
from algorithm import min as reduce_min
from algorithm import sum as reduce_sum
from algorithm import mean as reduce_mean
from algorithm import product as reduce_product
from memory import OpaquePointer
from linalg.matmul import matmul
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.runtime_layout import RuntimeLayout
from nn.softmax import softmax as nn_softmax
from reflection import get_base_type_name
from runtime.asyncrt import DeviceContextPtr
from tensor.managed_tensor_slice import ManagedTensorSlice
from tensor.io_spec import Input, Output
from compiler_internal import StaticTensorSpec
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
    StaticBroadcastTo,
    Pow,
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


fn _is_gpu_allowed_matmul_dtype[dtype: DType]() -> Bool:
    """Check if a dtype is allowed for GPU matmul at compile time.

    GPU matmul does not support int8, uint8, int16, uint16, or float64.
    """

    # TODO(MXF-109): Add support for other dtypes.
    return (
        dtype == DType.float32
        or dtype == DType.float16
        or dtype == DType.bfloat16
    )


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_mojo_ops() -> PythonObject:
    """Create a Python module with kernel function bindings."""
    try:
        var b = PythonModuleBuilder("mojo_ops")

        # Register dtype-dispatching functions

        # Binary arithmetic operations
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

        # Binary boolean operations
        @parameter
        for i in range(Variadic.size(BINARY_BOOLEAN_OPS)):
            comptime op = BINARY_BOOLEAN_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString(
                "Elementwise " + name + " (boolean only)"
            )
            b.def_function[bin_bool_dispatcher[op]](name, docstring=docstring)

        # Binary comparison operations
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

        # Unary elementwise operations
        @parameter
        for i in range(Variadic.size(UNARY_ELEMENTWISE_OPS)):
            comptime op = UNARY_ELEMENTWISE_OPS[i]
            comptime name = get_base_type_name[op]()
            comptime docstring = StaticString("Elementwise " + name)
            b.def_function[unary_elementwise_dispatcher[op]](
                name, docstring=docstring
            )

        # Unary float-only operations
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

        # Unary predicate operations (float -> bool)
        @parameter
        for i in range(Variadic.size(UNARY_PREDICATE_OPS)):
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

        # Matrix multiplication
        b.def_function[matmul_dispatcher](
            "Matmul", docstring="Matrix multiplication"
        )

        # Range operation
        b.def_function[range_dispatcher]("Range", docstring="Range operation")

        # Reduce operations
        b.def_function[reduce_max_dispatcher](
            "ReduceMax", docstring="Reduce max along axis"
        )
        b.def_function[reduce_min_dispatcher](
            "ReduceMin", docstring="Reduce min along axis"
        )
        b.def_function[reduce_sum_dispatcher](
            "ReduceAdd", docstring="Reduce add along axis"
        )
        b.def_function[mean_dispatcher]("Mean", docstring="Mean along axis")
        b.def_function[reduce_mul_dispatcher](
            "ReduceMul", docstring="Reduce mul along axis"
        )

        # Static broadcast to operation
        b.def_function[static_broadcast_to_dispatcher](
            "StaticBroadcastTo", docstring="Static broadcast to"
        )

        # Softmax operations
        b.def_function[softmax_dispatcher](
            "Softmax", docstring="Softmax along axis"
        )
        b.def_function[logsoftmax_dispatcher](
            "LogSoftmax", docstring="LogSoftmax along axis"
        )

        # Pow operation (custom dispatch - Pow doesn't conform to
        # ElementwiseBinaryOp)
        b.def_function[pow_dispatcher]("Pow", docstring="Elementwise Pow")

        # Random normal operation
        b.def_function[random_normal_dispatcher](
            "RandomNormal", docstring="Random normal distribution"
        )

        # Random uniform operation
        b.def_function[random_uniform_dispatcher](
            "RandomUniform", docstring="Random uniform distribution"
        )

        # Select operation (ternary: cond ? x : y)
        b.def_function[select_dispatcher](
            "Select", docstring="Elementwise select (cond ? x : y)"
        )

        return b.finalize()
    except e:
        abort(String("failed to create interpreter op bindings module: ", e))


fn _get_dtype(buffer: PythonObject) raises -> DType:
    return DType._from_ui8(UInt8(py=buffer.dtype.value)._mlir_value)


# Helper to extract buffer pointer with dtype
fn _get_buffer_ptr[
    dtype: DType
](buffer: PythonObject) raises -> UnsafePointer[
    Scalar[dtype], MutExternalOrigin
]:
    return UnsafePointer[Scalar[dtype], MutExternalOrigin](
        unsafe_from_address=Int(py=buffer._data_ptr())
    )


fn _get_size(buffer: PythonObject) raises -> Int:
    return Int(py=buffer.num_elements)


fn _get_ctx(
    device_context_ptr: PythonObject,
) raises -> OpaquePointer[MutExternalOrigin]:
    return OpaquePointer[MutExternalOrigin](
        unsafe_from_address=Int(py=device_context_ptr)
    )


comptime MAX_RANK = 5


fn _get_shape(
    shape_obj: PythonObject, rank: Int
) raises -> InlineArray[Int, MAX_RANK]:
    """Extract shape as InlineArray from Python sequence.

    Args:
        shape_obj: Python sequence containing the shape.
        rank: The rank of the shape.

    Returns:
        The shape as an InlineArray (only first `rank` elements are valid).
    """
    if rank > MAX_RANK:
        raise Error(
            "Tensor rank "
            + String(rank)
            + " exceeds MAX_RANK "
            + String(MAX_RANK)
        )
    var result = InlineArray[Int, MAX_RANK](fill=0)
    for i in range(rank):
        result[i] = Int(py=shape_obj[i])
    return result^


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

    @parameter
    if float_only:
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
        @parameter
        if has_accelerator():

            @parameter
            if _is_gpu_allowed_binary_op[op]() and dtype != DType.float64:
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
        @parameter
        if has_accelerator():

            @parameter
            if dtype != DType.float64:
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
        @parameter
        if has_accelerator():

            @parameter
            if _is_gpu_allowed_comparison_op[op]() and dtype != DType.float64:
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
        @parameter
        if has_accelerator():

            @parameter
            if _is_gpu_allowed_unary_op[op]() and dtype != DType.float64:
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
        @parameter
        if has_accelerator():

            @parameter
            if _is_gpu_allowed_mixed_unary_op[op]() and dtype != DType.float64:
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


# ===----------------------------------------------------------------------=== #
# Matmul operation
# ===----------------------------------------------------------------------=== #


fn matmul_dispatcher(
    out_buffer: PythonObject,
    lhs_buffer: PythonObject,
    rhs_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Matmul dispatcher with dtype dispatch.

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
            "Mismatched input dtypes for matmul: "
            + String(dtype)
            + " and "
            + String(rhs_dtype)
        )

    # Extract shapes: lhs is (M, K), rhs is (K, N), out is (M, N)
    var lhs_shape = lhs_buffer.shape
    var m = Int(py=lhs_shape[0])
    var k = Int(py=lhs_shape[1])
    var rhs_shape = rhs_buffer.shape
    var n = Int(py=rhs_shape[1])

    var ctx = _get_ctx(device_context_ptr)

    # Float types
    if dtype == DType.float16:
        matmul_op[DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](lhs_buffer),
            _get_buffer_ptr[DType.float16](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.float32:
        matmul_op[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](lhs_buffer),
            _get_buffer_ptr[DType.float32](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.float64:
        matmul_op[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](lhs_buffer),
            _get_buffer_ptr[DType.float64](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.bfloat16:
        matmul_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](lhs_buffer),
            _get_buffer_ptr[DType.bfloat16](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    # Integer types
    elif dtype == DType.int8:
        matmul_op[DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.int8](lhs_buffer),
            _get_buffer_ptr[DType.int8](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.int16:
        matmul_op[DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.int16](lhs_buffer),
            _get_buffer_ptr[DType.int16](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.int32:
        matmul_op[DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.int32](lhs_buffer),
            _get_buffer_ptr[DType.int32](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.int64:
        matmul_op[DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.int64](lhs_buffer),
            _get_buffer_ptr[DType.int64](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.uint8:
        matmul_op[DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.uint8](lhs_buffer),
            _get_buffer_ptr[DType.uint8](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.uint16:
        matmul_op[DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer),
            _get_buffer_ptr[DType.uint16](lhs_buffer),
            _get_buffer_ptr[DType.uint16](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.uint32:
        matmul_op[DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer),
            _get_buffer_ptr[DType.uint32](lhs_buffer),
            _get_buffer_ptr[DType.uint32](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    elif dtype == DType.uint64:
        matmul_op[DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer),
            _get_buffer_ptr[DType.uint64](lhs_buffer),
            _get_buffer_ptr[DType.uint64](rhs_buffer),
            m,
            k,
            n,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for matmul: " + String(dtype))


@always_inline
fn matmul_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    m: Int,
    k: Int,
    n: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Matrix multiplication: out = lhs @ rhs.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        lhs_ptr: Pointer to the left-hand side buffer data.
        rhs_ptr: Pointer to the right-hand side buffer data.
        m: Number of rows in lhs and output.
        k: Number of columns in lhs / rows in rhs.
        n: Number of columns in rhs and output.
        ctx: Device context pointer (null for CPU).
    """
    # Define static layout type with unknown dimensions for 2D row-major matrices
    comptime layout_2d = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime LayoutType = RuntimeLayout[layout_2d]

    # Create LayoutTensors with runtime shapes
    var c = LayoutTensor[dtype, layout_2d, MutExternalOrigin](
        out_ptr, LayoutType.row_major(IndexList[2](m, n))
    )
    var a = LayoutTensor[dtype, layout_2d, MutExternalOrigin](
        lhs_ptr, LayoutType.row_major(IndexList[2](m, k))
    )
    var b = LayoutTensor[dtype, layout_2d, MutExternalOrigin](
        rhs_ptr, LayoutType.row_major(IndexList[2](k, n))
    )

    if not ctx:
        # TODO(MXF-108): Remove single_thread_blocking_override
        matmul[target="cpu", single_thread_blocking_override=True](
            c, a, b, None
        )
    else:
        # GPU execution - check GPU availability and dtype support
        @parameter
        if has_accelerator():

            @parameter
            if _is_gpu_allowed_matmul_dtype[dtype]():
                var device_ctx = DeviceContextPtr(ctx)
                matmul[target="gpu"](c, a, b, device_ctx.get_device_context())
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for matmul with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")


# ===----------------------------------------------------------------------=== #
# Range operation
# ===----------------------------------------------------------------------=== #


fn range_dispatcher(
    out_buffer: PythonObject,
    start_buffer: PythonObject,
    step_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Range dispatcher with dtype dispatch.

    Fills output buffer with values: out[i] = start + i * step.

    Args:
        out_buffer: The output buffer object.
        start_buffer: Scalar buffer containing the start value.
        step_buffer: Scalar buffer containing the step value.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(out_buffer)
    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    # Float types
    if dtype == DType.float16:
        range_op[DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](start_buffer),
            _get_buffer_ptr[DType.float16](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float32:
        range_op[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](start_buffer),
            _get_buffer_ptr[DType.float32](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float64:
        range_op[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](start_buffer),
            _get_buffer_ptr[DType.float64](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        range_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](start_buffer),
            _get_buffer_ptr[DType.bfloat16](step_buffer),
            size,
            ctx,
        )
    # Integer types
    elif dtype == DType.int8:
        range_op[DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.int8](start_buffer),
            _get_buffer_ptr[DType.int8](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int16:
        range_op[DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.int16](start_buffer),
            _get_buffer_ptr[DType.int16](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int32:
        range_op[DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.int32](start_buffer),
            _get_buffer_ptr[DType.int32](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int64:
        range_op[DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.int64](start_buffer),
            _get_buffer_ptr[DType.int64](step_buffer),
            size,
            ctx,
        )
    # Unsigned integer types
    elif dtype == DType.uint8:
        range_op[DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.uint8](start_buffer),
            _get_buffer_ptr[DType.uint8](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint16:
        range_op[DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer),
            _get_buffer_ptr[DType.uint16](start_buffer),
            _get_buffer_ptr[DType.uint16](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint32:
        range_op[DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer),
            _get_buffer_ptr[DType.uint32](start_buffer),
            _get_buffer_ptr[DType.uint32](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint64:
        range_op[DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer),
            _get_buffer_ptr[DType.uint64](start_buffer),
            _get_buffer_ptr[DType.uint64](step_buffer),
            size,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for range: " + String(dtype))


fn range_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    start_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    step_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Range operation: out[i] = start + i * step.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        start_ptr: Pointer to the start scalar value.
        step_ptr: Pointer to the step scalar value.
        size: Number of elements to produce.
        ctx: Device context pointer (null for CPU).
    """
    var start = start_ptr.load()
    var step = step_ptr.load()

    @always_inline
    @parameter
    @__copy_capture(out_ptr, start, step)
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]
        var result = start + (iota[dtype, width](Scalar[dtype](i)) * step)
        out_ptr.store[width=width](i, result)

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl=True
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](size))
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for range with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


# ===----------------------------------------------------------------------=== #
# Random normal operation
# ===----------------------------------------------------------------------=== #


fn random_normal_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    mean: Float32,
    variance: Float32,
    seed_value: UInt64,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Random normal operation: fill output with normally distributed values.

    Parameters:
        dtype: The data type of the output array.

    Args:
        out_ptr: Pointer to the output buffer data.
        size: Number of elements to produce.
        mean: Mean of the normal distribution.
        variance: Standard deviation of the normal distribution.
        seed_value: Seed for the random number generator.
        ctx: Device context pointer (null for CPU).
    """
    if variance <= 0:
        raise Error("stddev must be positive")

    @always_inline
    @parameter
    @__copy_capture(out_ptr, mean, variance, seed_value)
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]
        var generator = NormalRandom(seed=seed_value, offset=UInt64(i))
        var values = generator.step_normal(mean=mean, stddev=variance)
        out_ptr.store[width=width](i, values.cast[dtype]().slice[width]())

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl=True
        elementwise[func, simd_width=8, use_blocking_impl=True](
            IndexList[1](size)
        )
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=8, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for random_normal"
                    " with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


fn random_normal_dispatcher(
    out_buffer: PythonObject,
    mean_val: PythonObject,
    variance_val: PythonObject,
    seed_val: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Random normal dispatcher with dtype dispatch.

    Args:
        out_buffer: The output buffer object.
        mean_val: Python float for the mean.
        variance_val: Python float for the standard deviation.
        seed_val: Python int for the seed.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(out_buffer)
    var size = _get_size(out_buffer)
    var mean = Float32(py=mean_val)
    var variance = Float32(py=variance_val)
    var seed = UInt64(Int(py=seed_val))
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        random_normal_op[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            size,
            mean,
            variance,
            seed,
            ctx,
        )
    elif dtype == DType.float64:
        random_normal_op[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            size,
            mean,
            variance,
            seed,
            ctx,
        )
    elif dtype == DType.float16:
        random_normal_op[DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            size,
            mean,
            variance,
            seed,
            ctx,
        )
    elif dtype == DType.bfloat16:
        random_normal_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            size,
            mean,
            variance,
            seed,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for random_normal: " + String(dtype))


# ===----------------------------------------------------------------------=== #
# Random uniform operation
# ===----------------------------------------------------------------------=== #


fn random_uniform_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    lower_bound: Float32,
    upper_bound: Float32,
    seed_value: UInt64,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Random uniform operation: fill output with uniformly distributed values.

    Parameters:
        dtype: The data type of the output array.

    Args:
        out_ptr: Pointer to the output buffer data.
        size: Number of elements to produce.
        lower_bound: Lower bound of the uniform distribution.
        upper_bound: Upper bound of the uniform distribution.
        seed_value: Seed for the random number generator.
        ctx: Device context pointer (null for CPU).
    """
    if lower_bound > upper_bound:
        raise Error("lower_bound must be less than or equal to upper_bound")

    var delta = upper_bound - lower_bound

    @always_inline
    @parameter
    @__copy_capture(out_ptr, lower_bound, delta, seed_value)
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]
        var generator = Random(seed=seed_value, offset=UInt64(i))
        var values: SIMD[DType.float32, 4] = generator.step_uniform()
        values = values * delta + lower_bound
        out_ptr.store[width=width](i, values.cast[dtype]().slice[width]())

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl=True
        elementwise[func, simd_width=4, use_blocking_impl=True](
            IndexList[1](size)
        )
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=4, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for random_uniform"
                    " with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


fn random_uniform_dispatcher(
    out_buffer: PythonObject,
    lower_val: PythonObject,
    upper_val: PythonObject,
    seed_val: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Random uniform dispatcher with dtype dispatch.

    Args:
        out_buffer: The output buffer object.
        lower_val: Python float for the lower bound.
        upper_val: Python float for the upper bound.
        seed_val: Python int for the seed.
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(out_buffer)
    var size = _get_size(out_buffer)
    var lower_bound = Float32(py=lower_val)
    var upper_bound = Float32(py=upper_val)
    var seed = UInt64(Int(py=seed_val))
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        random_uniform_op[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            size,
            lower_bound,
            upper_bound,
            seed,
            ctx,
        )
    elif dtype == DType.float64:
        random_uniform_op[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            size,
            lower_bound,
            upper_bound,
            seed,
            ctx,
        )
    elif dtype == DType.float16:
        random_uniform_op[DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            size,
            lower_bound,
            upper_bound,
            seed,
            ctx,
        )
    elif dtype == DType.bfloat16:
        random_uniform_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            size,
            lower_bound,
            upper_bound,
            seed,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for random_uniform: " + String(dtype))


# ===----------------------------------------------------------------------=== #
# Reduce operations (max, min, sum, mean)
# ===----------------------------------------------------------------------=== #

# Function type shared by reduce_max, reduce_min, reduce_sum, and
# _reduce_mean. Each takes (input_shape, reduce_dim, context) with
# compile-time dtype, input/output lambdas, and target parameters.
comptime ReduceFn = fn[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) capturing raises -> None


fn _reduce_max[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) raises:
    """Non-overloaded wrapper around algorithm.max for use with ReduceFn."""
    reduce_max[
        dtype,
        input_fn,
        output_fn,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](input_shape, reduce_dim, context)


fn _reduce_min[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) raises:
    """Non-overloaded wrapper around algorithm.min for use with ReduceFn."""
    reduce_min[
        dtype,
        input_fn,
        output_fn,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](input_shape, reduce_dim, context)


fn _reduce_sum[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) raises:
    """Non-overloaded wrapper around algorithm.sum for use with ReduceFn."""
    reduce_sum[
        dtype,
        input_fn,
        output_fn,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](input_shape, reduce_dim, context)


fn _reduce_mean[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) raises:
    """Wrapper around algorithm.mean matching the reduce_max/min/sum signature.

    Computes output_shape (reduction axis set to 1) and forwards to
    reduce_mean which requires it as an extra argument.
    """
    var output_shape = input_shape
    output_shape[reduce_dim] = 1
    reduce_mean[
        dtype,
        input_fn,
        output_fn,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](input_shape, reduce_dim, output_shape, context)


fn _reduce_mul[
    dtype: DType,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type = DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) raises:
    """Non-overloaded wrapper around algorithm.product for use with ReduceFn."""
    reduce_product[
        dtype,
        input_fn,
        output_fn,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](input_shape, reduce_dim, context)


fn reduce_dispatcher[
    reduce_fn: ReduceFn
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Reduce dispatcher with dtype dispatch.

    Parameters:
        reduce_fn: The reduction algorithm function (e.g. reduce_max,
            reduce_min, reduce_sum, _reduce_mean).

    Args:
        out_buffer: The output buffer object (reduced shape).
        in_buffer: The input buffer object.
        axis: The axis along which to reduce (integer).
        device_context_ptr: Device context pointer (must be null for CPU).
    """
    var dtype = _get_dtype(in_buffer)
    var axis_val = Int(py=axis)
    var ctx = _get_ctx(device_context_ptr)

    # Extract input shape and compute normalized rank-3 shape:
    # dim0: product of dims before axis
    # dim1: the reduction axis dimension
    # dim2: product of dims after axis
    var in_shape_py = in_buffer.shape
    var rank = Int(py=len(in_shape_py))
    var in_shape = _get_shape(in_shape_py, rank)

    var dim0 = 1
    for i in range(axis_val):
        dim0 *= in_shape[i]

    var dim1 = in_shape[axis_val]

    var dim2 = 1
    for i in range(axis_val + 1, rank):
        dim2 *= in_shape[i]

    var normalized_shape = IndexList[3](dim0, dim1, dim2)

    # Float types
    if dtype == DType.float16:
        reduce_op[DType.float16, reduce_fn](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.float32:
        reduce_op[DType.float32, reduce_fn](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.float64:
        reduce_op[DType.float64, reduce_fn](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.bfloat16:
        reduce_op[DType.bfloat16, reduce_fn](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](in_buffer),
            normalized_shape,
            ctx,
        )
    # Integer types
    elif dtype == DType.int8:
        reduce_op[DType.int8, reduce_fn](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.int8](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.int16:
        reduce_op[DType.int16, reduce_fn](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.int16](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.int32:
        reduce_op[DType.int32, reduce_fn](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.int32](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.int64:
        reduce_op[DType.int64, reduce_fn](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.int64](in_buffer),
            normalized_shape,
            ctx,
        )
    # Unsigned integer types
    elif dtype == DType.uint8:
        reduce_op[DType.uint8, reduce_fn](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.uint8](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.uint16:
        reduce_op[DType.uint16, reduce_fn](
            _get_buffer_ptr[DType.uint16](out_buffer),
            _get_buffer_ptr[DType.uint16](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.uint32:
        reduce_op[DType.uint32, reduce_fn](
            _get_buffer_ptr[DType.uint32](out_buffer),
            _get_buffer_ptr[DType.uint32](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.uint64:
        reduce_op[DType.uint64, reduce_fn](
            _get_buffer_ptr[DType.uint64](out_buffer),
            _get_buffer_ptr[DType.uint64](in_buffer),
            normalized_shape,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for reduce: " + String(dtype))


fn reduce_op[
    dtype: DType,
    reduce_fn: ReduceFn,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    normalized_shape: IndexList[3],
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Reduce operation on a rank-3 normalized tensor.

    Parameters:
        dtype: The data type of the arrays.
        reduce_fn: The reduction algorithm function.

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        normalized_shape: The normalized rank-3 shape [dim0, dim1, dim2] where
            dim0 is the product of dims before the reduction axis,
            dim1 is the reduction axis dimension,
            dim2 is the product of dims after the reduction axis.
        ctx: Device context pointer.
    """

    # Compute strides
    var dim1 = normalized_shape[1]
    var dim2 = normalized_shape[2]
    var inStride0 = dim1 * dim2
    var inStride1 = dim2
    var outStride0 = dim2

    # Define input function mapping rank-3 coords to flat index
    @always_inline
    @parameter
    @__copy_capture(in_ptr, inStride0, inStride1)
    fn input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var c = rebind[IndexList[3]](coords)
        var flat_idx = c[0] * inStride0 + c[1] * inStride1 + c[2]
        return in_ptr.load[width=width](flat_idx)

    # Define output function mapping rank-3 coords to flat index
    @always_inline
    @parameter
    @__copy_capture(out_ptr, outStride0)
    fn output_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]):
        var c = rebind[IndexList[3]](coords)
        var flat_idx = c[0] * outStride0 + c[2]
        out_ptr.store[width=width](flat_idx, val)

    # Always dispatch rank-3 reduction with axis=1
    if not ctx:
        # TODO(MXF-108): Remove single_thread_blocking_override
        reduce_fn[
            dtype,
            input_fn,
            output_fn,
            single_thread_blocking_override=True,
            target="cpu",
        ](normalized_shape, 1, DeviceContextPtr(ctx))
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype in (
                DType.float32,
                DType.float16,
                DType.bfloat16,
                DType.int32,
                DType.uint32,
                DType.int64,
                DType.uint64,
            ):
                var device_ctx = DeviceContextPtr(ctx)
                reduce_fn[
                    dtype,
                    input_fn,
                    output_fn,
                    target="gpu",
                ](normalized_shape, 1, device_ctx)
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for reduce with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")


# Concrete dispatcher functions for def_function registration.
# def_function requires fully concrete function types, so we can't pass
# reduce_dispatcher[_reduce_max] directly (parametric fn type can't be inferred).


fn reduce_max_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    reduce_dispatcher[_reduce_max](
        out_buffer, in_buffer, axis, device_context_ptr
    )


fn reduce_min_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    reduce_dispatcher[_reduce_min](
        out_buffer, in_buffer, axis, device_context_ptr
    )


fn reduce_sum_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    reduce_dispatcher[_reduce_sum](
        out_buffer, in_buffer, axis, device_context_ptr
    )


fn mean_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    reduce_dispatcher[_reduce_mean](
        out_buffer, in_buffer, axis, device_context_ptr
    )


fn reduce_mul_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    reduce_dispatcher[_reduce_mul](
        out_buffer, in_buffer, axis, device_context_ptr
    )


# ===----------------------------------------------------------------------=== #
# StaticBroadcastTo operation
# ===----------------------------------------------------------------------=== #


fn _pad_shape_to_rank5(
    shape_obj: PythonObject, rank: Int
) raises -> IndexList[MAX_RANK]:
    """Pad shape with leading 1s to make it rank-5."""
    var padded = IndexList[MAX_RANK]()
    var pad_count = MAX_RANK - rank
    for i in range(pad_count):
        padded[i] = 1
    for i in range(rank):
        padded[pad_count + i] = Int(py=shape_obj[i])
    return padded


@always_inline
fn static_broadcast_to_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_shape: IndexList[MAX_RANK],
    out_shape: IndexList[MAX_RANK],
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Call StaticBroadcastTo.execute with rank-5 tensors.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        in_ptr: Pointer to the input buffer data.
        in_shape: Padded input shape (rank-5).
        out_shape: Padded output shape (rank-5).
        ctx: Device context pointer (null for CPU).
    """
    # Create ManagedTensorSlice wrappers
    comptime in_spec = StaticTensorSpec[dtype, MAX_RANK].create_unknown()
    comptime out_spec = StaticTensorSpec[dtype, MAX_RANK].create_unknown()

    var input_tensor = ManagedTensorSlice[io_spec=Input, static_spec=in_spec](
        in_ptr, in_shape
    )

    var output_tensor = ManagedTensorSlice[
        io_spec=Output, static_spec=out_spec
    ](out_ptr, out_shape)

    if not ctx:
        # TODO(MXF-108): Remove use_blocking_impl
        StaticBroadcastTo.execute[
            target="cpu",
            dtype=dtype,
            in_rank=MAX_RANK,
            out_rank=MAX_RANK,
            _trace_name="interpreter.static_broadcast_to",
            use_blocking_impl=True,
        ](output_tensor, input_tensor, out_shape, DeviceContextPtr())
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                StaticBroadcastTo.execute[
                    target="gpu",
                    dtype=dtype,
                    in_rank=MAX_RANK,
                    out_rank=MAX_RANK,
                    _trace_name="interpreter.static_broadcast_to",
                ](output_tensor, input_tensor, out_shape, device_ctx)
                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for static_broadcast_to"
                    " with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")


fn static_broadcast_to_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    out_shape_obj: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """StaticBroadcastTo dispatcher - unwraps PythonObjects and dispatches.

    Pads shapes to rank-5 with leading 1s and dispatches a single rank-5
    broadcast operation.
    """
    # Unwrap all PythonObjects upfront
    var dtype = _get_dtype(in_buffer)
    var in_shape_obj = in_buffer.shape
    var in_rank = Int(py=len(in_shape_obj))
    var out_rank = Int(py=len(out_shape_obj))
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    # Validate ranks
    if in_rank > MAX_RANK or out_rank > MAX_RANK:
        raise Error(
            "Unsupported rank: in_rank="
            + String(in_rank)
            + ", out_rank="
            + String(out_rank)
            + ". Max supported rank is "
            + String(MAX_RANK)
        )

    # Pad shapes to rank-5 with leading 1s
    var padded_in_shape = _pad_shape_to_rank5(in_shape_obj, in_rank)
    var padded_out_shape = _pad_shape_to_rank5(out_shape_obj, out_rank)

    @always_inline
    fn _make_ptr[
        dtype: DType
    ](addr: Int) -> UnsafePointer[Scalar[dtype], MutExternalOrigin]:
        return UnsafePointer[Scalar[dtype], MutExternalOrigin](
            unsafe_from_address=addr
        )

    # Dispatch by dtype
    if dtype == DType.float32:
        static_broadcast_to_op[DType.float32](
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.float64:
        static_broadcast_to_op[DType.float64](
            _make_ptr[DType.float64](out_addr),
            _make_ptr[DType.float64](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.float16:
        static_broadcast_to_op[DType.float16](
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.bfloat16:
        static_broadcast_to_op[DType.bfloat16](
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.int8:
        static_broadcast_to_op[DType.int8](
            _make_ptr[DType.int8](out_addr),
            _make_ptr[DType.int8](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.int16:
        static_broadcast_to_op[DType.int16](
            _make_ptr[DType.int16](out_addr),
            _make_ptr[DType.int16](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.int32:
        static_broadcast_to_op[DType.int32](
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.int64:
        static_broadcast_to_op[DType.int64](
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.uint8:
        static_broadcast_to_op[DType.uint8](
            _make_ptr[DType.uint8](out_addr),
            _make_ptr[DType.uint8](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.uint16:
        static_broadcast_to_op[DType.uint16](
            _make_ptr[DType.uint16](out_addr),
            _make_ptr[DType.uint16](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.uint32:
        static_broadcast_to_op[DType.uint32](
            _make_ptr[DType.uint32](out_addr),
            _make_ptr[DType.uint32](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.uint64:
        static_broadcast_to_op[DType.uint64](
            _make_ptr[DType.uint64](out_addr),
            _make_ptr[DType.uint64](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    elif dtype == DType.bool:
        static_broadcast_to_op[DType.bool](
            _make_ptr[DType.bool](out_addr),
            _make_ptr[DType.bool](in_addr),
            padded_in_shape,
            padded_out_shape,
            ctx,
        )
    else:
        raise Error(
            "Unsupported dtype for static_broadcast_to: " + String(dtype)
        )


# ===----------------------------------------------------------------------=== #
# Softmax / LogSoftmax operations
# ===----------------------------------------------------------------------=== #


fn _softmax_cpu[
    dtype: DType,
    is_logsoftmax: Bool,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    batch_dim: Int,
    axis_dim: Int,
) where dtype.is_floating_point():
    """CPU softmax/logsoftmax on a rank-2 [batch, axis_dim] buffer.

    Uses a numerically stable 3-pass algorithm:
    1. Find max along axis for numerical stability
    2. Compute exp(x - max) and accumulate sum
    3. Normalize (divide by sum, or subtract log(sum) for logsoftmax)

    Parameters:
        dtype: The data type (must be floating point).
        is_logsoftmax: If True, compute log(softmax(x)).

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        batch_dim: Number of rows (batch dimension).
        axis_dim: Size of the softmax axis.
    """
    for row in range(batch_dim):
        var offset = row * axis_dim

        # Pass 1: find max for numerical stability
        var max_val = in_ptr[offset]
        for i in range(1, axis_dim):
            var v = in_ptr[offset + i]
            if v > max_val:
                max_val = v

        # Pass 2: compute exp(x - max) and accumulate sum
        var sum_val = Scalar[dtype](0)
        for i in range(axis_dim):
            var exp_val = exp(in_ptr[offset + i] - max_val)
            out_ptr[offset + i] = exp_val
            sum_val += exp_val

        # Pass 3: normalize
        @parameter
        if is_logsoftmax:
            var log_sum = log(sum_val)
            for i in range(axis_dim):
                out_ptr[offset + i] = in_ptr[offset + i] - max_val - log_sum
        else:
            for i in range(axis_dim):
                out_ptr[offset + i] /= sum_val


fn softmax_op[
    dtype: DType,
    is_logsoftmax: Bool,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    shape: IndexList[2],
    ctx: OpaquePointer[MutExternalOrigin],
) raises where dtype.is_floating_point():
    """Softmax/LogSoftmax operation on a rank-2 normalized tensor.

    The input is normalized to rank-2 [batch, axis_dim] with softmax applied
    along axis=1 (the last axis).

    Parameters:
        dtype: The data type of the arrays.
        is_logsoftmax: If True, compute log(softmax(x)) instead of softmax(x).

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        shape: The normalized rank-2 shape [batch_dim, axis_dim].
        ctx: Device context pointer.
    """
    var batch_dim = shape[0]
    var axis_dim = shape[1]

    if not ctx:
        # CPU path: use direct implementation to avoid runtime dependency
        # (nn.softmax requires AsyncRT parallelism_level which isn't
        # available in the interpreter context)
        _softmax_cpu[dtype, is_logsoftmax](out_ptr, in_ptr, batch_dim, axis_dim)
    else:

        @parameter
        if has_accelerator():

            @parameter
            if dtype in (DType.float32, DType.float16, DType.bfloat16):
                # GPU path: use nn.softmax kernel via input_fn + LayoutTensor
                @always_inline
                @parameter
                @__copy_capture(in_ptr, axis_dim)
                fn input_fn[
                    width: Int, rank: Int
                ](coords: IndexList[rank]) -> SIMD[dtype, width]:
                    var c = rebind[IndexList[2]](coords)
                    var flat_idx = c[0] * axis_dim + c[1]
                    return in_ptr.load[width=width](flat_idx)

                comptime out_layout = Layout.row_major(
                    UNKNOWN_VALUE, UNKNOWN_VALUE
                )
                var rt = RuntimeLayout[out_layout].row_major(shape)
                var output_tensor = LayoutTensor[
                    dtype,
                    out_layout,
                    MutExternalOrigin,
                ](out_ptr, rt)

                var device_ctx = DeviceContextPtr(ctx)

                # Always use nn_softmax (not nn_logsoftmax) on GPU
                # because the GPU logsoftmax kernel has a bug (KERN-2447)
                # where the log() is applied outside the normalization
                # loop. For logsoftmax, we compute softmax then apply
                # log element-wise below.
                nn_softmax[
                    dtype,
                    simd_width_of[dtype](),
                    2,
                    input_fn,
                    target="gpu",
                ](shape, output_tensor, 1, device_ctx)

                @parameter
                if is_logsoftmax:
                    var total = batch_dim * axis_dim

                    @always_inline
                    @parameter
                    @__copy_capture(out_ptr)
                    fn log_fn[
                        width: Int, rank: Int, alignment: Int = 1
                    ](idx: IndexList[rank]):
                        var i = rebind[IndexList[1]](idx)[0]
                        out_ptr.store(i, log(out_ptr.load[width=width](i)))

                    elementwise[log_fn, simd_width=1, target="gpu"](
                        IndexList[1](total), device_ctx
                    )

                # TODO(MXF-108): Remove device sync
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for softmax with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")


fn _softmax_dispatch[
    is_logsoftmax: Bool,
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Softmax/LogSoftmax dispatcher with dtype dispatch.

    Normalizes the input to rank-2 [batch, axis_dim] and dispatches by dtype.

    Parameters:
        is_logsoftmax: If True, compute log(softmax(x)).
    """
    var dtype = _get_dtype(in_buffer)
    var axis_val = Int(py=axis)
    var ctx = _get_ctx(device_context_ptr)

    # Extract input shape
    var in_shape_py = in_buffer.shape
    var rank = Int(py=len(in_shape_py))
    var in_shape = _get_shape(in_shape_py, rank)

    # Validate axis is the last dimension (kernel limitation)
    if axis_val != rank - 1:
        raise Error(
            "softmax only supports the last axis, got axis="
            + String(axis_val)
            + " for rank="
            + String(rank)
        )

    # Normalize to rank-2: [batch_dim, axis_dim]
    var axis_dim = in_shape[axis_val]
    var batch_dim = 1
    for i in range(rank - 1):
        batch_dim *= in_shape[i]

    var normalized_shape = IndexList[2](batch_dim, axis_dim)

    # Dispatch by dtype (float only)
    if dtype == DType.float16:
        softmax_op[DType.float16, is_logsoftmax](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.float32:
        softmax_op[DType.float32, is_logsoftmax](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.float64:
        softmax_op[DType.float64, is_logsoftmax](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](in_buffer),
            normalized_shape,
            ctx,
        )
    elif dtype == DType.bfloat16:
        softmax_op[DType.bfloat16, is_logsoftmax](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](in_buffer),
            normalized_shape,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for softmax: " + String(dtype))


# Concrete dispatcher functions for def_function registration.


fn softmax_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    _softmax_dispatch[is_logsoftmax=False](
        out_buffer, in_buffer, axis, device_context_ptr
    )


fn logsoftmax_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    _softmax_dispatch[is_logsoftmax=True](
        out_buffer, in_buffer, axis, device_context_ptr
    )


# ===----------------------------------------------------------------------=== #
# Select operation (ternary: cond ? x : y)
# ===----------------------------------------------------------------------=== #


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
        @parameter
        if has_accelerator():

            @parameter
            if dtype != DType.float64:
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
