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
from sys.info import (
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    simd_width_of,
)

from algorithm.functional import elementwise, IndexList
from memory import OpaquePointer
from reflection import get_base_type_name
from runtime.asyncrt import DeviceContextPtr
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

# =============================================================================
# GPU Support Configuration
# =============================================================================
# Operations that are allowed to run on GPU. Operations not in these lists
# will raise an error when GPU execution is attempted.
# Operations using libm (like atanh, log1p, erf) don't work on GPU.


fn _has_gpu() -> Bool:
    """Check if any GPU accelerator is available at compile time."""
    return has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()


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
        # CPU execution
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](size))
    else:
        # GPU execution - check GPU availability and op/dtype support
        @parameter
        if _has_gpu():

            @parameter
            if _is_gpu_allowed_binary_op[op]() and dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # Synchronize to ensure GPU operations complete
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for this binary elementwise"
                    " op or dtype"
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
        # CPU execution
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](size))
    else:
        # GPU execution - check GPU availability and op/dtype support
        @parameter
        if _has_gpu():

            @parameter
            if _is_gpu_allowed_comparison_op[op]() and dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # Synchronize to ensure GPU operations complete
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
        # CPU execution
        elementwise[
            func, simd_width = simd_width_of[dtype](), use_blocking_impl=True
        ](IndexList[1](size))
    else:
        # GPU execution - check GPU availability and op/dtype support
        @parameter
        if _has_gpu():

            @parameter
            if _is_gpu_allowed_unary_op[op]() and dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=1, target="gpu"](
                    IndexList[1](size), device_ctx
                )
                # Synchronize to ensure GPU operations complete
                device_ctx.get_device_context().synchronize()
            else:
                raise Error(
                    "GPU execution not supported for this unary elementwise"
                    " op or dtype"
                )
        else:
            raise Error("No GPU accelerator available")
