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

"""Mojo kernel wrappers for unary elementwise MO interpreter operations.

This module contains unary all-dtype ops (Negative, Abs, ReLU, Ceil, Floor,
Round), unary float-only ops (Exp, Log, Log1p, Sqrt, Rsqrt, Tanh, ATanh, Sin,
Cos, Erf, Trunc), unary boolean ops (Not), and unary predicate ops (IsNan,
IsInf).
"""

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator, simd_width_of

from algorithm.functional import elementwise, IndexList
from memory import OpaquePointer
from reflection import get_base_type_name
from runtime.asyncrt import DeviceContextPtr
from tensor import ElementwiseUnaryOp, ElementwiseUnaryMixedOp
from MOGGKernelAPI.MOGGKernelAPI import (
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
    IsNan,
    IsInf,
)

from op_utils import _get_dtype, _get_buffer_ptr, _get_size, _get_ctx


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
    return name == "IsNan" or name == "IsInf"


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_elementwise_unary_ops() -> PythonObject:
    """Create a Python module with unary elementwise kernel function bindings.
    """
    try:
        var b = PythonModuleBuilder("elementwise_unary_ops")

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

        return b.finalize()
    except e:
        abort(
            String("failed to create elementwise unary op bindings module: ", e)
        )


# =============================================================================
# Dispatchers
# =============================================================================


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


# =============================================================================
# Kernel implementations
# =============================================================================


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
