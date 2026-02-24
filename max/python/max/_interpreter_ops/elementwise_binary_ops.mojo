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

"""Mojo kernel wrappers for binary elementwise MO interpreter operations.

This module contains binary arithmetic ops (Add, Sub, Mul, Div, Mod, Max, Min),
binary boolean ops (And, Or, Xor), and Pow.
"""

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator, simd_width_of

from algorithm.functional import elementwise, IndexList
from memory import OpaquePointer
from reflection import get_base_type_name
from runtime.asyncrt import DeviceContextPtr
from tensor import ElementwiseBinaryOp
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

# =============================================================================
# GPU Support Configuration
# =============================================================================


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


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_elementwise_binary_ops() -> PythonObject:
    """Create a Python module with binary elementwise kernel function bindings.
    """
    try:
        var b = PythonModuleBuilder("elementwise_binary_ops")

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

        # Pow operation (custom dispatch - Pow doesn't conform to
        # ElementwiseBinaryOp)
        b.def_function[pow_dispatcher]("Pow", docstring="Elementwise Pow")

        return b.finalize()
    except e:
        abort(
            String(
                "failed to create elementwise binary op bindings module: ", e
            )
        )


# =============================================================================
# Dispatchers
# =============================================================================


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
