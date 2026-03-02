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

"""Mojo kernel wrappers for comparison and select MO interpreter operations.

This module contains binary comparison ops (Equal, Greater, GreaterEqual,
NotEqual) and the Select (ternary) operation.
"""

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator, simd_width_of

from algorithm.functional import elementwise, IndexList
from memory import OpaquePointer
from reflection import get_base_type_name
from runtime.asyncrt import DeviceContextPtr
from tensor import ElementwiseBinaryComparisonOp
from MOGGKernelAPI.MOGGKernelAPI import (
    Equal,
    Greater,
    GreaterEqual,
    NotEqual,
    Select,
)

from op_utils import _get_dtype, _get_buffer_ptr, _get_size, _get_ctx


# Binary comparison operations
comptime BINARY_COMPARISON_OPS = Variadic.types[
    T=ElementwiseBinaryComparisonOp, Equal, Greater, GreaterEqual, NotEqual
]

# =============================================================================
# GPU Support Configuration
# =============================================================================


fn _is_gpu_allowed_comparison_op[op: ElementwiseBinaryComparisonOp]() -> Bool:
    """Check if a comparison op is allowed on GPU at compile time."""
    comptime name = get_base_type_name[op]()
    return (
        name == "Equal"
        or name == "Greater"
        or name == "GreaterEqual"
        or name == "NotEqual"
    )


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_elementwise_comparison_ops() -> PythonObject:
    """Create a Python module with comparison elementwise kernel function bindings.
    """
    try:
        var b = PythonModuleBuilder("elementwise_comparison_ops")

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

        # Select operation (ternary: cond ? x : y)
        b.def_function[select_dispatcher](
            "Select", docstring="Elementwise select (cond ? x : y)"
        )

        return b.finalize()
    except e:
        abort(
            String(
                "failed to create elementwise comparison op bindings module: ",
                e,
            )
        )


# =============================================================================
# Dispatchers
# =============================================================================


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
