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

"""Mojo kernel wrappers for reduce MO interpreter operations."""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator

from std.algorithm import max as reduce_max
from std.algorithm import min as reduce_min
from std.algorithm import sum as reduce_sum
from std.algorithm import mean as reduce_mean
from std.algorithm import product as reduce_product
from std.algorithm.functional import IndexList
from std.memory import OpaquePointer
from std.runtime.asyncrt import DeviceContextPtr
from std.sys.info import has_apple_gpu_accelerator

from op_utils import _get_dtype, _get_buffer_ptr, _get_ctx, _get_shape, MAX_RANK


# =============================================================================
# Python bindings
# =============================================================================


@export
def PyInit_reduce_ops() -> PythonObject:
    """Create a Python module with reduce kernel function bindings."""
    try:
        var b = PythonModuleBuilder("reduce_ops")

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

        return b.finalize()
    except e:
        abort(t"failed to create reduce op bindings module: {e}")


# =============================================================================
# Reduce function type and wrappers
# =============================================================================

# Function type shared by reduce_max, reduce_min, reduce_sum, and
# _reduce_mean. Each takes (input_shape, reduce_dim, context) with
# compile-time dtype, input/output lambdas, and target parameters.
comptime ReduceFn = def[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) capturing raises -> None


def _reduce_max[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) raises:
    """Non-overloaded wrapper around algorithm.max for use with ReduceFn."""
    reduce_max[
        dtype,
        input_fn,
        output_fn,
        target=target,
    ](input_shape, reduce_dim, context)


def _reduce_min[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) raises:
    """Non-overloaded wrapper around algorithm.min for use with ReduceFn."""
    reduce_min[
        dtype,
        input_fn,
        output_fn,
        target=target,
    ](input_shape, reduce_dim, context)


def _reduce_sum[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) raises:
    """Non-overloaded wrapper around algorithm.sum for use with ReduceFn."""
    reduce_sum[
        dtype,
        input_fn,
        output_fn,
        target=target,
    ](input_shape, reduce_dim, context)


def _reduce_mean[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
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
        target=target,
    ](input_shape, reduce_dim, output_shape, context)


def _reduce_mul[
    dtype: DType,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing[_] -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing[_] -> None,
    /,
    target: StaticString = "cpu",
](
    input_shape: IndexList[_, element_type=DType.int64],
    reduce_dim: Int,
    context: DeviceContextPtr,
) raises:
    """Non-overloaded wrapper around algorithm.product for use with ReduceFn."""
    reduce_product[
        dtype,
        input_fn,
        output_fn,
        target=target,
    ](input_shape, reduce_dim, context)


# =============================================================================
# Dispatchers
# =============================================================================


def _normalize_reduce_shape(
    in_buffer: PythonObject,
    axis: PythonObject,
) raises -> IndexList[3]:
    """Compute normalized rank-3 shape for reduction along an axis.

    Collapses an N-dimensional tensor shape into three dimensions:
    - dim0: product of dims before the reduction axis
    - dim1: the reduction axis dimension
    - dim2: product of dims after the reduction axis

    Args:
        in_buffer: The input buffer object (must have a .shape attribute).
        axis: The axis along which to reduce (integer).

    Returns:
        An IndexList[3] with [dim0, dim1, dim2].
    """
    var axis_val = Int(py=axis)
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

    return IndexList[3](dim0, dim1, dim2)


def reduce_dispatcher[
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
    var ctx = _get_ctx(device_context_ptr)
    var normalized_shape = _normalize_reduce_shape(in_buffer, axis)

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
        comptime if not has_apple_gpu_accelerator():
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


# =============================================================================
# Kernel implementation
# =============================================================================


def reduce_op[
    dtype: DType,
    reduce_fn: ReduceFn,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    normalized_shape: IndexList[3],
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
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
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var c = rebind[IndexList[3]](coords)
        var flat_idx = c[0] * inStride0 + c[1] * inStride1 + c[2]
        return in_ptr.load[width=width](flat_idx)

    # Define output function mapping rank-3 coords to flat index
    @always_inline
    @parameter
    @__copy_capture(out_ptr, outStride0)
    def output_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]):
        var c = rebind[IndexList[3]](coords)
        var flat_idx = c[0] * outStride0 + c[2]
        out_ptr.store[width=width](flat_idx, val)

    # Always dispatch rank-3 reduction with axis=1
    if not ctx:
        reduce_fn[
            dtype,
            input_fn,
            output_fn,
            target="cpu",
        ](normalized_shape, 1, DeviceContextPtr())
    else:
        comptime if has_accelerator():
            comptime if dtype in (
                DType.float32,
                DType.float16,
                DType.bfloat16,
                DType.int32,
                DType.uint32,
                DType.int64,
                DType.uint64,
            ):
                var device_ctx = DeviceContextPtr(ctx.unsafe_value())
                reduce_fn[
                    dtype,
                    input_fn,
                    output_fn,
                    target="gpu",
                ](normalized_shape, 1, device_ctx)
            else:
                raise Error(
                    "GPU execution not supported for reduce with dtype "
                    + String(dtype)
                )
        else:
            raise Error("No GPU accelerator available")


# Concrete dispatcher functions for def_function registration.
# def_function requires fully concrete function types, so we can't pass
# reduce_dispatcher[_reduce_max] directly (parametric def type can't be inferred).


def _reduce_bool_dispatcher[
    reduce_fn: ReduceFn
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Handle DType.bool reduction for max/min only.

    Bool is not numeric so it cannot go through the generic
    reduce_dispatcher (sum/mean/mul would fail the SIMD numeric
    constraint). This helper dispatches directly for bool buffers.
    """
    var ctx = _get_ctx(device_context_ptr)
    var normalized_shape = _normalize_reduce_shape(in_buffer, axis)
    reduce_op[DType.bool, reduce_fn](
        _get_buffer_ptr[DType.bool](out_buffer),
        _get_buffer_ptr[DType.bool](in_buffer),
        normalized_shape,
        ctx,
    )


def reduce_max_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    if _get_dtype(in_buffer) == DType.bool:
        _reduce_bool_dispatcher[_reduce_max](
            out_buffer, in_buffer, axis, device_context_ptr
        )
    else:
        reduce_dispatcher[_reduce_max](
            out_buffer, in_buffer, axis, device_context_ptr
        )


def reduce_min_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    if _get_dtype(in_buffer) == DType.bool:
        _reduce_bool_dispatcher[_reduce_min](
            out_buffer, in_buffer, axis, device_context_ptr
        )
    else:
        reduce_dispatcher[_reduce_min](
            out_buffer, in_buffer, axis, device_context_ptr
        )


def reduce_sum_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    reduce_dispatcher[_reduce_sum](
        out_buffer, in_buffer, axis, device_context_ptr
    )


def mean_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    reduce_dispatcher[_reduce_mean](
        out_buffer, in_buffer, axis, device_context_ptr
    )


def reduce_mul_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    reduce_dispatcher[_reduce_mul](
        out_buffer, in_buffer, axis, device_context_ptr
    )
