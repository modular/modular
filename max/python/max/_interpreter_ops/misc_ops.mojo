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

"""Mojo kernel wrappers for miscellaneous MO interpreter operations.

Contains range and random operations.
"""

from os import abort
from python import PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator

from math import iota
from random import NormalRandom, Random
from algorithm.functional import elementwise, IndexList
from memory import OpaquePointer
from runtime.asyncrt import DeviceContextPtr
from tensor.managed_tensor_slice import ManagedTensorSlice
from tensor.io_spec import FusedOutput
from compiler_internal import StaticTensorSpec
from MOGGKernelAPI.MOGGKernelAPI import Range

from utils.numerics import get_accum_type

from op_utils import (
    _get_dtype,
    _get_buffer_ptr,
    _get_size,
    _get_ctx,
    _get_shape,
    MAX_RANK,
)


# =============================================================================
# Python bindings
# =============================================================================


@export
fn PyInit_misc_ops() -> PythonObject:
    """Create a Python module with miscellaneous kernel function bindings."""
    try:
        var b = PythonModuleBuilder("misc_ops")

        b.def_function[range_dispatcher]("Range", docstring="Range operation")
        b.def_function[range_shape_dispatcher](
            "RangeShape", docstring="Compute range output shape"
        )
        b.def_function[random_normal_dispatcher](
            "RandomNormal", docstring="Random normal distribution"
        )
        b.def_function[random_uniform_dispatcher](
            "RandomUniform", docstring="Random uniform distribution"
        )
        b.def_function[cumsum_dispatcher](
            "CumSum", docstring="Cumulative sum along axis"
        )
        return b.finalize()
    except e:
        abort(String("failed to create misc op bindings module: ", e))


# ===----------------------------------------------------------------------=== #
# Range operation
# ===----------------------------------------------------------------------=== #


fn range_dispatcher(
    out_buffer: PythonObject,
    start_buffer: PythonObject,
    stop_buffer: PythonObject,
    step_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Range dispatcher with dtype dispatch.

    Fills output buffer with values: out[i] = start + i * step.

    Args:
        out_buffer: The output buffer object.
        start_buffer: Scalar buffer containing the start value.
        stop_buffer: Scalar buffer containing the stop value.
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
            _get_buffer_ptr[DType.float16](stop_buffer),
            _get_buffer_ptr[DType.float16](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float32:
        range_op[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](start_buffer),
            _get_buffer_ptr[DType.float32](stop_buffer),
            _get_buffer_ptr[DType.float32](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float64:
        range_op[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](start_buffer),
            _get_buffer_ptr[DType.float64](stop_buffer),
            _get_buffer_ptr[DType.float64](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        range_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](start_buffer),
            _get_buffer_ptr[DType.bfloat16](stop_buffer),
            _get_buffer_ptr[DType.bfloat16](step_buffer),
            size,
            ctx,
        )
    # Integer types
    elif dtype == DType.int8:
        range_op[DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.int8](start_buffer),
            _get_buffer_ptr[DType.int8](stop_buffer),
            _get_buffer_ptr[DType.int8](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int16:
        range_op[DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.int16](start_buffer),
            _get_buffer_ptr[DType.int16](stop_buffer),
            _get_buffer_ptr[DType.int16](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int32:
        range_op[DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.int32](start_buffer),
            _get_buffer_ptr[DType.int32](stop_buffer),
            _get_buffer_ptr[DType.int32](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int64:
        range_op[DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.int64](start_buffer),
            _get_buffer_ptr[DType.int64](stop_buffer),
            _get_buffer_ptr[DType.int64](step_buffer),
            size,
            ctx,
        )
    # Unsigned integer types
    elif dtype == DType.uint8:
        range_op[DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.uint8](start_buffer),
            _get_buffer_ptr[DType.uint8](stop_buffer),
            _get_buffer_ptr[DType.uint8](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint16:
        range_op[DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer),
            _get_buffer_ptr[DType.uint16](start_buffer),
            _get_buffer_ptr[DType.uint16](stop_buffer),
            _get_buffer_ptr[DType.uint16](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint32:
        range_op[DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer),
            _get_buffer_ptr[DType.uint32](start_buffer),
            _get_buffer_ptr[DType.uint32](stop_buffer),
            _get_buffer_ptr[DType.uint32](step_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint64:
        range_op[DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer),
            _get_buffer_ptr[DType.uint64](start_buffer),
            _get_buffer_ptr[DType.uint64](stop_buffer),
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
    stop_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    step_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Range operation using Range.execute from MOGGKernelAPI.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        start_ptr: Pointer to the start scalar value.
        stop_ptr: Pointer to the stop scalar value.
        step_ptr: Pointer to the step scalar value.
        size: Number of elements to produce.
        ctx: Device context pointer (null for CPU).
    """
    var start = start_ptr.load()
    var stop = stop_ptr.load()
    var step = step_ptr.load()

    comptime out_spec = StaticTensorSpec[dtype, 1].create_unknown()
    var output_tensor = ManagedTensorSlice[
        io_spec=FusedOutput, static_spec=out_spec
    ](out_ptr, IndexList[1](size))

    if not ctx:
        Range.execute[
            dtype=dtype,
            target="cpu",
            _trace_name="interpreter.range",
            use_blocking_impl=True,
        ](output_tensor, start, stop, step, DeviceContextPtr())
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                # Range.execute uses iota with auto-selected SIMD width,
                # which triggers llvm.stepvector with 64-bit integers that
                # the Metal shader compiler cannot handle. Use elementwise
                # with simd_width=1 to avoid this issue on all GPU targets.
                @always_inline
                @parameter
                @__copy_capture(out_ptr, start, step)
                fn range_func[
                    width: Int, rank: Int, alignment: Int = 1
                ](idx: IndexList[rank]):
                    var i = rebind[IndexList[1]](idx)[0]
                    var result = start + (
                        iota[dtype, width](Scalar[dtype](i)) * step
                    )
                    out_ptr.store[width=width](i, result)

                var device_ctx = DeviceContextPtr(ctx)
                elementwise[range_func, simd_width=1, target="gpu"](
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
# Range shape computation
# ===----------------------------------------------------------------------=== #


fn range_shape_op[
    dtype: DType
](
    start_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    stop_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    step_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
) raises -> Int:
    """Compute range output size using Range.shape from MOGGKernelAPI.

    Parameters:
        dtype: The data type of the scalars.

    Args:
        start_ptr: Pointer to the start scalar value.
        stop_ptr: Pointer to the stop scalar value.
        step_ptr: Pointer to the step scalar value.

    Returns:
        The number of elements in the range output.
    """
    var start = start_ptr.load()
    var stop = stop_ptr.load()
    var step = step_ptr.load()
    var shape = Range.shape[dtype](start, stop, step)
    return shape[0]


fn range_shape_dispatcher(
    start_buffer: PythonObject,
    stop_buffer: PythonObject,
    step_buffer: PythonObject,
) raises -> PythonObject:
    """Compute range output shape, dispatching by dtype.

    Args:
        start_buffer: Scalar buffer containing the start value.
        stop_buffer: Scalar buffer containing the stop value.
        step_buffer: Scalar buffer containing the step value.

    Returns:
        The output size as a Python int.
    """
    var dtype = _get_dtype(start_buffer)

    # Float types
    if dtype == DType.float16:
        return PythonObject(
            range_shape_op[DType.float16](
                _get_buffer_ptr[DType.float16](start_buffer),
                _get_buffer_ptr[DType.float16](stop_buffer),
                _get_buffer_ptr[DType.float16](step_buffer),
            )
        )
    elif dtype == DType.float32:
        return PythonObject(
            range_shape_op[DType.float32](
                _get_buffer_ptr[DType.float32](start_buffer),
                _get_buffer_ptr[DType.float32](stop_buffer),
                _get_buffer_ptr[DType.float32](step_buffer),
            )
        )
    elif dtype == DType.float64:
        return PythonObject(
            range_shape_op[DType.float64](
                _get_buffer_ptr[DType.float64](start_buffer),
                _get_buffer_ptr[DType.float64](stop_buffer),
                _get_buffer_ptr[DType.float64](step_buffer),
            )
        )
    elif dtype == DType.bfloat16:
        return PythonObject(
            range_shape_op[DType.bfloat16](
                _get_buffer_ptr[DType.bfloat16](start_buffer),
                _get_buffer_ptr[DType.bfloat16](stop_buffer),
                _get_buffer_ptr[DType.bfloat16](step_buffer),
            )
        )
    # Integer types
    elif dtype == DType.int8:
        return PythonObject(
            range_shape_op[DType.int8](
                _get_buffer_ptr[DType.int8](start_buffer),
                _get_buffer_ptr[DType.int8](stop_buffer),
                _get_buffer_ptr[DType.int8](step_buffer),
            )
        )
    elif dtype == DType.int16:
        return PythonObject(
            range_shape_op[DType.int16](
                _get_buffer_ptr[DType.int16](start_buffer),
                _get_buffer_ptr[DType.int16](stop_buffer),
                _get_buffer_ptr[DType.int16](step_buffer),
            )
        )
    elif dtype == DType.int32:
        return PythonObject(
            range_shape_op[DType.int32](
                _get_buffer_ptr[DType.int32](start_buffer),
                _get_buffer_ptr[DType.int32](stop_buffer),
                _get_buffer_ptr[DType.int32](step_buffer),
            )
        )
    elif dtype == DType.int64:
        return PythonObject(
            range_shape_op[DType.int64](
                _get_buffer_ptr[DType.int64](start_buffer),
                _get_buffer_ptr[DType.int64](stop_buffer),
                _get_buffer_ptr[DType.int64](step_buffer),
            )
        )
    # Unsigned integer types
    elif dtype == DType.uint8:
        return PythonObject(
            range_shape_op[DType.uint8](
                _get_buffer_ptr[DType.uint8](start_buffer),
                _get_buffer_ptr[DType.uint8](stop_buffer),
                _get_buffer_ptr[DType.uint8](step_buffer),
            )
        )
    elif dtype == DType.uint16:
        return PythonObject(
            range_shape_op[DType.uint16](
                _get_buffer_ptr[DType.uint16](start_buffer),
                _get_buffer_ptr[DType.uint16](stop_buffer),
                _get_buffer_ptr[DType.uint16](step_buffer),
            )
        )
    elif dtype == DType.uint32:
        return PythonObject(
            range_shape_op[DType.uint32](
                _get_buffer_ptr[DType.uint32](start_buffer),
                _get_buffer_ptr[DType.uint32](stop_buffer),
                _get_buffer_ptr[DType.uint32](step_buffer),
            )
        )
    elif dtype == DType.uint64:
        return PythonObject(
            range_shape_op[DType.uint64](
                _get_buffer_ptr[DType.uint64](start_buffer),
                _get_buffer_ptr[DType.uint64](stop_buffer),
                _get_buffer_ptr[DType.uint64](step_buffer),
            )
        )
    else:
        raise Error("Unsupported dtype for range shape: " + String(dtype))


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
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
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
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
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
# Cumsum operation
# ===----------------------------------------------------------------------=== #


fn _cumsum_cpu[
    dtype: DType,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    dim0: Int,
    dim1: Int,
    dim2: Int,
    exclusive: Int,
    reverse: Int,
):
    """CPU cumsum on a rank-3 normalized buffer [dim0, dim1, dim2].

    Cumsum is applied along axis=1 (dim1). dim0 is the product of dimensions
    before the original axis, dim2 is the product of dimensions after.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer.
        in_ptr: Pointer to the input buffer.
        dim0: Product of dimensions before the cumsum axis.
        dim1: Size of the cumsum axis.
        dim2: Product of dimensions after the cumsum axis.
        exclusive: 1 for exclusive cumsum (first element is 0), 0 otherwise.
        reverse: 1 for reverse direction along the axis, 0 otherwise.
    """
    # Use float64 accumulator for float32 for precision, same type otherwise.
    # This matches the behavior in nn/cumsum.mojo.
    comptime accum_type = DType.float64 if dtype == DType.float32 else get_accum_type[
        dtype
    ]()

    # Strides for row-major [dim0, dim1, dim2] layout.
    var stride0 = dim1 * dim2
    var stride1 = dim2

    for i0 in range(dim0):
        for i2 in range(dim2):
            var accumulator: Scalar[accum_type] = 0

            for d in range(dim1):
                var d_adj = (dim1 - 1 - d) if reverse else d
                var idx = i0 * stride0 + d_adj * stride1 + i2

                if exclusive:
                    out_ptr[idx] = accumulator.cast[dtype]()
                    accumulator += in_ptr[idx].cast[accum_type]()
                else:
                    accumulator += in_ptr[idx].cast[accum_type]()
                    out_ptr[idx] = accumulator.cast[dtype]()


fn cumsum_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    axis: PythonObject,
    exclusive: PythonObject,
    reverse: PythonObject,
) raises:
    """Cumsum dispatcher with dtype dispatch.

    Normalizes the input to rank-3 [dim0, dim1, dim2] and dispatches by dtype.

    Args:
        out_buffer: The output buffer object (same shape as input).
        in_buffer: The input buffer object.
        axis: The axis along which to compute cumsum (non-negative integer).
        exclusive: 1 for exclusive cumsum, 0 otherwise.
        reverse: 1 for reverse cumsum, 0 otherwise.
    """
    var dtype = _get_dtype(in_buffer)
    var axis_val = Int(py=axis)
    var exclusive_val = Int(py=exclusive)
    var reverse_val = Int(py=reverse)

    # Extract input shape and compute normalized rank-3 shape:
    # dim0: product of dims before axis
    # dim1: the cumsum axis dimension
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

    # Float types
    if dtype == DType.float16:
        _cumsum_cpu[DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    elif dtype == DType.float32:
        _cumsum_cpu[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    elif dtype == DType.float64:
        _cumsum_cpu[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    elif dtype == DType.bfloat16:
        _cumsum_cpu[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    # Integer types
    elif dtype == DType.int8:
        _cumsum_cpu[DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.int8](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    elif dtype == DType.int16:
        _cumsum_cpu[DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.int16](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    elif dtype == DType.int32:
        _cumsum_cpu[DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.int32](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    elif dtype == DType.int64:
        _cumsum_cpu[DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.int64](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    # Unsigned integer types
    elif dtype == DType.uint8:
        _cumsum_cpu[DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.uint8](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    elif dtype == DType.uint16:
        _cumsum_cpu[DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer),
            _get_buffer_ptr[DType.uint16](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    elif dtype == DType.uint32:
        _cumsum_cpu[DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer),
            _get_buffer_ptr[DType.uint32](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    elif dtype == DType.uint64:
        _cumsum_cpu[DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer),
            _get_buffer_ptr[DType.uint64](in_buffer),
            dim0,
            dim1,
            dim2,
            exclusive_val,
            reverse_val,
        )
    else:
        raise Error("Unsupported dtype for cumsum: " + String(dtype))
