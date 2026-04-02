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

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator

from std.math import iota
from std.random import NormalRandom, Random
from std.algorithm.functional import elementwise, IndexList
from std.memory import OpaquePointer
from std.runtime.asyncrt import DeviceContextPtr
from tensor.managed_tensor_slice import (
    ManagedTensorSlice,
)
from tensor.io_spec import FusedOutput
from compiler_internal import StaticTensorSpec
from MOGGKernelAPI.MOGGKernelAPI import Range

from std.utils.numerics import get_accum_type

from op_utils import (
    _get_dtype,
    _get_buffer_ptr,
    _get_size,
    _get_ctx,
    _get_shape,
    _make_ptr,
    MAX_RANK,
    Dispatchable,
    dispatch_dtype,
)


# =============================================================================
# Python bindings
# =============================================================================


@export
def PyInit_misc_ops() -> PythonObject:
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
        abort(t"failed to create misc op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# Range operation
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _RangeShapeBody(Dispatchable):
    """Dispatch body for the RangeShape operation over data dtypes."""

    var start_addr: Int
    var stop_addr: Int
    var step_addr: Int
    var result_ptr: UnsafePointer[Int, MutAnyOrigin]

    def call[t: DType](self) raises -> None:
        comptime if t == DType.bool:
            raise Error("Unsupported dtype for range shape: bool")
        else:
            self.result_ptr[] = range_shape_op[t](
                _make_ptr[t](self.start_addr),
                _make_ptr[t](self.stop_addr),
                _make_ptr[t](self.step_addr),
            )


@fieldwise_init
struct _RangeBody(Dispatchable):
    """Dispatch body for the Range operation over data dtypes."""

    var out_addr: Int
    var start_addr: Int
    var stop_addr: Int
    var step_addr: Int
    var size: Int
    var ctx: OpaquePointer[MutExternalOrigin]

    def call[t: DType](self) raises -> None:
        comptime if t == DType.bool:
            raise Error("Unsupported dtype for range: bool")
        else:
            range_op[t](
                _make_ptr[t](self.out_addr),
                _make_ptr[t](self.start_addr),
                _make_ptr[t](self.stop_addr),
                _make_ptr[t](self.step_addr),
                self.size,
                self.ctx,
            )


def range_dispatcher(
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
    var out_addr = Int(py=out_buffer._data_ptr())
    var start_addr = Int(py=start_buffer._data_ptr())
    var stop_addr = Int(py=stop_buffer._data_ptr())
    var step_addr = Int(py=step_buffer._data_ptr())

    if dtype == DType.bool:
        raise Error("Unsupported dtype for range: " + String(dtype))
    dispatch_dtype(
        _RangeBody(out_addr, start_addr, stop_addr, step_addr, size, ctx),
        dtype,
    )


def range_op[
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

    comptime out_spec = StaticTensorSpec[dtype, 1, ...].get_unknown()
    var output_tensor = ManagedTensorSlice[
        io_spec=FusedOutput, static_spec=out_spec
    ](out_ptr, IndexList[1](size))

    if not ctx:
        Range.execute[
            dtype=dtype,
            target="cpu",
            _trace_name="interpreter.range",
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
                def range_func[
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
            else:
                raise Error(
                    "GPU execution not supported for range with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


# ===----------------------------------------------------------------------=== #
# Range shape computation
# ===----------------------------------------------------------------------=== #


def range_shape_op[
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


def range_shape_dispatcher(
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
    var start_addr = Int(py=start_buffer._data_ptr())
    var stop_addr = Int(py=stop_buffer._data_ptr())
    var step_addr = Int(py=step_buffer._data_ptr())

    if dtype == DType.bool:
        raise Error("Unsupported dtype for range shape: " + String(dtype))
    var result: Int = 0
    dispatch_dtype(
        _RangeShapeBody(
            start_addr, stop_addr, step_addr, UnsafePointer(to=result)
        ),
        dtype,
    )
    return PythonObject(result)


# ===----------------------------------------------------------------------=== #
# Random normal operation
# ===----------------------------------------------------------------------=== #


def random_normal_op[
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
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]
        var generator = NormalRandom(seed=seed_value, offset=UInt64(i))
        var values = generator.step_normal(mean=mean, stddev=variance)
        out_ptr.store[width=width](i, values.cast[dtype]().slice[width]())

    if not ctx:
        elementwise[func, simd_width=8](IndexList[1](size))
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=8, target="gpu"](
                    IndexList[1](size), device_ctx
                )
            else:
                raise Error(
                    "GPU execution not supported for random_normal"
                    " with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


def random_normal_dispatcher(
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


def random_uniform_op[
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
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = rebind[IndexList[1]](idx)[0]
        var generator = Random(seed=seed_value, offset=UInt64(i))
        var values: SIMD[DType.float32, 4] = generator.step_uniform()
        values = values * delta + lower_bound
        out_ptr.store[width=width](i, values.cast[dtype]().slice[width]())

    if not ctx:
        elementwise[func, simd_width=4](IndexList[1](size))
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                var device_ctx = DeviceContextPtr(ctx)
                elementwise[func, simd_width=4, target="gpu"](
                    IndexList[1](size), device_ctx
                )
            else:
                raise Error(
                    "GPU execution not supported for random_uniform"
                    " with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")


def random_uniform_dispatcher(
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


def _cumsum_cpu[
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


@fieldwise_init
struct _CumsumBody(Dispatchable):
    """Dispatch body for the CumSum operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var dim0: Int
    var dim1: Int
    var dim2: Int
    var exclusive: Int
    var reverse: Int

    def call[t: DType](self) raises -> None:
        comptime if t == DType.bool:
            raise Error("Unsupported dtype for cumsum: bool")
        else:
            _cumsum_cpu[t](
                _make_ptr[t](self.out_addr),
                _make_ptr[t](self.in_addr),
                self.dim0,
                self.dim1,
                self.dim2,
                self.exclusive,
                self.reverse,
            )


def cumsum_dispatcher(
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

    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())

    if dtype == DType.bool:
        raise Error("Unsupported dtype for cumsum: " + String(dtype))
    dispatch_dtype(
        _CumsumBody(
            out_addr, in_addr, dim0, dim1, dim2, exclusive_val, reverse_val
        ),
        dtype,
    )
