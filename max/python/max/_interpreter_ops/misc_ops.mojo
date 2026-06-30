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
from std.gpu.host import DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator

from std.math import iota
from std.random import NormalRandom, Random
from std.algorithm.functional import elementwise, IndexList

from extensibility import (
    ManagedTensorSlice,
)
from extensibility import FusedOutput
from extensibility import StaticTensorSpec
from builtin_kernels import Range, range_shape

from std.utils.coord import Coord

from op_utils import (
    _get_dtype,
    _get_buffer_ptr,
    _get_size,
    _get_ctx,
    _make_ptr,
    Dispatchable,
    dispatch_dtype,
)


# =============================================================================
# Python bindings
# =============================================================================


@export
def PyInit_misc_ops() abi("C") -> PythonObject:
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

    @__allow_legacy_any_origin_fields
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
    var ctx: DeviceContext

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
        device_context_ptr: Device context pointer.
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
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    start_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    stop_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    step_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    """Range operation using Range.execute from the `kernels` package.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        start_ptr: Pointer to the start scalar value.
        stop_ptr: Pointer to the stop scalar value.
        step_ptr: Pointer to the step scalar value.
        size: Number of elements to produce.
        ctx: Device context.
    """
    var start = start_ptr.load()
    var stop = stop_ptr.load()
    var step = step_ptr.load()

    comptime out_spec = StaticTensorSpec[dtype, 1, ...].get_unknown()
    var output_tensor = ManagedTensorSlice[
        io_spec=FusedOutput, static_spec=out_spec
    ](out_ptr, IndexList[1](size))

    if ctx.api() == "cpu":
        Range.execute[
            dtype=dtype,
            target="cpu",
            _trace_name="interpreter.range",
        ](output_tensor, start, stop, step, ctx)
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                # Range.execute uses iota with auto-selected SIMD width,
                # which triggers llvm.stepvector with 64-bit integers that
                # the Metal shader compiler cannot handle. Use elementwise
                # with simd_width=1 to avoid this issue on all GPU targets.
                @always_inline
                def range_func[
                    width: Int, alignment: Int = 1
                ](idx: Coord) {var}:
                    var i = Int(idx[0].value())
                    var result = start + (
                        iota[dtype, width](Scalar[dtype](i)) * step
                    )
                    out_ptr.store[width=width](i, result)

                elementwise[simd_width=1, target="gpu"](
                    range_func, Coord(size), ctx
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
    start_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    stop_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    step_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
) raises -> Int:
    """Compute range output size using Range.shape from the `kernels` package.

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
    var shape = range_shape[dtype](start, stop, step)
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
            start_addr,
            stop_addr,
            step_addr,
            UnsafePointer(to=result).as_unsafe_any_origin(),
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
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    mean: Float32,
    variance: Float32,
    seed_value: UInt64,
    ctx: DeviceContext,
) raises:
    """Random normal operation: fill output with normally distributed values.

    Mirrors PyTorch CUDA `torch.randn`'s element-to-counter mapping. For
    element `i`:

        thread_id     = i mod GRID_BLOCK
        within_thread = i div GRID_BLOCK   (0..3)

    where `GRID_BLOCK = 256 * min(num_SMs * blocks_per_sm, ceil(size/256))`.
    The per-element Box-Muller math is in
    :func:`std.random.NormalRandom.step_normal_4`. This op contributes the
    PyTorch-specific layout: which thread the element belongs to and which
    of the four normals from that thread's Philox step lands at `output[i]`.

    On CPU, `GRID_BLOCK = size` collapses every element to within_thread=0.

    Parameters:
        dtype: The data type of the output array.

    Args:
        out_ptr: Pointer to the output buffer data.
        size: Number of elements to produce.
        mean: Mean of the normal distribution.
        variance: Standard deviation of the normal distribution.
        seed_value: Seed for the random number generator.
        ctx: Device context.
    """
    if variance <= 0:
        raise Error("stddev must be positive")
    if size == 0:
        return

    comptime BLOCK_SIZE: Int = 256
    var grid_block: Int

    if ctx.api() == "cpu":
        grid_block = size
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                comptime info = DeviceContext.default_device_info
                comptime MAX_GRID = info.sm_count * (
                    info.threads_per_multiprocessor // BLOCK_SIZE
                )
                var nblocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
                var grid_x = MAX_GRID if nblocks > MAX_GRID else nblocks
                grid_block = grid_x * BLOCK_SIZE
            else:
                raise Error(
                    "GPU execution not supported for random_normal"
                    " with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")

    @always_inline
    def func[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert (
            width == 1
        ), "PyTorch-compat normal kernel uses scalar lanes"
        var i = Int(idx[0].value())
        var thread_id = UInt64(i % grid_block)
        var within_thread = i // grid_block

        var rng = NormalRandom(seed=seed_value, subsequence=thread_id)
        var four = rng.step_normal_4(mean=mean, stddev=variance)
        var value = four[within_thread].cast[dtype]()
        out_ptr.store[width=1](i, SIMD[dtype, 1](value))

    if ctx.api() == "cpu":
        elementwise[simd_width=1](func, Coord(size), ctx)
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                elementwise[simd_width=1, target="gpu"](func, Coord(size), ctx)


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
        device_context_ptr: Device context pointer.
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
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    lower_bound: Float32,
    upper_bound: Float32,
    seed_value: UInt64,
    ctx: DeviceContext,
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
        ctx: Device context.
    """
    if lower_bound > upper_bound:
        raise Error("lower_bound must be less than or equal to upper_bound")

    var delta = upper_bound - lower_bound

    @always_inline
    def func[width: Int, alignment: Int = 1](idx: Coord) {var}:
        var i = Int(idx[0].value())
        var generator = Random(seed=seed_value, offset=UInt64(i))
        var values: SIMD[DType.float32, 4] = generator.step_uniform()
        values = values * delta + lower_bound
        out_ptr.store[width=width](i, values.cast[dtype]().slice[width]())

    if ctx.api() == "cpu":
        elementwise[simd_width=4](func, Coord(size), ctx)
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                elementwise[simd_width=4, target="gpu"](func, Coord(size), ctx)
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
        device_context_ptr: Device context pointer.
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
