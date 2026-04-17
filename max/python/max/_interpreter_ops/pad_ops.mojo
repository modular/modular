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

"""Mojo kernel wrappers for pad MO interpreter operations.

All three ops pad a tensor along every axis.  Paddings are given as a flat
int64 array of length 2*rank laid out as [pre_0, post_0, pre_1, post_1, ...].
Output size per axis d: paddings[2d] + input.dim(d) + paddings[2d+1].

PadConstant fills padded cells with a scalar constant and runs on CPU and GPU.
PadReflect and PadRepeat are CPU-only (mo.pad.reflect / mo.pad.repeat are
MO_HostOnly ops).

Coordinate mapping (per output element, per axis):
  Constant: input_coord = coord - pre; write constant if out of [0, input_dim).
  Reflect:  period = max(1, 2*(input_dim-1));
            c = ((coord-pre) % period + period) % period;
            input_coord = c if c < input_dim else period - c.
  Repeat:   input_coord = clamp(coord - pre, 0, input_dim - 1).
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator

from std.algorithm.functional import elementwise, IndexList
from std.memory import OpaquePointer
from std.runtime.asyncrt import DeviceContextPtr

from op_utils import (
    _get_dtype,
    _get_shape,
    _make_ptr,
    _get_ctx,
    Dispatchable,
    dispatch_dtype,
    MAX_RANK,
)

comptime MAX_PAD_SIZE = 2 * MAX_RANK
"""Maximum number of padding values (2 * MAX_RANK axes)."""


@export
def PyInit_pad_ops() -> PythonObject:
    """Create a Python module with pad kernel function bindings."""
    try:
        var b = PythonModuleBuilder("pad_ops")
        b.def_function[pad_constant_dispatcher](
            "PadConstant", docstring="Constant padding along every axis"
        )
        b.def_function[pad_reflect_dispatcher](
            "PadReflect", docstring="Reflect padding along every axis (CPU)"
        )
        b.def_function[pad_repeat_dispatcher](
            "PadRepeat", docstring="Edge-repeat padding along every axis (CPU)"
        )
        return b.finalize()
    except e:
        abort(t"failed to create pad op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# Pad-constant kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def pad_constant_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    constant: Scalar[dtype],
    paddings: InlineArray[Int, MAX_PAD_SIZE],
    out_shape: InlineArray[Int, MAX_RANK],
    in_shape: InlineArray[Int, MAX_RANK],
    out_strides: InlineArray[Int, MAX_RANK],
    in_strides: InlineArray[Int, MAX_RANK],
    rank: Int,
    total: Int,
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
) raises:
    """Fill output with input values inside the content region, constant outside.

    For each flat output index the kernel recovers the N-D coordinate via
    out_strides, checks per axis whether the coordinate falls inside the content
    region [pre, pre+input_dim), and either copies the corresponding input
    element or writes the constant fill value.

    Parameters:
        dtype: Data type (inferred from pointers).

    Args:
        out_ptr: Output buffer (row-major, length total).
        in_ptr: Input buffer.
        constant: Scalar fill value for padded cells.
        paddings: Flat array [pre_0, post_0, pre_1, post_1, ...] (2*rank).
        out_shape: Output shape (first rank elements valid).
        in_shape: Input shape (first rank elements valid).
        out_strides: Row-major strides of the output tensor.
        in_strides: Row-major strides of the input tensor.
        rank: Number of tensor dimensions.
        total: Total number of output elements.
        ctx: Device context pointer (null for CPU).
    """

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        constant,
        paddings,
        out_shape,
        in_shape,
        out_strides,
        in_strides,
        rank,
    )
    def func[width: Int, rank_: Int, alignment: Int = 1](idx: IndexList[rank_]):
        var i = idx[0]
        var rem = i
        var in_flat = 0
        var in_bounds = True
        for d in range(rank):
            var coord, new_rem = divmod(rem, out_strides[d])
            rem = new_rem
            var pre = paddings[2 * d]
            var input_coord = coord - pre
            if input_coord < 0 or input_coord >= in_shape[d]:
                in_bounds = False
                break
            in_flat += input_coord * in_strides[d]
        if in_bounds:
            out_ptr[i] = in_ptr[in_flat]
        else:
            out_ptr[i] = constant

    if not ctx:
        elementwise[func, simd_width=1](IndexList[1](total))
    else:
        comptime if has_accelerator():
            var device_ctx = DeviceContextPtr(ctx.unsafe_value())
            elementwise[func, simd_width=1, target="gpu"](
                IndexList[1](total), device_ctx
            )
        else:
            raise Error("No GPU accelerator available")


# ===----------------------------------------------------------------------=== #
# Pad-reflect kernel (CPU-only)
# ===----------------------------------------------------------------------=== #


@always_inline
def pad_reflect_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    paddings: InlineArray[Int, MAX_PAD_SIZE],
    out_shape: InlineArray[Int, MAX_RANK],
    in_shape: InlineArray[Int, MAX_RANK],
    out_strides: InlineArray[Int, MAX_RANK],
    in_strides: InlineArray[Int, MAX_RANK],
    rank: Int,
    total: Int,
) raises:
    """Fill output by reflecting input values about the content-region edges.

    For each output coordinate the reflection formula maps it to the
    corresponding input coordinate using a periodic "bounce" with period
    2*(input_dim-1).  This matches mo.pad.reflect semantics for large
    paddings (wraps around rather than raising an error).

    Parameters:
        dtype: Data type (inferred from pointers).

    Args:
        out_ptr: Output buffer (row-major, length total).
        in_ptr: Input buffer.
        paddings: Flat array [pre_0, post_0, pre_1, post_1, ...] (2*rank).
        out_shape: Output shape (first rank elements valid).
        in_shape: Input shape (first rank elements valid).
        out_strides: Row-major strides of the output tensor.
        in_strides: Row-major strides of the input tensor.
        rank: Number of tensor dimensions.
        total: Total number of output elements.
    """

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        paddings,
        out_shape,
        in_shape,
        out_strides,
        in_strides,
        rank,
    )
    def func[width: Int, rank_: Int, alignment: Int = 1](idx: IndexList[rank_]):
        var i = idx[0]
        var rem = i
        var in_flat = 0
        for d in range(rank):
            var coord, new_rem = divmod(rem, out_strides[d])
            rem = new_rem
            var pre = paddings[2 * d]
            var input_dim = in_shape[d]
            # period = 2*(input_dim-1), clamped to 1 when input_dim==1.
            var period = max(1, 2 * (input_dim - 1))
            var c = ((coord - pre) % period + period) % period
            var input_coord = c if c < input_dim else (2 * (input_dim - 1) - c)
            in_flat += input_coord * in_strides[d]
        out_ptr[i] = in_ptr[in_flat]

    elementwise[func, simd_width=1](IndexList[1](total))


# ===----------------------------------------------------------------------=== #
# Pad-repeat (edge) kernel (CPU-only)
# ===----------------------------------------------------------------------=== #


@always_inline
def pad_repeat_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    paddings: InlineArray[Int, MAX_PAD_SIZE],
    out_shape: InlineArray[Int, MAX_RANK],
    in_shape: InlineArray[Int, MAX_RANK],
    out_strides: InlineArray[Int, MAX_RANK],
    in_strides: InlineArray[Int, MAX_RANK],
    rank: Int,
    total: Int,
) raises:
    """Fill output by repeating the nearest edge value of the input.

    For each output coordinate the input coordinate is clamped to
    [0, input_dim-1] per axis, which replicates the boundary element
    into the padded region.

    Parameters:
        dtype: Data type (inferred from pointers).

    Args:
        out_ptr: Output buffer (row-major, length total).
        in_ptr: Input buffer.
        paddings: Flat array [pre_0, post_0, pre_1, post_1, ...] (2*rank).
        out_shape: Output shape (first rank elements valid).
        in_shape: Input shape (first rank elements valid).
        out_strides: Row-major strides of the output tensor.
        in_strides: Row-major strides of the input tensor.
        rank: Number of tensor dimensions.
        total: Total number of output elements.
    """

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        paddings,
        out_shape,
        in_shape,
        out_strides,
        in_strides,
        rank,
    )
    def func[width: Int, rank_: Int, alignment: Int = 1](idx: IndexList[rank_]):
        var i = idx[0]
        var rem = i
        var in_flat = 0
        for d in range(rank):
            var coord, new_rem = divmod(rem, out_strides[d])
            rem = new_rem
            var pre = paddings[2 * d]
            var input_dim = in_shape[d]
            var input_coord = max(0, min(input_dim - 1, coord - pre))
            in_flat += input_coord * in_strides[d]
        out_ptr[i] = in_ptr[in_flat]

    elementwise[func, simd_width=1](IndexList[1](total))


# ===----------------------------------------------------------------------=== #
# Dispatcher helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _extract_paddings(
    py_paddings: PythonObject, rank: Int
) raises -> InlineArray[Int, MAX_PAD_SIZE]:
    """Extract 2*rank padding values from a Python sequence into an InlineArray.
    """
    var result = InlineArray[Int, MAX_PAD_SIZE](fill=0)
    for i in range(2 * rank):
        result[i] = Int(py=py_paddings[i])
    return result^


# ===----------------------------------------------------------------------=== #
# Pad-constant dispatcher
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _PadConstantBody(Dispatchable):
    """Dispatch body for the PadConstant operation over numeric data dtypes.

    Stores buffer addresses as plain integers together with pre-computed
    shape/stride/padding arrays.  The constant is cast to the dispatch
    dtype inside call[t].
    """

    var out_addr: Int
    var in_addr: Int
    var const_addr: Int
    var paddings: InlineArray[Int, MAX_PAD_SIZE]
    var out_shape: InlineArray[Int, MAX_RANK]
    var in_shape: InlineArray[Int, MAX_RANK]
    var out_strides: InlineArray[Int, MAX_RANK]
    var in_strides: InlineArray[Int, MAX_RANK]
    var rank: Int
    var total: Int
    var ctx: Optional[OpaquePointer[MutExternalOrigin]]

    def call[t: DType](self) raises -> None:
        var constant = _make_ptr[t](self.const_addr)[0]
        pad_constant_op(
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            constant,
            self.paddings,
            self.out_shape,
            self.in_shape,
            self.out_strides,
            self.in_strides,
            self.rank,
            self.total,
            self.ctx,
        )


def pad_constant_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """PadConstant dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Pre-allocated output buffer.
        in_buffer: Input tensor buffer.
        params: Python tuple
            (paddings, out_shape, in_shape, out_strides, in_strides, rank,
             total, const_addr).
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(in_buffer)
    var rank = Int(py=params[5])
    var paddings = _extract_paddings(params[0], rank)
    var out_shape = _get_shape(params[1], rank)
    var in_shape = _get_shape(params[2], rank)
    var out_strides = _get_shape(params[3], rank)
    var in_strides = _get_shape(params[4], rank)
    var total = Int(py=params[6])
    var const_addr = Int(py=params[7])
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    dispatch_dtype(
        _PadConstantBody(
            out_addr,
            in_addr,
            const_addr,
            paddings,
            out_shape,
            in_shape,
            out_strides,
            in_strides,
            rank,
            total,
            ctx,
        ),
        dtype,
    )


# ===----------------------------------------------------------------------=== #
# Pad-reflect dispatcher
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _PadReflectBody(Dispatchable):
    """Dispatch body for the PadReflect operation (CPU-only)."""

    var out_addr: Int
    var in_addr: Int
    var paddings: InlineArray[Int, MAX_PAD_SIZE]
    var out_shape: InlineArray[Int, MAX_RANK]
    var in_shape: InlineArray[Int, MAX_RANK]
    var out_strides: InlineArray[Int, MAX_RANK]
    var in_strides: InlineArray[Int, MAX_RANK]
    var rank: Int
    var total: Int

    def call[t: DType](self) raises -> None:
        pad_reflect_op(
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.paddings,
            self.out_shape,
            self.in_shape,
            self.out_strides,
            self.in_strides,
            self.rank,
            self.total,
        )


def pad_reflect_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """PadReflect dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Pre-allocated output buffer.
        in_buffer: Input tensor buffer.
        params: Python tuple
            (paddings, out_shape, in_shape, out_strides, in_strides, rank,
             total).
        device_context_ptr: Unused (CPU-only op).
    """
    var dtype = _get_dtype(in_buffer)
    var rank = Int(py=params[5])
    var paddings = _extract_paddings(params[0], rank)
    var out_shape = _get_shape(params[1], rank)
    var in_shape = _get_shape(params[2], rank)
    var out_strides = _get_shape(params[3], rank)
    var in_strides = _get_shape(params[4], rank)
    var total = Int(py=params[6])
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())

    dispatch_dtype(
        _PadReflectBody(
            out_addr,
            in_addr,
            paddings,
            out_shape,
            in_shape,
            out_strides,
            in_strides,
            rank,
            total,
        ),
        dtype,
    )


# ===----------------------------------------------------------------------=== #
# Pad-repeat dispatcher
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _PadRepeatBody(Dispatchable):
    """Dispatch body for the PadRepeat (edge) operation (CPU-only)."""

    var out_addr: Int
    var in_addr: Int
    var paddings: InlineArray[Int, MAX_PAD_SIZE]
    var out_shape: InlineArray[Int, MAX_RANK]
    var in_shape: InlineArray[Int, MAX_RANK]
    var out_strides: InlineArray[Int, MAX_RANK]
    var in_strides: InlineArray[Int, MAX_RANK]
    var rank: Int
    var total: Int

    def call[t: DType](self) raises -> None:
        pad_repeat_op(
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.paddings,
            self.out_shape,
            self.in_shape,
            self.out_strides,
            self.in_strides,
            self.rank,
            self.total,
        )


def pad_repeat_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """PadRepeat dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Pre-allocated output buffer.
        in_buffer: Input tensor buffer.
        params: Python tuple
            (paddings, out_shape, in_shape, out_strides, in_strides, rank,
             total).
        device_context_ptr: Unused (CPU-only op).
    """
    var dtype = _get_dtype(in_buffer)
    var rank = Int(py=params[5])
    var paddings = _extract_paddings(params[0], rank)
    var out_shape = _get_shape(params[1], rank)
    var in_shape = _get_shape(params[2], rank)
    var out_strides = _get_shape(params[3], rank)
    var in_strides = _get_shape(params[4], rank)
    var total = Int(py=params[6])
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())

    dispatch_dtype(
        _PadRepeatBody(
            out_addr,
            in_addr,
            paddings,
            out_shape,
            in_shape,
            out_strides,
            in_strides,
            rank,
            total,
        ),
        dtype,
    )
