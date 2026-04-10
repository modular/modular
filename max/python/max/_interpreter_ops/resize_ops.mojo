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

"""Mojo kernel wrappers for resize MO interpreter operations.

**Linear** and **nearest-neighbor** resizes are CPU-only (``MO_HostOnly``).
**Bicubic** uses ``MO_SingleDeviceWithHostOperands``; the interpreter
handler runs the CPU path.

Compile-time dispatch:
  - Linear / Nearest: ``coord_mode`` x ``dtype`` x ``rank``
    + ``antialias`` (linear) or ``round_mode`` (nearest)
  - Bicubic: ``dtype`` only (rank is fixed at 4, no configurable modes)

Runtime parameters passed through the Python ``params`` tuple::

    Linear:  (coord_mode, antialias, rank, in_shape, out_shape)
    Nearest: (coord_mode, round_mode, rank, in_shape, out_shape)
    Bicubic: (in_shape, out_shape)
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from layout import Coord, TileTensor, row_major
from std.utils import IndexList

from nn.bicubic import cpu_bicubic_kernel
from nn.resize import (
    CoordinateTransformationMode,
    RoundMode,
    resize_linear,
    resize_nearest_neighbor,
)

from op_utils import (
    MAX_RANK,
    Dispatchable,
    _get_dtype,
    _get_shape,
    _make_ptr,
)


def _dispatch_float_dtype[T: Dispatchable](body: T, dtype: DType) raises:
    """Narrowed dtype dispatch for resize ops (float types only).

    Resize is an interpolation operation so only floating-point types are
    meaningful.  Using this instead of the full ``dispatch_dtype`` (which
    covers 13+ types) drastically cuts the number of compile-time
    specializations and keeps ASAN build times within CI timeouts.
    """
    if dtype == DType.float32:
        body.call[DType.float32]()
    elif dtype == DType.float64:
        body.call[DType.float64]()
    elif dtype == DType.float16:
        body.call[DType.float16]()
    elif dtype == DType.bfloat16:
        body.call[DType.bfloat16]()
    else:
        raise Error(
            "resize: only floating-point dtypes are supported, got "
            + String(dtype)
        )


@export
def PyInit_resize_ops() -> PythonObject:
    """Create a Python module with resize kernel function bindings."""
    try:
        var b = PythonModuleBuilder("resize_ops")
        b.def_function[resize_linear_dispatcher](
            "ResizeLinear",
            docstring="Resize using linear (bilinear) interpolation (CPU-only)",
        )
        b.def_function[resize_nearest_dispatcher](
            "ResizeNearest",
            docstring="Resize using nearest-neighbor interpolation (CPU-only)",
        )
        b.def_function[resize_bicubic_dispatcher](
            "ResizeBicubic",
            docstring="Resize using bicubic interpolation (CPU path)",
        )
        return b.finalize()
    except e:
        abort(t"failed to create resize op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# Linear resize kernel
# ===----------------------------------------------------------------------=== #
#
# Wraps ``resize_linear[coord_mode, antialias, dtype](input, output)`` from
# ``nn.resize`` by constructing ``TileTensor`` views from raw pointers and
# runtime shapes.  Both ``coord_mode`` and ``antialias`` must be compile-time
# constants; ``rank`` must also be compile-time for ``TileTensor``.


@always_inline
def _resize_linear_impl[
    coord_mode: Int,
    antialias: Bool,
    dtype: DType,
    rank: Int,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_shape: InlineArray[Int, MAX_RANK],
    out_shape: InlineArray[Int, MAX_RANK],
) raises:
    """Resize input into output using a compile-time-specialised linear filter.

    Builds ``TileTensor`` views from raw pointers and delegates to
    ``nn.resize.resize_linear``.

    Parameters:
        coord_mode: Coordinate transformation mode (0=half_pixel,
            1=align_corners, 2=asymmetric, 3=half_pixel_1D).
        antialias: Apply an antialiasing filter when downscaling.
        dtype: Element data type of both tensors.
        rank: Tensor rank (compile-time constant required by ``TileTensor``).

    Args:
        out_ptr: Output buffer pointer.
        in_ptr: Input buffer pointer.
        in_shape: Input shape; entries 0..rank-1 are valid.
        out_shape: Output shape; entries 0..rank-1 are valid.
    """
    var in_idx = IndexList[rank](0)
    var out_idx = IndexList[rank](0)
    for i in range(rank):
        in_idx[i] = in_shape[i]
        out_idx[i] = out_shape[i]

    var in_tt = TileTensor(in_ptr, row_major(Coord(in_idx)))
    var out_tt = TileTensor(out_ptr, row_major(Coord(out_idx)))

    resize_linear[CoordinateTransformationMode(coord_mode), antialias](
        in_tt, out_tt
    )


# ===----------------------------------------------------------------------=== #
# Dispatcher
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _ResizeLinearBody[coord_mode: Int, antialias: Bool](Dispatchable):
    """Dispatch body for linear resize, specialised over coord_mode and antialias.

    Buffer addresses and shape metadata are stored as plain integers and cast
    to typed pointers inside ``call[t]``.  Rank dispatch (1–MAX_RANK) uses an
    if-elif chain so each branch is instantiated at compile-time.
    """

    var out_addr: Int
    var in_addr: Int
    var in_shape: InlineArray[Int, MAX_RANK]
    var out_shape: InlineArray[Int, MAX_RANK]
    var rank: Int

    def call[t: DType](self) raises -> None:
        var out_ptr = _make_ptr[t](self.out_addr)
        var in_ptr = _make_ptr[t](self.in_addr)
        if self.rank == 1:
            _resize_linear_impl[Self.coord_mode, Self.antialias, t, 1](
                out_ptr, in_ptr, self.in_shape, self.out_shape
            )
        elif self.rank == 2:
            _resize_linear_impl[Self.coord_mode, Self.antialias, t, 2](
                out_ptr, in_ptr, self.in_shape, self.out_shape
            )
        elif self.rank == 3:
            _resize_linear_impl[Self.coord_mode, Self.antialias, t, 3](
                out_ptr, in_ptr, self.in_shape, self.out_shape
            )
        elif self.rank == 4:
            _resize_linear_impl[Self.coord_mode, Self.antialias, t, 4](
                out_ptr, in_ptr, self.in_shape, self.out_shape
            )
        else:
            _resize_linear_impl[Self.coord_mode, Self.antialias, t, 5](
                out_ptr, in_ptr, self.in_shape, self.out_shape
            )


def _dispatch_coord_mode[
    antialias: Bool
](
    coord_mode: Int,
    dtype: DType,
    out_addr: Int,
    in_addr: Int,
    in_shape: InlineArray[Int, MAX_RANK],
    out_shape: InlineArray[Int, MAX_RANK],
    rank: Int,
) raises:
    """Dispatch over the four coordinate transformation modes.

    Parameters:
        antialias: Compile-time antialias flag.

    Args:
        coord_mode: Runtime coordinate mode (0–3).
        dtype: Element data type.
        out_addr: Raw address of the output buffer.
        in_addr: Raw address of the input buffer.
        in_shape: Input shape (first ``rank`` entries valid).
        out_shape: Output shape (first ``rank`` entries valid).
        rank: Tensor rank.

    Raises:
        Error: If ``coord_mode`` is not 0–3.
    """
    if coord_mode == 0:
        _dispatch_float_dtype(
            _ResizeLinearBody[0, antialias](
                out_addr, in_addr, in_shape, out_shape, rank
            ),
            dtype,
        )
    elif coord_mode == 1:
        _dispatch_float_dtype(
            _ResizeLinearBody[1, antialias](
                out_addr, in_addr, in_shape, out_shape, rank
            ),
            dtype,
        )
    elif coord_mode == 2:
        _dispatch_float_dtype(
            _ResizeLinearBody[2, antialias](
                out_addr, in_addr, in_shape, out_shape, rank
            ),
            dtype,
        )
    elif coord_mode == 3:
        _dispatch_float_dtype(
            _ResizeLinearBody[3, antialias](
                out_addr, in_addr, in_shape, out_shape, rank
            ),
            dtype,
        )
    else:
        raise Error(
            "resize_linear: unsupported coordinate_transform_mode "
            + String(coord_mode)
        )


def resize_linear_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Linear resize dispatcher: unwraps PythonObjects and dispatches by dtype.

    The ``device_context_ptr`` argument is accepted for API uniformity but
    unused — ``resize_linear`` is a CPU-only kernel.

    Args:
        out_buffer: Output buffer (pre-allocated with output shape).
        in_buffer: Input data buffer.
        params: Python tuple ``(coord_mode, antialias, rank,
            in_shape_list, out_shape_list)``.
        device_context_ptr: Device context pointer (unused, CPU-only).
    """
    var dtype = _get_dtype(in_buffer)
    var coord_mode = Int(py=params[0])
    var antialias = Bool(py=params[1])
    var rank = Int(py=params[2])
    var in_shape = _get_shape(params[3], rank)
    var out_shape = _get_shape(params[4], rank)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())

    if antialias:
        _dispatch_coord_mode[True](
            coord_mode, dtype, out_addr, in_addr, in_shape, out_shape, rank
        )
    else:
        _dispatch_coord_mode[False](
            coord_mode, dtype, out_addr, in_addr, in_shape, out_shape, rank
        )


# ===----------------------------------------------------------------------=== #
# Nearest-neighbor resize kernel
# ===----------------------------------------------------------------------=== #
#
# Wraps ``resize_nearest_neighbor[coord_mode, round_mode, dtype](in, out)``
# from ``nn.resize``.  Both ``coord_mode`` and ``round_mode`` must be
# compile-time constants; ``rank`` must also be compile-time for
# ``TileTensor``.


@always_inline
def _resize_nearest_impl[
    coord_mode: Int,
    round_mode: Int,
    dtype: DType,
    rank: Int,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_shape: InlineArray[Int, MAX_RANK],
    out_shape: InlineArray[Int, MAX_RANK],
) raises:
    """Resize input into output using nearest-neighbor interpolation.

    Parameters:
        coord_mode: Coordinate transformation mode (0–3).
        round_mode: Rounding mode (0=HalfDown, 1=HalfUp, 2=Floor, 3=Ceil).
        dtype: Element data type of both tensors.
        rank: Tensor rank (compile-time constant required by ``TileTensor``).

    Args:
        out_ptr: Output buffer pointer.
        in_ptr: Input buffer pointer.
        in_shape: Input shape; entries 0..rank-1 are valid.
        out_shape: Output shape; entries 0..rank-1 are valid.
    """
    var in_idx = IndexList[rank](0)
    var out_idx = IndexList[rank](0)
    for i in range(rank):
        in_idx[i] = in_shape[i]
        out_idx[i] = out_shape[i]

    var in_tt = TileTensor(in_ptr, row_major(Coord(in_idx)))
    var out_tt = TileTensor(out_ptr, row_major(Coord(out_idx)))

    resize_nearest_neighbor[
        CoordinateTransformationMode(coord_mode), RoundMode(round_mode)
    ](in_tt, out_tt)


# ===----------------------------------------------------------------------=== #
# Nearest-neighbor dispatcher
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _ResizeNearestBody[coord_mode: Int, round_mode: Int](Dispatchable):
    """Dispatch body for nearest resize, specialised over coord_mode and round_mode.

    Same layout as ``_ResizeLinearBody``: buffer addresses and shape metadata
    stored as plain integers, rank dispatched via if-elif chain.
    """

    var out_addr: Int
    var in_addr: Int
    var in_shape: InlineArray[Int, MAX_RANK]
    var out_shape: InlineArray[Int, MAX_RANK]
    var rank: Int

    def call[t: DType](self) raises -> None:
        var out_ptr = _make_ptr[t](self.out_addr)
        var in_ptr = _make_ptr[t](self.in_addr)
        if self.rank == 1:
            _resize_nearest_impl[Self.coord_mode, Self.round_mode, t, 1](
                out_ptr, in_ptr, self.in_shape, self.out_shape
            )
        elif self.rank == 2:
            _resize_nearest_impl[Self.coord_mode, Self.round_mode, t, 2](
                out_ptr, in_ptr, self.in_shape, self.out_shape
            )
        elif self.rank == 3:
            _resize_nearest_impl[Self.coord_mode, Self.round_mode, t, 3](
                out_ptr, in_ptr, self.in_shape, self.out_shape
            )
        elif self.rank == 4:
            _resize_nearest_impl[Self.coord_mode, Self.round_mode, t, 4](
                out_ptr, in_ptr, self.in_shape, self.out_shape
            )
        else:
            _resize_nearest_impl[Self.coord_mode, Self.round_mode, t, 5](
                out_ptr, in_ptr, self.in_shape, self.out_shape
            )


def _dispatch_nearest_round_mode[
    coord_mode: Int,
](
    round_mode: Int,
    dtype: DType,
    out_addr: Int,
    in_addr: Int,
    in_shape: InlineArray[Int, MAX_RANK],
    out_shape: InlineArray[Int, MAX_RANK],
    rank: Int,
) raises:
    """Dispatch over the four rounding modes for nearest resize.

    Parameters:
        coord_mode: Compile-time coordinate transformation mode.

    Args:
        round_mode: Runtime rounding mode (0–3).
        dtype: Element data type.
        out_addr: Raw address of the output buffer.
        in_addr: Raw address of the input buffer.
        in_shape: Input shape (first ``rank`` entries valid).
        out_shape: Output shape (first ``rank`` entries valid).
        rank: Tensor rank.
    """
    if round_mode == 0:
        _dispatch_float_dtype(
            _ResizeNearestBody[coord_mode, 0](
                out_addr, in_addr, in_shape, out_shape, rank
            ),
            dtype,
        )
    elif round_mode == 1:
        _dispatch_float_dtype(
            _ResizeNearestBody[coord_mode, 1](
                out_addr, in_addr, in_shape, out_shape, rank
            ),
            dtype,
        )
    elif round_mode == 2:
        _dispatch_float_dtype(
            _ResizeNearestBody[coord_mode, 2](
                out_addr, in_addr, in_shape, out_shape, rank
            ),
            dtype,
        )
    elif round_mode == 3:
        _dispatch_float_dtype(
            _ResizeNearestBody[coord_mode, 3](
                out_addr, in_addr, in_shape, out_shape, rank
            ),
            dtype,
        )
    else:
        raise Error(
            "resize_nearest: unsupported round_mode " + String(round_mode)
        )


def _dispatch_nearest_coord_mode(
    coord_mode: Int,
    round_mode: Int,
    dtype: DType,
    out_addr: Int,
    in_addr: Int,
    in_shape: InlineArray[Int, MAX_RANK],
    out_shape: InlineArray[Int, MAX_RANK],
    rank: Int,
) raises:
    """Dispatch over coordinate modes, then delegate to round-mode dispatch.

    Args:
        coord_mode: Runtime coordinate mode (0–3).
        round_mode: Runtime rounding mode (0–3).
        dtype: Element data type.
        out_addr: Raw address of the output buffer.
        in_addr: Raw address of the input buffer.
        in_shape: Input shape (first ``rank`` entries valid).
        out_shape: Output shape (first ``rank`` entries valid).
        rank: Tensor rank.
    """
    if coord_mode == 0:
        _dispatch_nearest_round_mode[0](
            round_mode, dtype, out_addr, in_addr, in_shape, out_shape, rank
        )
    elif coord_mode == 1:
        _dispatch_nearest_round_mode[1](
            round_mode, dtype, out_addr, in_addr, in_shape, out_shape, rank
        )
    elif coord_mode == 2:
        _dispatch_nearest_round_mode[2](
            round_mode, dtype, out_addr, in_addr, in_shape, out_shape, rank
        )
    elif coord_mode == 3:
        _dispatch_nearest_round_mode[3](
            round_mode, dtype, out_addr, in_addr, in_shape, out_shape, rank
        )
    else:
        raise Error(
            "resize_nearest: unsupported coordinate_transform_mode "
            + String(coord_mode)
        )


def resize_nearest_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Nearest-neighbor resize dispatcher: unwraps PythonObjects and dispatches.

    Args:
        out_buffer: Output buffer (pre-allocated with output shape).
        in_buffer: Input data buffer.
        params: Python tuple ``(coord_mode, round_mode, rank,
            in_shape_list, out_shape_list)``.
        device_context_ptr: Device context pointer (unused, CPU-only).
    """
    var dtype = _get_dtype(in_buffer)
    var coord_mode = Int(py=params[0])
    var round_mode = Int(py=params[1])
    var rank = Int(py=params[2])
    var in_shape = _get_shape(params[3], rank)
    var out_shape = _get_shape(params[4], rank)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())

    _dispatch_nearest_coord_mode(
        coord_mode,
        round_mode,
        dtype,
        out_addr,
        in_addr,
        in_shape,
        out_shape,
        rank,
    )


# ===----------------------------------------------------------------------=== #
# Bicubic resize kernel
# ===----------------------------------------------------------------------=== #
#
# Wraps ``cpu_bicubic_kernel(output, input)`` from ``nn.bicubic``.
# The kernel is rank-4 only (NCHW), uses hardcoded half_pixel coordinate
# mapping and a=-0.75 Catmull-Rom cubic filter.  No compile-time mode
# parameters — just dtype dispatch.


@always_inline
def _resize_bicubic_impl[
    dtype: DType,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_shape: InlineArray[Int, MAX_RANK],
    out_shape: InlineArray[Int, MAX_RANK],
) raises:
    """Resize input into output using bicubic interpolation (rank-4 only).

    Parameters:
        dtype: Element data type of both tensors.

    Args:
        out_ptr: Output buffer pointer.
        in_ptr: Input buffer pointer.
        in_shape: Input shape; entries 0..3 are valid (NCHW).
        out_shape: Output shape; entries 0..3 are valid (NCHW).
    """
    var in_idx = IndexList[4](0)
    var out_idx = IndexList[4](0)
    for i in range(4):
        in_idx[i] = in_shape[i]
        out_idx[i] = out_shape[i]

    var in_tt = TileTensor(in_ptr, row_major(Coord(in_idx)))
    var out_tt = TileTensor(out_ptr, row_major(Coord(out_idx)))

    cpu_bicubic_kernel(out_tt, in_tt)


# ===----------------------------------------------------------------------=== #
# Bicubic dispatcher
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _ResizeBicubicBody(Dispatchable):
    """Dispatch body for bicubic resize — dtype dispatch only.

    The kernel is rank-4 (NCHW) with no configurable modes, so this is
    the simplest of the three resize dispatchers.
    """

    var out_addr: Int
    var in_addr: Int
    var in_shape: InlineArray[Int, MAX_RANK]
    var out_shape: InlineArray[Int, MAX_RANK]

    def call[t: DType](self) raises -> None:
        var out_ptr = _make_ptr[t](self.out_addr)
        var in_ptr = _make_ptr[t](self.in_addr)
        _resize_bicubic_impl[t](out_ptr, in_ptr, self.in_shape, self.out_shape)


def resize_bicubic_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Bicubic resize dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer (pre-allocated with output shape).
        in_buffer: Input data buffer.
        params: Python tuple ``(in_shape_list, out_shape_list)``.
        device_context_ptr: Device context pointer (unused, CPU path).
    """
    var dtype = _get_dtype(in_buffer)
    var in_shape = _get_shape(params[0], 4)
    var out_shape = _get_shape(params[1], 4)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())

    _dispatch_float_dtype(
        _ResizeBicubicBody(out_addr, in_addr, in_shape, out_shape), dtype
    )
