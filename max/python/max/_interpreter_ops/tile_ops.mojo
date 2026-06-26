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

"""Mojo kernel wrappers for tile MO interpreter operations.

Tile repeats the input tensor along each dimension. For each output
coordinate (i0, i1, ...), the value comes from
input[i0 % d0, i1 % d1, ...] where d_k = in_shape[k].
CPU-only (mo.tile is MO_HostOnly).
"""

from std.gpu.host import DeviceContext
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from std.algorithm.functional import elementwise, IndexList
from std.utils.coord import Coord

from op_utils import (
    _get_ctx,
    _get_dtype,
    _get_shape,
    _make_ptr,
    Dispatchable,
    dispatch_dtype,
    MAX_RANK,
)


@export
def PyInit_tile_ops() abi("C") -> PythonObject:
    """Create a Python module with tile kernel function bindings."""
    try:
        var b = PythonModuleBuilder("tile_ops")
        b.def_function[tile_dispatcher](
            "Tile", docstring="Tile (repeat) tensor along each dimension"
        )
        return b.finalize()
    except e:
        abort(t"failed to create tile op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# Tile kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def tile_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    in_shape: InlineArray[Int, MAX_RANK],
    out_strides: InlineArray[Int, MAX_RANK],
    in_strides: InlineArray[Int, MAX_RANK],
    rank: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    """Copy input into output, tiling (repeating) along every dimension.

    For each flat output index the N-D coordinate is recovered via
    out_strides, then each coordinate is wrapped with modulo into
    the input shape to compute the source index.

    Parameters:
        dtype: Data type (inferred from pointers).

    Args:
        out_ptr: Output buffer, length total.
        in_ptr: Input buffer.
        in_shape: Input shape (first `rank` elements valid).
        out_strides: Row-major strides of the output tensor.
        in_strides: Row-major strides of the input tensor.
        rank: Number of dimensions.
        total: Total number of output elements.
        ctx: Device context (CPU).
    """

    # TODO: This body captures `InlineArray` shape/stride args, which makes the
    # closure memory-only, so it cannot satisfy the unified-closure
    # `elementwise` overload's `RegisterPassable` bound. Keep using the
    # deprecated parameter-closure overload until the array data is staged in
    # device memory (or the unified overload accepts memory-only closures).
    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, in_shape, out_strides, in_strides, rank)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var rem = i
        var in_flat = 0
        for d in range(rank):
            var coord, new_rem = divmod(rem, out_strides[d])
            rem = new_rem
            in_flat += (coord % in_shape[d]) * in_strides[d]
        out_ptr[i] = in_ptr[in_flat]

    elementwise[func, simd_width=1](Coord(total), ctx)


# ===----------------------------------------------------------------------=== #
# Dispatcher
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _TileBody(Dispatchable):
    """Dispatch body for the Tile operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var in_shape: InlineArray[Int, MAX_RANK]
    var o_strides: InlineArray[Int, MAX_RANK]
    var i_strides: InlineArray[Int, MAX_RANK]
    var rank: Int
    var total: Int
    var ctx: DeviceContext

    def call[t: DType](self) raises -> None:
        tile_op(
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.in_shape,
            self.o_strides,
            self.i_strides,
            self.rank,
            self.total,
            self.ctx,
        )


def tile_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Tile dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer (pre-allocated by Python handler).
        in_buffer: Input data buffer.
        params: Python tuple (in_shape, out_strides, in_strides, rank).
        device_context_ptr: Device context pointer (unused, CPU-only).
    """
    var dtype = _get_dtype(in_buffer)
    var rank = Int(py=params[3])
    var in_shape = _get_shape(params[0], rank)
    var o_strides = _get_shape(params[1], rank)
    var i_strides = _get_shape(params[2], rank)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var total = Int(py=out_buffer.num_elements)
    var ctx = _get_ctx(device_context_ptr)

    dispatch_dtype(
        _TileBody(
            out_addr, in_addr, in_shape, o_strides, i_strides, rank, total, ctx
        ),
        dtype,
    )
