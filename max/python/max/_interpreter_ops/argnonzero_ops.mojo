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

"""Mojo kernel wrappers for arg_nonzero MO interpreter operations.

ArgNonzero returns a 2D int64 tensor of shape [nnz, rank(input)] containing
the row-major coordinates of every nonzero element in the input tensor.

Because the number of nonzero elements (nnz) is data-dependent, the kernel
is split into two passes:

1. ArgNonZeroCount — writes the nonzero count into a 1-element int64 buffer.
2. ArgNonZeroFill  — writes the [nnz, rank] coordinate matrix into a
   pre-allocated int64 output buffer.

Both passes are CPU-only: `mo.arg_nonzero` carries the `MO_HostOnly` trait,
so the Python handler always receives a CPU input buffer.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.memory import OpaquePointer

from op_utils import (
    _get_dtype,
    _get_shape,
    _make_ptr,
    Dispatchable,
    dispatch_dtype,
    MAX_RANK,
)


@export
def PyInit_argnonzero_ops() -> PythonObject:
    """Create a Python module with arg_nonzero kernel function bindings."""
    try:
        var b = PythonModuleBuilder("argnonzero_ops")
        b.def_function[argnonzero_count_dispatcher](
            "ArgNonZeroCount",
            docstring=(
                "Count nonzero elements and write result to 1-elem int64 buffer"
            ),
        )
        b.def_function[argnonzero_fill_dispatcher](
            "ArgNonZeroFill",
            docstring=(
                "Fill [nnz, rank] int64 coordinate buffer for nonzero elements"
            ),
        )
        return b.finalize()
    except e:
        abort(t"failed to create argnonzero op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# Count kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def argnonzero_count_op[
    dtype: DType, //
](
    count_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    numel: Int,
):
    """Count the nonzero elements in a flat input buffer.

    Parameters:
        dtype: Element dtype of the input buffer.

    Args:
        count_ptr: Output buffer of length 1; receives the nonzero count.
        in_ptr: Flat input data buffer of length numel.
        numel: Total number of elements in the input tensor.
    """
    var count: Int64 = 0
    for i in range(numel):
        if in_ptr[i] != 0:
            count += 1
    count_ptr[0] = count


@fieldwise_init
struct _ArgNonZeroCountBody(Dispatchable):
    """Dispatch body for the ArgNonZeroCount operation over all data dtypes."""

    var count_addr: Int
    var in_addr: Int
    var numel: Int

    def call[t: DType](self) raises -> None:
        argnonzero_count_op(
            _make_ptr[DType.int64](self.count_addr),
            _make_ptr[t](self.in_addr),
            self.numel,
        )


def argnonzero_count_dispatcher(
    count_buffer: PythonObject,
    in_buffer: PythonObject,
    numel: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """ArgNonZeroCount dispatcher: count nonzeros and write to 1-elem int64 buffer.

    Args:
        count_buffer: Pre-allocated 1-element int64 output buffer.
        in_buffer: Input data buffer.
        numel: Total number of elements in the input tensor (Python int).
        device_context_ptr: Device context pointer (unused; op is CPU-only).
    """
    var dtype = _get_dtype(in_buffer)
    dispatch_dtype(
        _ArgNonZeroCountBody(
            Int(py=count_buffer._data_ptr()),
            Int(py=in_buffer._data_ptr()),
            Int(py=numel),
        ),
        dtype,
    )


# ===----------------------------------------------------------------------=== #
# Fill kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def argnonzero_fill_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    numel: Int,
    rank: Int,
    shape: InlineArray[Int, MAX_RANK],
):
    """Fill the [nnz, rank] coordinate buffer for all nonzero elements.

    Each nonzero element at flat index i is decoded into its multi-dimensional
    row-major coordinates and written to consecutive rows of out_ptr.

    Parameters:
        dtype: Element dtype of the input buffer.

    Args:
        out_ptr: Pre-allocated int64 output buffer of shape [nnz, rank].
        in_ptr: Flat input data buffer of length numel.
        numel: Total number of elements in the input tensor.
        rank: Rank of the input tensor (number of dimensions).
        shape: Shape of the input tensor (first `rank` elements are valid).
    """
    var j: Int = 0
    for i in range(numel):
        if in_ptr[i] != 0:
            # Decode flat index i into row-major multi-dim coordinates.
            var remaining = i
            var coords = InlineArray[Int, MAX_RANK](fill=0)
            var k = rank - 1
            while k >= 0:
                coords[k] = remaining % shape[k]
                remaining //= shape[k]
                k -= 1

            # Write coordinates to the j-th row of the output buffer.
            for c in range(rank):
                out_ptr[j * rank + c] = Int64(coords[c])
            j += 1


@fieldwise_init
struct _ArgNonZeroFillBody(Dispatchable):
    """Dispatch body for the ArgNonZeroFill operation over all data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var numel: Int
    var rank: Int
    var shape: InlineArray[Int, MAX_RANK]

    def call[t: DType](self) raises -> None:
        argnonzero_fill_op(
            _make_ptr[DType.int64](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.numel,
            self.rank,
            self.shape,
        )


def argnonzero_fill_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    shape: PythonObject,
    rank: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """ArgNonZeroFill dispatcher: fill [nnz, rank] coordinate buffer.

    Args:
        out_buffer: Pre-allocated int64 output buffer of shape [nnz, rank].
        in_buffer: Input data buffer.
        shape: Python sequence of input tensor dimensions.
        rank: Rank of the input tensor (Python int).
        device_context_ptr: Device context pointer (unused; op is CPU-only).
    """
    var dtype = _get_dtype(in_buffer)
    var r = Int(py=rank)
    var sh = _get_shape(shape, r)
    var numel = 1
    for i in range(r):
        numel *= sh[i]

    dispatch_dtype(
        _ArgNonZeroFillBody(
            Int(py=out_buffer._data_ptr()),
            Int(py=in_buffer._data_ptr()),
            numel,
            r,
            sh,
        ),
        dtype,
    )
