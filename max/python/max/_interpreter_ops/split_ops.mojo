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

"""Mojo kernel wrappers for split MO interpreter operations.

Split copies one chunk of the input tensor along the split axis into an
output buffer. Input is normalized to 3D [dim0, axis_dim, dim2] where
axis_dim is the full input axis size. Each call copies the slice
[dim0, axis_offset:axis_offset+out_dim1, dim2] into the output.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator

from std.algorithm.functional import elementwise, IndexList
from std.memory import OpaquePointer
from std.runtime.asyncrt import DeviceContextPtr
from std.sys.info import has_apple_gpu_accelerator

from op_utils import (
    _get_dtype,
    _get_ctx,
    _make_ptr,
    Dispatchable,
    dispatch_dtype,
)


@export
def PyInit_split_ops() -> PythonObject:
    """Create a Python module with split kernel function bindings."""
    try:
        var b = PythonModuleBuilder("split_ops")
        b.def_function[split_copy_dispatcher](
            "SplitCopy", docstring="Copy one split chunk along axis"
        )
        return b.finalize()
    except e:
        abort(t"failed to create split op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# SplitCopy kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def split_copy_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    dim0: Int,
    out_dim1: Int,
    dim2: Int,
    axis_offset: Int,
    in_dim1: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Copy one split chunk from input to output along the normalized axis.

    Parameters:
        dtype: Data type (inferred from pointers).

    Args:
        out_ptr: Output buffer, length dim0 * out_dim1 * dim2.
        in_ptr: Input buffer, length dim0 * in_dim1 * dim2.
        dim0: Product of dimensions before the split axis.
        out_dim1: Size of this chunk along the split axis.
        dim2: Product of dimensions after the split axis.
        axis_offset: Starting index along the split axis for this chunk.
        in_dim1: Full input size along the split axis.
        ctx: Device context pointer (null for CPU).
    """
    var total = dim0 * out_dim1 * dim2
    var in_stride0 = in_dim1 * dim2
    var out_stride0 = out_dim1 * dim2

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, out_stride0, dim2, axis_offset, in_stride0)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var i0, rem = divmod(i, out_stride0)
        var j, i2 = divmod(rem, dim2)
        var in_flat = i0 * in_stride0 + (j + axis_offset) * dim2 + i2
        out_ptr[i] = in_ptr[in_flat]

    if not ctx:
        elementwise[func, simd_width=1](IndexList[1](total))
    else:
        comptime if has_accelerator():
            var device_ctx = DeviceContextPtr(ctx)
            elementwise[func, simd_width=1, target="gpu"](
                IndexList[1](total), device_ctx
            )
        else:
            raise Error("No GPU accelerator available")


# ===----------------------------------------------------------------------=== #
# Dispatcher
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _SplitCopyBody(Dispatchable):
    """Dispatch body for the SplitCopy operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var dim0: Int
    var out_dim1: Int
    var dim2: Int
    var axis_offset: Int
    var in_dim1: Int
    var ctx: OpaquePointer[MutExternalOrigin]

    def call[t: DType](self) raises -> None:
        split_copy_op(
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.dim0,
            self.out_dim1,
            self.dim2,
            self.axis_offset,
            self.in_dim1,
            self.ctx,
        )


def split_copy_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """SplitCopy dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer for this split chunk.
        in_buffer: Input data buffer.
        params: Python tuple (dim0, out_dim1, dim2, axis_offset, in_dim1).
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(in_buffer)
    var d0 = Int(py=params[0])
    var od1 = Int(py=params[1])
    var d2 = Int(py=params[2])
    var ax_off = Int(py=params[3])
    var id1 = Int(py=params[4])
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    dispatch_dtype(
        _SplitCopyBody(out_addr, in_addr, d0, od1, d2, ax_off, id1, ctx),
        dtype,
    )
