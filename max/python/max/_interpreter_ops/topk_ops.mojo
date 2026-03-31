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

"""Mojo kernel wrappers for top_k MO interpreter operations.

Top-K selects the k largest values and their original indices along a given
axis. Input is normalized to 3-D [dim0, axis_dim, dim2] and the kernel
launches one elementwise thread per (dim0, dim2) row-pair; each thread
performs an O(axis_dim × k + k²) selection scan that requires no heap
allocation, so the same kernel runs identically on CPU and GPU.

Returned values are in descending order (stable); the `sorted` flag is
accepted but currently ignored since the implementation always sorts.
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
    _get_ctx,
    _make_ptr,
    Dispatchable,
    dispatch_dtype,
)


@export
def PyInit_topk_ops() -> PythonObject:
    """Create a Python module with top_k kernel function bindings."""
    try:
        var b = PythonModuleBuilder("topk_ops")
        b.def_function[topk_dispatcher]("TopK", docstring="Top-K along axis")
        return b.finalize()
    except e:
        abort(t"failed to create topk op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# Top-K kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def topk_op[
    dtype: DType, //
](
    out_val_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_idx_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    dim0: Int,
    dim1: Int,
    dim2: Int,
    k: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Select the top-k largest values and their indices along the axis dim.

    The input is assumed to be laid out as [dim0, dim1, dim2] where dim1 is
    the axis being reduced. For each (i0, i2) pair the kernel performs k
    passes of a linear scan, skipping any index already selected in a
    prior pass. This is allocation-free and runs on both CPU and GPU.

    Parameters:
        dtype: Element dtype of the input and value output buffers.

    Args:
        out_val_ptr: Output values buffer, shape [dim0, k, dim2].
        out_idx_ptr: Output indices buffer (int64), shape [dim0, k, dim2].
        in_ptr: Input buffer, shape [dim0, dim1, dim2].
        dim0: Product of dimensions before the top-k axis.
        dim1: Size of the top-k axis.
        dim2: Product of dimensions after the top-k axis.
        k: Number of top elements to select.
        ctx: Device context pointer (null for CPU).
    """
    var total = dim0 * dim2

    @always_inline
    @parameter
    @__copy_capture(out_val_ptr, out_idx_ptr, in_ptr, dim1, dim2, k)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var pair = idx[0]
        var i0, i2 = divmod(pair, dim2)
        var in_base = i0 * dim1 * dim2 + i2
        var out_base = i0 * k * dim2 + i2

        # Selection scan: k passes, each O(dim1 + ki) with no allocation.
        # The already-selected indices are tracked via the output index buffer
        # filled in previous passes of this same thread.
        for ki in range(k):
            var best_val = Scalar[dtype].MIN_FINITE
            var best_idx = Int64(-1)

            for d in range(dim1):
                var val = in_ptr[in_base + d * dim2]

                # Skip indices already committed to earlier ki slots.
                var already = False
                for prev in range(ki):
                    if out_idx_ptr[out_base + prev * dim2] == Int64(d):
                        already = True
                        break

                if not already and (best_idx == -1 or val > best_val):
                    best_val = val
                    best_idx = Int64(d)

            out_val_ptr[out_base + ki * dim2] = best_val
            out_idx_ptr[out_base + ki * dim2] = best_idx

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
struct _TopKBody(Dispatchable):
    """Dispatch body for the TopK operation over numeric data dtypes.

    Stores all buffer addresses as plain integers and casts them to typed
    pointers inside `call[t]`. The values output uses the dispatch dtype t;
    the indices output is always int64 regardless of t.
    """

    var out_val_addr: Int
    var out_idx_addr: Int
    var in_addr: Int
    var dim0: Int
    var dim1: Int
    var dim2: Int
    var k: Int
    var ctx: OpaquePointer[MutExternalOrigin]

    def call[t: DType](self) raises -> None:
        comptime if t.is_numeric():
            topk_op(
                _make_ptr[t](self.out_val_addr),
                _make_ptr[DType.int64](self.out_idx_addr),
                _make_ptr[t](self.in_addr),
                self.dim0,
                self.dim1,
                self.dim2,
                self.k,
                self.ctx,
            )
        else:
            raise Error("top_k requires a numeric dtype, got " + String(t))


def topk_dispatcher(
    out_val_buffer: PythonObject,
    out_idx_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Top-K dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_val_buffer: Output values buffer (same dtype as input).
        out_idx_buffer: Output indices buffer (int64).
        in_buffer: Input data buffer.
        params: Python tuple (dim0, dim1, dim2, k).
        device_context_ptr: Device context pointer (null for CPU).
    """
    var dtype = _get_dtype(in_buffer)
    var out_val_addr = Int(py=out_val_buffer._data_ptr())
    var out_idx_addr = Int(py=out_idx_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    dispatch_dtype(
        _TopKBody(
            out_val_addr,
            out_idx_addr,
            in_addr,
            Int(py=params[0]),
            Int(py=params[1]),
            Int(py=params[2]),
            Int(py=params[3]),
            ctx,
        ),
        dtype,
    )
