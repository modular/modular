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

"""Mojo kernel wrappers for argmax/argmin MO interpreter operations.

ArgMax/ArgMin reduce along an axis but return indices (int64) instead of
values. Input is normalized to 3D [dim0, dim1, dim2] where dim1 is the
reduction axis. Each output element independently scans dim1 to find the
index of the extreme value.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator

from std.algorithm.functional import elementwise, IndexList
from std.memory import OpaquePointer
from std.runtime.asyncrt import DeviceContextPtr

from op_utils import _get_dtype, _get_ctx, _make_ptr


@export
def PyInit_argmax_ops() -> PythonObject:
    """Create a Python module with argmax/argmin kernel function bindings."""
    try:
        var b = PythonModuleBuilder("argmax_ops")
        b.def_function[argmax_dispatcher](
            "ArgMax", docstring="ArgMax along axis"
        )
        b.def_function[argmin_dispatcher](
            "ArgMin", docstring="ArgMin along axis"
        )
        return b.finalize()
    except e:
        abort(t"failed to create argmax op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# ArgReduce kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def argminmax_reduce_op[
    dtype: DType, //, is_max: Bool
](
    out_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    dim0: Int,
    dim1: Int,
    dim2: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Scan the reduction axis for each output element to find extreme index.

    The existing optimized argmax/argmin kernels in max/kernels/src/nn use
    TileTensor with compile-time layouts, which is incompatible with the
    interpreter's runtime-dynamic raw-pointer interface. This follows the
    same pattern as reduce_ops.mojo.

    Parameters:
        dtype: Input data type (inferred from in_ptr).
        is_max: True for argmax, False for argmin.

    Args:
        out_ptr: Output buffer (int64), length dim0 * dim2.
        in_ptr: Input buffer (dtype), length dim0 * dim1 * dim2.
        dim0: Product of dimensions before the reduction axis.
        dim1: Size of the reduction axis.
        dim2: Product of dimensions after the reduction axis.
        ctx: Device context pointer (null for CPU).
    """
    var total = dim0 * dim2
    var in_stride0 = dim1 * dim2

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, dim1, dim2, in_stride0)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var i0, i2 = divmod(i, dim2)
        var base = i0 * in_stride0 + i2

        var best_idx: Int64 = 0
        var best_val = in_ptr[base]

        for d in range(1, dim1):
            var val = in_ptr[base + d * dim2]

            comptime if is_max:
                if val > best_val:
                    best_val = val
                    best_idx = Int64(d)
            else:
                if val < best_val:
                    best_val = val
                    best_idx = Int64(d)

        out_ptr[i] = best_idx

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
# Dispatchers
# ===----------------------------------------------------------------------=== #


def argmax_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """ArgMax dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer (int64).
        in_buffer: Input data buffer.
        params: Python tuple (dim0, dim1, dim2).
        device_context_ptr: Device context pointer (null for CPU).
    """
    _arg_dispatch[True](out_buffer, in_buffer, params, device_context_ptr)


def argmin_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """ArgMin dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer (int64).
        in_buffer: Input data buffer.
        params: Python tuple (dim0, dim1, dim2).
        device_context_ptr: Device context pointer (null for CPU).
    """
    _arg_dispatch[False](out_buffer, in_buffer, params, device_context_ptr)


def _arg_dispatch[
    is_max: Bool
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var d0 = Int(py=params[0])
    var d1 = Int(py=params[1])
    var d2 = Int(py=params[2])
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)
    var out_ptr = _make_ptr[DType.int64](out_addr)

    if dtype == DType.float32:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.float32](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.float64:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.float64](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.float16:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.float16](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.bfloat16:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.bfloat16](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.int8:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.int8](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.int16:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.int16](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.int32:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.int32](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.int64:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.int64](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.uint8:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.uint8](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.uint16:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.uint16](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.uint32:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.uint32](in_addr), d0, d1, d2, ctx
        )
    elif dtype == DType.uint64:
        argminmax_reduce_op[is_max](
            out_ptr, _make_ptr[DType.uint64](in_addr), d0, d1, d2, ctx
        )
    else:
        raise Error("Unsupported dtype for argmax/argmin: " + String(dtype))
