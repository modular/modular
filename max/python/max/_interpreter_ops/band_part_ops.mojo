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

"""Mojo kernel wrappers for band_part MO interpreter operations.

Masks a tensor's last two dims (M, N) based on a diagonal band.
out[..., m, n] = in_band(m,n) ? input[..., m,n] : 0  (exclude=false)
out[..., m, n] = in_band(m,n) ? 0 : input[..., m,n]  (exclude=true)

where in_band(m,n) = (num_lower<0 || (m-n)<=num_lower) &&
                      (num_upper<0 || (n-m)<=num_upper)

CPU and GPU via the elementwise + DeviceContextPtr pattern.
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
def PyInit_band_part_ops() -> PythonObject:
    """Create a Python module with band_part kernel function bindings."""
    try:
        var b = PythonModuleBuilder("band_part_ops")
        b.def_function[band_part_dispatcher](
            "BandPart", docstring="Matrix band part masking"
        )
        return b.finalize()
    except e:
        abort(t"failed to create band_part op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# BandPart kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def band_part_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    mn_stride: Int,
    M: Int,
    N: Int,
    num_lower: Int,
    num_upper: Int,
    exclude_flag: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Apply band_part masking over the last two dims of a tensor.

    Parameters:
        dtype: Data type (inferred from pointers).

    Args:
        out_ptr: Output buffer, same shape as input.
        in_ptr: Input buffer [..., M, N].
        mn_stride: M * N (stride of batch dimensions).
        M: Second-to-last dimension size.
        N: Last dimension size.
        num_lower: Lower band count (-1 means keep all).
        num_upper: Upper band count (-1 means keep all).
        exclude_flag: 1 to invert the mask, 0 for normal.
        ctx: Device context pointer (null for CPU).
    """
    var total = mn_stride  # caller passes batch * M * N

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        M,
        N,
        num_lower,
        num_upper,
        exclude_flag,
    )
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var mn_rem = i % (M * N)
        var m, n = divmod(mn_rem, N)

        var in_band = Int(1)
        if num_lower >= 0 and (m - n) > num_lower:
            in_band = 0
        if num_upper >= 0 and (n - m) > num_upper:
            in_band = 0

        var keep = in_band ^ exclude_flag
        if keep:
            out_ptr[i] = in_ptr[i]
        else:
            out_ptr[i] = Scalar[dtype](0)

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
struct _BandPartBody(Dispatchable):
    """Dispatch body for the BandPart operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var total: Int
    var M: Int
    var N: Int
    var num_lower: Int
    var num_upper: Int
    var exclude_flag: Int
    var ctx: OpaquePointer[MutExternalOrigin]

    def call[t: DType](self) raises -> None:
        band_part_op(
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.total,
            self.M,
            self.N,
            self.num_lower,
            self.num_upper,
            self.exclude_flag,
            self.ctx,
        )


def band_part_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """BandPart dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer (same shape as input).
        in_buffer: Input data buffer.
        params: Python tuple (total, M, N, num_lower, num_upper, exclude).
        device_context_ptr: Device context pointer.
    """
    var dtype = _get_dtype(in_buffer)
    var total = Int(py=params[0])
    var p_M = Int(py=params[1])
    var p_N = Int(py=params[2])
    var p_lower = Int(py=params[3])
    var p_upper = Int(py=params[4])
    var p_exclude = Int(py=params[5])
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    dispatch_dtype(
        _BandPartBody(
            out_addr,
            in_addr,
            total,
            p_M,
            p_N,
            p_lower,
            p_upper,
            p_exclude,
            ctx,
        ),
        dtype,
    )
