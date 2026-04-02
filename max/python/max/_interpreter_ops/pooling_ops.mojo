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

"""Mojo kernel wrappers for pooling MO interpreter operations."""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator
from std.utils.numerics import min_or_neg_inf

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
def PyInit_pooling_ops() -> PythonObject:
    """Create a Python module with pooling kernel function bindings."""
    try:
        var b = PythonModuleBuilder("pooling_ops")
        b.def_function[max_pool_dispatcher](
            "MaxPool", docstring="2D max pooling (floor mode)"
        )
        b.def_function[max_pool_ceil_dispatcher](
            "MaxPoolCeil", docstring="2D max pooling (ceil mode)"
        )
        return b.finalize()
    except e:
        abort(t"failed to create pooling op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# Max pooling kernel
# ===----------------------------------------------------------------------=== #
#
# Input layout: NHWC [batch, height, width, channels]
# Output layout: NHWC [batch, out_h, out_w, channels]
# For each output element (n, oh, ow, c):
#   max over kernel window (kh, kw) of input[n, oh*sh-ph+kh*dh, ow*sw-pw+kw*dw, c]


@always_inline
def max_pool_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    batch: Int,
    in_h: Int,
    in_w: Int,
    channels: Int,
    out_h: Int,
    out_w: Int,
    filter_h: Int,
    filter_w: Int,
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
    pad_h: Int,
    pad_w: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    var total = batch * out_h * out_w * channels
    var in_row_stride = in_w * channels
    var in_batch_stride = in_h * in_row_stride
    var out_row_stride = out_w * channels
    var out_batch_stride = out_h * out_row_stride

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        in_h,
        in_w,
        channels,
        filter_h,
        filter_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
        in_row_stride,
        in_batch_stride,
        out_row_stride,
        out_batch_stride,
    )
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var n, r = divmod(i, out_batch_stride)
        var oh, r2 = divmod(r, out_row_stride)
        var ow, c = divmod(r2, channels)

        var max_val = min_or_neg_inf[dtype]()
        for kh in range(filter_h):
            var ih = oh * stride_h - pad_h + kh * dilation_h
            if ih < 0 or ih >= in_h:
                continue
            for kw in range(filter_w):
                var iw = ow * stride_w - pad_w + kw * dilation_w
                if iw < 0 or iw >= in_w:
                    continue
                var in_flat = (
                    n * in_batch_stride + ih * in_row_stride + iw * channels + c
                )
                var val = in_ptr[in_flat]
                if val > max_val:
                    max_val = val

        out_ptr[i] = max_val

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
struct _MaxPoolBody(Dispatchable):
    """Dispatch body for the MaxPool operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var batch: Int
    var in_h: Int
    var in_w: Int
    var channels: Int
    var out_h: Int
    var out_w: Int
    var filter_h: Int
    var filter_w: Int
    var stride_h: Int
    var stride_w: Int
    var dilation_h: Int
    var dilation_w: Int
    var pad_h: Int
    var pad_w: Int
    var ctx: OpaquePointer[MutExternalOrigin]

    def call[t: DType](self) raises -> None:
        max_pool_op(
            _make_ptr[t](self.out_addr),
            _make_ptr[t](self.in_addr),
            self.batch,
            self.in_h,
            self.in_w,
            self.channels,
            self.out_h,
            self.out_w,
            self.filter_h,
            self.filter_w,
            self.stride_h,
            self.stride_w,
            self.dilation_h,
            self.dilation_w,
            self.pad_h,
            self.pad_w,
            self.ctx,
        )


# ===----------------------------------------------------------------------=== #
# Python dispatchers
# ===----------------------------------------------------------------------=== #


def _unpack_and_dispatch(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Shared logic for both floor and ceil max_pool dispatchers."""
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var batch = Int(py=params[0])
    var in_h = Int(py=params[1])
    var in_w = Int(py=params[2])
    var channels = Int(py=params[3])
    var out_h = Int(py=params[4])
    var out_w = Int(py=params[5])
    var filter_h = Int(py=params[6])
    var filter_w = Int(py=params[7])
    var stride_h = Int(py=params[8])
    var stride_w = Int(py=params[9])
    var dilation_h = Int(py=params[10])
    var dilation_w = Int(py=params[11])
    var pad_h = Int(py=params[12])
    var pad_w = Int(py=params[13])
    var ctx = _get_ctx(device_context_ptr)

    dispatch_dtype(
        _MaxPoolBody(
            out_addr,
            in_addr,
            batch,
            in_h,
            in_w,
            channels,
            out_h,
            out_w,
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            pad_h,
            pad_w,
            ctx,
        ),
        dtype,
    )


def max_pool_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Max pool dispatcher (floor mode).

    Args:
        out_buffer: Output buffer [N, out_h, out_w, C].
        in_buffer: Input buffer [N, H, W, C].
        params: Python tuple of pooling parameters.
        device_context_ptr: Device context pointer (null for CPU).
    """
    _unpack_and_dispatch(out_buffer, in_buffer, params, device_context_ptr)


def max_pool_ceil_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Max pool dispatcher (ceil mode).

    Args:
        out_buffer: Output buffer [N, out_h, out_w, C].
        in_buffer: Input buffer [N, H, W, C].
        params: Python tuple of pooling parameters.
        device_context_ptr: Device context pointer (null for CPU).
    """
    _unpack_and_dispatch(out_buffer, in_buffer, params, device_context_ptr)
