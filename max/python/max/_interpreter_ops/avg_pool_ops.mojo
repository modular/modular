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

"""Mojo kernel wrappers for avg_pool2d MO interpreter operations.

2D average pooling over NHWC input with configurable kernel size, stride,
dilation, padding, and count_boundary semantics.

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
def PyInit_avg_pool_ops() -> PythonObject:
    """Create a Python module with avg_pool2d kernel function bindings."""
    try:
        var b = PythonModuleBuilder("avg_pool_ops")
        b.def_function[avg_pool2d_dispatcher](
            "AvgPool2d", docstring="2D average pooling (NHWC)"
        )
        return b.finalize()
    except e:
        abort(t"failed to create avg_pool op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# AvgPool2d kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def avg_pool2d_op[
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
    kH: Int,
    kW: Int,
    stride_h: Int,
    stride_w: Int,
    dil_h: Int,
    dil_w: Int,
    pad_h_before: Int,
    pad_w_before: Int,
    count_boundary_flag: Int,
    ctx: OpaquePointer[MutExternalOrigin],
) raises:
    """Compute 2D average pooling over NHWC input.

    Parameters:
        dtype: Data type (inferred from pointers).

    Args:
        out_ptr: Output buffer [N, out_h, out_w, C].
        in_ptr: Input buffer [N, in_h, in_w, C].
        batch: Batch size (N).
        in_h: Input height.
        in_w: Input width.
        channels: Number of channels (C).
        out_h: Output height.
        out_w: Output width.
        kH: Kernel height.
        kW: Kernel width.
        stride_h: Stride along height.
        stride_w: Stride along width.
        dil_h: Dilation along height.
        dil_w: Dilation along width.
        pad_h_before: Padding before height.
        pad_w_before: Padding before width.
        count_boundary_flag: 1 = include padding in divisor, 0 = exclude.
        ctx: Device context pointer (null for CPU).
    """
    var total = batch * out_h * out_w * channels

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        in_h,
        in_w,
        channels,
        out_h,
        out_w,
        kH,
        kW,
        stride_h,
        stride_w,
        dil_h,
        dil_w,
        pad_h_before,
        pad_w_before,
        count_boundary_flag,
    )
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var rem, c = divmod(i, channels)
        var rem2, ow = divmod(rem, out_w)
        var n, oh = divmod(rem2, out_h)

        var sum = Scalar[dtype](0)
        var count = Int(0)

        for fh in range(kH):
            var ih = oh * stride_h - pad_h_before + fh * dil_h
            if ih < 0 or ih >= in_h:
                if count_boundary_flag:
                    count += kW
                continue
            for fw in range(kW):
                var iw = ow * stride_w - pad_w_before + fw * dil_w
                if iw < 0 or iw >= in_w:
                    if count_boundary_flag:
                        count += 1
                    continue
                var in_idx = ((n * in_h + ih) * in_w + iw) * channels + c
                sum += in_ptr[in_idx]
                count += 1

        if count > 0:
            out_ptr[i] = sum / Scalar[dtype](count)
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
struct _AvgPool2dBody(Dispatchable):
    """Dispatch body for the AvgPool2d operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var batch: Int
    var in_h: Int
    var in_w: Int
    var channels: Int
    var out_h: Int
    var out_w: Int
    var kH: Int
    var kW: Int
    var stride_h: Int
    var stride_w: Int
    var dil_h: Int
    var dil_w: Int
    var pad_h_before: Int
    var pad_w_before: Int
    var count_boundary_flag: Int
    var ctx: OpaquePointer[MutExternalOrigin]

    def call[t: DType](self) raises -> None:
        comptime if t.is_numeric():
            avg_pool2d_op(
                _make_ptr[t](self.out_addr),
                _make_ptr[t](self.in_addr),
                self.batch,
                self.in_h,
                self.in_w,
                self.channels,
                self.out_h,
                self.out_w,
                self.kH,
                self.kW,
                self.stride_h,
                self.stride_w,
                self.dil_h,
                self.dil_w,
                self.pad_h_before,
                self.pad_w_before,
                self.count_boundary_flag,
                self.ctx,
            )
        else:
            raise Error("avg_pool2d requires a numeric dtype, got " + String(t))


def avg_pool2d_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """AvgPool2d dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer [N, out_h, out_w, C].
        in_buffer: Input data buffer [N, in_h, in_w, C].
        params: Python tuple of pooling parameters.
        device_context_ptr: Device context pointer.
    """
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    dispatch_dtype(
        _AvgPool2dBody(
            out_addr,
            in_addr,
            Int(py=params[0]),
            Int(py=params[1]),
            Int(py=params[2]),
            Int(py=params[3]),
            Int(py=params[4]),
            Int(py=params[5]),
            Int(py=params[6]),
            Int(py=params[7]),
            Int(py=params[8]),
            Int(py=params[9]),
            Int(py=params[10]),
            Int(py=params[11]),
            Int(py=params[12]),
            Int(py=params[13]),
            Int(py=params[14]),
            ctx,
        ),
        dtype,
    )
