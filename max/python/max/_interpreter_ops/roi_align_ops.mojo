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

"""Mojo kernel wrapper for ROI Align MO interpreter operation.

Implements ROI Align pooling over NHWC input with configurable spatial scale,
sampling ratio, alignment mode (aligned/unaligned), and pooling mode (AVG/MAX).

CPU-only via the elementwise + DeviceContextPtr pattern.
"""

from std.math import ceil
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
def PyInit_roi_align_ops() -> PythonObject:
    """Create a Python module with ROI Align kernel function bindings."""
    try:
        var b = PythonModuleBuilder("roi_align_ops")
        b.def_function[roi_align_dispatcher](
            "RoiAlign", docstring="ROI Align pooling (NHWC)"
        )
        return b.finalize()
    except e:
        abort(t"failed to create roi_align op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# ROI Align kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def roi_align_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rois_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n_regions: Int,
    height: Int,
    width: Int,
    channels: Int,
    out_h: Int,
    out_w: Int,
    spatial_scale_val: Float32,
    sampling_ratio_val: Float32,
    aligned_flag: Int,
    mode_flag: Int,
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
) raises:
    """Compute ROI Align pooling over NHWC input.

    Each ROI is [batch_idx, x0, y0, x1, y1] (5 columns). Output layout is
    [M, out_h, out_w, C].

    Parameters:
        dtype: Data type (inferred from pointers).

    Args:
        out_ptr: Output buffer [M, out_h, out_w, C].
        in_ptr: Input buffer [N, H, W, C].
        rois_ptr: ROIs buffer [M, 5].
        n_regions: Number of ROIs (M).
        height: Input height (H).
        width: Input width (W).
        channels: Number of channels (C).
        out_h: Pooled output height.
        out_w: Pooled output width.
        spatial_scale_val: Scale factor mapping ROI coords to input space.
        sampling_ratio_val: Sampling points per bin (0 = adaptive).
        aligned_flag: 1 = apply half-pixel offset, 0 = no offset.
        mode_flag: 0 = average pooling, 1 = max pooling.
        ctx: Device context pointer (null for CPU).
    """
    var total = n_regions * out_h * out_w * channels
    var offset = Float32(0.5) if aligned_flag else Float32(0.0)

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        rois_ptr,
        n_regions,
        height,
        width,
        channels,
        out_h,
        out_w,
        spatial_scale_val,
        sampling_ratio_val,
        aligned_flag,
        mode_flag,
        offset,
    )
    def func[
        width_param: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank],):
        var i = idx[0]
        var rem, c = divmod(i, channels)
        var rem2, pw = divmod(rem, out_w)
        var ri, ph = divmod(rem2, out_h)

        # ROI layout: [batch_idx, x0, y0, x1, y1] (5 columns per ROI)
        var roi_base = ri * 5
        var roi_batch_idx = Int(rois_ptr[roi_base])
        var roi_start_w = (
            Float32(rois_ptr[roi_base + 1]) * spatial_scale_val - offset
        )
        var roi_start_h = (
            Float32(rois_ptr[roi_base + 2]) * spatial_scale_val - offset
        )
        var roi_end_w = (
            Float32(rois_ptr[roi_base + 3]) * spatial_scale_val - offset
        )
        var roi_end_h = (
            Float32(rois_ptr[roi_base + 4]) * spatial_scale_val - offset
        )

        var roi_height: Float32
        var roi_width: Float32
        if aligned_flag:
            roi_height = roi_end_h - roi_start_h
            roi_width = roi_end_w - roi_start_w
        else:
            roi_height = max(roi_end_h - roi_start_h, Float32(1.0))
            roi_width = max(roi_end_w - roi_start_w, Float32(1.0))

        var bin_size_h = roi_height / Float32(out_h)
        var bin_size_w = roi_width / Float32(out_w)

        var roi_bin_grid_h = Int(
            sampling_ratio_val if sampling_ratio_val > 0 else ceil(bin_size_h)
        )
        var roi_bin_grid_w = Int(
            sampling_ratio_val if sampling_ratio_val > 0 else ceil(bin_size_w)
        )

        var pool_elem_num = max(roi_bin_grid_h * roi_bin_grid_w, 1)

        var pool_val: Scalar[dtype]
        if mode_flag == 0:
            pool_val = Scalar[dtype](0)
        else:
            pool_val = min_or_neg_inf[dtype]()

        for iy in range(roi_bin_grid_h):
            for ix in range(roi_bin_grid_w):
                var y = (
                    roi_start_h
                    + Float32(ph) * bin_size_h
                    + (Float32(iy) + Float32(0.5))
                    * bin_size_h
                    / Float32(roi_bin_grid_h)
                )
                var x = (
                    roi_start_w
                    + Float32(pw) * bin_size_w
                    + (Float32(ix) + Float32(0.5))
                    * bin_size_w
                    / Float32(roi_bin_grid_w)
                )

                if not (Float32(-1.0) <= y <= Float32(height)) or not (
                    Float32(-1.0) <= x <= Float32(width)
                ):
                    if mode_flag == 0:
                        pass
                    continue

                y = max(y, Float32(0.0))
                x = max(x, Float32(0.0))

                var y_low = min(Int(y), height - 1)
                var x_low = min(Int(x), width - 1)
                var y_high = min(y_low + 1, height - 1)
                var x_high = min(x_low + 1, width - 1)

                var ly = y - Float32(y_low)
                var lx = x - Float32(x_low)
                var hy = Float32(1.0) - ly
                var hx = Float32(1.0) - lx

                var w1 = (hy * hx).cast[dtype]()
                var w2 = (hy * lx).cast[dtype]()
                var w3 = (ly * hx).cast[dtype]()
                var w4 = (ly * lx).cast[dtype]()

                var v1 = (
                    w1
                    * in_ptr[
                        ((roi_batch_idx * height + y_low) * width + x_low)
                        * channels
                        + c
                    ]
                )
                var v2 = (
                    w2
                    * in_ptr[
                        ((roi_batch_idx * height + y_low) * width + x_high)
                        * channels
                        + c
                    ]
                )
                var v3 = (
                    w3
                    * in_ptr[
                        ((roi_batch_idx * height + y_high) * width + x_low)
                        * channels
                        + c
                    ]
                )
                var v4 = (
                    w4
                    * in_ptr[
                        ((roi_batch_idx * height + y_high) * width + x_high)
                        * channels
                        + c
                    ]
                )

                if mode_flag == 0:
                    pool_val += v1 + v2 + v3 + v4
                else:
                    pool_val = max(pool_val, v1)
                    pool_val = max(pool_val, v2)
                    pool_val = max(pool_val, v3)
                    pool_val = max(pool_val, v4)

        if mode_flag == 0:
            out_ptr[i] = pool_val / Scalar[dtype](pool_elem_num)
        else:
            out_ptr[i] = pool_val

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
# Dispatcher
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _RoiAlignBody(Dispatchable):
    """Dispatch body for the RoiAlign operation over data dtypes."""

    var out_addr: Int
    var in_addr: Int
    var rois_addr: Int
    var n_regions: Int
    var height: Int
    var width: Int
    var channels: Int
    var out_h: Int
    var out_w: Int
    var spatial_scale_val: Float32
    var sampling_ratio_val: Float32
    var aligned_flag: Int
    var mode_flag: Int
    var ctx: Optional[OpaquePointer[MutExternalOrigin]]

    def call[t: DType](self) raises -> None:
        comptime if t.is_floating_point():
            roi_align_op(
                _make_ptr[t](self.out_addr),
                _make_ptr[t](self.in_addr),
                _make_ptr[t](self.rois_addr),
                self.n_regions,
                self.height,
                self.width,
                self.channels,
                self.out_h,
                self.out_w,
                self.spatial_scale_val,
                self.sampling_ratio_val,
                self.aligned_flag,
                self.mode_flag,
                self.ctx,
            )
        else:
            raise Error(
                "roi_align requires a floating-point dtype, got " + String(t)
            )


def roi_align_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    rois_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """RoiAlign dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer [M, out_h, out_w, C].
        in_buffer: Input data buffer [N, H, W, C].
        rois_buffer: ROIs buffer [M, 5].
        params: Python tuple (n_regions, H, W, C, out_h, out_w,
            spatial_scale, sampling_ratio, aligned_flag, mode_flag).
        device_context_ptr: Device context pointer.
    """
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var rois_addr = Int(py=rois_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    dispatch_dtype(
        _RoiAlignBody(
            out_addr,
            in_addr,
            rois_addr,
            Int(py=params[0]),
            Int(py=params[1]),
            Int(py=params[2]),
            Int(py=params[3]),
            Int(py=params[4]),
            Int(py=params[5]),
            Float32(py=params[6]),
            Float32(py=params[7]),
            Int(py=params[8]),
            Int(py=params[9]),
            ctx,
        ),
        dtype,
    )
