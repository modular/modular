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

"""Mojo kernel wrappers for convolution MO interpreter operations.

Naive elementwise conv2d and conv_transpose2d kernels for the eager
interpreter. Supports NHWC input layout and RSCF filter layout.
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

from op_utils import _get_dtype, _get_ctx, _make_ptr


@export
def PyInit_conv_ops() -> PythonObject:
    """Create a Python module with convolution kernel function bindings."""
    try:
        var b = PythonModuleBuilder("conv_ops")
        b.def_function[conv2d_dispatcher](
            "Conv2d", docstring="2D convolution (NHWC, RSCF)"
        )
        b.def_function[conv_transpose2d_dispatcher](
            "ConvTranspose2d",
            docstring="2D transposed convolution (NHWC, RSCF)",
        )
        return b.finalize()
    except e:
        abort(t"failed to create conv op bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# Conv2d kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def conv2d_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    filt_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    kh: Int,
    kw: Int,
    stride_h: Int,
    stride_w: Int,
    dil_h: Int,
    dil_w: Int,
    pad_h: Int,
    pad_w: Int,
    groups: Int,
    out_h: Int,
    out_w: Int,
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
) raises:
    """Naive 2D convolution over NHWC input with RSCF filter.

    Parameters:
        dtype: Data type (inferred from pointers).

    Args:
        out_ptr: Output buffer [batch, out_h, out_w, out_c].
        in_ptr: Input buffer [batch, in_h, in_w, in_c].
        filt_ptr: Filter buffer [kh, kw, in_c/groups, out_c].
        batch: Batch size (N).
        in_h: Input height (H).
        in_w: Input width (W).
        in_c: Input channels (C).
        out_c: Output channels.
        kh: Kernel height.
        kw: Kernel width.
        stride_h: Stride along height.
        stride_w: Stride along width.
        dil_h: Dilation along height.
        dil_w: Dilation along width.
        pad_h: Padding before height.
        pad_w: Padding before width.
        groups: Number of groups.
        out_h: Output height.
        out_w: Output width.
        ctx: Device context pointer (null for CPU).
    """
    var total = batch * out_h * out_w * out_c
    var ic_per_group = in_c // groups
    var oc_per_group = out_c // groups
    var in_hw_stride = in_w * in_c
    var out_hw_stride = out_w * out_c
    var filt_w_stride = ic_per_group * out_c
    var filt_h_stride = kw * filt_w_stride

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        filt_ptr,
        in_h,
        in_w,
        in_c,
        out_c,
        kh,
        kw,
        stride_h,
        stride_w,
        dil_h,
        dil_w,
        pad_h,
        pad_w,
        out_h,
        ic_per_group,
        oc_per_group,
        in_hw_stride,
        out_hw_stride,
        filt_w_stride,
        filt_h_stride,
    )
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var n, rem1 = divmod(i, out_h * out_hw_stride)
        var oh, rem2 = divmod(rem1, out_hw_stride)
        var ow, oc = divmod(rem2, out_c)

        var g = oc // oc_per_group
        var ic_start = g * ic_per_group

        var accum = Scalar[dtype](0)
        for fh in range(kh):
            var ih = oh * stride_h - pad_h + fh * dil_h
            if ih < 0 or ih >= in_h:
                continue
            var in_h_off = n * in_h * in_hw_stride + ih * in_hw_stride
            var filt_h_off = fh * filt_h_stride
            for fw in range(kw):
                var iw = ow * stride_w - pad_w + fw * dil_w
                if iw < 0 or iw >= in_w:
                    continue
                var in_hw_off = in_h_off + iw * in_c + ic_start
                var filt_hw_off = filt_h_off + fw * filt_w_stride
                for ic in range(ic_per_group):
                    accum += (
                        in_ptr[in_hw_off + ic]
                        * filt_ptr[filt_hw_off + ic * out_c + oc]
                    )
        out_ptr[i] = accum

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
# ConvTranspose2d kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def conv_transpose2d_op[
    dtype: DType, //
](
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    filt_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    kh: Int,
    kw: Int,
    stride_h: Int,
    stride_w: Int,
    dil_h: Int,
    dil_w: Int,
    pad_h: Int,
    pad_w: Int,
    out_h: Int,
    out_w: Int,
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
) raises:
    """Naive 2D transposed convolution over NHWC input with RSCF filter.

    Filter layout is RSCF = [kH, kW, C_out_transpose, C_in_transpose].
    For conv_transpose the filter's 3rd dim is the *output* channels and
    4th dim is the *input* channels (reversed vs forward conv).

    Parameters:
        dtype: Data type (inferred from pointers).

    Args:
        out_ptr: Output buffer [batch, out_h, out_w, out_c].
        in_ptr: Input buffer [batch, in_h, in_w, in_c].
        filt_ptr: Filter [kh, kw, out_c, in_c] (RSCF for transpose).
        batch: Batch size (N).
        in_h: Input height.
        in_w: Input width.
        in_c: Input channels.
        out_c: Output channels.
        kh: Kernel height.
        kw: Kernel width.
        stride_h: Stride along height.
        stride_w: Stride along width.
        dil_h: Dilation along height.
        dil_w: Dilation along width.
        pad_h: Padding before height (on output side).
        pad_w: Padding before width (on output side).
        out_h: Output height.
        out_w: Output width.
        ctx: Device context pointer (null for CPU).
    """
    var total = batch * out_h * out_w * out_c
    var in_hw_stride = in_w * in_c
    var out_hw_stride = out_w * out_c
    var filt_c_stride = in_c
    var filt_w_stride = out_c * in_c
    var filt_h_stride = kw * filt_w_stride

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr,
        in_ptr,
        filt_ptr,
        in_h,
        in_w,
        in_c,
        out_c,
        kh,
        kw,
        stride_h,
        stride_w,
        dil_h,
        dil_w,
        pad_h,
        pad_w,
        out_h,
        in_hw_stride,
        out_hw_stride,
        filt_c_stride,
        filt_w_stride,
        filt_h_stride,
    )
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var n, rem1 = divmod(i, out_h * out_hw_stride)
        var oh, rem2 = divmod(rem1, out_hw_stride)
        var ow, oc = divmod(rem2, out_c)

        var accum = Scalar[dtype](0)
        for fh in range(kh):
            var h_cand = oh + pad_h - fh * dil_h
            if h_cand < 0:
                continue
            var h_rem, h_check = divmod(h_cand, stride_h)
            if h_check != 0:
                continue
            var ih = h_rem
            if ih >= in_h:
                continue
            var in_h_off = n * in_h * in_hw_stride + ih * in_hw_stride
            var filt_h_off = fh * filt_h_stride
            for fw in range(kw):
                var w_cand = ow + pad_w - fw * dil_w
                if w_cand < 0:
                    continue
                var w_rem, w_check = divmod(w_cand, stride_w)
                if w_check != 0:
                    continue
                var iw = w_rem
                if iw >= in_w:
                    continue
                var in_hw_off = in_h_off + iw * in_c
                var filt_hw_off = (
                    filt_h_off + fw * filt_w_stride + oc * filt_c_stride
                )
                for ic in range(in_c):
                    accum += in_ptr[in_hw_off + ic] * filt_ptr[filt_hw_off + ic]
        out_ptr[i] = accum

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
# Dispatchers
# ===----------------------------------------------------------------------=== #


def conv2d_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    filt_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Conv2d dispatcher: unwraps PythonObjects and dispatches by dtype.

    Args:
        out_buffer: Output buffer.
        in_buffer: Input data buffer.
        filt_buffer: Filter data buffer.
        params: Python tuple (batch, in_h, in_w, in_c, out_c, kh, kw,
            stride_h, stride_w, dil_h, dil_w, pad_h, pad_w, groups,
            out_h, out_w).
        device_context_ptr: Device context pointer.
    """
    var dtype = _get_dtype(in_buffer)
    var p_batch = Int(py=params[0])
    var p_in_h = Int(py=params[1])
    var p_in_w = Int(py=params[2])
    var p_in_c = Int(py=params[3])
    var p_out_c = Int(py=params[4])
    var p_kh = Int(py=params[5])
    var p_kw = Int(py=params[6])
    var p_sh = Int(py=params[7])
    var p_sw = Int(py=params[8])
    var p_dh = Int(py=params[9])
    var p_dw = Int(py=params[10])
    var p_ph = Int(py=params[11])
    var p_pw = Int(py=params[12])
    var p_groups = Int(py=params[13])
    var p_oh = Int(py=params[14])
    var p_ow = Int(py=params[15])
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var filt_addr = Int(py=filt_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    _conv2d_dtype_dispatch(
        dtype,
        out_addr,
        in_addr,
        filt_addr,
        p_batch,
        p_in_h,
        p_in_w,
        p_in_c,
        p_out_c,
        p_kh,
        p_kw,
        p_sh,
        p_sw,
        p_dh,
        p_dw,
        p_ph,
        p_pw,
        p_groups,
        p_oh,
        p_ow,
        ctx,
    )


def _conv2d_dtype_dispatch(
    dtype: DType,
    out_addr: Int,
    in_addr: Int,
    filt_addr: Int,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    kh: Int,
    kw: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    ph: Int,
    pw: Int,
    groups: Int,
    oh: Int,
    ow: Int,
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
) raises:
    if dtype == DType.float32:
        conv2d_op(
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](in_addr),
            _make_ptr[DType.float32](filt_addr),
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            ph,
            pw,
            groups,
            oh,
            ow,
            ctx,
        )
    elif dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            conv2d_op(
                _make_ptr[DType.float64](out_addr),
                _make_ptr[DType.float64](in_addr),
                _make_ptr[DType.float64](filt_addr),
                batch,
                in_h,
                in_w,
                in_c,
                out_c,
                kh,
                kw,
                sh,
                sw,
                dh,
                dw,
                ph,
                pw,
                groups,
                oh,
                ow,
                ctx,
            )
    elif dtype == DType.float16:
        conv2d_op(
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](in_addr),
            _make_ptr[DType.float16](filt_addr),
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            ph,
            pw,
            groups,
            oh,
            ow,
            ctx,
        )
    elif dtype == DType.bfloat16:
        conv2d_op(
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](in_addr),
            _make_ptr[DType.bfloat16](filt_addr),
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            ph,
            pw,
            groups,
            oh,
            ow,
            ctx,
        )
    elif dtype == DType.int32:
        conv2d_op(
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](in_addr),
            _make_ptr[DType.int32](filt_addr),
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            ph,
            pw,
            groups,
            oh,
            ow,
            ctx,
        )
    elif dtype == DType.int64:
        conv2d_op(
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](in_addr),
            _make_ptr[DType.int64](filt_addr),
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            ph,
            pw,
            groups,
            oh,
            ow,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for conv2d: " + String(dtype))


def conv_transpose2d_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    filt_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """ConvTranspose2d dispatcher: unwraps PythonObjects and dispatches.

    Args:
        out_buffer: Output buffer.
        in_buffer: Input data buffer.
        filt_buffer: Filter data buffer.
        params: Python tuple (batch, in_h, in_w, in_c, out_c, kh, kw,
            stride_h, stride_w, dil_h, dil_w, pad_h, pad_w, out_h, out_w).
        device_context_ptr: Device context pointer.
    """
    var dtype = _get_dtype(in_buffer)
    var p_batch = Int(py=params[0])
    var p_in_h = Int(py=params[1])
    var p_in_w = Int(py=params[2])
    var p_in_c = Int(py=params[3])
    var p_out_c = Int(py=params[4])
    var p_kh = Int(py=params[5])
    var p_kw = Int(py=params[6])
    var p_sh = Int(py=params[7])
    var p_sw = Int(py=params[8])
    var p_dh = Int(py=params[9])
    var p_dw = Int(py=params[10])
    var p_ph = Int(py=params[11])
    var p_pw = Int(py=params[12])
    var p_oh = Int(py=params[13])
    var p_ow = Int(py=params[14])
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var filt_addr = Int(py=filt_buffer._data_ptr())
    var ctx = _get_ctx(device_context_ptr)

    _conv_transpose2d_dtype_dispatch(
        dtype,
        out_addr,
        in_addr,
        filt_addr,
        p_batch,
        p_in_h,
        p_in_w,
        p_in_c,
        p_out_c,
        p_kh,
        p_kw,
        p_sh,
        p_sw,
        p_dh,
        p_dw,
        p_ph,
        p_pw,
        p_oh,
        p_ow,
        ctx,
    )


def _conv_transpose2d_dtype_dispatch(
    dtype: DType,
    out_addr: Int,
    in_addr: Int,
    filt_addr: Int,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    kh: Int,
    kw: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    ph: Int,
    pw: Int,
    oh: Int,
    ow: Int,
    ctx: Optional[OpaquePointer[MutExternalOrigin]],
) raises:
    if dtype == DType.float32:
        conv_transpose2d_op(
            _make_ptr[DType.float32](out_addr),
            _make_ptr[DType.float32](in_addr),
            _make_ptr[DType.float32](filt_addr),
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            ph,
            pw,
            oh,
            ow,
            ctx,
        )
    elif dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            conv_transpose2d_op(
                _make_ptr[DType.float64](out_addr),
                _make_ptr[DType.float64](in_addr),
                _make_ptr[DType.float64](filt_addr),
                batch,
                in_h,
                in_w,
                in_c,
                out_c,
                kh,
                kw,
                sh,
                sw,
                dh,
                dw,
                ph,
                pw,
                oh,
                ow,
                ctx,
            )
    elif dtype == DType.float16:
        conv_transpose2d_op(
            _make_ptr[DType.float16](out_addr),
            _make_ptr[DType.float16](in_addr),
            _make_ptr[DType.float16](filt_addr),
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            ph,
            pw,
            oh,
            ow,
            ctx,
        )
    elif dtype == DType.bfloat16:
        conv_transpose2d_op(
            _make_ptr[DType.bfloat16](out_addr),
            _make_ptr[DType.bfloat16](in_addr),
            _make_ptr[DType.bfloat16](filt_addr),
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            ph,
            pw,
            oh,
            ow,
            ctx,
        )
    elif dtype == DType.int32:
        conv_transpose2d_op(
            _make_ptr[DType.int32](out_addr),
            _make_ptr[DType.int32](in_addr),
            _make_ptr[DType.int32](filt_addr),
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            ph,
            pw,
            oh,
            ow,
            ctx,
        )
    elif dtype == DType.int64:
        conv_transpose2d_op(
            _make_ptr[DType.int64](out_addr),
            _make_ptr[DType.int64](in_addr),
            _make_ptr[DType.int64](filt_addr),
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            ph,
            pw,
            oh,
            ow,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for conv_transpose2d: " + String(dtype))
