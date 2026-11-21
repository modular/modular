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
"""SM100 tap-GEMM conv3d kernel tests (B200): the time conv and the input conv.

The oracle is a naive GPU kernel that accumulates in fp32, not cuDNN. Operands
are small integers drawn from a hash, so every fp32 partial sum is exact in any
accumulation order and a mismatch can only be an indexing, padding, filter
layout or epilogue error; each output element is compared bit for bit with the
oracle's value rounded to the output dtype.

Both kernels run plain and with a storing epilogue and a compute epilogue, in
both filter layouts, through their launch functions directly (so the kernel is
pinned even if a dispatch guard changes). The epilogues scale by a power of two
chosen from the output coordinate, which is exact at every dtype and fails on a
wrong coordinate. One random-operand case per dtype checks a full-mantissa
input against the oracle: fp32 (tf32 operands) against an oracle that drops
the same operand bits.
"""

from std.math import ceildiv, isclose
from std.memory import bitcast
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true

from layout import (
    Coord,
    ImmTileTensor,
    MutTileTensor,
    TensorLayout,
    TileTensor,
    row_major,
)
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from nn.conv.conv_utils import (
    no_conv_simd_compute_fn,
    no_conv_simd_epilogue_fn,
)
from nn.conv.gpu.nvidia.sm100.conv3d_tap_umma import (
    conv3d_depth_umma_supported,
    conv3d_window_umma_supported,
    launch_conv3d_depth_umma,
    launch_conv3d_window_umma,
)

comptime PLAIN = 0
comptime STORE = 1
comptime COMPUTE = 2


def _hash(i: Int, salt: Int) -> Int:
    return ((i * 2654435761 + salt * 40503 + 7) % 4294967296) >> 9


def _exact_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random integer in [-2, 2]."""
    return Float32(_hash(i, salt) % 5 - 2)


def _random_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random fp32 in [-1, 1] with a full mantissa."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return Float32((h >> 8) % 2000001) / 1000000.0 - 1.0


def _scale(c: Coord) -> Float32:
    """A power of two chosen from the coordinate: exact at every dtype. The
    channel enters in groups of eight, so every lane of a closure call (four
    or eight channels) shares it."""
    var s = (
        Int(c[1]) + 3 * Int(c[2]) + 5 * Int(c[3]) + 7 * (Int(c[4]) // 8)
    ) % 4
    return Float32(1 << s)


@always_inline
def _operand[dt: DType, tf32: Bool](v: Scalar[dt]) -> Float32:
    """An operand as the oracle multiplies it: fp32, with the low 13 mantissa
    bits dropped when the tensor core reads fp32 operands as tf32."""
    var x = v.cast[.float32]()
    comptime if tf32:
        return bitcast[.float32](bitcast[.uint32](x) & UInt32(0xFFFFE000))
    return x


def naive_conv3d_kernel[
    dt: DType,
    tf32: Bool,
    taps_d: Int,
    taps_h: Int,
    taps_w: Int,
    ActLayout: TensorLayout,
    FiltLayout: TensorLayout,
    DstLayout: TensorLayout,
    act_origin: ImmOrigin,
    filt_origin: ImmOrigin,
    dst_origin: MutOrigin,
](
    act: ImmTileTensor[dt, ActLayout, act_origin],
    filt: ImmTileTensor[dt, FiltLayout, filt_origin],
    dst: MutTileTensor[.float32, DstLayout, dst_origin],
    pad_d32: Int32,
    pad_hw32: Int32,
):
    """One output element per thread: NDHWC activation, QRSCF filter, stride
    1, padding `pad_d` along D and `pad_hw` along H and W, fp32 accumulation."""
    var in_d = Int(act.dim[0]())
    var in_h = Int(act.dim[1]())
    var in_w = Int(act.dim[2]())
    var cin = Int(act.dim[3]())
    var fout = Int(filt.dim[4]())
    var pad_d = Int(pad_d32)
    var pad_hw = Int(pad_hw32)
    var out_d = in_d + 2 * pad_d - taps_d + 1
    var out_h = in_h + 2 * pad_hw - taps_h + 1
    var out_w = in_w + 2 * pad_hw - taps_w + 1
    var tid = global_idx.x
    var total = out_d * out_h * out_w * fout
    if tid >= total:
        return
    var vox, f = divmod(tid, fout)
    var dd, rem = divmod(vox, out_h * out_w)
    var h, w = divmod(rem, out_w)
    var acc = Float32(0)
    for q in range(taps_d):
        var di = dd + q - pad_d
        if di < 0 or di >= in_d:
            continue
        for r in range(taps_h):
            var hi = h + r - pad_hw
            if hi < 0 or hi >= in_h:
                continue
            for s in range(taps_w):
                var wi = w + s - pad_hw
                if wi < 0 or wi >= in_w:
                    continue
                for c in range(cin):
                    acc += _operand[dt, tf32](act[di, hi, wi, c]) * _operand[
                        dt, tf32
                    ](filt[q, r, s, c, f])
    dst[tid] = acc


def run_case[
    dt: DType,
    window: Bool,
    cin: Int,
    fout: Int,
    height: Int,
    width: Int,
    fcrs: Bool,
    mode: Int,
    exact: Bool = True,
](ctx: DeviceContext, depth: Int) raises:
    """Runs one shape in one launch form and compares it with the oracle.

    Parameters:
        dt: Element type of the activation, filter and output.
        window: The 3x3x3 input conv (padding 1, 16 channels) rather than the
            3x1x1 time conv (no padding).
        cin: Input channels.
        fout: Output channels.
        height: Input height.
        width: Input width.
        fcrs: The filter is FCQRS rather than QRSCF.
        mode: PLAIN, STORE (storing epilogue) or COMPUTE (compute epilogue).
        exact: Small-integer operands compared bit for bit; otherwise random
            operands compared with the dtype's tolerance.

    Args:
        ctx: Device context.
        depth: Input depth.
    """
    comptime taps_h = 3 if window else 1
    comptime taps_w = 3 if window else 1
    comptime pad_d = 1 if window else 0
    comptime pad_hw = 1 if window else 0
    comptime taps = 3 * taps_h * taps_w
    var out_d = depth + 2 * pad_d - 2
    var n_act = depth * height * width * cin
    var n_filt = taps * cin * fout
    var n_out = out_d * height * width * fout

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var filt_dev_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var ref_host = ctx.enqueue_create_host_buffer[.float32](n_out)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    for i in range(n_act):
        comptime if exact:
            act_host[i] = _exact_value(i, 11).cast[dt]()
        else:
            act_host[i] = _random_value(i, 11).cast[dt]()
    for i in range(n_filt):
        comptime if exact:
            filt_host[i] = _exact_value(i, 3).cast[dt]()
        else:
            filt_host[i] = _random_value(i, 3).cast[dt]()
    # QRSCF -> FCQRS for the other layout.
    comptime if fcrs:
        for k in range(taps):
            for c in range(cin):
                for f in range(fout):
                    filt_dev_host[(f * cin + c) * taps + k] = filt_host[
                        (k * cin + c) * fout + f
                    ]
    else:
        for i in range(n_filt):
            filt_dev_host[i] = filt_host[i]

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_ref_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var ref_dev = ctx.enqueue_create_buffer[.float32](n_out)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_ref_dev, filt_host)
    ctx.enqueue_copy(filt_dev, filt_dev_host)

    var act_ref = TileTensor(
        act_dev.unsafe_ptr(), row_major(depth, height, width, cin)
    ).as_imm()
    var filt_ref = TileTensor(
        filt_ref_dev.unsafe_ptr(), row_major[3, taps_h, taps_w, cin, fout]()
    ).as_imm()
    var ref_tt = TileTensor(ref_dev.unsafe_ptr(), row_major(n_out))
    comptime naive = naive_conv3d_kernel[
        dt,
        dt == .float32 and not exact,
        3,
        taps_h,
        taps_w,
        act_ref.LayoutType,
        filt_ref.LayoutType,
        ref_tt.LayoutType,
        act_ref.origin,
        filt_ref.origin,
        ref_tt.origin,
    ]
    ctx.enqueue_function[naive](
        act_ref,
        filt_ref,
        ref_tt,
        Int32(pad_d),
        Int32(pad_hw),
        grid_dim=ceildiv(n_out, 256),
        block_dim=256,
    )
    ctx.enqueue_copy(ref_host, ref_dev)

    var act_tt = TileTensor(act_dev, row_major(1, depth, height, width, cin))
    var out_tt = TileTensor(out_dev, row_major(1, out_d, height, width, fout))
    # The storing closure writes through a view of its own: it may not borrow
    # the tensor the launch receives alongside it.
    var out_view = TileTensor(
        out_dev, row_major(1, out_d, height, width, fout)
    ).as_unsafe_any_origin()

    @always_inline
    @__allow_legacy_any_origin_fields
    def scale_store[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_view}:
        out_view.store[width=Int(width)](
            idx, (val * _scale(idx).cast[_dtype]()).cast[dt]()
        )

    @always_inline
    def scale_compute[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
        return val * _scale(idx).cast[_dtype]()

    out_dev.enqueue_fill(Scalar[dt](-1000))
    var launched: Bool
    comptime epi = mode == STORE
    comptime cmp = mode == COMPUTE
    comptime if fcrs:
        var filt_tt = TileTensor(
            filt_dev, row_major[fout, cin, 3, taps_h, taps_w]()
        )
        comptime if window:
            comptime if mode == STORE:
                launched = launch_conv3d_window_umma[
                    dt, fout, height, width, True, True, False
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    scale_store,
                    no_conv_simd_compute_fn,
                )
            elif mode == COMPUTE:
                launched = launch_conv3d_window_umma[
                    dt, fout, height, width, True, False, True
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    no_conv_simd_epilogue_fn,
                    scale_compute,
                )
            else:
                launched = launch_conv3d_window_umma[
                    dt, fout, height, width, True, False, False
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    no_conv_simd_epilogue_fn,
                    no_conv_simd_compute_fn,
                )
        else:
            comptime if mode == STORE:
                launched = launch_conv3d_depth_umma[
                    dt, cin, fout, height, width, 3, True, True, False
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    scale_store,
                    no_conv_simd_compute_fn,
                )
            elif mode == COMPUTE:
                launched = launch_conv3d_depth_umma[
                    dt, cin, fout, height, width, 3, True, False, True
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    no_conv_simd_epilogue_fn,
                    scale_compute,
                )
            else:
                launched = launch_conv3d_depth_umma[
                    dt, cin, fout, height, width, 3, True, False, False
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    no_conv_simd_epilogue_fn,
                    no_conv_simd_compute_fn,
                )
    else:
        var filt_tt = TileTensor(
            filt_dev, row_major[3, taps_h, taps_w, cin, fout]()
        )
        comptime if window:
            comptime if mode == STORE:
                launched = launch_conv3d_window_umma[
                    dt, fout, height, width, False, True, False
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    scale_store,
                    no_conv_simd_compute_fn,
                )
            elif mode == COMPUTE:
                launched = launch_conv3d_window_umma[
                    dt, fout, height, width, False, False, True
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    no_conv_simd_epilogue_fn,
                    scale_compute,
                )
            else:
                launched = launch_conv3d_window_umma[
                    dt, fout, height, width, False, False, False
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    no_conv_simd_epilogue_fn,
                    no_conv_simd_compute_fn,
                )
        else:
            comptime if mode == STORE:
                launched = launch_conv3d_depth_umma[
                    dt, cin, fout, height, width, 3, False, True, False
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    scale_store,
                    no_conv_simd_compute_fn,
                )
            elif mode == COMPUTE:
                launched = launch_conv3d_depth_umma[
                    dt, cin, fout, height, width, 3, False, False, True
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    no_conv_simd_epilogue_fn,
                    scale_compute,
                )
            else:
                launched = launch_conv3d_depth_umma[
                    dt, cin, fout, height, width, 3, False, False, False
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    ctx,
                    no_conv_simd_epilogue_fn,
                    no_conv_simd_compute_fn,
                )
    assert_true(launched)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var bad = 0
    for i in range(n_out):
        var want = ref_host[i]
        comptime if mode != PLAIN:
            var vox, f = divmod(i, fout)
            var dd, rem = divmod(vox, height * width)
            var h, w = divmod(rem, width)
            want = want.cast[dt]().cast[.float32]() * _scale(
                Coord(0, dd, h, w, f)
            )
        want = want.cast[dt]().cast[.float32]()
        var got = out_host[i].cast[.float32]()
        comptime if exact:
            if got != want:
                bad += 1
        else:
            # The oracle sums the same (tf32-truncated, for fp32) operands in
            # fp32, so the only difference is the output rounding (bf16 and
            # fp16 round the same fp32 sum) and the summation order.
            comptime rtol = Float64(8e-3) if dt == .bfloat16 else (
                Float64(1e-3) if dt == .float16 else Float64(1e-4)
            )
            if not isclose[symmetrical=False](got, want, atol=1e-3, rtol=rtol):
                bad += 1
    assert_equal(bad, 0)
    var kernel = "window" if window else "depth"
    var layout = "fcrs" if fcrs else "qrscf"
    var epilogue = "plain" if mode == PLAIN else (
        "store" if mode == STORE else "compute"
    )
    var operands = "exact" if exact else "random"
    print(
        t"PASS {dt} {kernel} {cin} -> {fout} {height} x {width} depth {depth}"
        t" {layout} {epilogue} {operands}"
    )
    _ = act_dev^
    _ = filt_dev^
    _ = filt_ref_dev^
    _ = ref_dev^
    _ = out_dev^


def run_shape[
    dt: DType, window: Bool, cin: Int, fout: Int, height: Int, width: Int
](ctx: DeviceContext, depth: Int) raises:
    comptime if window:
        comptime assert conv3d_window_umma_supported[
            dt, cin, fout, height, width
        ]()
    else:
        comptime assert conv3d_depth_umma_supported[
            dt, cin, fout, height, width, 3
        ]()
    run_case[dt, window, cin, fout, height, width, False, PLAIN](ctx, depth)
    run_case[dt, window, cin, fout, height, width, True, PLAIN](ctx, depth)
    run_case[dt, window, cin, fout, height, width, False, STORE](ctx, depth)
    run_case[dt, window, cin, fout, height, width, True, COMPUTE](ctx, depth)
    run_case[dt, window, cin, fout, height, width, False, PLAIN, False](
        ctx, depth
    )


def run_dtype[dt: DType](ctx: DeviceContext) raises:
    # Depths 3 and 4: an output depth (1, 2) that is not a multiple of the
    # pass, so the depth mask and the clip of the last output depth run.
    run_shape[dt, False, 192, 384, 60, 104](ctx, 3)
    run_shape[dt, False, 192, 384, 60, 104](ctx, 4)
    # Depth 5: one output-depth block of the time conv; 23 is the production
    # extent (seven blocks).
    run_shape[dt, False, 192, 384, 60, 104](ctx, 5)
    run_shape[dt, False, 192, 384, 60, 104](ctx, 23)
    # Depths 3 and 21: three output frames, and the production extent.
    run_shape[dt, True, 16, 384, 30, 52](ctx, 3)
    run_shape[dt, True, 16, 384, 30, 52](ctx, 21)


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator")
    else:
        with DeviceContext() as ctx:
            run_dtype[.bfloat16](ctx)
            run_dtype[.float16](ctx)
            run_dtype[.float32](ctx)
