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
"""MI355X: the FCRS-in-place and the staged conv2d kernels against a host loop.

FCRS-in-place: the dispatcher's 30x30 config (6 compute waves) and a 3-wave
variant, a ragged last strip, a batch of two, stride 2, a 1x1 filter, fp16, and a
full-mantissa random case.

Staged: the WAN row (240x416, 96 -> 96, a sample of its output), groups of
two and three strips, an odd split of strips over the compute waves, one and
two filter-plane load waves, and an RSCF filter.

Both arms: the three epilogue forms (store, `compute_fn` + `epilogue_fn`,
`compute_fn` + residual).
"""

from std.random import rand, random_float64, seed
from std.sys import size_of
from std.testing import assert_equal

from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext, HostBuffer
from nn.conv.gpu.amd.conv2d_fcrs_fprop import (
    Conv2dFcrsConfig,
    launch_conv2d_fcrs_fprop,
)
from nn.conv.gpu.amd.conv2d_staged_fprop import (
    Conv2dStagedConfig,
    conv2d_staged_fprop,
)

comptime _BIAS = Float32(0.5)
comptime _BETA = Float32(0.75)


def _host_conv[
    dt: DType
](
    x: TileTensor[dt, ...],
    w: TileTensor[dt, ...],
    source: TileTensor[dt, ...],
    y: TileTensor[mut=True, dt, ...],
    *,
    stride: Int,
    pad: Int,
    bias: Float32,
    with_residual: Bool,
    sample: Int,
):
    """`y = round(round(conv + bias) + beta * source)`, the order the kernels
    apply their epilogue in, for every `sample`-th output element (and the
    first and last 64); `x`, `source` and `y` are NHWC and the filter `w` is
    FCRS `[F, C, R, S]`."""
    var H = Int(x.dim[1]())
    var W = Int(x.dim[2]())
    var C = Int(x.dim[3]())
    var F = Int(w.dim[0]())
    var R = Int(w.dim[2]())
    var S = Int(w.dim[3]())
    var H_out = Int(y.dim[1]())
    var W_out = Int(y.dim[2]())
    var total = y.num_elements()
    for at in range(total):
        if at % sample != 0 and at >= 64 and at < total - 64:
            continue
        var pixel, f = divmod(at, F)
        var image_row, ox = divmod(pixel, W_out)
        var n, oy = divmod(image_row, H_out)
        var acc = Float32(0)
        for r in range(R):
            for s in range(S):
                var iy = oy * stride + r - pad
                var ix = ox * stride + s - pad
                if 0 <= iy < H and 0 <= ix < W:
                    for c in range(C):
                        acc += Float32(
                            x.load[width=1]((n, iy, ix, c))
                        ) * Float32(w.load[width=1]((f, c, r, s)))
        var v = (acc.cast[dt]().cast[.float32]() + bias).cast[dt]()
        if with_residual:
            v = (
                v.cast[.float32]()
                + _BETA * Float32(source.load[width=1]((n, oy, ox, f)))
            ).cast[dt]()
        y.store[width=1]((n, oy, ox, f), v)


def _permute_filter[
    dt: DType, //, fcrs: Bool
](w_fcrs: TileTensor[dt, ...], mut device_filter: HostBuffer[dt]):
    """Fills `device_filter` with the FCRS filter `w_fcrs`, or the same values
    in RSCF order."""
    var F = Int(w_fcrs.dim[0]())
    var C = Int(w_fcrs.dim[1]())
    var R = Int(w_fcrs.dim[2]())
    var S = Int(w_fcrs.dim[3]())
    comptime if fcrs:
        var dst = TileTensor(device_filter, row_major(F, C, R, S))
        for f in range(F):
            for c in range(C):
                for r in range(R):
                    for s in range(S):
                        dst[f, c, r, s] = w_fcrs.load[width=1]((f, c, r, s))
    else:
        var dst = TileTensor(device_filter, row_major(R, S, C, F))
        for f in range(F):
            for c in range(C):
                for r in range(R):
                    for s in range(S):
                        dst[r, s, c, f] = w_fcrs.load[width=1]((f, c, r, s))


def _check[
    dt: DType
](
    got: HostBuffer[dt],
    want: HostBuffer[dt],
    label: StaticString,
    *,
    sample: Int,
) raises:
    """Compares the elements `_host_conv` computed, which are the ones at
    multiples of `sample` and the first and last 64."""
    # the kernel and the host loop sum in different orders and each rounds the
    # fp32 sum once: two bf16 ulps of the result
    comptime rel_tol = Float32(0.0157) if dt == .bfloat16 else Float32(0.002)
    var n = len(got)
    var bad = 0
    var checked = 0
    var worst = Float32(0)
    for i in range(n):
        if i % sample != 0 and i >= 64 and i < n - 64:
            continue
        checked += 1
        var a = Float32(got[i])
        var b = Float32(want[i])
        var d = abs(a - b)
        worst = max(worst, d)
        if not (d <= rel_tol * abs(b) + Float32(1e-3)):
            if bad < 3:
                print(t"   {label}: mismatch at {i}: got {a} want {b}")
            bad += 1
    print(t"   {label}: max_abs={worst} mismatches={bad}/{checked}")
    assert_equal(bad, 0)


def run_case[
    dt: DType,
    *,
    staged: Bool,
    N: Int,
    H: Int,
    W: Int,
    C: Int,
    F: Int,
    NL: Int,
    NOW: Int,
    NLB: Int = 1,
    G: Int = 2,
    max_grid: Int = 1 << 20,
    R: Int = 3,
    S: Int = 3,
    stride: Int = 1,
    pad: Int = 1,
    mode: Int = 0,
    filter_is_fcrs: Bool = True,
    full_mantissa: Bool = False,
    sample: Int = 1,
](ctx: DeviceContext, label: StaticString) raises:
    """`mode` 0: plain store. 1: `compute_fn` + `epilogue_fn`. 2: `compute_fn`
    + residual."""
    comptime H_out = (H + 2 * pad - R) // stride + 1
    comptime W_out = (W + 2 * pad - S) // stride + 1
    comptime n_in = N * H * W * C
    comptime n_w = F * C * R * S
    comptime n_out = N * H_out * W_out * F
    print(t"== {label}: N={N} {H}x{W} C={C} F={F} mode={mode}")

    var x_host = ctx.enqueue_create_host_buffer[dt](n_in)
    var w_host = ctx.enqueue_create_host_buffer[dt](n_w)
    var w_dev_host = ctx.enqueue_create_host_buffer[dt](n_w)
    var src_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var want = ctx.enqueue_create_host_buffer[dt](n_out)
    var got = ctx.enqueue_create_host_buffer[dt](n_out)
    seed(7)
    comptime if full_mantissa:
        for i in range(n_in):
            x_host[i] = random_float64(-2.0, 2.0).cast[dt]()
        for i in range(n_w):
            w_host[i] = random_float64(-1.0, 1.0).cast[dt]()
    else:
        rand(x_host.unsafe_ptr(), n_in)
        rand(w_host.unsafe_ptr(), n_w)
    rand(src_host.unsafe_ptr(), n_out)
    var w_fcrs = TileTensor(w_host, row_major[F, C, R, S]())
    _permute_filter[filter_is_fcrs](w_fcrs, w_dev_host)

    var x_buf = ctx.enqueue_create_buffer[dt](n_in)
    var w_buf = ctx.enqueue_create_buffer[dt](n_w)
    var src_buf = ctx.enqueue_create_buffer[dt](n_out)
    var y_buf = ctx.enqueue_create_buffer[dt](n_out)
    var out_buf = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(x_buf, x_host)
    ctx.enqueue_copy(w_buf, w_dev_host)
    ctx.enqueue_copy(src_buf, src_host)
    # a poisoned output: a skipped or misplaced store shows as a mismatch
    y_buf.enqueue_fill(Scalar[dt].MAX_FINITE)
    out_buf.enqueue_fill(Scalar[dt].MAX_FINITE)

    var x = TileTensor(x_buf, row_major[N, H, W, C]())
    var y = TileTensor(y_buf, row_major[N, H_out, W_out, F]())
    # the epilogue closure's coordinates are `[pixel, channel]`
    var out = TileTensor(out_buf, row_major[N * H_out * W_out, F]())
    var src = TileTensor(src_buf, row_major[N, H_out, W_out, F]())

    # set by a closure call for a pixel or channel the output does not have
    var flags_buf = ctx.enqueue_create_buffer[.int32](1)
    flags_buf.enqueue_fill(Int32(0))
    # Untracked, so both closures can hold it at once.
    var flags = TileTensor(
        flags_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        row_major[1](),
    )
    var flags_host = ctx.enqueue_create_host_buffer[.int32](1)
    var bias = _BIAS
    var beta = _BETA

    @inline(.always)
    @__allow_legacy_any_origin_fields
    def add_bias[
        _dtype: DType, _width: SIMDLength, *, alignment: Int
    ](coords: Coord, val: SIMD[_dtype, _width]) {var bias, var flags} -> SIMD[
        _dtype, _width
    ]:
        comptime assert coords.rank == 2
        if Int(coords[0]) >= N * H_out * W_out or Int(coords[1]) >= F:
            flags[0] = 1
        return (val.cast[.float32]() + bias).cast[_dtype]()

    @inline(.always)
    @__allow_legacy_any_origin_fields
    def store_to_out[
        _dtype: DType, _width: SIMDLength, *, alignment: Int
    ](coords: Coord, val: SIMD[_dtype, _width]) {var out, var flags}:
        comptime assert coords.rank == 2
        if Int(coords[0]) >= N * H_out * W_out or Int(coords[1]) >= F:
            flags[0] = 1
            return
        out.store[width=_width, alignment=alignment](coords, val.cast[dt]())

    comptime if staged:
        comptime config = Conv2dStagedConfig(
            C,
            F,
            R,
            S,
            H,
            W,
            pad,
            G,
            NL,
            NLB,
            NOW,
            2,
            2,
            max_grid,
            size_of[dt](),
        )
        comptime if filter_is_fcrs:
            var w = TileTensor(w_buf, row_major[F, C, R, S]())
            comptime if mode == 0:
                conv2d_staged_fprop[config, True](x, w, y, ctx, source=src)
            elif mode == 1:
                conv2d_staged_fprop[
                    config, True, has_epilogue_fn=True, has_compute_fn=True
                ](
                    x,
                    w,
                    y,
                    ctx,
                    source=src,
                    epilogue_fn=store_to_out,
                    compute_fn=add_bias,
                )
            else:
                conv2d_staged_fprop[
                    config, True, has_residual=True, has_compute_fn=True
                ](x, w, y, ctx, source=src, beta=beta, compute_fn=add_bias)
        else:
            var w = TileTensor(w_buf, row_major[R, S, C, F]())
            conv2d_staged_fprop[config, False](x, w, y, ctx, source=src)
    else:
        comptime config = Conv2dFcrsConfig(
            C, F, R, S, H, W, pad, stride, NL, NOW, size_of[dt]()
        )
        var w = TileTensor(w_buf, row_major[F, C, R, S]())
        comptime if mode == 0:
            launch_conv2d_fcrs_fprop[config](x, w, y, ctx, source=src)
        elif mode == 1:
            launch_conv2d_fcrs_fprop[
                config, has_epilogue_fn=True, has_compute_fn=True
            ](
                x,
                w,
                y,
                ctx,
                source=src,
                epilogue_fn=store_to_out,
                compute_fn=add_bias,
            )
        else:
            launch_conv2d_fcrs_fprop[
                config, has_residual=True, has_compute_fn=True
            ](x, w, y, ctx, source=src, beta=beta, compute_fn=add_bias)

    ctx.synchronize()
    _host_conv[dt](
        TileTensor(x_host, row_major[N, H, W, C]()),
        w_fcrs,
        TileTensor(src_host, row_major[N, H_out, W_out, F]()),
        TileTensor(want, row_major[N, H_out, W_out, F]()),
        stride=stride,
        pad=pad,
        bias=Float32(0) if mode == 0 else bias,
        with_residual=mode == 2,
        sample=sample,
    )
    comptime if mode == 1:
        ctx.enqueue_copy(got, out_buf)
    else:
        ctx.enqueue_copy(got, y_buf)
    ctx.enqueue_copy(flags_host, flags_buf)
    ctx.synchronize()
    assert_equal(flags_host[0], 0, "epilogue closure called out of range")
    _check(got, want, label, sample=sample)
    _ = x_buf^
    _ = w_buf^
    _ = src_buf^
    _ = y_buf^
    _ = out_buf^
    _ = flags_buf^


def main() raises:
    with DeviceContext() as ctx:
        comptime bf = DType.bfloat16
        comptime fp16 = DType.float16

        # FCRS in place: the dispatcher's 30x30 config (6 compute waves) and a 3-wave
        # variant
        run_case[bf, staged=False, N=1, H=30, W=30, C=96, F=96, NL=8, NOW=6](
            ctx, "fcrs 30x30 n6"
        )
        run_case[bf, staged=False, N=1, H=30, W=30, C=96, F=96, NL=8, NOW=3](
            ctx, "fcrs 30x30 n3"
        )
        # a ragged last strip (14*14 = 196 rows = 12.25 strips), a batch of two
        run_case[bf, staged=False, N=2, H=14, W=14, C=32, F=32, NL=4, NOW=2](
            ctx, "fcrs ragged"
        )
        run_case[fp16, staged=False, N=1, H=30, W=30, C=96, F=96, NL=8, NOW=3](
            ctx, "fcrs fp16"
        )
        # non-square image, stride 2, a 1x1 filter
        run_case[
            bf, staged=False, N=1, H=18, W=26, C=64, F=64, NL=4, NOW=4, stride=2
        ](ctx, "fcrs stride 2")
        run_case[
            bf,
            staged=False,
            N=1,
            H=20,
            W=12,
            C=64,
            F=32,
            NL=2,
            NOW=2,
            R=1,
            S=1,
            pad=0,
        ](ctx, "fcrs 1x1")
        run_case[
            bf,
            staged=False,
            N=1,
            H=30,
            W=30,
            C=96,
            F=96,
            NL=8,
            NOW=6,
            full_mantissa=True,
        ](ctx, "fcrs full mantissa")
        run_case[
            bf, staged=False, N=1, H=30, W=30, C=96, F=96, NL=8, NOW=6, mode=1
        ](ctx, "fcrs compute+epilogue")
        run_case[
            bf, staged=False, N=2, H=14, W=14, C=32, F=32, NL=4, NOW=2, mode=1
        ](ctx, "fcrs ragged compute+epilogue")
        run_case[
            bf, staged=False, N=1, H=30, W=30, C=96, F=96, NL=8, NOW=3, mode=2
        ](ctx, "fcrs compute+residual")
        run_case[
            bf, staged=False, N=2, H=14, W=14, C=32, F=32, NL=4, NOW=2, mode=2
        ](ctx, "fcrs ragged compute+residual")

        # staged: the WAN row, a sample of its 9.6M outputs
        run_case[
            bf,
            staged=True,
            N=1,
            H=240,
            W=416,
            C=96,
            F=96,
            G=13,
            NL=2,
            NLB=1,
            NOW=4,
            max_grid=240,
            sample=997,
        ](ctx, "staged WAN")
        # whole outputs of small images: two strips a group, a batch of two,
        # fewer workgroups than groups
        run_case[
            bf,
            staged=True,
            N=2,
            H=8,
            W=32,
            C=32,
            F=32,
            G=2,
            NL=2,
            NLB=1,
            NOW=2,
            max_grid=5,
        ](ctx, "staged G=2")
        # three strips over two compute waves: one wave repeats a strip
        run_case[
            bf,
            staged=True,
            N=1,
            H=6,
            W=48,
            C=64,
            F=64,
            G=3,
            NL=1,
            NLB=2,
            NOW=2,
        ](ctx, "staged G=3 odd split")
        run_case[
            fp16,
            staged=True,
            N=1,
            H=6,
            W=48,
            C=64,
            F=64,
            G=3,
            NL=1,
            NLB=2,
            NOW=2,
        ](ctx, "staged fp16")
        run_case[
            bf,
            staged=True,
            N=1,
            H=6,
            W=32,
            C=32,
            F=48,
            G=2,
            NL=2,
            NLB=1,
            NOW=2,
            filter_is_fcrs=False,
        ](ctx, "staged RSCF filter")
        run_case[
            bf,
            staged=True,
            N=1,
            H=6,
            W=32,
            C=32,
            F=32,
            G=2,
            NL=2,
            NLB=1,
            NOW=2,
            full_mantissa=True,
        ](ctx, "staged full mantissa")
        run_case[
            bf,
            staged=True,
            N=2,
            H=8,
            W=32,
            C=32,
            F=32,
            G=2,
            NL=2,
            NLB=1,
            NOW=2,
            mode=1,
        ](ctx, "staged compute+epilogue")
        run_case[
            bf,
            staged=True,
            N=1,
            H=240,
            W=416,
            C=96,
            F=96,
            G=13,
            NL=2,
            NLB=1,
            NOW=4,
            max_grid=240,
            sample=997,
            mode=2,
        ](ctx, "staged WAN compute+residual")
        run_case[
            bf,
            staged=True,
            N=1,
            H=240,
            W=416,
            C=96,
            F=96,
            G=13,
            NL=2,
            NLB=1,
            NOW=4,
            max_grid=240,
            sample=997,
            mode=1,
        ](ctx, "staged WAN compute+epilogue")
