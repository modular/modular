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
"""The fp32 transposed convolution route of `conv_transposed_gpu` (B200).

The oracle is a host loop, not cuDNN. Operands are small integers drawn from a
hash, so every fp32 partial sum is exact in any accumulation order and a
mismatch can only be an indexing, padding or epilogue error; each output
element is compared bit for bit. The 1-D 512 -> 1024 channel geometry and the
2-D geometry also run on random full-mantissa fp32 operands against a tight
tolerance, so a tf32 or bf16 operand rounding would fail there.

Every geometry the dedicated kernels serve runs four ways: through both
`conv_dgrad_fp32` entry points directly (pinning the kernel even if the
dispatch guard moves), and through both `conv_transposed_gpu` overloads, the
fused ones with the storing epilogue `acc + bias[channel] + 1`. The output is
poisoned with NaN first, with a guard band after it, so an element the kernel skipped or a write
past the end shows up as a NaN.

A real PDL chain covers the operands: a slow producer releases its dependents
at entry, spins, and only then copies the input and the filter into place over
a NaN sentinel. The convolution is launched right behind it with no host sync,
so a kernel that reads an operand before its own `wait_on_dependent_grids()`
produces NaN.

Geometries the kernels do not serve (another batch, another channel count,
another tap count) run through `conv_transposed_gpu` and are compared with the
same host loop, and `conv_dgrad_fp32` must decline them with nothing enqueued.
"""

from std.math import ceildiv, isclose, isnan, nan
from std.sys.intrinsics import llvm_intrinsic
from std.testing import assert_equal, assert_true

from layout import Coord, TileTensor, row_major
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.primitives.grid_controls import launch_dependent_grids
from nn.conv.conv_transpose import conv_transposed_gpu
from nn.conv.gpu.nvidia.sm100.conv_dgrad_fp32 import conv_dgrad_fp32

comptime dt = DType.float32
comptime GUARD = 1024
"""NaN-filled elements after the output."""


def _hash_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random integer in [-2, 2]."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return Float32((h >> 9) % 5 - 2)


def _float_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random fp32 in [-1, 1] with a full mantissa."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return Float32((h >> 8) % 2000001) / 1000000.0 - 1.0


def _fill[exact: Bool](buf: HostBuffer[dt], n: Int, salt: Int):
    for i in range(n):
        comptime if exact:
            buf[i] = _hash_value(i, salt)
        else:
            buf[i] = _float_value(i, salt)


def _out_extent(n: Int, k: Int, stride: Int, dilation: Int, pad: Int) -> Int:
    return stride * (n - 1) + dilation * (k - 1) - 2 * pad + 1


def host_dgrad(
    dy: HostBuffer[dt],
    filt: HostBuffer[dt],
    dst: HostBuffer[dt],
    batch: Int,
    c_in: Int,
    c_out: Int,
    h_in: Int,
    w_in: Int,
    r: Int,
    s: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    ph: Int,
    pw: Int,
):
    """The transposed convolution as a loop: NCHW input and output, CKRS
    filter."""
    var h_out = _out_extent(h_in, r, sh, dh, ph)
    var w_out = _out_extent(w_in, s, sw, dw, pw)
    for i in range(batch * c_out * h_out * w_out):
        dst[i] = 0
    for b in range(batch):
        for co in range(c_out):
            for k in range(c_in):
                for ri in range(r):
                    for si in range(s):
                        var wgt = filt[((k * c_out + co) * r + ri) * s + si]
                        for hi in range(h_in):
                            var ho = hi * sh + ri * dh - ph
                            if ho < 0 or ho >= h_out:
                                continue
                            var src = ((b * c_in + k) * h_in + hi) * w_in
                            var out = ((b * c_out + co) * h_out + ho) * w_out
                            for wi in range(w_in):
                                var wo = wi * sw + si * dw - pw
                                if wo >= 0 and wo < w_out:
                                    dst[out + wo] += dy[src + wi] * wgt


def _count_bad[
    exact: Bool
](
    got: HostBuffer[dt],
    expect: HostBuffer[dt],
    bias: HostBuffer[dt],
    n_out: Int,
    c_out: Int,
    plane: Int,
    biased: Bool,
    bit_exact: Bool,
) -> Int:
    """Elements of `got` that differ from the host loop (plus the epilogue)."""
    var bad = 0
    for i in range(n_out):
        var want = expect[i]
        if biased:
            want += bias[(i // plane) % c_out] + 1
        if isnan(got[i]):
            bad += 1
        elif exact and bit_exact:
            if got[i] != want:
                bad += 1
        elif not isclose[symmetrical=False](got[i], want, atol=1e-4, rtol=1e-5):
            bad += 1
    return bad


def _guard_clean(buf: HostBuffer[dt], n_out: Int) -> Bool:
    for i in range(GUARD):
        if not isnan(buf[n_out + i]):
            return False
    return True


def run_case[
    exact: Bool = True
](
    ctx: DeviceContext,
    c_in: Int,
    c_out: Int,
    h_in: Int,
    w_in: Int,
    r: Int,
    s: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    ph: Int,
    pw: Int,
    served: Bool,
    batch: Int = 1,
) raises:
    """Runs one geometry through every launch form and checks each.

    Parameters:
        exact: Small-integer operands compared bit for bit; otherwise random
            fp32 operands compared with a tight tolerance.

    Args:
        ctx: Device context.
        c_in: Input channels.
        c_out: Output channels.
        h_in: Input rows.
        w_in: Input columns.
        r: Filter rows.
        s: Filter columns.
        sh: Row stride.
        sw: Column stride.
        dh: Row dilation.
        dw: Column dilation.
        ph: Row padding.
        pw: Column padding.
        served: Whether a dedicated kernel serves the geometry.
        batch: Images.
    """
    var h_out = _out_extent(h_in, r, sh, dh, ph)
    var w_out = _out_extent(w_in, s, sw, dw, pw)
    var n_in = batch * c_in * h_in * w_in
    var n_filt = c_in * c_out * r * s
    var n_out = batch * c_out * h_out * w_out

    var dy_host = ctx.enqueue_create_host_buffer[dt](n_in)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](c_out)
    var ref_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out + GUARD)
    ctx.synchronize()
    _fill[exact](dy_host, n_in, h_in * 7 + w_in)
    _fill[exact](filt_host, n_filt, 3)
    _fill[exact](bias_host, c_out, 5)
    host_dgrad(
        dy_host,
        filt_host,
        ref_host,
        batch,
        c_in,
        c_out,
        h_in,
        w_in,
        r,
        s,
        sh,
        sw,
        dh,
        dw,
        ph,
        pw,
    )

    var dy_dev = ctx.enqueue_create_buffer[dt](n_in)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[dt](c_out)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out + GUARD)
    ctx.enqueue_copy(dy_dev, dy_host)
    ctx.enqueue_copy(filt_dev, filt_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    var dy_tt = TileTensor(dy_dev, row_major(batch, c_in, h_in, w_in))
    var filt_tt = TileTensor(filt_dev, row_major(c_in, c_out, r, s))
    var out_tt = TileTensor(out_dev, row_major(batch, c_out, h_out, w_out))
    var bias_tt = TileTensor(bias_dev, row_major(c_out))
    # The storing closures write through a view of their own, so they borrow
    # nothing the entry point receives alongside them.
    var out_view = TileTensor(
        out_dev.unsafe_ptr(), row_major(batch, c_out, h_out, w_out)
    )
    var stride = Coord(sh, sw)
    var dilation = Coord(dh, dw)
    var padding = Coord(ph, pw)
    var out_shape = Coord(batch, c_out, h_out, w_out)
    var plane = h_out * w_out

    @always_inline
    def store[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_view}:
        out_view.store[width=Int(width), alignment=alignment](
            idx, val.cast[dt]()
        )

    @always_inline
    def bias_plus_one[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt, var out_view}:
        # A vector runs along the row, so every lane has one channel's bias.
        var b = SIMD[_dtype, width](bias_tt[idx[1]].cast[_dtype]())
        out_view.store[width=Int(width), alignment=alignment](
            idx, (val + b + 1).cast[dt]()
        )

    # 0: `conv_dgrad_fp32` with a storing epilogue. 1: `conv_dgrad_fp32` into
    # the output. 2: `conv_transposed_gpu`. 3: the same with the storing
    # epilogue.
    comptime for form in range(4):
        out_dev.enqueue_fill(nan[dt]())
        comptime if form == 0:
            var ran = conv_dgrad_fp32(
                dy_tt, filt_tt, out_shape, stride, dilation, padding, store, ctx
            )
            assert_equal(ran, served)
        elif form == 1:
            var ran = conv_dgrad_fp32(
                dy_tt, filt_tt, out_tt, stride, dilation, padding, ctx
            )
            assert_equal(ran, served)
        elif form == 2:
            conv_transposed_gpu[dt, dt, dt](
                out_tt, dy_tt, filt_tt, stride, dilation, padding, ctx
            )
        else:
            conv_transposed_gpu[dt, dt, dt](
                row_major(batch, c_out, h_out, w_out),
                dy_tt,
                filt_tt,
                stride,
                dilation,
                padding,
                bias_plus_one,
                ctx,
            )
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        # A declined direct call enqueues nothing, so there is no output.
        if form > 1 or served:
            assert_equal(
                _count_bad[exact](
                    out_host,
                    ref_host,
                    bias_host,
                    n_out,
                    c_out,
                    plane,
                    form == 3,
                    served,
                ),
                0,
            )
            assert_true(_guard_clean(out_host, n_out))
    var routed = "served" if served else "declined"
    print(
        t"PASS {routed} batch {batch} {c_in} -> {c_out} channels {h_in} x"
        t" {w_in} taps {r} x {s} stride {sh} {sw} dilation {dh} {dw} pad {ph}"
        t" {pw}"
    )


def slow_copy(
    dy_dst: MutPointer[Scalar[dt], MutUntrackedOrigin],
    dy_src: MutPointer[Scalar[dt], MutUntrackedOrigin],
    dy_length: Int32,
    filt_dst: MutPointer[Scalar[dt], MutUntrackedOrigin],
    filt_src: MutPointer[Scalar[dt], MutUntrackedOrigin],
    filt_length: Int32,
    delay_us: Int32,
):
    """Releases the dependent grid at entry, spins, then publishes the input
    and the filter."""
    launch_dependent_grids()
    # nanosleep takes nanoseconds and caps at ~1ms per call.
    var spins = Int(delay_us)
    while spins > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        spins -= 1000
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(dy_length), stride):
        dy_dst[i] = dy_src[i]
    for i in range(tid, Int(filt_length), stride):
        filt_dst[i] = filt_src[i]


def run_pdl_race(
    ctx: DeviceContext,
    c_in: Int,
    c_out: Int,
    h_in: Int,
    w_in: Int,
    r: Int,
    s: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    ph: Int,
    pw: Int,
) raises:
    """Runs a served geometry right behind a producer that publishes its
    operands late, and checks the result against the host loop."""
    var h_out = _out_extent(h_in, r, sh, dh, ph)
    var w_out = _out_extent(w_in, s, sw, dw, pw)
    var n_in = c_in * h_in * w_in
    var n_filt = c_in * c_out * r * s
    var n_out = c_out * h_out * w_out

    var dy_host = ctx.enqueue_create_host_buffer[dt](n_in)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var ref_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out + GUARD)
    ctx.synchronize()
    _fill[True](dy_host, n_in, 11)
    _fill[True](filt_host, n_filt, 13)
    host_dgrad(
        dy_host,
        filt_host,
        ref_host,
        1,
        c_in,
        c_out,
        h_in,
        w_in,
        r,
        s,
        sh,
        sw,
        dh,
        dw,
        ph,
        pw,
    )

    var dy_src = ctx.enqueue_create_buffer[dt](n_in)
    var filt_src = ctx.enqueue_create_buffer[dt](n_filt)
    var dy_dev = ctx.enqueue_create_buffer[dt](n_in)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out + GUARD)
    ctx.enqueue_copy(dy_src, dy_host)
    ctx.enqueue_copy(filt_src, filt_host)
    dy_dev.enqueue_fill(nan[dt]())
    filt_dev.enqueue_fill(nan[dt]())
    out_dev.enqueue_fill(nan[dt]())
    ctx.synchronize()

    var dy_tt = TileTensor(dy_dev, row_major(1, c_in, h_in, w_in))
    var filt_tt = TileTensor(filt_dev, row_major(c_in, c_out, r, s))
    var out_tt = TileTensor(out_dev, row_major(1, c_out, h_out, w_out))

    ctx.enqueue_function[slow_copy](
        dy_dev.unsafe_ptr(),
        dy_src.unsafe_ptr(),
        Int32(n_in),
        filt_dev.unsafe_ptr(),
        filt_src.unsafe_ptr(),
        Int32(n_filt),
        Int32(200),
        grid_dim=ceildiv(max(n_in, n_filt), 256),
        block_dim=256,
    )
    var ran = conv_dgrad_fp32(
        dy_tt,
        filt_tt,
        out_tt,
        Coord(sh, sw),
        Coord(dh, dw),
        Coord(ph, pw),
        ctx,
    )
    assert_true(ran)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var no_bias = ctx.enqueue_create_host_buffer[dt](1)
    assert_equal(
        _count_bad[True](
            out_host,
            ref_host,
            no_bias,
            n_out,
            c_out,
            h_out * w_out,
            False,
            True,
        ),
        0,
    )
    print(t"PASS behind a late producer {c_in} -> {c_out} channels")


def main() raises:
    with DeviceContext() as ctx:
        if ctx.default_device_info.api != "cuda":
            print("Skipping - not running on an NVIDIA GPU")
            return

        # The served geometries.
        run_case(ctx, 2, 3, 1, 9, 1, 5, 1, 1, 1, 1, 0, 0, True)
        run_case(ctx, 4, 6, 8, 9, 3, 5, 1, 2, 2, 2, 1, 1, True)
        run_case[exact=False](ctx, 4, 6, 8, 9, 3, 5, 1, 2, 2, 2, 1, 1, True)
        run_case(ctx, 512, 1024, 1, 550, 1, 7, 1, 1, 1, 1, 0, 3, True)
        run_case[exact=False](
            ctx, 512, 1024, 1, 550, 1, 7, 1, 1, 1, 1, 0, 3, True
        )

        # Near misses: each served geometry with one extent changed.
        run_case(ctx, 3, 3, 1, 9, 1, 5, 1, 1, 1, 1, 0, 0, False)
        run_case(ctx, 512, 1024, 1, 32, 1, 7, 1, 1, 1, 1, 0, 3, False)
        run_case(ctx, 256, 1024, 1, 32, 1, 7, 1, 1, 1, 1, 0, 3, False)
        run_case(ctx, 2, 3, 1, 10, 1, 5, 1, 1, 1, 1, 0, 0, False)
        run_case(ctx, 2, 3, 1, 9, 1, 4, 1, 1, 1, 1, 0, 0, False)
        run_case(ctx, 4, 6, 8, 9, 3, 5, 1, 2, 2, 2, 1, 1, False, batch=2)
        run_case(ctx, 4, 6, 8, 9, 3, 5, 2, 2, 2, 2, 1, 1, False)
        run_case(ctx, 4, 6, 9, 9, 3, 5, 1, 2, 2, 2, 1, 1, False)
        run_case(ctx, 8, 70, 1, 31, 1, 7, 1, 1, 1, 1, 0, 3, False)
        run_case(ctx, 8, 70, 3, 31, 3, 7, 2, 1, 1, 1, 1, 3, False)

        # The operands arrive from a producer still running.
        run_pdl_race(ctx, 4, 6, 8, 9, 3, 5, 1, 2, 2, 2, 1, 1)
        run_pdl_race(ctx, 512, 1024, 1, 550, 1, 7, 1, 1, 1, 1, 0, 3)
