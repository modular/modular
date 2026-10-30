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

from std.math import abs, exp, max, sqrt
from std.random import randn, random_float64, random_ui64, seed

from layout import Coord, Idx, ImmTileTensor, TileTensor, row_major
from layout.tile_layout import Layout
from max.algorithm import sync_parallelize
from max.gpu import global_idx
from max.gpu.host import DeviceContext, HostBuffer
from nn.mhc import mhc_fn_split, mhc_pre, mhc_pre_launch_config
from std.testing import assert_raises, assert_true
from std.utils.numerics import inf, isnan, nan

comptime hc = 4
comptime mix_width = 2 * hc + hc * hc
comptime sinkhorn_iters = 20
comptime hc_eps = Float32(1e-6)
comptime rms_eps = Float32(1e-5)
comptime norm_eps = Float32(1e-5)

# post/comb/pre against fp64: max abs. The collapsed (or normed) output passes
# when |err| <= rtol * |ref| + value_atol_frac * rms(ref).
comptime gate_atol = 1e-4
comptime collapsed_rtol = 2.0**-7
comptime normed_rtol = 2.0**-6
comptime value_atol_frac = 1e-2


def _sigmoid(x: Float64) -> Float64:
    return 1 / (1 + exp(-x))


comptime guard_f32 = Float32(-1234.5)
comptime guard_bf16 = guard_f32.cast[.bfloat16]()


def _guard_intact[
    dtype: DType
](buf: HostBuffer[dtype], start: Int, sentinel: Scalar[dtype]) -> Bool:
    """Returns whether `buf` holds `sentinel` from `start` to its end."""
    for i in range(start, len(buf)):
        if buf[i].to_bits() != sentinel.to_bits():
            return False
    return True


def _worst(a: Float64, b: Float64) -> Float64:
    """Returns the larger error, a NaN counting as infinite.

    `max` lowers to `maxnum`, which drops a NaN operand.
    """
    if isnan(a) or isnan(b):
        return inf[DType.float64]()
    return max(a, b)


@fieldwise_init
struct Errors(Copyable, Writable):
    var post: Float64
    var comb: Float64
    var pre: Float64
    var comb_transposed: Float64
    # Worst `|err| - rtol * |ref|` over `collapsed`, in units of its atol.
    var collapsed_over_atol: Float64
    # Whether a second launch on the same inputs matched bit for bit.
    var repeatable: Bool

    def gates(self) -> Float64:
        return _worst(self.post, _worst(self.comb, self.pre))


def run_mhc_pre[
    hidden: Int,
    fn_dtype: DType,
    has_norm: Bool,
    fn_terms: Int = 3,
](
    ctx: DeviceContext,
    num_rows: Int,
    adversarial_fn: Bool = False,
    force_k_splits: Int = 0,
    repeat: Bool = False,
    zero_odd_rows: Bool = False,
    nan_output: String = "",
) raises -> Errors:
    """Runs `mhc_pre` and returns its errors against fp64.

    With `adversarial_fn`, the streams are positive and every fp32 weight
    leaves a positive remainder past its first two bf16 terms, so dropping a
    term biases the projections instead of averaging out. With `repeat`, a
    second launch into re-poisoned outputs must reproduce every output bit.
    With `zero_odd_rows`, every odd token's streams are all zero, so its RMS
    scale and its norm see only their epsilons. `nan_output` names an output
    (`pre`, `post`, `comb` or `collapsed`) that gets one NaN on the host before
    the check, so tests can show the check catches it.
    """
    comptime K = hc * hidden
    var n_streams = num_rows * hc * hidden
    var n_gate = num_rows * hc
    var n_comb = num_rows * hc * hc
    var n_collapsed = num_rows * hidden

    var streams_h = ctx.enqueue_create_host_buffer[.bfloat16](n_streams)
    var fn_h = ctx.enqueue_create_host_buffer[fn_dtype](mix_width * K)
    var base_h = ctx.enqueue_create_host_buffer[.float32](mix_width)
    var scale_h = ctx.enqueue_create_host_buffer[.float32](3)
    var gamma_h = ctx.enqueue_create_host_buffer[.bfloat16](hidden)

    randn[.bfloat16](streams_h.unsafe_ptr(), n_streams, 0.0, 1.0)
    if adversarial_fn:
        comptime if fn_dtype == .float32:
            for i in range(n_streams):
                streams_h[i] = abs(streams_h[i])
            # Each weight is h + l + d with bf16 h and l of random signs and
            # d > 0, placed so the split returns exactly (h, l, d): the
            # remainder past two terms is then positive in every weight, and
            # against positive streams dropping it biases every projection by
            # about 2^-18 * sqrt(K) of its spread. Projections spread ~6, so
            # the gates near their midpoints move by > 1e-4.
            comptime e = 2.0**-5
            for i in range(mix_width * K):
                var h = e * (1 + Float64(random_ui64(1, 127)) / 128)
                var l = e * 2.0**-9 * (1 + Float64(random_ui64(1, 127)) / 128)
                var d = e * 2.0**-18 * (1 + Float64(random_ui64(0, 31)) / 32)
                if random_ui64(0, 1) == 1:
                    h = -h
                if random_ui64(0, 1) == 1:
                    l = -l
                fn_h[i] = Float32(h + l + d).cast[fn_dtype]()
        else:
            raise Error("the adversarial hc_fn is fp32")
    else:
        randn[fn_dtype](
            fn_h.unsafe_ptr(), mix_width * K, 0.0, 1.0 / sqrt(Float64(K))
        )
    var streams = TileTensor(
        streams_h, row_major((num_rows, Idx[hc], Idx[hidden]))
    )
    if zero_odd_rows:
        for t in range(1, num_rows, 2):
            for j in range(hc):
                for d in range(hidden):
                    streams[t, j, d] = 0
    for i in range(mix_width):
        base_h[i] = Float32(0.5 * random_float64(-1.7, 1.7))
    scale_h[0] = 0.5
    scale_h[1] = 0.5
    scale_h[2] = 1.0
    for i in range(hidden):
        gamma_h[i] = Float32(1 + 0.05 * random_float64(-1.7, 1.7)).cast[
            .bfloat16
        ]()

    var streams_d = ctx.enqueue_create_buffer[.bfloat16](n_streams)
    var fn_d = ctx.enqueue_create_buffer[fn_dtype](mix_width * K)
    var base_d = ctx.enqueue_create_buffer[.float32](mix_width)
    var scale_d = ctx.enqueue_create_buffer[.float32](3)
    var gamma_d = ctx.enqueue_create_buffer[.bfloat16](hidden)
    # One guard row past the last token on each output.
    var post_d = ctx.enqueue_create_buffer[.float32](n_gate + hc)
    var pre_d = ctx.enqueue_create_buffer[.float32](n_gate + hc)
    var comb_d = ctx.enqueue_create_buffer[.float32](n_comb + hc * hc)
    var coll_d = ctx.enqueue_create_buffer[.bfloat16](n_collapsed + hidden)
    var post_rows = post_d.create_sub_buffer[.float32](0, n_gate)
    var pre_rows = pre_d.create_sub_buffer[.float32](0, n_gate)
    var comb_rows = comb_d.create_sub_buffer[.float32](0, n_comb)
    var coll_rows = coll_d.create_sub_buffer[.bfloat16](0, n_collapsed)
    ctx.enqueue_copy(streams_d, streams_h)
    ctx.enqueue_copy(fn_d, fn_h)
    ctx.enqueue_copy(base_d, base_h)
    ctx.enqueue_copy(scale_d, scale_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    comptime gamma_size = hidden if has_norm else 1

    var post_h = ctx.enqueue_create_host_buffer[.float32](n_gate + hc)
    var pre_h = ctx.enqueue_create_host_buffer[.float32](n_gate + hc)
    var comb_h = ctx.enqueue_create_host_buffer[.float32](n_comb + hc * hc)
    var coll_h = ctx.enqueue_create_host_buffer[.bfloat16](n_collapsed + hidden)
    var post_2 = ctx.enqueue_create_host_buffer[.float32](n_gate + hc)
    var pre_2 = ctx.enqueue_create_host_buffer[.float32](n_gate + hc)
    var comb_2 = ctx.enqueue_create_host_buffer[.float32](n_comb + hc * hc)
    var coll_2 = ctx.enqueue_create_host_buffer[.bfloat16](n_collapsed + hidden)
    for rep in range(2 if repeat else 1):
        # NaN-poison the outputs so a skipped store cannot pass, and fill the
        # guard rows with a sentinel so a store past the last row cannot.
        post_d.enqueue_fill(guard_f32)
        pre_d.enqueue_fill(guard_f32)
        comb_d.enqueue_fill(guard_f32)
        coll_d.enqueue_fill(guard_bf16)
        post_rows.enqueue_fill(nan[.float32]())
        pre_rows.enqueue_fill(nan[.float32]())
        comb_rows.enqueue_fill(nan[.float32]())
        coll_rows.enqueue_fill(nan[.bfloat16]())
        mhc_pre[
            hc_mult=hc,
            hidden_size=hidden,
            hc_sinkhorn_iters=sinkhorn_iters,
            has_norm=has_norm,
            target="gpu",
            fn_terms=fn_terms,
        ](
            TileTensor(post_d, row_major((num_rows, Idx[hc]))),
            TileTensor(comb_d, row_major((num_rows, Idx[hc], Idx[hc]))),
            TileTensor(coll_d, row_major((num_rows, Idx[hidden]))),
            TileTensor(pre_d, row_major((num_rows, Idx[hc]))),
            TileTensor(
                streams_d, row_major((num_rows, Idx[hc], Idx[hidden]))
            ).as_imm(),
            TileTensor(fn_d, row_major((Idx[mix_width], Idx[K]))).as_imm(),
            TileTensor(base_d, row_major((Idx[mix_width],))).as_imm(),
            TileTensor(scale_d, row_major((Idx[3],))).as_imm(),
            TileTensor(gamma_d, row_major((Idx[gamma_size],))).as_imm(),
            rms_eps,
            hc_eps,
            norm_eps,
            ctx,
            force_k_splits=force_k_splits,
        )
        if rep == 0:
            ctx.enqueue_copy(post_h, post_d)
            ctx.enqueue_copy(pre_h, pre_d)
            ctx.enqueue_copy(comb_h, comb_d)
            ctx.enqueue_copy(coll_h, coll_d)
        else:
            ctx.enqueue_copy(post_2, post_d)
            ctx.enqueue_copy(pre_2, pre_d)
            ctx.enqueue_copy(comb_2, comb_d)
            ctx.enqueue_copy(coll_2, coll_d)
        ctx.synchronize()
        var guard_intact = (
            _guard_intact(post_h if rep == 0 else post_2, n_gate, guard_f32)
            and _guard_intact(pre_h if rep == 0 else pre_2, n_gate, guard_f32)
            and _guard_intact(comb_h if rep == 0 else comb_2, n_comb, guard_f32)
            and _guard_intact(
                coll_h if rep == 0 else coll_2, n_collapsed, guard_bf16
            )
        )
        assert_true(guard_intact, "mhc_pre stores past the last row")

    var nan_row = num_rows // 2
    if nan_output == "pre":
        pre_h[nan_row * hc] = nan[.float32]()
    elif nan_output == "post":
        post_h[nan_row * hc] = nan[.float32]()
    elif nan_output == "comb":
        comb_h[nan_row * hc * hc] = nan[.float32]()
    elif nan_output == "collapsed":
        coll_h[nan_row * hidden] = nan[.bfloat16]()

    var repeatable = True
    if repeat:
        for i in range(n_gate):
            if (
                post_2[i].to_bits() != post_h[i].to_bits()
                or pre_2[i].to_bits() != pre_h[i].to_bits()
            ):
                repeatable = False
        for i in range(n_comb):
            if comb_2[i].to_bits() != comb_h[i].to_bits():
                repeatable = False
        for i in range(n_collapsed):
            if coll_2[i].to_bits() != coll_h[i].to_bits():
                repeatable = False

    # Per-row results of the fp64 reference, reduced after the parallel pass.
    var row_post = List[Float64](length=num_rows, fill=0)
    var row_comb = List[Float64](length=num_rows, fill=0)
    var row_pre = List[Float64](length=num_rows, fill=0)
    var row_comb_t = List[Float64](length=num_rows, fill=0)
    var row_excess = List[Float64](length=num_rows, fill=0)
    var row_ref_sq = List[Float64](length=num_rows, fill=0)
    var post_err = TileTensor(Span(row_post), row_major((num_rows,)))
    var comb_err = TileTensor(Span(row_comb), row_major((num_rows,)))
    var pre_err = TileTensor(Span(row_pre), row_major((num_rows,)))
    var comb_t_err = TileTensor(Span(row_comb_t), row_major((num_rows,)))
    var excess_out = TileTensor(Span(row_excess), row_major((num_rows,)))
    var ref_sq_out = TileTensor(Span(row_ref_sq), row_major((num_rows,)))
    var hc_fn = TileTensor(
        fn_h, row_major((Idx[mix_width], Idx[hc], Idx[hidden]))
    )
    var gamma = TileTensor(gamma_h, row_major((Idx[hidden],)))
    var base = TileTensor(base_h, row_major((Idx[mix_width],)))
    var scale = TileTensor(scale_h, row_major((Idx[3],)))
    var post_out = TileTensor(post_h, row_major((num_rows, Idx[hc])))
    var pre_out = TileTensor(pre_h, row_major((num_rows, Idx[hc])))
    var comb_out = TileTensor(comb_h, row_major((num_rows, Idx[hc], Idx[hc])))
    var coll_out = TileTensor(coll_h, row_major((num_rows, Idx[hidden])))
    comptime rtol = normed_rtol if has_norm else collapsed_rtol

    def ref_row(t: Int) {imm}:
        var ss = Float64(0)
        for j in range(hc):
            for d in range(hidden):
                var v = Float64(streams[t, j, d])
                ss += v * v
        var rms = 1 / sqrt(ss / Float64(K) + Float64(rms_eps))
        var mix = Array[Float64, mix_width](fill=0)
        for n in range(mix_width):
            var acc = Float64(0)
            for j in range(hc):
                for d in range(hidden):
                    acc += Float64(streams[t, j, d]) * Float64(hc_fn[n, j, d])
            mix[n] = acc * rms

        var pre = Array[Float64, hc](fill=0)
        var e_post = Float64(0)
        var e_pre = Float64(0)
        for j in range(hc):
            pre[j] = _sigmoid(
                mix[j] * Float64(scale[0]) + Float64(base[j])
            ) + Float64(hc_eps)
            var post = 2 * _sigmoid(
                mix[hc + j] * Float64(scale[1]) + Float64(base[hc + j])
            )
            e_pre = _worst(e_pre, abs(Float64(pre_out[t, j]) - pre[j]))
            e_post = _worst(e_post, abs(Float64(post_out[t, j]) - post))

        var c = Array[Float64, hc * hc](fill=0)
        for j in range(hc):
            var mx = Float64(-1e300)
            for k in range(hc):
                c[j * hc + k] = mix[2 * hc + j * hc + k] * Float64(
                    scale[2]
                ) + Float64(base[2 * hc + j * hc + k])
                mx = max(mx, c[j * hc + k])
            var s = Float64(0)
            for k in range(hc):
                c[j * hc + k] = exp(c[j * hc + k] - mx)
                s += c[j * hc + k]
            for k in range(hc):
                c[j * hc + k] = c[j * hc + k] / s + Float64(hc_eps)
        for it in range(sinkhorn_iters):
            if it > 0:
                for j in range(hc):
                    var s = Float64(0)
                    for k in range(hc):
                        s += c[j * hc + k]
                    for k in range(hc):
                        c[j * hc + k] /= s + Float64(hc_eps)
            for k in range(hc):
                var s = Float64(0)
                for j in range(hc):
                    s += c[j * hc + k]
                for j in range(hc):
                    c[j * hc + k] /= s + Float64(hc_eps)
        var e_comb = Float64(0)
        var e_comb_t = Float64(0)
        for j in range(hc):
            for k in range(hc):
                var got = Float64(comb_out[t, j, k])
                e_comb = _worst(e_comb, abs(got - c[j * hc + k]))
                e_comb_t = _worst(e_comb_t, abs(got - c[k * hc + j]))

        var y = List[Float64](length=hidden, fill=0)
        for d in range(hidden):
            var acc = Float64(0)
            for j in range(hc):
                acc += pre[j] * Float64(streams[t, j, d])
            y[d] = acc
        comptime if has_norm:
            # The norm reads the bf16-rounded collapse, as the op chain does.
            var css = Float64(0)
            for d in range(hidden):
                y[d] = Float64(Float32(y[d]).cast[.bfloat16]())
                css += y[d] * y[d]
            var f = 1 / sqrt(css / Float64(hidden) + Float64(norm_eps))
            for d in range(hidden):
                y[d] *= f * Float64(gamma[d])
        var excess = Float64(-1e300)
        var ref_sq = Float64(0)
        for d in range(hidden):
            var got = Float64(coll_out[t, d])
            excess = _worst(excess, abs(got - y[d]) - rtol * abs(y[d]))
            ref_sq += y[d] * y[d]

        post_err[t] = e_post
        pre_err[t] = e_pre
        comb_err[t] = e_comb
        comb_t_err[t] = e_comb_t
        excess_out[t] = excess
        ref_sq_out[t] = ref_sq

    sync_parallelize(ref_row, num_rows)

    var errs = Errors(0, 0, 0, 1e300, -1e300, repeatable)
    var ref_sq = Float64(0)
    var excess = Float64(-1e300)
    for t in range(num_rows):
        errs.post = _worst(errs.post, row_post[t])
        errs.comb = _worst(errs.comb, row_comb[t])
        errs.pre = _worst(errs.pre, row_pre[t])
        # The transposed comb must miss somewhere, so keep its largest miss.
        errs.comb_transposed = row_comb_t[t] if t == 0 else _worst(
            errs.comb_transposed, row_comb_t[t]
        )
        excess = _worst(excess, row_excess[t])
        ref_sq += row_ref_sq[t]
    var atol = value_atol_frac * sqrt(ref_sq / Float64(n_collapsed))
    errs.collapsed_over_atol = excess / atol
    var config = mhc_pre_launch_config(num_rows, hidden, force_k_splits)
    print(
        t"mhc_pre KS= {config[1]}  D= {hidden}  fn= {fn_dtype} x {fn_terms} "
        t" norm= {has_norm}  adversarial= {adversarial_fn}  T= {num_rows} "
        t" post= {errs.post}  comb= {errs.comb}  pre= {errs.pre}  comb_T="
        t" {errs.comb_transposed}  collapsed_excess/atol="
        t""" {errs.collapsed_over_atol} {" repeatable=" if repeat else ""} {String(repeatable) if repeat else ""}"""
    )
    return errs^


def check_mhc_pre[
    hidden: Int, fn_dtype: DType, has_norm: Bool
](
    ctx: DeviceContext,
    num_rows: Int,
    force_k_splits: Int = 0,
    zero_odd_rows: Bool = False,
) raises:
    var errs = run_mhc_pre[hidden, fn_dtype, has_norm](
        ctx,
        num_rows,
        force_k_splits=force_k_splits,
        zero_odd_rows=zero_odd_rows,
    )
    assert_true(errs.post <= gate_atol, "post differs from fp64")
    assert_true(errs.comb <= gate_atol, "comb differs from fp64")
    assert_true(errs.pre <= gate_atol, "pre differs from fp64")
    assert_true(errs.collapsed_over_atol <= 1.0, "collapsed differs from fp64")
    # A comb with j and k swapped must fail the same gate.
    assert_true(
        errs.comb_transposed > gate_atol, "the comb check misses a transpose"
    )


def test_fn_terms_mutants(ctx: DeviceContext) raises:
    """Dropping fp32 hc_fn terms must fail the gate; all three must pass it."""
    comptime hidden = 4096
    comptime num_rows = 1024
    var full = run_mhc_pre[hidden, .float32, False, fn_terms=3](
        ctx, num_rows, adversarial_fn=True
    )
    assert_true(full.gates() <= gate_atol, "3 hc_fn terms miss the gate")
    var two = run_mhc_pre[hidden, .float32, False, fn_terms=2](
        ctx, num_rows, adversarial_fn=True
    )
    assert_true(two.gates() > gate_atol, "the 2-term mutant passes the gate")
    var one = run_mhc_pre[hidden, .float32, False, fn_terms=1](
        ctx, num_rows, adversarial_fn=True
    )
    assert_true(one.gates() > gate_atol, "the 1-term mutant passes the gate")


def test_nan_outputs_fail_the_gate(ctx: DeviceContext) raises:
    """One NaN in any output must fail the check, whichever output it is in."""
    for output in ["pre", "post", "comb"]:
        var errs = run_mhc_pre[4096, .float32, False](
            ctx, 17, nan_output=output
        )
        assert_true(errs.gates() > gate_atol, "a NaN gate passes the check")
    var errs = run_mhc_pre[4096, .float32, False](
        ctx, 17, nan_output="collapsed"
    )
    assert_true(errs.collapsed_over_atol > 1.0, "a NaN collapse passes")


comptime split_n = 1 << 16
comptime split_layout = row_major[split_n]()


def split_kernel(
    v: TileTensor[.float32, type_of(split_layout), MutAnyOrigin],
    hi: TileTensor[.bfloat16, type_of(split_layout), MutAnyOrigin],
    lo: TileTensor[.bfloat16, type_of(split_layout), MutAnyOrigin],
    lo2: TileTensor[.bfloat16, type_of(split_layout), MutAnyOrigin],
):
    comptime assert v.flat_rank == 1
    comptime assert hi.flat_rank == 1
    comptime assert lo.flat_rank == 1
    comptime assert lo2.flat_rank == 1
    var i = global_idx.x
    if i < split_n:
        var parts = mhc_fn_split(v.load((i,)))
        hi.store((i,), parts[0])
        lo.store((i,), parts[1])
        lo2.store((i,), parts[2])


def test_fn_split_is_exact(ctx: DeviceContext) raises:
    """`hi + lo + lo2` is exactly the fp32 weight, down to 1e-30."""
    comptime n = split_n
    var v_h = ctx.enqueue_create_host_buffer[.float32](n)
    for i in range(n):
        # Random 24-bit significands over magnitudes 1e-30 to 1e30, plus a
        # band pinned near 1e-30.
        var mag = Float64(1e-30) if i % 4 == 0 else Float64(
            10
        ) ** random_float64(-30, 30)
        var sig = Float64(1) + Float64(random_ui64(0, (1 << 23) - 1)) / Float64(
            1 << 23
        )
        var sign = Float64(1) if random_ui64(0, 1) == 1 else Float64(-1)
        v_h[i] = Float32(sign * sig * mag)
    var v_d = ctx.enqueue_create_buffer[.float32](n)
    var hi_d = ctx.enqueue_create_buffer[.bfloat16](n)
    var lo_d = ctx.enqueue_create_buffer[.bfloat16](n)
    var lo2_d = ctx.enqueue_create_buffer[.bfloat16](n)
    ctx.enqueue_copy(v_d, v_h)
    ctx.enqueue_function[split_kernel](
        TileTensor(v_d, split_layout),
        TileTensor(hi_d, split_layout),
        TileTensor(lo_d, split_layout),
        TileTensor(lo2_d, split_layout),
        grid_dim=n // 256,
        block_dim=256,
    )
    var hi_h = ctx.enqueue_create_host_buffer[.bfloat16](n)
    var lo_h = ctx.enqueue_create_host_buffer[.bfloat16](n)
    var lo2_h = ctx.enqueue_create_host_buffer[.bfloat16](n)
    ctx.enqueue_copy(hi_h, hi_d)
    ctx.enqueue_copy(lo_h, lo_d)
    ctx.enqueue_copy(lo2_h, lo2_d)
    ctx.synchronize()
    var bad = 0
    var low_bits = 0
    for i in range(n):
        var sum = Float64(hi_h[i]) + Float64(lo_h[i]) + Float64(lo2_h[i])
        if sum != Float64(v_h[i]):
            bad += 1
        if v_h[i].to_bits() & 0xFFFF != 0:
            low_bits += 1
    print(t"hc_fn split: values= {n}  inexact= {bad}  low16 set= {low_bits}")
    assert_true(bad == 0, "hi + lo + lo2 differs from the fp32 weight")
    assert_true(low_bits > n // 2, "the split test lacks low mantissa bits")


def _launch[
    hidden: Int, has_norm: Bool
](
    ctx: DeviceContext,
    streams: ImmTileTensor[.bfloat16, ...],
    hc_fn: ImmTileTensor[.float32, ...],
    gamma: ImmTileTensor[.bfloat16, ...],
    num_rows: Int,
    force_k_splits: Int = 0,
) raises:
    var alloc_rows = max(num_rows, 1)
    var post_d = ctx.enqueue_create_buffer[.float32](alloc_rows * hc)
    var pre_d = ctx.enqueue_create_buffer[.float32](alloc_rows * hc)
    var comb_d = ctx.enqueue_create_buffer[.float32](alloc_rows * hc * hc)
    var coll_d = ctx.enqueue_create_buffer[.bfloat16](alloc_rows * hidden)
    var base_d = ctx.enqueue_create_buffer[.float32](mix_width)
    var scale_d = ctx.enqueue_create_buffer[.float32](3)
    mhc_pre[
        hc_mult=hc,
        hidden_size=hidden,
        hc_sinkhorn_iters=sinkhorn_iters,
        has_norm=has_norm,
        target="gpu",
    ](
        TileTensor(post_d, row_major((num_rows, hc))),
        TileTensor(comb_d, row_major((num_rows, hc, hc))),
        TileTensor(coll_d, row_major((num_rows, hidden))),
        TileTensor(pre_d, row_major((num_rows, hc))),
        streams,
        hc_fn,
        TileTensor(base_d, row_major((mix_width,))).as_imm(),
        TileTensor(scale_d, row_major((3,))).as_imm(),
        gamma,
        rms_eps,
        hc_eps,
        norm_eps,
        ctx,
        force_k_splits=force_k_splits,
    )
    ctx.synchronize()


def test_mhc_pre_rejects_bad_inputs(ctx: DeviceContext) raises:
    """Bad layouts, shapes and K-splits raise before anything launches."""
    comptime hidden = 4096
    comptime K = hc * hidden
    comptime num_rows = 2
    var streams_d = ctx.enqueue_create_buffer[.bfloat16](
        2 * num_rows * hc * hidden + 8
    )
    var fn_d = ctx.enqueue_create_buffer[.float32](mix_width * K + 8)
    var gamma_d = ctx.enqueue_create_buffer[.bfloat16](hidden + 8)
    var streams_shape = row_major((num_rows, hc, hidden))
    var streams = TileTensor(streams_d, streams_shape).as_imm()
    var hc_fn = TileTensor(fn_d, row_major((mix_width, K))).as_imm()
    var no_gamma = TileTensor(gamma_d, row_major((1,))).as_imm()

    # K-splits must be powers of two whose slices are whole 64-wide steps.
    for bad_k_splits in [-1, 3, 512]:
        with assert_raises(contains="force_k_splits"):
            _launch[hidden, False](
                ctx,
                streams,
                hc_fn,
                no_gamma,
                num_rows,
                force_k_splits=bad_k_splits,
            )
    with assert_raises(contains="force_k_splits"):
        _ = mhc_pre_launch_config(num_rows, 5120, force_k_splits=128)

    # Every other element along the hidden dimension.
    var strided = TileTensor(
        streams_d,
        Layout(
            Coord(num_rows, hc, hidden),
            Coord(2 * hc * hidden, 2 * hidden, 2),
        ),
    ).as_imm()
    with assert_raises(contains="unit inner strides"):
        _launch[hidden, False](ctx, strided, hc_fn, no_gamma, num_rows)

    # Shifted by one element, so no row starts on a 16-byte boundary.
    var misaligned_fn = TileTensor(
        fn_d.unsafe_ptr().unsafe_offset(1), row_major((mix_width, K))
    ).as_imm()
    with assert_raises(contains="unit inner strides"):
        _launch[hidden, False](ctx, streams, misaligned_fn, no_gamma, num_rows)
    var misaligned_gamma = TileTensor(
        gamma_d.unsafe_ptr().unsafe_offset(1), row_major((hidden,))
    ).as_imm()
    with assert_raises(contains="unit inner strides"):
        _launch[hidden, True](ctx, streams, hc_fn, misaligned_gamma, num_rows)

    # Wrong shapes: hc_fn one column short, gamma not [hidden] under the norm.
    var short_fn = TileTensor(fn_d, row_major((mix_width, K - 8))).as_imm()
    with assert_raises(contains="mhc_pre expects"):
        _launch[hidden, False](ctx, streams, short_fn, no_gamma, num_rows)
    with assert_raises(contains="mhc_pre expects"):
        _launch[hidden, True](ctx, streams, hc_fn, no_gamma, num_rows)


def test_mhc_pre_zero_rows(ctx: DeviceContext) raises:
    """An empty batch passes the checks and launches nothing."""
    comptime hidden = 4096
    var streams_d = ctx.enqueue_create_buffer[.bfloat16](hc * hidden)
    var fn_d = ctx.enqueue_create_buffer[.float32](mix_width * hc * hidden)
    var gamma_d = ctx.enqueue_create_buffer[.bfloat16](hidden)
    _launch[hidden, True](
        ctx,
        TileTensor(streams_d, row_major((0, hc, hidden))).as_imm(),
        TileTensor(fn_d, row_major((mix_width, hc * hidden))).as_imm(),
        TileTensor(gamma_d, row_major((hidden,))).as_imm(),
        0,
    )


def test_launch_config() raises:
    """Small batches split K across two launches; large ones stay fused."""
    var decode = mhc_pre_launch_config(8, 4096)
    print("mhc_pre launch config T=8 D=4096: ", decode)
    assert_true(decode[0] == 1, "T=8 is one 16-token block")
    assert_true(decode[1] == 256, "T=8 splits K 256 ways")
    assert_true(decode[2] == 2, "a K-split runs partials, then finish")
    var prefill = mhc_pre_launch_config(8192, 4096)
    assert_true(
        prefill[1] == 1 and prefill[2] == 1, "T=8192 runs the fused kernel"
    )
    # Above 128 token blocks K splits 4 ways up to 192 blocks, then not at all.
    assert_true(mhc_pre_launch_config(2048, 4096)[1] == 2, "T=2048 splits 2")
    assert_true(mhc_pre_launch_config(2064, 4096)[1] == 4, "T=2064 splits 4")
    assert_true(mhc_pre_launch_config(3072, 4096)[1] == 4, "T=3072 splits 4")
    assert_true(mhc_pre_launch_config(3073, 4096)[1] == 1, "T=3073 is fused")
    assert_true(mhc_pre_launch_config(0, 4096)[2] == 0, "T=0 launches nothing")
    var forced = mhc_pre_launch_config(8, 5120, force_k_splits=64)
    assert_true(forced[1] == 64 and forced[2] == 2, "forced K-splits hold")


def test_zero_stream_rows(ctx: DeviceContext) raises:
    """All-zero tokens next to live ones match fp64 on both paths."""
    for k_splits in [1, 2, 256]:
        for num_rows in [2, 17]:
            check_mhc_pre[4096, .bfloat16, True](
                ctx, num_rows, k_splits, zero_odd_rows=True
            )
            check_mhc_pre[4096, .float32, False](
                ctx, num_rows, k_splits, zero_odd_rows=True
            )


def test_k_split_is_repeatable(ctx: DeviceContext) raises:
    """Two launches at 256 K-splits agree bit for bit."""
    for num_rows in [8, 17]:
        var f32 = run_mhc_pre[4096, .float32, True](
            ctx, num_rows, force_k_splits=256, repeat=True
        )
        assert_true(f32.repeatable, "K-split outputs differ across runs")
        var bf16 = run_mhc_pre[4096, .bfloat16, False](
            ctx, num_rows, force_k_splits=256, repeat=True
        )
        assert_true(bf16.repeatable, "K-split outputs differ across runs")


def check_mid_width[hidden: Int](ctx: DeviceContext) raises:
    """Checks K-splits 128, 4 and 1, both hc_fn dtypes and the norm on/off."""
    check_mhc_pre[hidden, .float32, True](ctx, 8, 128)
    check_mhc_pre[hidden, .float32, True](ctx, 1024, 4)
    check_mhc_pre[hidden, .float32, True](ctx, 8192, 1)
    check_mhc_pre[hidden, .bfloat16, False](ctx, 17, 128)
    check_mhc_pre[hidden, .bfloat16, False](ctx, 17, 4)
    check_mhc_pre[hidden, .bfloat16, False](ctx, 1024, 1)


def main() raises:
    seed(0)
    test_launch_config()
    with DeviceContext() as ctx:
        test_mhc_pre_rejects_bad_inputs(ctx)
        test_mhc_pre_zero_rows(ctx)
        test_fn_split_is_exact(ctx)
        test_fn_terms_mutants(ctx)
        test_nan_outputs_fail_the_gate(ctx)
        test_k_split_is_repeatable(ctx)
        test_zero_stream_rows(ctx)
        # The fused kernel (one K-split), forced where the default splits:
        # every T at D=4096 with fp32 hc_fn and no norm; every D, hc_fn dtype
        # and norm setting elsewhere. T = 15, 17 and 129 leave a partial
        # 16-row block.
        for num_rows in [1, 2, 15, 16, 17, 128, 129, 1024, 8192]:
            check_mhc_pre[4096, .float32, False](ctx, num_rows, 1)
        for num_rows in [16, 1024]:
            check_mhc_pre[4096, .float32, True](ctx, num_rows, 1)
        for num_rows in [1, 17, 129, 8192]:
            check_mhc_pre[4096, .bfloat16, True](ctx, num_rows, 1)
        for num_rows in [2, 16, 129, 1024, 8192]:
            check_mhc_pre[5120, .float32, True](ctx, num_rows, 1)
        for num_rows in [15, 128, 8192]:
            check_mhc_pre[5120, .bfloat16, False](ctx, num_rows, 1)

        # The K-split path at every forced split, in a reduced cross product.
        # D=5120 has 320 64-wide steps, so it splits at most 64 ways.
        var small_rows = [1, 2, 8, 15, 16, 17, 128]
        for k_splits in [2, 16, 64, 256]:
            for num_rows in small_rows:
                check_mhc_pre[4096, .float32, False](ctx, num_rows, k_splits)
                check_mhc_pre[4096, .bfloat16, True](ctx, num_rows, k_splits)
        for k_splits in [4, 32]:
            for num_rows in [8, 17]:
                check_mhc_pre[4096, .float32, False](ctx, num_rows, k_splits)
                check_mhc_pre[4096, .bfloat16, True](ctx, num_rows, k_splits)
        for k_splits in [2, 256]:
            for num_rows in [8, 17]:
                check_mhc_pre[4096, .float32, True](ctx, num_rows, k_splits)
        for k_splits in [16, 64]:
            for num_rows in [8, 128]:
                check_mhc_pre[4096, .bfloat16, False](ctx, num_rows, k_splits)
        for k_splits in [2, 16, 64]:
            for num_rows in [1, 8, 17, 128]:
                check_mhc_pre[5120, .float32, True](ctx, num_rows, k_splits)
            for num_rows in [2, 15, 16]:
                check_mhc_pre[5120, .bfloat16, False](ctx, num_rows, k_splits)

        # The widest and narrowest supported streams. D=1024 has 64 64-wide
        # steps, so it splits at most 64 ways; D=8192 splits 256 ways with
        # two steps per slice.
        for k_splits in [1, 2, 256]:
            for num_rows in [8, 17, 8192]:
                if num_rows > 17 and k_splits > 1:
                    continue
                check_mhc_pre[8192, .float32, True](ctx, num_rows, k_splits)
                check_mhc_pre[8192, .bfloat16, False](ctx, num_rows, k_splits)
        check_mhc_pre[8192, .bfloat16, True](ctx, 17, 256)
        check_mhc_pre[8192, .bfloat16, True](ctx, 8192, 2)
        for k_splits in [1, 2, 64]:
            for num_rows in [8, 17, 8192]:
                if num_rows > 17 and k_splits > 1:
                    continue
                check_mhc_pre[1024, .float32, False](ctx, num_rows, k_splits)
                check_mhc_pre[1024, .bfloat16, True](ctx, num_rows, k_splits)

        # Widths between the ends, at the default split and fused.
        for num_rows in [8, 1024]:
            check_mhc_pre[3072, .float32, True](ctx, num_rows)
            check_mhc_pre[7168, .bfloat16, False](ctx, num_rows)
        check_mhc_pre[3072, .float32, True](ctx, 17, 1)
        check_mhc_pre[7168, .bfloat16, False](ctx, 17, 1)

        # The remaining widths, which split at most 128 ways.
        check_mid_width[2048](ctx)
        check_mid_width[6144](ctx)

        # The default split across the mid band.
        for num_rows in [256, 1024, 3000, 4096]:
            check_mhc_pre[4096, .float32, False](ctx, num_rows)
            check_mhc_pre[4096, .bfloat16, True](ctx, num_rows)
        for num_rows in [256, 1024]:
            check_mhc_pre[5120, .float32, True](ctx, num_rows)
            check_mhc_pre[5120, .bfloat16, False](ctx, num_rows)
