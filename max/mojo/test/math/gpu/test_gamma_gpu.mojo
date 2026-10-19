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

from std.math import gamma, inf, isfinite, isinf, isnan, nan
from std.math.math import _call_libm
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, default_accelerator
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from std.utils.coord import Coord

comptime length = 4096

# glibc's tgamma/tgammaf are themselves off by up to about 4.5 ulp (measured
# against mpmath), while `gamma` has measured maxima of 1.97 ulp for float32
# and 3.16 ulp for float64 against MPFR. The 8 ulp bound covers 3.2 + 4.5 ulp.
comptime max_ulp_vs_libm = 8
# Half-precision results round a float32 evaluation once.
comptime half_max_ulp = 1
# Offset from each negative integer pole, chosen per dtype so the offset
# point survives the `Scalar[dtype]` cast in `_run`: 1e-6 rounds to the
# pole itself in bfloat16/float16 at every k and in float32 for k = 33..39.
# Each offset is 2 ULP at the largest pole's binade in that type.
comptime pole_offset_f32 = Float64(7.62939453125e-06)  # 2^-17
comptime pole_offset_f16 = Float64(0.0625)  # 2^-4
comptime pole_offset_bf16 = Float64(0.25)  # 2^-2
comptime pole_offset_f64 = Float64(1e-6)
comptime num_poles = 40


def _gamma_ulp_distance[
    dtype: DType, //
](got: Scalar[dtype], want: Scalar[dtype], msg: String) raises -> Int:
    if isnan(want):
        assert_true(isnan(got), msg)
        return 0
    if isinf(want):
        assert_true(got == want, msg)
        return 0
    assert_true(isfinite(got), msg)
    var got_bits = got.to_bits[.uint64]()
    var want_bits = want.to_bits[.uint64]()
    if want == 0:
        assert_true(got_bits == want_bits, msg)
        return 0
    assert_true(
        (got._to_bits_signed() < 0) == (want._to_bits_signed() < 0), msg
    )
    # Subtract unsigned encodings only after excluding other classes and signs.
    var distance = max(got_bits, want_bits) - min(got_bits, want_bits)
    assert_true(distance <= UInt64(Int.MAX), msg)
    return Int(distance)


def _run[
    dtype: DType
](
    ctx: DeviceContext, values: List[Float64], max_ulp: Int
) raises -> Bool where dtype.is_floating_point():
    """Checks GPU and host `gamma` against libm `tgamma` evaluated on host."""
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    var n = len(values)
    var in_device = ctx.enqueue_create_buffer[dtype](n)
    var out_device = ctx.enqueue_create_buffer[dtype](n)
    with in_device.map_to_host() as in_host:
        for i in range(n):
            in_host[i] = Scalar[dtype](values[i])

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=n)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=n)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, gamma(val))

    elementwise[pack_size, target="gpu"](func, Coord(n), ctx)

    var worst_gpu = 0
    var worst_host = 0
    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(n):
            var x = in_host[i]
            # `_call_libm` evaluates narrower types in float32 libm.
            var want = _call_libm["tgamma"](x)
            var got_gpu = out_host[i]
            var got_host = gamma(x)
            var msg = String(
                t"gamma({x}): gpu={got_gpu} host={got_host} libm={want}"
            )
            var d_gpu = _gamma_ulp_distance(got_gpu, want, msg)
            var d_host = _gamma_ulp_distance(got_host, want, msg)
            if d_gpu > max_ulp or d_host > max_ulp:
                print(msg)
            worst_gpu = max(worst_gpu, d_gpu)
            worst_host = max(worst_host, d_host)
    print(t"gamma {dtype} max ulp vs libm: gpu {worst_gpu} host {worst_host}")
    return worst_gpu <= max_ulp and worst_host <= max_ulp


def _sweep(lo: Float64, hi: Float64) -> List[Float64]:
    var values = List[Float64](capacity=length)
    for i in range(length):
        values.append(lo + (hi - lo) * Float64(i) / length)
    return values^


def _specials(pole_offset: Float64) -> List[Float64]:
    var values: List[Float64] = [
        0.0,
        -0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        10.0,
        -0.5,
        -1.0,
        -2.0,
        -1.5,
        -2.5,
        -100.0,
        # Subnormal results, then subnormal inputs.
        -34.5,
        -36.5,
        -38.5,
        -42.5,
        -45.5,
        1e-30,
        1e-40,
        -1e-40,
        1.4e-45,
        -1.4e-45,
        1e-38,
        -1e-30,
        1e-300,
        171.6,
        171.7,
        35.0,
        35.1,
        -170.5,
        -180.5,
        -189.5,
        -200.5,
        1000.0,
        -1000.5,
        -1001.5,
        inf[.float64](),
        -inf[.float64](),
        nan[.float64](),
    ]
    # Gamma is ill-conditioned in x near the poles.
    for k in range(num_poles):
        values.append(-Float64(k) - pole_offset)
        values.append(-Float64(k) + pole_offset)
        values.append(-Float64(k) - 0.5)
    return values^


def test_gamma_gpu() raises:
    var ok = True
    with DeviceContext() as ctx:
        ok &= _run[.float32](ctx, _sweep(-49.99, 35.9), max_ulp_vs_libm)
        ok &= _run[.float32](ctx, _sweep(-1.5, 3.5), max_ulp_vs_libm)
        ok &= _run[.float32](ctx, _specials(pole_offset_f32), max_ulp_vs_libm)
        ok &= _run[.float16](ctx, _sweep(-12.0, 10.0), half_max_ulp)
        ok &= _run[.float16](ctx, _specials(pole_offset_f16), half_max_ulp)
        ok &= _run[.bfloat16](ctx, _sweep(-40.0, 35.9), half_max_ulp)
        ok &= _run[.bfloat16](ctx, _specials(pole_offset_bf16), half_max_ulp)
        comptime if not default_accelerator().is_apple_gpu():
            ok &= _run[.float64](ctx, _sweep(-189.99, 171.7), max_ulp_vs_libm)
            ok &= _run[.float64](ctx, _sweep(-1.5, 3.5), max_ulp_vs_libm)
            ok &= _run[.float64](
                ctx, _specials(pole_offset_f64), max_ulp_vs_libm
            )
    assert_true(ok)


def _check_gamma_oracle[dtype: DType]() raises:
    comptime S = Scalar[dtype]
    comptime budget = half_max_ulp if dtype == .float16 or dtype == .bfloat16 else max_ulp_vs_libm
    var one = S(1.0)
    var infinity = inf[dtype]()
    var max_finite = S(from_bits=infinity._to_bits_signed() - 1)
    var nan_next_to_inf = S(from_bits=infinity._to_bits_signed() + 1)
    var smallest = S(from_bits=S(0.0)._to_bits_signed() + 1)
    for valid in [
        one,
        -one,
        max_finite,
        -max_finite,
        smallest,
        -smallest,
        S(0.0),
        S(-0.0),
        infinity,
        -infinity,
        nan[dtype](),
    ]:
        assert_equal(_gamma_ulp_distance(valid, valid, "oracle"), 0)
    assert_equal(
        _gamma_ulp_distance(
            S(from_bits=one._to_bits_signed() + 4), one, "oracle"
        ),
        4,
    )
    assert_equal(
        _gamma_ulp_distance(
            S(from_bits=one._to_bits_signed() + 5), one, "oracle"
        ),
        5,
    )
    # A same-sign zero is one representable step from the smallest subnormal.
    assert_equal(_gamma_ulp_distance(S(0.0), smallest, "oracle"), 1)
    assert_equal(_gamma_ulp_distance(S(-0.0), -smallest, "oracle"), 1)
    for mutant in [S(-4.0), -one, infinity, -infinity, nan_next_to_inf]:
        with assert_raises():
            assert_true(
                _gamma_ulp_distance(mutant, one, "oracle mutant") <= budget
            )
    for got, want in [
        (infinity, max_finite),
        (nan_next_to_inf, max_finite),
        (max_finite, infinity),
        (nan_next_to_inf, infinity),
        (-infinity, infinity),
        (one, nan[dtype]()),
        (S(0.0), S(-0.0)),
        (S(-0.0), S(0.0)),
        (smallest, S(0.0)),
    ]:
        with assert_raises():
            assert_true(
                _gamma_ulp_distance(got, want, "oracle mutant") <= budget
            )


def test_gamma_oracle_f32() raises:
    _check_gamma_oracle[.float32]()


def test_gamma_oracle_f64() raises:
    _check_gamma_oracle[.float64]()


def test_gamma_oracle_running_max() raises:
    # The signed-distance overflow used to disappear into max(0, distance).
    with assert_raises():
        var worst = 0
        for got in [Float64(1.0), Float64(-4.0), Float64(1.0)]:
            worst = max(
                worst, _gamma_ulp_distance(got, Float64(1.0), "oracle mutant")
            )
        assert_true(worst <= max_ulp_vs_libm)


def test_gamma_oracle_f16() raises:
    _check_gamma_oracle[.float16]()


def test_gamma_oracle_bf16() raises:
    _check_gamma_oracle[.bfloat16]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
