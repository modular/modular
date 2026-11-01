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
from std.math.math import (
    _call_libm,
    _gamma_max_scalings,
)
from std.sys import CompilationTarget
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

# glibc's tgamma/tgammaf are themselves off by up to about 4.5 ulp (measured
# against mpmath at the worst points of these sweeps; glibc documents up to
# 9 ulp), while `gamma` has measured maxima of 1.97 ulp for float32 and
# 3.16 ulp for float64 against MPFR. The 8 ulp bound covers 3.2 + 4.5 ulp.
# Apple's tgamma is off by hundreds of ulp, so the libm comparisons run only
# on Linux; every platform checks the exact reference values below.
comptime max_ulp_vs_libm = 8
comptime max_ulp_vs_exact = 4
comptime max_ulp_vs_libm_specials = 4
comptime euler_gamma = 0.5772156649015329


# Correctly rounded gamma at integers, (n-1)!, and at half-integers,
# (2n)! sqrt(pi) / (4^n n!) and (-4)^n n! sqrt(pi) / (2n)!, evaluated in
# exact/120-digit arithmetic and rounded to the target precision.
def _ref_f64() -> List[Tuple[Float64, Float64]]:
    return [
        (1.0, 1.0),
        (2.0, 1.0),
        (3.0, 2.0),
        (5.0, 24.0),
        (10.0, 362880.0),
        (20.0, 1.21645100408832e17),
        (23.0, 1.1240007277776077e21),
        (30.0, 8.841761993739702e30),
        (50.0, 6.082818640342675e62),
        (100.0, 9.332621544394415e155),
        (150.0, 3.80892263763057e260),
        (170.0, 4.269068009004705e304),
        (171.0, 7.257415615307999e306),
        (0.5, 1.772453850905516),
        (1.5, 0.886226925452758),
        (2.5, 1.329340388179137),
        (5.5, 52.34277778455352),
        (10.5, 1133278.3889487856),
        (20.5, 5.406242982335075e17),
        (50.5, 4.29046291235196e63),
        (100.5, 9.320963104082716e156),
        (150.5, 4.661072627097378e261),
        (170.5, 5.56209241456e305),
        (171.5, 9.4833675668248e307),
        (-0.5, -3.544907701811032),
        (-1.5, 2.363271801207355),
        (-2.5, -0.9453087204829419),
        (-4.5, -0.060019601300504245),
        (-9.5, 2.772127911575102e-06),
        (-19.5, 5.811045977502236e-18),
        (-49.5, 7.322269689234127e-64),
        (-99.5, 3.3704592739067173e-157),
        (-149.5, 6.740063725516714e-262),
        (-169.5, 5.6482208842233253e-306),
        (-170.5, -3.3127395215386074e-308),
    ]


def _ref_f32() -> List[Tuple[Float64, Float64]]:
    return [
        (1.0, 1.0),
        (2.0, 1.0),
        (3.0, 2.0),
        (5.0, 24.0),
        (10.0, 362880.0),
        (20.0, 1.2164510459415757e17),
        (30.0, 8.84176187039338e30),
        (35.0, 2.952328027141237e38),
        (0.5, 1.7724539041519165),
        (1.5, 0.8862269520759583),
        (2.5, 1.3293403387069702),
        (5.5, 52.342777252197266),
        (10.5, 1133278.375),
        (20.5, 5.406243010998436e17),
        (30.5, 4.822697170920775e31),
        (34.5, 5.04462063421064e37),
        (-0.5, -3.544907808303833),
        (-1.5, 2.363271713256836),
        (-2.5, -0.9453087449073792),
        (-4.5, -0.060019601136446),
        (-9.5, 2.7721278001990868e-06),
        (-19.5, 5.81104595999067e-18),
        (-29.5, 6.514182383464557e-32),
        (-32.5, -2.0862490282641087e-36),
        (-33.5, 6.227609156717808e-38),
        # Subnormal results, kept on targets that flush subnormals too.
        (-34.5, -1.805103636296978e-39),
        (-36.5, -1.3928906735388682e-42),
    ]


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
    assert_true(
        (got._to_bits_signed() < 0) == (want._to_bits_signed() < 0), msg
    )
    # Subtract unsigned encodings only after excluding other classes and signs.
    var distance = max(got_bits, want_bits) - min(got_bits, want_bits)
    assert_true(distance <= UInt64(Int.MAX), msg)
    return Int(distance)


def _check_exact[
    dtype: DType
](refs: List[Tuple[Float64, Float64]]) raises where dtype.is_floating_point():
    for r in refs:
        var x = Scalar[dtype](r[0])
        var want = Scalar[dtype](r[1])
        var got = gamma(x)
        assert_true(
            _gamma_ulp_distance(got, want, String(t"gamma({x})"))
            <= max_ulp_vs_exact,
            String(t"gamma({x}) = {got}, exact = {want}"),
        )
        var got_v = gamma(SIMD[dtype, 4](x))
        comptime for k in range(4):
            assert_equal(got_v[k], got, String(t"SIMD gamma({x})"))


def _check_edge_cases[dtype: DType]() raises where dtype.is_floating_point():
    comptime S = Scalar[dtype]
    assert_equal(gamma(S(0.0)), inf[dtype]())
    assert_equal(gamma(S(-0.0)), -inf[dtype]())
    assert_equal(gamma(inf[dtype]()), inf[dtype]())
    assert_equal(gamma(S(1000.0)), inf[dtype]())
    for x in [S(-1.0), S(-2.0), S(-100.0), -inf[dtype](), nan[dtype]()]:
        assert_true(isnan(gamma(x)), String(t"gamma({x}) is not nan"))
    assert_equal(gamma(S(-1000.5)), 0.0)
    assert_equal(1 / gamma(S(-1000.5)), -inf[dtype]())
    assert_equal(1 / gamma(S(-1001.5)), inf[dtype]())
    # gamma(x) = 1/x - euler_gamma + O(x) near zero.
    for x in [S(1e-30), S(-1e-30)]:
        var x64 = x.cast[.float64]()
        var want = (1.0 / x64 - euler_gamma).cast[dtype]()
        assert_true(
            _gamma_ulp_distance(gamma(x), want, String(t"gamma({x})"))
            <= max_ulp_vs_exact,
            String(t"gamma({x}) = {gamma(x)}, want {want}"),
        )


def _max_ulp[
    dtype: DType, width: Int
](
    lo: Float64, hi: Float64, n: Int
) raises -> Int where dtype.is_floating_point():
    """Returns the max ulp distance between `gamma` and libm `tgamma`."""
    var worst = 0
    var i = 0
    while i < n:
        var v = SIMD[dtype, width]()
        comptime for k in range(width):
            v[k] = Scalar[dtype](lo + (hi - lo) * Float64(i + k) / Float64(n))
        var got = gamma(v)
        var want = _call_libm["tgamma"](v)
        comptime for k in range(width):
            worst = max(
                worst,
                _gamma_ulp_distance(got[k], want[k], String(t"gamma({v[k]})")),
            )
        i += width
    return worst


def _check_specials[dtype: DType]() raises where dtype.is_floating_point():
    comptime S = Scalar[dtype]
    var values: List[S] = [
        S(0.0),
        S(-0.0),
        S(1.0),
        S(2.0),
        S(0.5),
        S(-0.5),
        S(-1.0),
        S(-2.0),
        S(-100.0),
        S(-1.5),
        S(1e-30),
        S(-1e-30),
        S(1000.0),
        S(-1000.5),
        inf[dtype](),
        -inf[dtype](),
        nan[dtype](),
    ]
    for x in values:
        var got = gamma(x)
        var want = _call_libm["tgamma"](x)
        assert_true(
            _gamma_ulp_distance(got, want, String(t"gamma({x})"))
            <= max_ulp_vs_libm_specials,
            String(t"gamma({x}) = {got}, libm = {want}"),
        )


def test_gamma_float64() raises:
    _check_exact[.float64](_ref_f64())
    _check_edge_cases[.float64]()
    comptime if CompilationTarget.is_linux():
        var worst = _max_ulp[.float64, 4](-189.99, 171.7, 40000)
        worst = max(worst, _max_ulp[.float64, 1](-1.5, 3.5, 20001))
        worst = max(worst, _max_ulp[.float64, 8](-1e-6, 1e-6, 2001))
        print("gamma float64 max ulp vs libm:", worst)
        assert_true(worst <= max_ulp_vs_libm)
        _check_specials[.float64]()


def test_gamma_float32() raises:
    _check_exact[.float32](_ref_f32())
    _check_edge_cases[.float32]()
    comptime if CompilationTarget.is_linux():
        var worst = _max_ulp[.float32, 4](-49.99, 35.9, 40000)
        worst = max(worst, _max_ulp[.float32, 1](-1.5, 3.5, 20001))
        print("gamma float32 max ulp vs libm:", worst)
        assert_true(worst <= max_ulp_vs_libm)
        _check_specials[.float32]()


def test_gamma_max_scalings() raises:
    # 2 * 50! is about 2^215.2, so float32 needs 3 rescalings by 2^64 near
    # its -50 cutoff; 2 * 190! is about 2^1170.3, one rescaling by 2^1000.
    assert_equal(_gamma_max_scalings(50, 64), 3)
    assert_equal(_gamma_max_scalings(190, 1000), 1)
    assert_equal(_gamma_max_scalings(35, 64), 2)


def _check_gamma_oracle[dtype: DType]() raises:
    comptime S = Scalar[dtype]
    comptime budget = max_ulp_vs_exact
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
    assert_equal(_gamma_ulp_distance(smallest, S(0.0), "oracle"), 1)
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
