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

from std.math import exp2, inf, isinf, isnan, nan, nextafter, pi, tan, ulp
from std.math.math import _call_libm
from std.testing import assert_equal, assert_true, TestSuite

comptime max_ulp = 2.0
comptime max_ulp_reference = 1.0
comptime width = 4
comptime num_dense = 20000
comptime dense_bound = 100.0
comptime num_half_pi_multiples = 2000
comptime num_huge = 8000
comptime two_pow_neg53 = 1.0 / 9007199254740992.0


def _ulp_error[
    dtype: DType, //
](
    got: Scalar[dtype], expected: Scalar[dtype]
) -> Float64 where dtype.is_floating_point():
    if isnan(expected) or isnan(got):
        return 0.0 if isnan(expected) and isnan(got) else inf[.float64]()
    if got == expected:
        if (
            got == 0
            and 1.0 / got.cast[.float64]() != 1.0 / expected.cast[.float64]()
        ):
            return inf[.float64]()
        return 0.0
    if isinf(got) or isinf(expected):
        return inf[.float64]()
    var diff = abs(got.cast[.float64]() - expected.cast[.float64]())
    return diff / ulp(expected).cast[.float64]()


def _check_inputs[
    dtype: DType, //
](
    name: String, inputs: List[Scalar[dtype]]
) raises where dtype.is_floating_point():
    # Evaluate in vectors so lanes on both sides of the Cody-Waite /
    # Payne-Hanek switch share a vector.
    var worst = 0.0
    var worst_x = Scalar[dtype](0)
    for i in range(0, len(inputs) - width + 1, width):
        var v = SIMD[dtype, width]()
        comptime for j in range(width):
            v[j] = inputs[i + j]
        var got = tan(v)
        for j in range(width):
            var want = _call_libm["tan"](v[j])
            var e = max(_ulp_error(got[j], want), _ulp_error(tan(v[j]), want))
            if e > worst:
                worst = e
                worst_x = v[j]
    print(t"tan {dtype} {name} max ulp = {worst} at {worst_x}")
    assert_true(worst <= max_ulp, String(t"tan {dtype} {name} > {max_ulp} ulp"))


def _sweep[dtype: DType]() raises where dtype.is_floating_point():
    var dense = List[Scalar[dtype]]()
    for i in range(num_dense):
        dense.append(
            Scalar[dtype](
                -dense_bound
                + 2 * dense_bound * Float64(i) / Float64(num_dense - 1)
            )
        )
    _check_inputs(String(t"[{-dense_bound}, {dense_bound}]"), dense)

    var near = List[Scalar[dtype]]()
    var up = Scalar[dtype].MAX_FINITE
    for k in range(1, num_half_pi_multiples + 1):
        var c = Scalar[dtype](Float64(k) * (pi / 2))
        near.append(c)
        near.append(nextafter(c, -up))
        near.append(nextafter(c, up))
        near.append(-c)
    _check_inputs("near k*pi/2", near)

    var huge = List[Scalar[dtype]]()
    var max_exp = 127.9 if dtype == .float32 else 1023.9
    var state: UInt64 = 7
    for _ in range(num_huge):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float64(state >> 11) * two_pow_neg53
        var mag = Scalar[dtype](exp2(3.0 + (max_exp - 3.0) * u))
        huge.append(mag if (state >> 7) & 1 == 0 else -mag)
    _check_inputs("huge", huge)


def test_tan_float32() raises:
    _sweep[.float32]()


def test_tan_float64() raises:
    _sweep[.float64]()


def test_tan_hard_cases() raises:
    # Inputs that once had more than 1.5 ulp of error, with correctly rounded
    # results from mpmath.
    var x32: List[Float32] = [
        -3992.179,
        -1541.7366943359375,
        -5.193172931671143,
    ]
    var want32: List[Float32] = [0.999819, 0.99980086, 1.9171497]
    for i in range(len(x32)):
        assert_true(
            _ulp_error(tan(x32[i]), want32[i]) <= 1.0, String(t"x = {x32[i]}")
        )
    var x64: List[Float64] = [1.3745834964292172e291, 7.386753707202125]
    var want64: List[Float64] = [1.9923528895765197, 1.9822256111149423]
    for i in range(len(x64)):
        assert_true(
            _ulp_error(tan(x64[i]), want64[i]) <= 1.0, String(t"x = {x64[i]}")
        )


def test_tan_near_quarter_pi() raises:
    # Source emulation found a 1.494-ULP case outside the pole sweeps.
    # The reference is rounded once from mpmath at 450 decimal digits.
    var x = Float64(-960366.8100446416)
    var want = Float64(-0.9999999999096455)
    assert_true(_ulp_error(tan(x), want) <= max_ulp)
    assert_true(_ulp_error(tan(-x), -want) <= max_ulp)
    var got = tan(SIMD[.float64, 4](x, -x, x, -x))
    for i in range(4):
        assert_true(
            _ulp_error(got[i], want if i % 2 == 0 else -want) <= max_ulp
        )


def _check_reference[
    dtype: DType, //
](
    x: List[Scalar[dtype]], want: List[Scalar[dtype]]
) raises where dtype.is_floating_point():
    var worst = 0.0
    for i in range(len(x)):
        var e = _ulp_error(tan(x[i]), want[i])
        worst = max(worst, e)
        assert_true(
            e <= max_ulp_reference,
            String(t"tan {dtype} x = {x[i]} want {want[i]} ulp = {e}"),
        )
    print(t"tan {dtype} vs mpmath max ulp = {worst}")


def test_tan_correctly_rounded_reference() raises:
    # Inputs spread over the dense range, multiples of pi/2 and the full
    # exponent range, with results correctly rounded from mpmath. Unlike the
    # libm comparison above, these have no reference error.
    var x32: List[Float32] = [
        24.580339431762695,
        48.357398986816406,
        59.03871154785156,
        88.49005889892578,
        47.979713439941406,
        84.46499633789062,
        -94.1989517211914,
        -6.875469207763672,
        88.67134094238281,
        29.794910430908203,
        80.18009948730469,
        -77.35881042480469,
        1.5707963705062866,
        -1.5707963705062866,
        3.1415927410125732,
        -3.1415927410125732,
        4.71238899230957,
        -4.71238899230957,
        15.707962989807129,
        -15.707962989807129,
        157.0796356201172,
        -157.0796356201172,
        1570.79638671875,
        -1570.79638671875,
        3.593053805778685e33,
        2.2272157892193393e21,
        -24.898075103759766,
        -1.7742597946605568e16,
        -2.272142877961267e35,
        8016639.0,
        -1319948.875,
        464311.71875,
        -2.8950492854759483e37,
        -600297408.0,
    ]
    var w32: List[Float32] = [
        -0.6164147207149244,
        2.8515417642609635,
        -0.76265126131732,
        0.5798407186257248,
        1.151792001879568,
        -0.3741272601543554,
        0.04886672808332359,
        -0.6728685015431658,
        0.8538875149264434,
        19.895730709297577,
        -14.368048472168038,
        2.4342157169747596,
        -22877332.42885646,
        22877332.42885646,
        8.742278000372508e-08,
        -8.742278000372508e-08,
        -83858283.00668408,
        83858283.00668408,
        -2.7814183728607036e-07,
        2.7814183728607036e-07,
        2.940627525585344e-06,
        -2.940627525585344e-06,
        5.9923853452494896e-05,
        -5.9923853452494896e-05,
        -17.220598462135435,
        1.1941920285355052,
        0.23907071791573364,
        -4.37527126837597,
        -1.3644325015370085,
        6.0274370094121315,
        0.8477454287981488,
        0.032524157478649854,
        0.505377602935248,
        2.599281231677147,
    ]
    var x64: List[Float64] = [
        -99.83999199959997,
        -3.0049329364669744e152,
        -56.903766155053546,
        96.48422176518505,
        74.48155308736037,
        -42.1389664506147,
        92.2955977900167,
        7.844693774162124,
        35.56609545011847,
        -59.04409709324143,
        88.19520021759982,
        38.12838822138164,
        93.31286246343907,
        78.74833551529571,
        1.5707963267948966,
        -1.5707963267948966,
        3.141592653589793,
        -3.141592653589793,
        4.71238898038469,
        -4.71238898038469,
        15.707963267948966,
        -15.707963267948966,
        157.07963267948966,
        -157.07963267948966,
        1570.7963267948967,
        -1570.7963267948967,
        34047032.74514626,
        8.034486851756135e51,
        -9.509917253533272e81,
        3.290240749653878e93,
        -1.148810747970704e184,
        1.5603068473104005e21,
        2.8269032253368798e252,
        8.470043124657052e214,
        -6.19237338839139e148,
        -2.634525703167884e18,
    ]
    var w64: List[Float64] = [
        0.8269730656298917,
        -1.3654261010554134,
        -0.37081704737656795,
        -1.2735748936979772,
        -1.304231666016098,
        -3.577964104191514,
        2.4936404005589696,
        107.66433443838628,
        1.5871843149184384,
        0.754168044791791,
        0.23478258846478495,
        0.45774549050021635,
        -1.3547231402843223,
        0.21159483385641853,
        1.633123935319537e16,
        -1.633123935319537e16,
        -1.2246467991473532e-16,
        1.2246467991473532e-16,
        5443746451065123.0,
        -5443746451065123.0,
        -6.123233995736766e-16,
        6.123233995736766e-16,
        9.82193361864236e-16,
        -9.82193361864236e-16,
        6.666535247945037e-14,
        -6.666535247945037e-14,
        0.38868975194525696,
        -0.5098829404068315,
        -1.19390283139637,
        -0.7654524057585489,
        0.5640005779445127,
        0.17429582615396438,
        -0.5110688225219037,
        -4.422809497415507,
        10.8243303193624,
        0.8897804755781987,
    ]
    _check_reference(x32, w32)
    _check_reference(x64, w64)


def test_tan_half_overflow() raises:
    assert_equal(tan(Float16(177.5)), -inf[.float16]())
    assert_equal(tan(Float16(-177.5)), inf[.float16]())


def test_ulp_error() raises:
    assert_equal(_ulp_error(Float64(0), -inf[.float64]()), inf[.float64]())
    assert_equal(_ulp_error(inf[.float64](), -inf[.float64]()), inf[.float64]())
    assert_equal(_ulp_error(Float32(0), -inf[.float32]()), inf[.float64]())
    assert_equal(_ulp_error(inf[.float32](), -inf[.float32]()), inf[.float64]())
    assert_equal(_ulp_error(nan[.float64](), Float64(1)), inf[.float64]())
    assert_equal(_ulp_error(Float64(0), Float64(-0.0)), inf[.float64]())


def _check_special_values[
    dtype: DType
]() raises where dtype.is_floating_point():
    assert_true(isnan(tan(inf[dtype]())))
    assert_true(isnan(tan(-inf[dtype]())))
    assert_true(isnan(tan(nan[dtype]())))
    var pz = tan(Scalar[dtype](0))
    var nz = tan(-Scalar[dtype](0))
    assert_equal(String(pz), String(Scalar[dtype](0)))
    assert_equal(String(nz), String(-Scalar[dtype](0)), "tan(-0) must be -0")


def test_tan_special_values() raises:
    _check_special_values[.float16]()
    _check_special_values[.bfloat16]()
    _check_special_values[.float32]()
    _check_special_values[.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
