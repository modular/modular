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

from std.math import (
    atan,
    atan2,
    cos,
    exp2,
    inf,
    isinf,
    isnan,
    nan,
    nextafter,
    pi,
    sin,
)
from std.math.math import _call_libm
from std.testing import TestSuite, assert_true
from std.utils.numerics import FPUtils

from test_utils.math import libm_call_binary

comptime log2_10 = 3.321928094887362
comptime atan2_libm = libm_call_binary["atan2f", "atan2"]


def _ulp_diff[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Int where dtype.is_floating_point():
    """Returns the ULP distance between `a` and `b`, or -1 when exactly one is
    NaN or their signs differ (atan and atan2 never flip the sign, so this also
    catches signed-zero mismatches)."""
    if isnan(a) or isnan(b):
        return 0 if isnan(a) and isnan(b) else -1
    if isinf(a) or isinf(b):
        return 0 if a == b else -1
    var ia = FPUtils[dtype].bitcast_to_integer(a)
    var ib = FPUtils[dtype].bitcast_to_integer(b)
    if (ia < 0) != (ib < 0):
        return -1
    return abs(ia - ib)


def _max_ulp[dtype: DType](name: StaticString) -> Int:
    # Compare approximations with libm, which has its own rounding error.
    # f16/bf16 compute in f32 and allow one ULP of double rounding.
    comptime if dtype in (DType.float32, DType.float64):
        return 2 if name.startswith("atan2") else 1
    return 1


def _check[
    dtype: DType
](
    name: StaticString,
    got: Scalar[dtype],
    expected: Scalar[dtype],
    y: Float64,
    x: Float64,
    mut worst: Int,
) raises where dtype.is_floating_point():
    var d = _ulp_diff(got, expected)
    if d < 0 or d > _max_ulp[dtype](name):
        raise Error(
            t"{name}[{dtype}]({y}, {x}): got {got}, expected {expected},"
            t" ulp={d}"
        )
    worst = max(worst, d)


def _specials() -> List[Float64]:
    var v: List[Float64] = [
        0.0,
        1.0,
        0.5,
        2.0,
        0.4142135623730950,
        0.41421357,
        0.66,
        2.414213562373095,
        2.4142137,
        1e-5,
        1e-30,
        1e30,
        1e300,
        5e-324,
        1.401298464324817e-45,
        6.0e-8,
        65504.0,
        3.4028234663852886e38,
        1.7976931348623157e308,
        inf[DType.float64](),
    ]
    # The atan reduction breakpoints 7/16, 11/16, 19/16, 39/16, and the old
    # Cephes worst case near 0.432, with their float32 and float64 neighbors.
    var edges: List[Float64] = [
        0.4375,
        0.6875,
        1.1875,
        2.4375,
        0.43206868,
        0.44,
    ]
    for e in edges:
        var f = e.cast[.float32]()
        v.append(nextafter(f, 0.0).cast[.float64]())
        v.append(nextafter(f, 4.0).cast[.float64]())
        v.append(nextafter(e, 0.0))
        v.append(nextafter(e, 4.0))
        v.append(e)
    var out = List[Float64]()
    for x in v:
        out.append(x)
        out.append(-x)
    out.append(nan[DType.float64]())
    return out^


def _atan_inputs() -> List[Float64]:
    comptime num_sweep = 8192
    var out = _specials()
    # Linear sweep across the reduction splits.
    for i in range(num_sweep):
        out.append(-8.0 + 16.0 * Float64(i) / num_sweep)
    # Logarithmic sweep across magnitudes, 1e-40 to 1e40.
    for i in range(num_sweep):
        var e = -40.0 + 80.0 * Float64(i) / num_sweep
        var m = exp2(e * log2_10)
        out.append(m)
        out.append(-m)
    return out^


def _atan2_inputs() -> Tuple[List[Float64], List[Float64]]:
    comptime num_angles = 2048
    comptime num_aspect_ratios = 512
    var ys = List[Float64]()
    var xs = List[Float64]()
    var sp = _specials()
    for a in sp:
        for b in sp:
            ys.append(a)
            xs.append(b)
    # Points on circles of varied radius, densely in angle.
    var radii: List[Float64] = [1e-30, 1e-6, 0.3, 1.0, 7.5, 1e5, 1e30]
    for r in radii:
        for i in range(num_angles):
            var th = -pi + 2.0 * pi * (Float64(i) + 0.5) / num_angles
            ys.append(r * sin(th))
            xs.append(r * cos(th))
    # Former worst cases, where the ratio lands near an atan breakpoint.
    var worst: List[Float64] = [
        3.0178835,
        6.7271781,
        203.92819,
        149.56111,
        -3.8479858657274626,
        3.695254715704384,
        1e-310,
        3e-310,
    ]
    for i in range(0, len(worst), 2):
        ys.append(worst[i])
        xs.append(worst[i + 1])
        ys.append(worst[i + 1])
        xs.append(-worst[i])
    # Extreme aspect ratios.
    for i in range(num_aspect_ratios):
        var e = -30.0 + 60.0 * Float64(i) / num_aspect_ratios
        var s = exp2(e * log2_10)
        ys.append(s)
        xs.append(1.0)
        ys.append(-1.0)
        xs.append(-s)
    return (ys^, xs^)


def _check_atan[dtype: DType]() raises where dtype.is_floating_point():
    var worst = 0
    for v in _atan_inputs():
        var x = v.cast[dtype]()
        _check("atan", atan(x), _call_libm["atan"](x), 0.0, v, worst)
    # A SIMD call must agree with the scalar one lane by lane.
    var xs = SIMD[dtype, 4](-3.0, -0.5, 0.25, 7.0)
    var r = atan(xs)
    for i in range(4):
        _check("atan simd", r[i], _call_libm["atan"](xs[i]), 0.0, 0.0, worst)
    print(t"atan {dtype} max ulp vs libm: {worst}")


def _check_atan2[dtype: DType]() raises where dtype.is_floating_point():
    var worst = 0
    var inputs = _atan2_inputs()
    ref ys = inputs[0]
    ref xs = inputs[1]
    for i in range(len(ys)):
        var y = ys[i].cast[dtype]()
        var x = xs[i].cast[dtype]()
        _check("atan2", atan2(y, x), atan2_libm(y, x), ys[i], xs[i], worst)
    print(t"atan2 {dtype} max ulp vs libm: {worst}")


def test_atan() raises:
    _check_atan[.float32]()
    _check_atan[.float64]()
    _check_atan[.float16]()
    _check_atan[.bfloat16]()


def test_atan2() raises:
    _check_atan2[.float32]()
    _check_atan2[.float64]()
    _check_atan2[.float16]()
    _check_atan2[.bfloat16]()


@inline(.never)
def _check_atan2_overflow(y: Float64, x: Float64, expected: Float64) raises:
    # High-precision atan2 rounded to float64; this is a witness, not a
    # universal correctly-rounded contract for the approximation.
    var distance = _ulp_diff(atan2(y, x), expected)
    assert_true(0 <= distance <= 1)


def test_atan2_correction_overflow() raises:
    _check_atan2_overflow(
        Float64(from_bits=UInt64(0x7FEBFA9CF132315E)),
        Float64(from_bits=UInt64(0x7FE2CD1DB8C38B16)),
        Float64(from_bits=UInt64(0x3FEF54FFC9B3006D)),
    )
    _check_atan2_overflow(
        Float64(from_bits=UInt64(0x7FED37A80A3AC399)),
        Float64(from_bits=UInt64(0x7FE3B9CDA8638A28)),
        Float64(from_bits=UInt64(0x3FEF432FF4D1A03B)),
    )


def test_atan2_simd() raises:
    var ys = SIMD[.float32, 8](
        Float32(from_bits=UInt32(3)),
        Float32(from_bits=UInt32(1)),
        -0.0,
        0.0,
        inf[DType.float32](),
        -1.0,
        1.0,
        nan[DType.float32](),
    )
    var xs = SIMD[.float32, 8](
        Float32(from_bits=UInt32(1)),
        Float32(from_bits=UInt32(4)),
        -0.0,
        -1.0,
        inf[DType.float32](),
        1.0,
        -1.0,
        1.0,
    )
    var got = atan2(ys, xs)
    var worst = 0
    for i in range(8):
        _check("atan2 simd", got[i], atan2_libm(ys[i], xs[i]), 0.0, 0.0, worst)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
