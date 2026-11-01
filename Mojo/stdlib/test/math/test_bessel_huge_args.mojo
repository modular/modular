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

# Inputs whose phase `x - pi/4` lies within 2^-25 .. 2^-61 of a multiple of
# pi/2, so one of the Bessel asymptotic terms nearly vanishes. Reducing `x`
# and subtracting a double-word pi/4 afterwards lost up to 62 ulp (float64)
# and 46 ulp (float32) here; folding the shift into the reduction keeps the
# result within an ulp. References are correctly rounded (MPFR via mpmath).

from std.math import j0, j1, y0, y1
from std.testing import TestSuite, assert_true

from test_utils.math import ulp_distance

comptime f32_max_ulp = 2


def _check[
    dtype: DType, //, max_ulp: Int
](
    xs: List[Scalar[dtype]], want: List[Scalar[dtype]], name: String
) raises where dtype.is_floating_point():
    for i in range(len(xs)):
        var got: Scalar[dtype]
        if name == "j0":
            got = j0(xs[i])
        elif name == "y0":
            got = y0(xs[i])
        elif name == "j1":
            got = j1(xs[i])
        else:
            got = y1(xs[i])
        var d = ulp_distance(got, want[i])
        assert_true(
            d <= max_ulp,
            String(t"{name}({xs[i]}) = {got}, expected {want[i]} ({d} ulp)"),
        )


def test_f64_phase() raises:
    comptime f64_max_ulp = 1
    # The closest a float64 comes to an odd multiple of pi/4 (half the worst
    # case of pi/2 reduction, 6381956970095103 * 2^796).
    var xs: List[Float64] = [2.6596863241632707e255]
    _check[f64_max_ulp](xs, [-1.5471230315595117e-128], "j0")
    _check[f64_max_ulp](xs, [-3.62581117707763e-147], "y0")
    _check[f64_max_ulp](xs, [-3.62581117707763e-147], "j1")
    _check[f64_max_ulp](xs, [1.5471230315595117e-128], "y1")


def test_f32_phase() raises:
    # The float32 values closest to odd multiples of pi/4 in the Cody-Waite
    # and Payne-Hanek ranges of the reduction.
    var xs: List[Float32] = [
        126.44910430908203,
        2.356194496154785,
        7.0685834884643555,
        1119.1923828125,
        11.780972480773926,
        3.864589459726451e28,
        10999692288.0,
        2.319091691391733e25,
        761394495488.0,
        26258.216796875,
        1354837.75,
    ]
    _check[f32_max_ulp](
        xs,
        [
            0.07095455378293991,
            0.025495409965515137,
            0.2996954321861267,
            0.023849960416555405,
            -0.002456285525113344,
            -4.058713373798328e-15,
            7.607637144246837e-06,
            1.6568419813206559e-13,
            -3.162302429125601e-15,
            0.004923879634588957,
            0.0006854825769551098,
        ],
        "j0",
    )
    _check[f32_max_ulp](
        xs,
        [
            -7.013900903984904e-05,
            0.5143632888793945,
            -0.005248181521892548,
            -2.6644340778148035e-06,
            -0.23234421014785767,
            -3.276943780290961e-24,
            7.569287163472927e-15,
            5.358108865942336e-22,
            -9.14397844553605e-07,
            -2.3399774562449238e-08,
            -5.669753208592532e-11,
        ],
        "y0",
    )
    _check[f32_max_ulp](
        xs,
        [
            0.0002104217273881659,
            0.5292403101921082,
            0.0158381387591362,
            7.990550329850521e-06,
            -0.23265525698661804,
            -3.276943780290961e-24,
            7.915098528584862e-15,
            5.358108865942336e-22,
            -9.14397844553605e-07,
            7.035906435248762e-08,
            1.9627835468050137e-10,
        ],
        "j1",
    )
    _check[f32_max_ulp](
        xs,
        [
            -0.07095538824796677,
            0.07969999313354492,
            -0.3007933497428894,
            -0.023849964141845703,
            -0.007385152857750654,
            4.058713373798328e-15,
            -7.607637144246837e-06,
            -1.6568419813206559e-13,
            3.1617018827659977e-15,
            -0.004923879634588957,
            -0.0006854825769551098,
        ],
        "y1",
    )


def test_f32_phase_near_j0_zero() raises:
    # x - pi/4 is within 2^-27 of 3pi/2 and j0's zero sits 1/(8x) beyond that,
    # so j0 cancels to 1e-11 of its envelope: the P cos - Q sin combination
    # cannot resolve it in float32 and j0 is checked against the envelope.
    # The other three functions have no such cancellation and stay exact.
    var xs: List[Float32] = [21390802.0]
    _check[f32_max_ulp](xs, [-0.00017251486133318394], "y0")
    _check[f32_max_ulp](xs, [-0.00017251486133318394], "j1")
    _check[f32_max_ulp](xs, [-4.034744326836259e-12], "y1")
    var j = j0(xs[0])
    assert_true(
        abs(j - 2.2899446790560548e-15) < 1.0e-18,
        String(t"j0({xs[0]}) = {j} is off the envelope"),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
