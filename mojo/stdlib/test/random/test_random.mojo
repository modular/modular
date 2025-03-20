# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from random import (
    randn_float64,
    random_float64,
    random_si64,
    random_ui64,
    seed,
    shuffle,
)

from testing import assert_equal, assert_true


def test_random():
    for _ in range(100):
        var random_float = random_float64(0, 1)
        assert_true(
            random_float >= 0,
            String("Value ", random_float, " is not above or equal to 0"),
        )
        assert_true(
            random_float <= 1,
            String("Value ", random_float, " is not below or equal to 1"),
        )

        var random_signed = random_si64(-255, 255)
        assert_true(
            random_signed >= -255,
            String(
                "Signed value ", random_signed, " is not above or equal to -255"
            ),
        )
        assert_true(
            random_signed <= 255,
            String(
                "Signed value ", random_signed, " is not below or equal to 255"
            ),
        )

        var random_unsigned = random_ui64(0, 255)
        assert_true(
            random_unsigned >= 0,
            String(
                "Unsigned value ",
                random_unsigned,
                " is not above or equal to 0",
            ),
        )
        assert_true(
            random_unsigned <= 255,
            String(
                "Unsigned value ",
                random_unsigned,
                " is not below or equal to 255",
            ),
        )


def test_seed_normal():
    seed(42)
    # verify `randn_float64` values are normally distributed
    var num_samples = 1000
    var samples = List[Float64](capacity=num_samples)
    for _ in range(num_samples):
        samples.append(randn_float64(0, 2))

    var sum: Float64 = 0.0
    for sample in samples:
        sum += sample[]

    var mean: Float64 = sum / num_samples

    var sum_sq: Float64 = 0.0
    for sample in samples:
        sum_sq += (sample[] - mean) ** 2

    var variance = sum_sq / num_samples

    # Calculate absolute differences (errors)
    var mean_error = abs(mean)
    var variance_error = abs(variance - 4)

    var mean_tolerance: Float64 = 0.06  # SE_μ = σ / √n
    assert_true(
        mean_error < mean_tolerance,
        String(
            "Mean error ",
            mean_error,
            " is above the accepted tolerance ",
            mean_tolerance,
        ),
    )
    var variance_tolerance: Float64 = 0.57  # SE_S² = √(2 * σ^4 / (n - 1))
    assert_true(
        variance_error < variance_tolerance,
        String(
            "Variance error ",
            variance_error,
            " is above the accepted tolerance ",
            variance_tolerance,
        ),
    )


def test_seed():
    seed(5)
    var some_float = random_float64(0, 1)
    var some_signed_integer = random_si64(-255, 255)
    var some_unsigned_integer = random_ui64(0, 255)

    seed(5)
    assert_equal(some_float, random_float64(0, 1))
    assert_equal(some_signed_integer, random_si64(-255, 255))
    assert_equal(some_unsigned_integer, random_ui64(0, 255))


def test_shuffle():
    # TODO: Clean up with list comprehension when possible.

    # Property tests
    alias L_i = List[Int]
    alias L_s = List[String]
    var a = L_i(1, 2, 3, 4)
    var b = L_i(1, 2, 3, 4)
    var c = L_s("Random", "shuffle", "in", "Mojo")
    var d = L_s("Random", "shuffle", "in", "Mojo")

    shuffle(b)
    assert_equal(len(a), len(b))
    assert_true(a != b)
    for i in range(len(b)):
        assert_true(b[i] in a)

    shuffle(d)
    assert_equal(len(c), len(d))
    assert_true(c != d)
    for i in range(len(d)):
        assert_true(d[i] in c)

    var e = L_i(21)
    shuffle(e)
    assert_true(e == L_i(21))
    var f = L_s("Mojo")
    shuffle(f)
    assert_true(f == L_s("Mojo"))

    alias L_l = List[List[Int]]
    var g = L_l()
    var h = L_l()
    for i in range(10):
        g.append(L_i(i, i + 1, i + 3))
        h.append(L_i(i, i + 1, i + 3))
    shuffle(g)
    # TODO: Uncomment when possible
    # assert_true(g != h)
    assert_equal(len(g), len(h))
    for i in range(10):
        # Currently, the below does not compile.
        # assert_true(g.__contains__(L_i(i, i + 1, i + 3)))
        var target: List[Int] = L_i(i, i + 1, i + 3)
        var found = False
        for j in range(len(g)):
            if g[j] == target:
                found = True
                break
        assert_true(found)

    alias L_l_s = List[List[String]]
    var i = L_l_s()
    var j = L_l_s()
    for x in range(10):
        i.append(L_s(String(x), String(x + 1), String(x + 3)))
        j.append(L_s(String(x), String(x + 1), String(x + 3)))
    shuffle(i)
    # TODO: Uncomment when possible
    # assert_true(g != h)
    assert_equal(len(i), len(j))
    for x in range(10):
        var target: List[String] = L_s(String(x), String(x + 1), String(x + 3))
        var found = False
        for y in range(len(i)):
            if j[y] == target:
                found = True
                break
        assert_true(found)

    # Given the number of permutations of size 1000 is 1000!,
    # we rely on the assertion that a truly random shuffle should not
    # result in the same order as the to pre-shuffle list with extremely
    # high probability.
    var l = L_i()
    var m = L_i()
    for i in range(1000):
        l.append(i)
        m.append(i)
    shuffle(l)
    assert_equal(len(l), len(m))
    assert_true(l != m)
    shuffle(m)
    assert_equal(len(l), len(m))
    assert_true(l != m)


def main():
    test_random()
    test_seed()
    test_shuffle()
