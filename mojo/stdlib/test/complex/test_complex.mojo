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
# RUN: %mojo-no-debug %s

import math
from complex import ComplexFloat32, ComplexFloat64, ComplexSIMD, abs
from testing import assert_almost_equal, assert_equal


def test_init():
    var x = ComplexFloat32(1, 2)
    assert_equal(x.re, 1)
    assert_equal(x.im, 2)

    var y = ComplexFloat32(3)
    assert_equal(y.re, 3)
    assert_equal(y.im, 0)

    var z = ComplexSIMD[DType.float32, 2](SIMD[DType.float32, 2](1, 2))
    assert_equal(z.re, SIMD[DType.float32, 2](1, 2))
    assert_equal(z.im, SIMD[DType.float32, 2](0, 0))


def test_math():
    var x = ComplexFloat32(1, 2)
    var y = ComplexFloat32(3, 4)
    var z = x + y
    assert_equal(z.re, 4)
    assert_equal(z.im, 6)

    var w = x - y
    assert_equal(w.re, -2)
    assert_equal(w.im, -2)

    var v = x * y
    assert_equal(v.re, -5)
    assert_equal(v.im, 10)

    var u = x / y
    assert_almost_equal(u.re, 0.44)
    assert_almost_equal(u.im, -0.08)


def test_abs():
    assert_equal(abs(ComplexFloat32(0, 0)), 0)
    assert_equal(abs(ComplexFloat32(1, 0)), 1)
    assert_equal(abs(ComplexFloat32(0, 1)), 1)
    assert_almost_equal(abs(ComplexFloat32(-1, -1)), 1.41421)
    assert_almost_equal(abs(ComplexFloat32(-93, -23)), 95.801)


def test_complex_str():
    assert_equal(String(ComplexFloat32(0, 0)), "0.0")
    assert_equal(String(ComplexFloat32(1, 0)), "1.0")
    assert_equal(String(ComplexFloat32(0, 1)), "0.0 + 1.0i")
    assert_equal(String(ComplexFloat32(1, 1)), "1.0 + 1.0i")

    assert_equal(
        String(
            ComplexSIMD[DType.float32, 2](
                SIMD[DType.float32, 2](1, 0),
                SIMD[DType.float32, 2](0, 1),
            )
        ),
        "[1.0, 0.0 + 1.0i]",
    )


def test_fma():
    var x = ComplexFloat32(17, 31)
    var y = ComplexFloat32(42, 1337)
    var c = ComplexFloat32(13, 37)
    var res1 = x * y + c
    var res2 = x.fma(y, c)
    assert_almost_equal(res1.re, res2.re)
    assert_almost_equal(res1.im, res2.im)


def test_exp():
    var a = math.exp(ComplexFloat32(1, 2))
    assert_almost_equal(a.re, -1.1312)
    assert_almost_equal(a.im, 2.47173)

    var b = math.exp(ComplexFloat32(0, 0))
    assert_equal(b.re, 1)
    assert_equal(b.im, 0)

    var c = math.exp(ComplexFloat64(0, math.pi))
    assert_almost_equal(c.re, -1)
    assert_almost_equal(c.im, 0)


def main():
    test_init()
    test_math()
    test_abs()
    test_complex_str()
    test_fma()
    test_exp()
