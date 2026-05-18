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

from std.builtin.simd import FastMathFlag
from std.compile import compile_info
from std.testing import TestSuite, assert_false, assert_true
from test_utils import check_write_to


def test_simd_fma_fastmath() raises:
    def my_fma(a: Float32, b: Float32, c: Float32) -> Float32:
        return a.fma[FastMathFlag.FAST](c, b)

    var asm = compile_info[my_fma, emission_kind="llvm"]()

    assert_true(" call fast float @llvm.fma.f32" in asm)


def test_fast_math_flag_write_to() raises:
    check_write_to(FastMathFlag.NONE, expected="NONE", is_repr=False)
    check_write_to(FastMathFlag.NNAN, expected="NNAN", is_repr=False)
    check_write_to(FastMathFlag.NINF, expected="NINF", is_repr=False)
    check_write_to(FastMathFlag.NSZ, expected="NSZ", is_repr=False)
    check_write_to(FastMathFlag.ARCP, expected="ARCP", is_repr=False)
    check_write_to(FastMathFlag.CONTRACT, expected="CONTRACT", is_repr=False)
    check_write_to(FastMathFlag.AFN, expected="AFN", is_repr=False)
    check_write_to(FastMathFlag.REASSOC, expected="REASSOC", is_repr=False)
    check_write_to(FastMathFlag.FAST, expected="FAST", is_repr=False)


def test_fast_math_flag_write_repr_to() raises:
    check_write_to(
        FastMathFlag.NONE, expected="FastMathFlag.NONE", is_repr=True
    )
    check_write_to(
        FastMathFlag.NNAN, expected="FastMathFlag.NNAN", is_repr=True
    )
    check_write_to(
        FastMathFlag.NINF, expected="FastMathFlag.NINF", is_repr=True
    )
    check_write_to(FastMathFlag.NSZ, expected="FastMathFlag.NSZ", is_repr=True)
    check_write_to(
        FastMathFlag.ARCP, expected="FastMathFlag.ARCP", is_repr=True
    )
    check_write_to(
        FastMathFlag.CONTRACT, expected="FastMathFlag.CONTRACT", is_repr=True
    )
    check_write_to(FastMathFlag.AFN, expected="FastMathFlag.AFN", is_repr=True)
    check_write_to(
        FastMathFlag.REASSOC, expected="FastMathFlag.REASSOC", is_repr=True
    )
    check_write_to(
        FastMathFlag.FAST, expected="FastMathFlag.FAST", is_repr=True
    )


def main() raises:
    var suite = TestSuite()

    suite.test[test_simd_fma_fastmath]()
    suite.test[test_fast_math_flag_write_to]()
    suite.test[test_fast_math_flag_write_repr_to]()

    suite^.run()
