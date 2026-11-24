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

from std.simd import FastMathFlag
from std.compile import compile_info
from std.testing import TestSuite, assert_true
from test_utils import check_write_to


def test_simd_fma_fastmath() raises:
    def my_fma(a: Float32, b: Float32, c: Float32) -> Float32:
        return a.fma[.fast](c, b)

    var asm = compile_info[my_fma, emission_kind="llvm"]()

    assert_true(" call fast float @llvm.fma.f32" in asm)


def test_fast_math_flag_write_to() raises:
    check_write_to(FastMathFlag.none, expected="none", is_repr=False)
    check_write_to(FastMathFlag.nnan, expected="nnan", is_repr=False)
    check_write_to(FastMathFlag.ninf, expected="ninf", is_repr=False)
    check_write_to(FastMathFlag.nsz, expected="nsz", is_repr=False)
    check_write_to(FastMathFlag.arcp, expected="arcp", is_repr=False)
    check_write_to(FastMathFlag.contract, expected="contract", is_repr=False)
    check_write_to(FastMathFlag.afn, expected="afn", is_repr=False)
    check_write_to(FastMathFlag.reassoc, expected="reassoc", is_repr=False)
    check_write_to(FastMathFlag.fast, expected="fast", is_repr=False)


def test_fast_math_flag_write_repr_to() raises:
    check_write_to(
        FastMathFlag.none, expected="FastMathFlag.none", is_repr=True
    )
    check_write_to(
        FastMathFlag.nnan, expected="FastMathFlag.nnan", is_repr=True
    )
    check_write_to(
        FastMathFlag.ninf, expected="FastMathFlag.ninf", is_repr=True
    )
    check_write_to(FastMathFlag.nsz, expected="FastMathFlag.nsz", is_repr=True)
    check_write_to(
        FastMathFlag.arcp, expected="FastMathFlag.arcp", is_repr=True
    )
    check_write_to(
        FastMathFlag.contract, expected="FastMathFlag.contract", is_repr=True
    )
    check_write_to(FastMathFlag.afn, expected="FastMathFlag.afn", is_repr=True)
    check_write_to(
        FastMathFlag.reassoc, expected="FastMathFlag.reassoc", is_repr=True
    )
    check_write_to(
        FastMathFlag.fast, expected="FastMathFlag.fast", is_repr=True
    )


def main() raises:
    var suite = TestSuite()

    suite.test[test_simd_fma_fastmath]()
    suite.test[test_fast_math_flag_write_to]()
    suite.test[test_fast_math_flag_write_repr_to]()

    suite^.run()
