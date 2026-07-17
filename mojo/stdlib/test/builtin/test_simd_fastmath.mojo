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
from std.math import mul_no_contraction
from std.sys.info import CompilationTarget
from std.testing import TestSuite, assert_equal, assert_false, assert_true


def test_simd_fma_fastmath() raises:
    def my_fma(a: Float32, b: Float32, c: Float32) -> Float32:
        return a.fma[FastMathFlag.FAST](c, b)

    var asm = compile_info[my_fma, emission_kind="llvm"]()

    assert_true(" call fast float @llvm.fma.f32" in asm)


def test_mul_no_contraction_no_fma() raises:
    def my_mul_add(
        a: SIMD[DType.float32, 4],
        b: SIMD[DType.float32, 4],
        c: SIMD[DType.float32, 4],
    ) -> SIMD[DType.float32, 4]:
        return mul_no_contraction(a, b) + c

    var asm = compile_info[my_mul_add, emission_kind="llvm"]()

    comptime if CompilationTarget.has_fma() or CompilationTarget.has_neon():
        assert_true(
            "call asm" in asm,
            "mul_no_contraction should use inline assembly to prevent FMA"
            " contraction",
        )
        assert_false(
            " call float @llvm.fma.f32" in asm,
            "mul_no_contraction + add should not be contracted into FMA",
        )
    else:
        assert_true("fmul" in asm or "mul" in asm)


def test_mul_no_contraction_correctness() raises:
    var a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0)

    var product = mul_no_contraction(a, b)
    assert_equal(product, SIMD[DType.float32, 4](5.0, 12.0, 21.0, 32.0))

    var a64 = SIMD[DType.float64, 2](1.5, 2.5)
    var b64 = SIMD[DType.float64, 2](3.0, 4.0)
    var product64 = mul_no_contraction(a64, b64)
    assert_equal(product64, SIMD[DType.float64, 2](4.5, 10.0))


def test_mul_no_contraction_scalar() raises:
    var a = Float32(3.0)
    var b = Float32(4.0)

    var product = mul_no_contraction(a, b)
    assert_equal(product, Float32(12.0))

    var a64 = Float64(2.5)
    var b64 = Float64(6.0)
    var product64 = mul_no_contraction(a64, b64)
    assert_equal(product64, Float64(15.0))


def main() raises:
    var suite = TestSuite()

    suite.test[test_simd_fma_fastmath]()
    suite.test[test_mul_no_contraction_no_fma]()
    suite.test[test_mul_no_contraction_correctness]()
    suite.test[test_mul_no_contraction_scalar]()

    suite^.run()
