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
# test_types_and_literals.mojo
# Tests for the Mojo "Types & Literals" cheat-sheet card.
#
# Not tested (compile-error claims or no portable runtime API to assert):
#   - "no implicit numeric conversion" is a compile error, not runtime
#   - \u and \U reject surrogate code points (U+D800 to U+DFFF) at compile time
#   - a float literal never becomes an integer type (compile error)
#   - len(s) on a string and t-string format specifiers are compile errors
#   - Int128 / Int256 are software-emulated (no observable runtime difference)
#   - a shift at or above the bit width has no defined result
from std.collections import Set
from std.math import cos, fma, inf, nan, sin, sqrt
from std.testing import assert_equal, assert_false, assert_true
from std.sys.info import bit_width_of


def test_simd_is_the_foundation() raises:
    var v = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    var d = v * 2.0
    assert_equal(d[1], 4.0)  # every lane scaled
    v[0] = 5.0  # write one lane
    assert_equal(v.reduce_add(), 14.0)  # 5 + 2 + 3 + 4


def test_conversions_explicit() raises:
    var i = 42
    assert_equal(Float64(i), 42.0)
    var g = Float64(i).cast[.int32]()  # between SIMD types
    assert_equal(g, Int32(42))


def test_integer_overflow_wraps() raises:
    # signed overflow wraps (two's complement)
    assert_equal(Int8(127) + 1, Int8(-128))


def test_float_to_int_truncates() raises:
    # truncates toward zero
    assert_equal(Int(Float64(3.9)), 3)


def test_bounds() raises:
    assert_equal(UInt8.MAX, UInt8(255))
    assert_equal(Int8.MIN, Int8(-128))
    assert_equal(Float32.MAX, inf[DType.float32]())  # MAX is inf for floats
    assert_true(Float32.MAX_FINITE < Float32.MAX)
    var n = nan[DType.float64]()
    assert_false(n == n)  # nan never equals itself


def test_contextual_dtype() raises:
    var a = SIMD[.float32, 4](1, 2, 3, 4)  # DType inferred from context
    var b = SIMD[DType.float32, 4](1, 2, 3, 4)
    assert_true(a == b)


def test_literals_wrap_silently() raises:
    var b: Int8 = 300
    var u: UInt8 = -1
    assert_equal(b, 44)
    assert_equal(u, 255)


def test_bit_width() raises:
    # bit_width_of[Int]() from std.sys.info; 64 on this platform
    assert_equal(bit_width_of[Int](), 64)


def test_number_literals() raises:
    assert_equal(0xFF, 255)
    assert_equal(0o52, 42)
    assert_equal(0b1010, 10)
    assert_equal(1_000_000, 1000000)


def test_triple_quote_keeps_layout() raises:
    # newlines AND indentation are part of the string
    var s = """line one
    line two"""
    assert_equal(s, "line one\n    line two")


def test_adjacent_literals_join() raises:
    var same_line = "abcd"
    assert_equal(same_line, "abcd")
    var across_lines = "Content of line 1. Content of line 2."
    assert_equal(across_lines, "Content of line 1. Content of line 2.")


def test_unicode_escapes() raises:
    assert_equal("\u20AC", "€")  # lowercase \u, 4 digits (EURO)
    assert_equal("\U0001F44B", "👋")  # uppercase \U, 8 digits (above U+FFFF)


def test_tstring_interpolates() raises:
    var who = "Mojo"
    assert_equal(String(t"x = {who}"), "x = Mojo")
    assert_equal(String(t"sum = {1 + 2}"), "sum = 3")


def test_raw_tstring() raises:
    var who = "Mojo"
    # raw: backslash stays literal, interpolation still happens
    assert_equal(String(rt"raw\path {who}"), "raw\\path Mojo")


def test_collection_literals() raises:
    # A bracket literal infers a fixed-size Array; annotating gets a List, and
    # only the List grows.
    var fixed = [1, 2, 3]
    assert_equal(len(fixed), 3)
    var nums: List[Int] = [1, 2, 3]
    nums.append(4)
    assert_equal(len(nums), 4)
    var d = {"id": 1, "qty": 9}
    assert_equal(d["qty"], 9)
    var s = {1, 2, 3}  # Set
    assert_equal(len(s), 3)
    assert_true(2 in s)
    var pair = (1, "a", 2.0)
    assert_equal(pair[0], 1)


def test_string_length() raises:
    var s = "héllo"
    assert_equal(s.byte_length(), 6)  # UTF-8 bytes
    assert_equal(len(s.codepoints()), 5)  # Unicode code points
    assert_equal(len(s.graphemes()), 5)  # user-visible characters


def test_worth_knowing() raises:
    # fmt: off
    assert_equal('A', "A")  # no character type: single quotes make a string
    # fmt: on
    assert_equal(ord("A"), 65)
    assert_equal(chr(66), "B")
    var a = -7
    assert_equal(a / 2, -3)  # / truncates toward zero
    assert_equal(a // 2, -4)  # // floors
    assert_equal(7 / 2, 3.5)  # the literal expression stays exact
    var f: Float64 = -7.0
    assert_equal(f / 2, -3.5)  # / on floats is true division
    assert_equal(f // 2, -4.0)  # // floors for floats too


def test_scalar_aliases_cast() raises:
    var float: Float64 = 42.0
    var float32 = float.cast[.float32]()  # cast works on named aliases
    var scalar = Scalar[.float64](float)
    var scalar16 = scalar.cast[.float16]()
    assert_equal(float32, Float32(42.0))
    assert_equal(scalar16, Float16(42.0))


def test_simd_construction() raises:
    var broadcast = SIMD[DType.float64, 2](42.0)  # broadcast all lanes
    var specific = SIMD[DType.float64, 2](1.0, 2.0)
    var zeros = SIMD[DType.float64, 2]()  # zero-initialized
    assert_equal(broadcast[1], 42.0)
    assert_equal(specific[1], 2.0)
    assert_equal(zeros.reduce_add(), 0.0)


def test_element_operations() raises:
    var v = SIMD[DType.int32, 2](6, 3)
    assert_true((v % 4) == SIMD[DType.int32, 2](2, 3))
    assert_true((v // 2) == SIMD[DType.int32, 2](3, 1))
    assert_true((v ^ 1) == SIMD[DType.int32, 2](7, 2))  # ^ is xor here
    assert_true((~v) == SIMD[DType.int32, 2](-7, -4))
    assert_true((v << 1) == SIMD[DType.int32, 2](12, 6))
    var f = SIMD[DType.float32, 2](1, 4)
    assert_true(sqrt(f) == SIMD[DType.float32, 2](1, 2))
    assert_true(fma(f, f, f) == SIMD[DType.float32, 2](2, 20))
    assert_true(sin(Float32(0)) == 0 and cos(Float32(0)) == 1)


def test_vector_operations() raises:
    var v = SIMD[DType.int32, 4](1, 2, 3, 4)
    assert_equal(v.reduce_mul(), 24)
    assert_equal(v.reduce_min(), 1)
    assert_equal(v.reduce_max(), 4)
    assert_true(v.shuffle[3, 2, 1, 0]() == SIMD[DType.int32, 4](4, 3, 2, 1))
    assert_true(v.slice[2]() == SIMD[DType.int32, 2](1, 2))
    assert_equal(len(v.join(v)), 8)
    var first, second = v.split()
    # The halves' width stays the unevaluated `4 // 2`, so compare after a
    # rebind. TODO(#101323): drop the rebind once split() halves unify with
    # width 2.
    assert_true(rebind[SIMD[.int32, 2]](first) == SIMD[.int32, 2](1, 2))
    assert_true(rebind[SIMD[.int32, 2]](second) == SIMD[.int32, 2](3, 4))


def test_optionals() raises:
    var foo: Optional[Int] = None
    var bar: Optional[Int] = 42
    assert_equal(foo.or_else(0), 0)  # default fallback
    assert_equal(bar.or_else(0), 42)
    assert_false(Bool(foo))  # check then access
    assert_true(Bool(bar))
    assert_equal(bar.value(), 42)


def test_empty_collections() raises:
    var list: List[Int] = []
    var dict: Dict[String, Int] = {}
    var set: Set[Int] = {}  # naming Set needs the import
    assert_equal(len(list), 0)
    assert_equal(len(dict), 0)
    assert_equal(len(set), 0)


def test_string_count_methods() raises:
    var text = "café"
    assert_equal(text.byte_length(), 5)  # é is 2 bytes
    assert_equal(text.count_codepoints(), 4)
    assert_equal(text.count_graphemes(), 4)


def main() raises:
    test_simd_is_the_foundation()
    test_conversions_explicit()
    test_integer_overflow_wraps()
    test_float_to_int_truncates()
    test_bounds()
    test_contextual_dtype()
    test_literals_wrap_silently()
    test_bit_width()
    test_number_literals()
    test_triple_quote_keeps_layout()
    test_adjacent_literals_join()
    test_unicode_escapes()
    test_tstring_interpolates()
    test_raw_tstring()
    test_collection_literals()
    test_string_length()
    test_worth_knowing()
    test_scalar_aliases_cast()
    test_simd_construction()
    test_element_operations()
    test_vector_operations()
    test_optionals()
    test_empty_collections()
    test_string_count_methods()
