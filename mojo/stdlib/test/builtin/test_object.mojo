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

from random import random_float64

from testing import assert_equal, assert_false, assert_raises, assert_true


def test_object_ctors():
    a = object()
    assert_true(a._value.is_none())
    a = 5
    assert_true(a._value.is_int())
    assert_equal(a._value.get_as_int(), 5)
    a = 6.5
    assert_true(a._value.is_float())
    assert_equal(a._value.get_as_float(), 6.5)
    a = False
    assert_true(a._value.is_bool())
    assert_false(a._value.get_as_bool())

    a = "foobar"
    assert_true(a)
    a = ""
    assert_false(a)
    a = []
    assert_false(a)
    a = [1, 2]
    assert_true(a)
    b = object([2, 4])
    assert_true(a < b)


def test_comparison_ops():
    assert_true(object(False) < True)
    assert_false(object(True) < True)
    assert_true(object(1) > False)
    assert_true(object(2) == 2)
    assert_false(object(True) != 1)
    assert_true(object(True) <= 1.0)
    assert_false(object(False) >= 0.5)

    lhs = object("aaa")
    rhs = object("bbb")
    assert_false(lhs == rhs)
    assert_true(lhs != rhs)
    assert_true(lhs < rhs)
    assert_true(lhs <= rhs)
    assert_false(lhs > rhs)
    assert_false(lhs >= rhs)


def test_arithmetic_ops():
    a = object(False)
    a += True
    assert_true(a == True)

    a = object(1)
    a -= 5.5
    assert_true(a == -4.5)

    a = object(2.5)
    a *= 2

    assert_true(a == 5)
    assert_true(-object(True) == -1)
    assert_true(~object(5) == -6)
    assert_true((object(True) + True) == 2)
    assert_true(5 - object(6) == -1)

    assert_false(object(False) and True)
    assert_true(object(False) or True)

    assert_true(object(5) ** 2 == 25)
    assert_true(5 ** object(2) == 25)
    assert_true(object(5) ** object(2) == 25)
    assert_true(object(4.5) ** 2 == 20.25)

    a = 5
    a **= 2
    assert_true(a == 25)

    lhs = object("foo")
    rhs = object("bar")
    concatted = lhs + rhs
    lhs += rhs
    assert_true(lhs == concatted)


def test_arithmetic_ops_div():
    # test mod
    lhs = object(5.5)
    rhs = object(2.0)
    assert_true((lhs % rhs) == 1.5)
    lhs %= rhs
    assert_true(lhs == 1.5)
    assert_true(5.5 % object(2.0) == 1.5)

    lhs = object(5)
    rhs = object(2)
    assert_true((lhs % rhs) == 1)
    lhs %= rhs
    assert_true(lhs == 1)
    assert_true(5 % object(2) == 1)

    # truediv
    lhs = object(5.5)
    rhs = object(2.0)
    assert_true(lhs / rhs == 2.75)
    lhs /= rhs
    assert_true(lhs == 2.75)
    assert_true(5.5 / object(2.0) == 2.75)

    lhs = object(5)
    rhs = object(2)
    assert_true(lhs / rhs == 2)
    lhs /= rhs
    assert_true(lhs == 2)
    assert_true(5 / object(2) == 2)

    # floor div
    lhs = object(5.5)
    rhs = object(2.0)
    assert_true(lhs // rhs == 2)
    lhs //= rhs
    assert_true(lhs == 2)
    assert_true(5.5 // object(2.0) == 2)

    lhs = object(5)
    rhs = object(2)
    assert_true(lhs // rhs == 2)
    lhs //= rhs
    assert_true(lhs == 2)
    assert_true(5 // object(2) == 2)


def test_object_bitwise():
    a = object(1)
    b = object(2)
    assert_true(a << b == 4)
    assert_true(b >> a == 1)

    b <<= a
    assert_true(b == 4)
    b >>= a
    assert_true(b == 2)

    assert_true(2 << object(1) == 4)
    assert_true(2 >> object(1) == 1)

    assert_true(object(15) & object(7) == 7)
    assert_true(object(15) | object(7) == 15)
    assert_true(object(15) ^ object(7) == 8)

    a = object(15)
    b = object(7)
    a &= b
    assert_true(a == 7)
    a = object(15)
    a |= b
    assert_true(a == 15)
    a = object(15)
    a ^= b
    assert_true(a == 8)

    assert_true(15 & object(7) == 7)
    assert_true(15 | object(7) == 15)
    assert_true(15 ^ object(7) == 8)


def test_function(lhs: object, rhs: object) -> object:
    return lhs + rhs


# These are all marked read-only because 'object' doesn't support function
# types with owned arguments.
def test_function_raises(a: object) -> object:
    raise Error("Error from function type")


def test_object_function():
    var a: object = test_function
    assert_true(String(a).startswith("Function at address 0x"))
    assert_equal(String(a(1, 2)), String(3))
    a = test_function_raises
    with assert_raises(contains="Error from function type"):
        a(1)


def test_non_object_getattr():
    var a: object = [2, 3, 4]
    with assert_raises(contains="Type 'list' does not have attribute 'foo'"):
        a.foo(2)


def matrix_getitem(self: object, i: object) -> object:
    return self.value[i]


def matrix_setitem(self: object, i: object, value: object) -> object:
    self.value[i] = value
    return None


def matrix_append(self: object, value: object) -> object:
    var impl = self.value
    impl.append(value)
    return None


def matrix_init(rows: Int, cols: Int) -> object:
    value = object([])
    return object(
        Attr("value", value),
        Attr("__getitem__", matrix_getitem),
        Attr("__setitem__", matrix_setitem),
        Attr("rows", rows),
        Attr("cols", cols),
        Attr("append", matrix_append),
    )


def matmul_untyped(C: object, A: object, B: object):
    for m in range(C.rows):
        for n in range(C.cols):
            for k in range(A.rows):
                C[m, n] += A[m, k] * B[k, n]


def test_matrix():
    alias size = 3
    A = matrix_init(size, size)
    B = matrix_init(size, size)
    C = matrix_init(size, size)
    for i in range(size):
        row = object([])
        row_zero = object([])
        for j in range(size):
            row_zero.append(0)
            row.append(i + j)
        A.append(row)
        B.append(row)
        C.append(row_zero)

    matmul_untyped(C, A, B)
    assert_equal(String(C[0]), "[5, 8, 11]")
    assert_equal(String(C[1]), "[8, 14, 20]")
    assert_equal(String(C[2]), "[11, 20, 29]")


def test_convert_to_string():
    var a: object = True
    assert_equal(String(a), "True")
    a = 42
    assert_equal(String(a), "42")
    a = 2.5
    assert_equal(String(a), "2.5")
    a = "hello"
    assert_equal(String(a), "'hello'")
    a = []
    assert_equal(String(a), "[]")
    a.append(3)
    a.append(False)
    a.append(5.5)
    var b: object = []
    b.append("foo")
    b.append("baz")
    a.append(b)
    assert_equal(String(a), "[3, False, 5.5, ['foo', 'baz']]")
    assert_equal(String(a[3, 1]), "'baz'")
    a[3, 1] = "bar"
    assert_equal(String(a[3, 1]), "'bar'")
    var c = a + b
    assert_equal(String(c), "[3, False, 5.5, ['foo', 'bar'], 'foo', 'bar']")
    b.append(False)
    assert_equal(
        String(c), "[3, False, 5.5, ['foo', 'bar', False], 'foo', 'bar']"
    )
    assert_equal(String(a), "[3, False, 5.5, ['foo', 'bar', False]]")
    assert_equal(String(c[3]), "['foo', 'bar', False]")
    b[1] = object()
    assert_equal(String(a), "[3, False, 5.5, ['foo', None, False]]")
    a = "abc"
    b = a[True]
    assert_equal(String(b), "'b'")
    b = a[2]
    assert_equal(String(b), "'c'")
    a = [1, 1.2, False, "true"]
    assert_equal(String(a), "[1, 1.2, False, 'true']")

    a = object(Attr("foo", 5), Attr("bar", "hello"), Attr("baz", False))
    assert_equal(String(a.bar), "'hello'")
    a.bar = [1, 2]
    assert_equal(String(a), "{'foo' = 5, 'bar' = [1, 2], 'baz' = False}")
    assert_equal(repr(a), "{'foo' = 5, 'bar' = [1, 2], 'baz' = False}")


def main():
    test_object_ctors()
    test_comparison_ops()
    test_arithmetic_ops()
    test_arithmetic_ops_div()
    test_object_bitwise()
    test_object_function()
    test_non_object_getattr()
    test_matrix()
    test_convert_to_string()
