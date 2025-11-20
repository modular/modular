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

from builtin.variadics import (
    VariadicOf,
    Reversed,
    Concatenated,
    variadic_size,
)
from sys.intrinsics import _type_is_eq
from testing import assert_equal, assert_false, assert_true, TestSuite


fn test_variadic_iterator() raises:
    fn helper(*args: Int) raises:
        var n = 5
        var count = 0

        for i, e in enumerate(args):
            assert_equal(e, n)
            assert_equal(i, count)
            count += 1
            n -= 1

    helper(5, 4, 3, 2, 1)


def test_variadic_reverse_empty():
    var _tup = ()
    comptime ReversedVariadic = Reversed[*type_of(_tup).element_types]
    assert_equal(variadic_size(_tup.element_types), 0)
    assert_equal(variadic_size(ReversedVariadic), 0)


def test_variadic_reverse_odd():
    var _tup = (String("hi"), Int(42), Float32(3.14))
    comptime ReversedVariadic = Reversed[*type_of(_tup).element_types]
    assert_equal(variadic_size(_tup.element_types), 3)
    assert_equal(variadic_size(ReversedVariadic), 3)
    assert_true(_type_is_eq[ReversedVariadic[0], Float32]())
    assert_true(_type_is_eq[ReversedVariadic[1], Int]())
    assert_true(_type_is_eq[ReversedVariadic[2], String]())


def test_variadic_reverse_even():
    var _tup = (Int(1), String("a"))
    comptime ReversedVariadic3 = Reversed[*type_of(_tup).element_types]
    assert_equal(variadic_size(_tup.element_types), 2)
    assert_equal(variadic_size(ReversedVariadic3), 2)
    assert_true(_type_is_eq[ReversedVariadic3[0], String]())
    assert_true(_type_is_eq[ReversedVariadic3[1], Int]())


def test_variadic_concat_empty():
    var _tup = ()
    comptime ConcattedVariadic = Concatenated[
        type_of(_tup).element_types, type_of(_tup).element_types
    ]
    assert_equal(variadic_size(_tup.element_types), 0)
    assert_equal(variadic_size(ConcattedVariadic), 0)


def test_variadic_concat_singleton():
    var _tup = (String("hi"), Int(42), Float32(3.14))
    var _tup2 = (Bool(True),)
    comptime ConcattedVariadic = Concatenated[
        type_of(_tup).element_types, type_of(_tup2).element_types
    ]
    assert_equal(variadic_size(_tup.element_types), 3)
    assert_equal(variadic_size(ConcattedVariadic), 4)
    assert_true(_type_is_eq[ConcattedVariadic[0], String]())
    assert_true(_type_is_eq[ConcattedVariadic[1], Int]())
    assert_true(_type_is_eq[ConcattedVariadic[2], Float32]())
    assert_true(_type_is_eq[ConcattedVariadic[3], Bool]())


def test_variadic_concat_identity():
    var _tup = (Int(1), String("a"))
    var _tup2 = ()
    comptime ConcattedVariadic = Concatenated[
        type_of(_tup).element_types, type_of(_tup2).element_types
    ]
    assert_equal(variadic_size(_tup.element_types), 2)
    assert_equal(variadic_size(ConcattedVariadic), 2)
    assert_true(_type_is_eq[ConcattedVariadic[0], Int]())
    assert_true(_type_is_eq[ConcattedVariadic[1], String]())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
