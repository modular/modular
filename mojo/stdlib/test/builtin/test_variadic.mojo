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
    EmptyVariadic,
    EmptyVariadicValue,
    VariadicOf,
    Reversed,
    Concatenated,
    Contains,
    variadic_size,
    MakeVariadic,
    MakeVariadicValue,
    Variadic,
    VariadicOf,
    Splatted,
    _ReduceValueAndIdxToVariadic,
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


trait HasStaticValue:
    comptime STATIC_VALUE: Int


@fieldwise_init
struct WithValue[value: Int](HasStaticValue, ImplicitlyCopyable, Movable):
    comptime STATIC_VALUE = Self.value


comptime _IntToWithValueMapper[
    Prev: VariadicOf[HasStaticValue],
    From: Variadic[Int],
    idx: Int,
] = Concatenated[Prev, MakeVariadic[WithValue[From[idx]]]]

comptime IntToWithValue[*values: Int] = _ReduceValueAndIdxToVariadic[
    BaseVal = EmptyVariadic[HasStaticValue],
    VariadicType=values,
    Reducer=_IntToWithValueMapper,
]


def test_variadic_value_reducer():
    comptime mapped_values = IntToWithValue[1, 2, 3]
    assert_true(_type_is_eq[mapped_values[0], WithValue[1]]())
    assert_true(_type_is_eq[mapped_values[1], WithValue[2]]())
    assert_true(_type_is_eq[mapped_values[2], WithValue[3]]())
    assert_equal(variadic_size(mapped_values), 3)


def test_variadic_value_reducer_empty():
    comptime mapped_values = IntToWithValue[*EmptyVariadicValue[Int]]
    assert_equal(variadic_size(mapped_values), 0)


def test_variadic_splatted():
    comptime splatted_variadic = Splatted[String, 3]
    assert_equal(variadic_size(splatted_variadic), 3)
    assert_true(_type_is_eq[splatted_variadic[0], String]())
    assert_true(_type_is_eq[splatted_variadic[1], String]())
    assert_true(_type_is_eq[splatted_variadic[2], String]())


def test_variadic_splatted_zero():
    comptime splatted_variadic = Splatted[Float64, 0]
    assert_equal(variadic_size(splatted_variadic), 0)


def test_variadic_contains():
    comptime variadic = MakeVariadic[T=Writable, Int, String, Float32]
    assert_equal(variadic_size(variadic), 3)
    comptime ContainsWritable = Contains[Trait=Writable]
    assert_true(ContainsWritable[Int, variadic])
    assert_true(ContainsWritable[String, variadic])
    assert_true(ContainsWritable[Float32, variadic])
    assert_false(ContainsWritable[Bool, variadic])


def test_variadic_contains_empty():
    comptime variadic = MakeVariadic[T=Writable, *EmptyVariadic[Writable]]
    assert_equal(variadic_size(variadic), 0)
    comptime ContainsWritable = Contains[Trait=Writable]
    assert_false(ContainsWritable[Bool, variadic])


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
