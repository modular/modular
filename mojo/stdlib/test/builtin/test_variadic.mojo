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

from builtin.variadics import Variadic, _ReduceValueAndIdxToVariadic
from sys.intrinsics import _type_is_eq
from testing import assert_equal, assert_false, assert_true, TestSuite
from test_utils import ExplicitDelOnly


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
    comptime ReversedVariadic = Variadic.reverse[*type_of(_tup).element_types]
    assert_equal(Variadic.size(_tup.element_types), 0)
    assert_equal(Variadic.size(ReversedVariadic), 0)


def test_variadic_reverse_odd():
    var _tup = (String("hi"), Int(42), Float32(3.14))
    comptime ReversedVariadic = Variadic.reverse[*type_of(_tup).element_types]
    assert_equal(Variadic.size(_tup.element_types), 3)
    assert_equal(Variadic.size(ReversedVariadic), 3)
    assert_true(_type_is_eq[ReversedVariadic[0], Float32]())
    assert_true(_type_is_eq[ReversedVariadic[1], Int]())
    assert_true(_type_is_eq[ReversedVariadic[2], String]())


def test_variadic_reverse_even():
    var _tup = (Int(1), String("a"))
    comptime ReversedVariadic3 = Variadic.reverse[*type_of(_tup).element_types]
    assert_equal(Variadic.size(_tup.element_types), 2)
    assert_equal(Variadic.size(ReversedVariadic3), 2)
    assert_true(_type_is_eq[ReversedVariadic3[0], String]())
    assert_true(_type_is_eq[ReversedVariadic3[1], Int]())


def test_variadic_concat_empty():
    var _tup = ()
    comptime ConcattedVariadic = Variadic.concat[
        type_of(_tup).element_types, type_of(_tup).element_types
    ]
    assert_equal(Variadic.size(_tup.element_types), 0)
    assert_equal(Variadic.size(ConcattedVariadic), 0)


def test_variadic_concat_singleton():
    var _tup = (String("hi"), Int(42), Float32(3.14))
    var _tup2 = (Bool(True),)
    comptime ConcattedVariadic = Variadic.concat[
        type_of(_tup).element_types, type_of(_tup2).element_types
    ]
    assert_equal(Variadic.size(_tup.element_types), 3)
    assert_equal(Variadic.size(ConcattedVariadic), 4)
    assert_true(_type_is_eq[ConcattedVariadic[0], String]())
    assert_true(_type_is_eq[ConcattedVariadic[1], Int]())
    assert_true(_type_is_eq[ConcattedVariadic[2], Float32]())
    assert_true(_type_is_eq[ConcattedVariadic[3], Bool]())


def test_variadic_concat_identity():
    var _tup = (Int(1), String("a"))
    var _tup2 = ()
    comptime ConcattedVariadic = Variadic.concat[
        type_of(_tup).element_types, type_of(_tup2).element_types
    ]
    assert_equal(Variadic.size(_tup.element_types), 2)
    assert_equal(Variadic.size(ConcattedVariadic), 2)
    assert_true(_type_is_eq[ConcattedVariadic[0], Int]())
    assert_true(_type_is_eq[ConcattedVariadic[1], String]())


trait HasStaticValue:
    comptime STATIC_VALUE: Int


@fieldwise_init
struct WithValue[value: Int](HasStaticValue, ImplicitlyCopyable):
    comptime STATIC_VALUE = Self.value


comptime _IntToWithValueMapper[
    Prev: Variadic.TypesOfTrait[HasStaticValue],
    From: Variadic.ValuesOfType[Int],
    idx: Int,
] = Variadic.concat[Prev, Variadic.types[WithValue[From[idx]]]]

comptime IntToWithValue[*values: Int] = _ReduceValueAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[HasStaticValue],
    VariadicType=values,
    Reducer=_IntToWithValueMapper,
]


def test_variadic_value_reducer():
    comptime mapped_values = IntToWithValue[1, 2, 3]
    assert_true(_type_is_eq[mapped_values[0], WithValue[1]]())
    assert_true(_type_is_eq[mapped_values[1], WithValue[2]]())
    assert_true(_type_is_eq[mapped_values[2], WithValue[3]]())
    assert_equal(Variadic.size(mapped_values), 3)


def test_variadic_value_reducer_empty():
    comptime mapped_values = IntToWithValue[*Variadic.empty_of_type[Int]]
    assert_equal(Variadic.size(mapped_values), 0)


def test_variadic_splatted():
    comptime splatted_variadic = Variadic.splat[String, 3]
    assert_equal(Variadic.size(splatted_variadic), 3)
    assert_true(_type_is_eq[splatted_variadic[0], String]())
    assert_true(_type_is_eq[splatted_variadic[1], String]())
    assert_true(_type_is_eq[splatted_variadic[2], String]())


def test_variadic_splatted_zero():
    comptime splatted_variadic = Variadic.splat[Float64, 0]
    assert_equal(Variadic.size(splatted_variadic), 0)


def test_variadic_contains():
    comptime variadic = Variadic.types[T=Writable, Int, String, Float32]
    assert_equal(Variadic.size(variadic), 3)
    comptime ContainsWritable = Variadic.contains[Trait=Writable]
    assert_true(ContainsWritable[Int, variadic])
    assert_true(ContainsWritable[String, variadic])
    assert_true(ContainsWritable[Float32, variadic])
    assert_false(ContainsWritable[Bool, variadic])


def test_variadic_contains_empty():
    comptime variadic = Variadic.types[
        T=Writable, *Variadic.empty_of_trait[Writable]
    ]
    assert_equal(Variadic.size(variadic), 0)
    comptime ContainsWritable = Variadic.contains[Trait=Writable]
    assert_false(ContainsWritable[Bool, variadic])


def test_zip_types_empty():
    comptime v1 = Variadic.empty_of_trait[Writable]
    comptime v2 = Variadic.empty_of_trait[Writable]
    comptime v_zip = Variadic.zip_types[v1, v2]
    assert_equal(Variadic.size(v_zip), 1)
    assert_equal(Variadic.size(v_zip[0]), 0)
    assert_equal(Variadic.size(v_zip[1]), 0)


def test_zip_types_uneven():
    comptime v1 = Variadic.types[T=Writable, String, Float32, Bool]
    comptime v2 = Variadic.types[T=Writable, StaticString, Int]
    comptime v_zip = Variadic.zip_types[v1, v2]
    assert_equal(Variadic.size(v_zip), 2)
    assert_true(_type_is_eq[v_zip[0][0], String]())
    assert_true(_type_is_eq[v_zip[0][1], StaticString]())
    assert_true(_type_is_eq[v_zip[1][0], Float32]())
    assert_true(_type_is_eq[v_zip[1][1], Int]())


def test_zip_types():
    comptime v1 = Variadic.types[T=Writable, String, Float32, Bool]
    comptime v2 = Variadic.types[T=Writable, StaticString, Int, Float64]
    comptime v_zip = Variadic.zip_types[v1, v2]
    assert_equal(Variadic.size(v_zip), 3)
    assert_true(_type_is_eq[v_zip[0][0], String]())
    assert_true(_type_is_eq[v_zip[0][1], StaticString]())
    assert_true(_type_is_eq[v_zip[1][0], Float32]())
    assert_true(_type_is_eq[v_zip[1][1], Int]())
    assert_true(_type_is_eq[v_zip[2][0], Bool]())
    assert_true(_type_is_eq[v_zip[2][1], Float64]())


def test_zip_types_triple():
    comptime v1 = Variadic.types[T=Writable, String, Float32, Bool]
    comptime v2 = Variadic.types[T=Writable, StaticString, Int, Float64]
    comptime v3 = Variadic.types[T=Writable, UInt8, UInt32, UInt64]
    comptime v_zip = Variadic.zip_types[v1, v2, v3]
    assert_equal(Variadic.size(v_zip), 3)
    assert_true(_type_is_eq[v_zip[0][0], String]())
    assert_true(_type_is_eq[v_zip[0][1], StaticString]())
    assert_true(_type_is_eq[v_zip[0][2], UInt8]())
    assert_true(_type_is_eq[v_zip[1][0], Float32]())
    assert_true(_type_is_eq[v_zip[1][1], Int]())
    assert_true(_type_is_eq[v_zip[1][2], UInt32]())
    assert_true(_type_is_eq[v_zip[2][0], Bool]())
    assert_true(_type_is_eq[v_zip[2][1], Float64]())
    assert_true(_type_is_eq[v_zip[2][2], UInt64]())


def test_zip_values_empty():
    comptime v1 = Variadic.empty_of_type[Int]
    comptime v2 = Variadic.empty_of_type[Int]
    comptime v_zip = Variadic.zip_values[v1, v2]
    assert_equal(Variadic.size(v_zip), 1)
    assert_equal(Variadic.size(v_zip[0]), 0)
    assert_equal(Variadic.size(v_zip[1]), 0)


def test_zip_values_uneven():
    comptime v1 = Variadic.values[1, 2, 3]
    comptime v2 = Variadic.values[4, 5]
    comptime v_zip = Variadic.zip_values[v1, v2]
    assert_equal(Variadic.size(v_zip), 2)
    assert_equal(v_zip[0][0], 1)
    assert_equal(v_zip[0][1], 4)
    assert_equal(v_zip[1][0], 2)
    assert_equal(v_zip[1][1], 5)


def test_zip_values():
    comptime v1 = Variadic.values[1, 2, 3]
    comptime v2 = Variadic.values[4, 5, 6]
    comptime v_zip = Variadic.zip_values[v1, v2]
    assert_equal(Variadic.size(v_zip), 3)
    assert_equal(v_zip[0][0], 1)
    assert_equal(v_zip[0][1], 4)
    assert_equal(v_zip[1][0], 2)
    assert_equal(v_zip[1][1], 5)
    assert_equal(v_zip[2][0], 3)
    assert_equal(v_zip[2][1], 6)


def test_zip_values_triple():
    comptime v1 = Variadic.values[1, 2, 3]
    comptime v2 = Variadic.values[4, 5, 6]
    comptime v3 = Variadic.values[7, 8, 9]
    comptime v_zip = Variadic.zip_values[v1, v2, v3]
    assert_equal(Variadic.size(v_zip), 3)
    assert_equal(v_zip[0][0], 1)
    assert_equal(v_zip[0][1], 4)
    assert_equal(v_zip[0][2], 7)
    assert_equal(v_zip[1][0], 2)
    assert_equal(v_zip[1][1], 5)
    assert_equal(v_zip[1][2], 8)
    assert_equal(v_zip[2][0], 3)
    assert_equal(v_zip[2][1], 6)
    assert_equal(v_zip[2][2], 9)


def test_slice_types_empty():
    comptime variadic = Variadic.slice_types[
        Variadic.empty_of_trait[Writable], start=0, end=0
    ]
    assert_equal(Variadic.size(variadic), 0)


def test_slice_types():
    comptime variadic = Variadic.slice_types[
        Variadic.types[T=AnyType, Int, String, Float32], start=0, end=2
    ]
    assert_equal(Variadic.size(variadic), 2)
    assert_true(_type_is_eq[variadic[0], Int]())
    assert_true(_type_is_eq[variadic[1], String]())


def test_map_types_to_types_empty():
    comptime mapper[T: AnyType] = Int
    comptime variadic = Variadic.map_types_to_types[
        Variadic.empty_of_trait[AnyType], mapper
    ]
    assert_equal(Variadic.size(variadic), 0)


trait TestErrable:
    comptime ErrorType: AnyType


struct Foo(TestErrable):
    comptime ErrorType = Int


struct Baz(TestErrable):
    comptime ErrorType = String


def test_map_types_to_types():
    comptime Mapper[T: TestErrable] = T.ErrorType
    comptime variadic = Variadic.map_types_to_types[
        Variadic.types[T=TestErrable, Foo, Baz], Mapper
    ]
    assert_equal(Variadic.size(variadic), 2)
    assert_true(_type_is_eq[variadic[0], Int]())
    assert_true(_type_is_eq[variadic[1], String]())


def test_variadic_list_linear_type():
    """Test owned variadics with a linear type (ExplicitDelOnly)."""

    @parameter
    fn destroy_elem(_idx: Int, var arg: ExplicitDelOnly):
        arg^.destroy()

    fn take_owned_linear(var *args: ExplicitDelOnly):
        args^.consume_elements[destroy_elem]()

    take_owned_linear(ExplicitDelOnly(5), ExplicitDelOnly(10))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
