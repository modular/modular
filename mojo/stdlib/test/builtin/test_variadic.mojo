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

from std.builtin.variadics import (
    TypeList,
    Variadic,
    _ReduceValueAndIdxToVariadic,
)
from std.sys.intrinsics import _type_is_eq
from std.testing import assert_equal, assert_false, assert_true, TestSuite
from test_utils import ExplicitDelOnly


def test_variadic_iterator() raises:
    def helper(*args: Int) raises:
        var n = 5
        for e in args:
            assert_equal(e, n)
            n -= 1

    helper(5, 4, 3, 2, 1)


def test_variadic_reverse_empty() raises:
    var _tup = ()
    comptime ReversedVariadic = Variadic.reverse[*type_of(_tup).element_types]
    assert_equal(TypeList[*_tup.element_types].size, 0)
    assert_equal(TypeList[*ReversedVariadic].size, 0)


def test_variadic_reverse_odd() raises:
    var _tup = (String("hi"), Int(42), Float32(3.14))
    comptime ReversedVariadic = Variadic.reverse[*type_of(_tup).element_types]
    assert_equal(TypeList[*_tup.element_types].size, 3)
    assert_equal(TypeList[*ReversedVariadic].size, 3)
    assert_true(_type_is_eq[ReversedVariadic[0], Float32]())
    assert_true(_type_is_eq[ReversedVariadic[1], Int]())
    assert_true(_type_is_eq[ReversedVariadic[2], String]())


def test_variadic_reverse_even() raises:
    var _tup = (Int(1), String("a"))
    comptime ReversedVariadic3 = Variadic.reverse[*type_of(_tup).element_types]
    assert_equal(TypeList[*_tup.element_types].size, 2)
    assert_equal(TypeList[*ReversedVariadic3].size, 2)
    assert_true(_type_is_eq[ReversedVariadic3[0], String]())
    assert_true(_type_is_eq[ReversedVariadic3[1], Int]())


def test_variadic_concat_empty() raises:
    var _tup = ()
    comptime ConcattedVariadic = Variadic.concat_types[
        type_of(_tup).element_types, type_of(_tup).element_types
    ]
    assert_equal(TypeList[*_tup.element_types].size, 0)
    assert_equal(TypeList[*ConcattedVariadic].size, 0)


def test_variadic_concat_singleton() raises:
    var _tup = (String("hi"), Int(42), Float32(3.14))
    var _tup2 = (Bool(True),)
    comptime ConcattedVariadic = Variadic.concat_types[
        type_of(_tup).element_types, type_of(_tup2).element_types
    ]
    assert_equal(TypeList[*_tup.element_types].size, 3)
    assert_equal(TypeList[*ConcattedVariadic].size, 4)
    assert_true(_type_is_eq[ConcattedVariadic[0], String]())
    assert_true(_type_is_eq[ConcattedVariadic[1], Int]())
    assert_true(_type_is_eq[ConcattedVariadic[2], Float32]())
    assert_true(_type_is_eq[ConcattedVariadic[3], Bool]())


def test_variadic_concat_identity() raises:
    var _tup = (Int(1), String("a"))
    var _tup2 = ()
    comptime ConcattedVariadic = Variadic.concat_types[
        type_of(_tup).element_types, type_of(_tup2).element_types
    ]
    assert_equal(TypeList[*_tup.element_types].size, 2)
    assert_equal(TypeList[*ConcattedVariadic].size, 2)
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
] = Variadic.concat_types[Prev, Variadic.types[WithValue[From[idx]]]]

comptime IntToWithValue[*values: Int] = _ReduceValueAndIdxToVariadic[
    BaseVal=Variadic.empty_of_trait[HasStaticValue],
    ParamListType=values.values,
    Reducer=_IntToWithValueMapper,
]


def test_variadic_value_reducer() raises:
    comptime mapped_values = IntToWithValue[1, 2, 3]
    assert_true(_type_is_eq[mapped_values[0], WithValue[1]]())
    assert_true(_type_is_eq[mapped_values[1], WithValue[2]]())
    assert_true(_type_is_eq[mapped_values[2], WithValue[3]]())
    assert_equal(TypeList[*mapped_values].size, 3)


def test_variadic_value_reducer_empty() raises:
    comptime mapped_values = IntToWithValue[
        *ParameterList[Variadic.empty_of_type[Int]]()
    ]
    assert_equal(TypeList[*mapped_values].size, 0)


def test_variadic_splatted() raises:
    comptime splatted_variadic = Variadic.splat_type[3, String]
    assert_equal(TypeList[*splatted_variadic].size, 3)
    assert_true(_type_is_eq[splatted_variadic[0], String]())
    assert_true(_type_is_eq[splatted_variadic[1], String]())
    assert_true(_type_is_eq[splatted_variadic[2], String]())


def test_variadic_splatted_zero() raises:
    comptime splatted_variadic = Variadic.splat_type[0, Float64]
    assert_equal(TypeList[*splatted_variadic].size, 0)


def test_variadic_contains() raises:
    comptime variadic = Variadic.types[T=Writable, Int, String, Float32]
    assert_equal(TypeList[*variadic].size, 3)
    comptime ContainsWritable = Variadic.contains[Trait=Writable, ...]
    assert_true(ContainsWritable[Int, variadic])
    assert_true(ContainsWritable[String, variadic])
    assert_true(ContainsWritable[Float32, variadic])
    assert_false(ContainsWritable[Bool, variadic])


def test_variadic_contains_empty() raises:
    comptime variadic = Variadic.types[
        T=Writable, *Variadic.empty_of_trait[Writable]
    ]
    assert_equal(TypeList[*variadic].size, 0)
    comptime ContainsWritable = Variadic.contains[Trait=Writable, ...]
    assert_false(ContainsWritable[Bool, variadic])


def test_variadic_contains_value() raises:
    comptime variadic = Variadic.values[1, 2, 3]
    assert_equal(ParameterList[variadic].size, 3)
    assert_true(Variadic.contains_value[1, variadic])
    assert_true(Variadic.contains_value[2, variadic])
    assert_true(Variadic.contains_value[3, variadic])
    assert_false(Variadic.contains_value[4, variadic])


def test_variadic_contains_value_empty() raises:
    comptime variadic = Variadic.empty_of_type[Int]
    assert_equal(ParameterList[variadic].size, 0)
    assert_false(Variadic.contains_value[1, variadic])


def test_zip_types_empty() raises:
    comptime v1 = Variadic.empty_of_trait[Writable]
    comptime v2 = Variadic.empty_of_trait[Writable]
    comptime v_zip = Variadic.zip_types[v1, v2]
    assert_equal(ParameterList[v_zip].size, 1)
    assert_equal(TypeList[*v_zip[0]].size, 0)
    assert_equal(TypeList[*v_zip[1]].size, 0)


def test_zip_types_uneven() raises:
    comptime v1 = Variadic.types[T=Writable, String, Float32, Bool]
    comptime v2 = Variadic.types[T=Writable, StaticString, Int]
    comptime v_zip = Variadic.zip_types[v1, v2]
    assert_equal(ParameterList[v_zip].size, 2)
    assert_true(_type_is_eq[v_zip[0][0], String]())
    assert_true(_type_is_eq[v_zip[0][1], StaticString]())
    assert_true(_type_is_eq[v_zip[1][0], Float32]())
    assert_true(_type_is_eq[v_zip[1][1], Int]())


def test_zip_types() raises:
    comptime v1 = Variadic.types[T=Writable, String, Float32, Bool]
    comptime v2 = Variadic.types[T=Writable, StaticString, Int, Float64]
    comptime v_zip = Variadic.zip_types[v1, v2]
    assert_equal(ParameterList[v_zip].size, 3)
    assert_true(_type_is_eq[v_zip[0][0], String]())
    assert_true(_type_is_eq[v_zip[0][1], StaticString]())
    assert_true(_type_is_eq[v_zip[1][0], Float32]())
    assert_true(_type_is_eq[v_zip[1][1], Int]())
    assert_true(_type_is_eq[v_zip[2][0], Bool]())
    assert_true(_type_is_eq[v_zip[2][1], Float64]())


def test_zip_types_triple() raises:
    comptime v1 = Variadic.types[T=Writable, String, Float32, Bool]
    comptime v2 = Variadic.types[T=Writable, StaticString, Int, Float64]
    comptime v3 = Variadic.types[T=Writable, UInt8, UInt32, UInt64]
    comptime v_zip = Variadic.zip_types[v1, v2, v3]
    assert_equal(ParameterList[v_zip].size, 3)
    assert_true(_type_is_eq[v_zip[0][0], String]())
    assert_true(_type_is_eq[v_zip[0][1], StaticString]())
    assert_true(_type_is_eq[v_zip[0][2], UInt8]())
    assert_true(_type_is_eq[v_zip[1][0], Float32]())
    assert_true(_type_is_eq[v_zip[1][1], Int]())
    assert_true(_type_is_eq[v_zip[1][2], UInt32]())
    assert_true(_type_is_eq[v_zip[2][0], Bool]())
    assert_true(_type_is_eq[v_zip[2][1], Float64]())
    assert_true(_type_is_eq[v_zip[2][2], UInt64]())


def test_zip_values_empty() raises:
    comptime v1 = Variadic.empty_of_type[Int]
    comptime v2 = Variadic.empty_of_type[Int]
    comptime v_zip = Variadic.zip_values[v1, v2]
    assert_equal(ParameterList[v_zip].size, 1)
    assert_equal(ParameterList[v_zip[0]].size, 0)
    assert_equal(ParameterList[v_zip[1]].size, 0)


def test_zip_values_uneven() raises:
    comptime v1 = Variadic.values[1, 2, 3]
    comptime v2 = Variadic.values[4, 5]
    comptime v_zip = Variadic.zip_values[v1, v2]
    assert_equal(ParameterList[v_zip].size, 2)
    assert_equal(v_zip[0][0], 1)
    assert_equal(v_zip[0][1], 4)
    assert_equal(v_zip[1][0], 2)
    assert_equal(v_zip[1][1], 5)


def test_zip_values() raises:
    comptime v1 = Variadic.values[1, 2, 3]
    comptime v2 = Variadic.values[4, 5, 6]
    comptime v_zip = Variadic.zip_values[v1, v2]
    assert_equal(ParameterList[v_zip].size, 3)
    assert_equal(v_zip[0][0], 1)
    assert_equal(v_zip[0][1], 4)
    assert_equal(v_zip[1][0], 2)
    assert_equal(v_zip[1][1], 5)
    assert_equal(v_zip[2][0], 3)
    assert_equal(v_zip[2][1], 6)


def test_zip_values_triple() raises:
    comptime v1 = Variadic.values[1, 2, 3]
    comptime v2 = Variadic.values[4, 5, 6]
    comptime v3 = Variadic.values[7, 8, 9]
    comptime v_zip = Variadic.zip_values[v1, v2, v3]
    assert_equal(ParameterList[v_zip].size, 3)
    assert_equal(v_zip[0][0], 1)
    assert_equal(v_zip[0][1], 4)
    assert_equal(v_zip[0][2], 7)
    assert_equal(v_zip[1][0], 2)
    assert_equal(v_zip[1][1], 5)
    assert_equal(v_zip[1][2], 8)
    assert_equal(v_zip[2][0], 3)
    assert_equal(v_zip[2][1], 6)
    assert_equal(v_zip[2][2], 9)


def test_slice_types_empty() raises:
    comptime variadic = Variadic.slice_types[
        Variadic.empty_of_trait[Writable], start=0, end=0
    ]
    assert_equal(TypeList[*variadic].size, 0)


def test_slice_types() raises:
    comptime variadic = Variadic.slice_types[
        Variadic.types[T=AnyType, Int, String, Float32], start=0, end=2
    ]
    assert_equal(TypeList[*variadic].size, 2)
    assert_true(_type_is_eq[variadic[0], Int]())
    assert_true(_type_is_eq[variadic[1], String]())


def test_map_types_to_types_empty() raises:
    comptime mapper[T: AnyType] = Int
    comptime variadic = Variadic.map_types_to_types[
        Variadic.empty_of_trait[AnyType], mapper
    ]
    assert_equal(TypeList[*variadic].size, 0)


trait TestErrable:
    comptime ErrorType: AnyType


struct Foo(TestErrable):
    comptime ErrorType = Int


struct Baz(TestErrable):
    comptime ErrorType = String


def test_map_types_to_types() raises:
    comptime Mapper[T: TestErrable] = T.ErrorType
    comptime variadic = Variadic.map_types_to_types[
        Variadic.types[T=TestErrable, Foo, Baz], Mapper
    ]
    assert_equal(TypeList[*variadic].size, 2)
    assert_true(_type_is_eq[variadic[0], Int]())
    assert_true(_type_is_eq[variadic[1], String]())


def test_filter_types_exclude_one() raises:
    comptime IsNotInt[Type: Movable] = not _type_is_eq[Type, Int]()
    comptime without_int = Variadic.filter_types[
        *Tuple[Int, String, Float64, Bool].element_types, predicate=IsNotInt
    ]
    assert_equal(TypeList[*without_int].size, 3)
    assert_true(_type_is_eq[without_int[0], String]())
    assert_true(_type_is_eq[without_int[1], Float64]())
    assert_true(_type_is_eq[without_int[2], Bool]())


def test_filter_types_keep_only() raises:
    comptime IsStringOrFloat[Type: Movable] = (
        _type_is_eq[Type, String]() or _type_is_eq[Type, Float64]()
    )
    comptime kept = Variadic.filter_types[
        *Tuple[Int, String, Float64, Bool].element_types,
        predicate=IsStringOrFloat,
    ]
    assert_equal(TypeList[*kept].size, 2)
    assert_true(_type_is_eq[kept[0], String]())
    assert_true(_type_is_eq[kept[1], Float64]())


def test_filter_types_exclude_many() raises:
    comptime NotIntOrBool[Type: Movable] = (
        not _type_is_eq[Type, Int]() and not _type_is_eq[Type, Bool]()
    )
    comptime filtered = Variadic.filter_types[
        *Tuple[Int, String, Float64, Bool].element_types,
        predicate=NotIntOrBool,
    ]
    assert_equal(TypeList[*filtered].size, 2)
    assert_true(_type_is_eq[filtered[0], String]())
    assert_true(_type_is_eq[filtered[1], Float64]())


def test_filter_types_chained() raises:
    comptime IsNotBool[Type: Movable] = not _type_is_eq[Type, Bool]()
    comptime IsNotInt[Type: Movable] = not _type_is_eq[Type, Int]()
    comptime step1 = Variadic.filter_types[
        *Tuple[Int, String, Float64, Bool].element_types, predicate=IsNotBool
    ]
    comptime step2 = Variadic.filter_types[*step1, predicate=IsNotInt]
    assert_equal(TypeList[*step2].size, 2)
    assert_true(_type_is_eq[step2[0], String]())
    assert_true(_type_is_eq[step2[1], Float64]())


def test_filter_types_empty_result() raises:
    comptime AlwaysFalse[Type: Movable] = False
    comptime empty = Variadic.filter_types[
        *Tuple[Int, String, Float64, Bool].element_types, predicate=AlwaysFalse
    ]
    assert_equal(TypeList[*empty].size, 0)


def test_variadic_list_linear_type() raises:
    """Test owned variadics with a linear type (ExplicitDelOnly)."""

    @parameter
    def destroy_elem(_idx: Int, var arg: ExplicitDelOnly):
        arg^.destroy()

    def take_owned_linear(var *args: ExplicitDelOnly):
        args^.consume_elements[destroy_elem]()

    take_owned_linear(ExplicitDelOnly(5), ExplicitDelOnly(10))


def test_variadic_list_write_to() raises:
    def check_three(*args: Int) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(1, 2, 3)")

    def check_single(*args: Int) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(42)")

    def check_empty(*args: Int) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "()")

    check_three(1, 2, 3)
    check_single(42)
    check_empty()


def test_variadic_list_write_repr_to() raises:
    def check_three(*args: Int) raises:
        var s = String()
        args.write_repr_to(s)
        assert_equal(s, "VariadicList[Int]((Int(1), Int(2), Int(3)))")

    def check_single(*args: Int) raises:
        var s = String()
        args.write_repr_to(s)
        assert_equal(s, "VariadicList[Int]((Int(42)))")

    check_three(1, 2, 3)
    check_single(42)


def test_variadic_list_mem_write_to() raises:
    def check_two(*args: String) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(hello, world)")

    def check_single(*args: String) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(hi)")

    check_two("hello", "world")
    check_single("hi")


def test_variadic_list_mem_write_repr_to() raises:
    def check_two(*args: String) raises:
        var s = String()
        args.write_repr_to(s)
        assert_equal(s, "VariadicList[String](('hello', 'world'))")

    check_two("hello", "world")


def test_variadic_pack_write_to() raises:
    def helper[*Ts: Writable](*args: *Ts) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(1, hello, True)")

    helper(1, "hello", True)


def test_variadic_pack_write_repr_to() raises:
    def helper[*Ts: Writable](*args: *Ts) raises:
        var s = String()
        args.write_repr_to(s)
        assert_equal(s, "(Int(1), 'hello', True)")

    helper(1, "hello", True)


def test_variadic_pack_forwarding() raises:
    """Test that variadic packs can be forwarded with *pack syntax."""

    def callee[*Ts: Writable](*args: *Ts) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(1, hello, 3.14)")

    def forwarder[*Ts: Writable](*args: *Ts) raises:
        callee(*args)

    forwarder(1, "hello", 3.14)


def test_variadic_pack_forwarding_single_element() raises:
    """Test forwarding a single-element variadic pack."""

    def callee[*Ts: Writable](*args: *Ts) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(42)")

    def forwarder[*Ts: Writable](*args: *Ts) raises:
        callee(*args)

    forwarder(42)


def test_variadic_pack_forwarding_empty() raises:
    """Test forwarding an empty variadic pack."""

    def callee[*Ts: Writable](*args: *Ts) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "()")

    def forwarder[*Ts: Writable](*args: *Ts) raises:
        callee(*args)

    forwarder()


def test_variadic_pack_forwarding_through_two_levels() raises:
    """Test forwarding a variadic pack through two levels of indirection."""

    def callee[*Ts: Writable](*args: *Ts) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(a, True)")

    def middle[*Ts: Writable](*args: *Ts) raises:
        callee(*args)

    def outer[*Ts: Writable](*args: *Ts) raises:
        middle(*args)

    outer("a", True)


# ===-----------------------------------------------------------------------===#
# TypeList tests
# ===-----------------------------------------------------------------------===#


def test_typelist_size() raises:
    assert_equal(TypeList[type=AnyType, Int, String, Float64].size, 3)
    assert_equal(TypeList[Bool].size, 1)


def test_typelist_getitem() raises:
    comptime TL = TypeList[type=AnyType, Int, String, Float64]()
    assert_true(_type_is_eq[TL[0], Int]())
    assert_true(_type_is_eq[TL[1], String]())
    assert_true(_type_is_eq[TL[2], Float64]())


def test_typelist_reversed() raises:
    comptime rev = TypeList[type=AnyType, Int, String, Float64]().reverse()
    assert_equal(rev.size, 3)
    assert_true(_type_is_eq[rev[0], Float64]())
    assert_true(_type_is_eq[rev[1], String]())
    assert_true(_type_is_eq[rev[2], Int]())


def test_typelist_contains() raises:
    comptime TL = TypeList[type=AnyType, Int, String, Float64]()
    comptime assert TL.contains[Int]
    comptime assert TL.contains[String]
    comptime assert TL.contains[Float64]
    comptime assert not TL.contains[Bool]


def test_typelist_slice() raises:
    comptime TL = TypeList[type=AnyType, Int, String, Float64, Bool]()

    # Slice middle
    comptime middle = TL.slice[start=1, end=3]()
    assert_equal(middle.size, 2)
    assert_true(_type_is_eq[middle[0], String]())
    assert_true(_type_is_eq[middle[1], Float64]())

    # Slice from start
    comptime first2 = TL.slice[start=0, end=2]()
    assert_equal(first2.size, 2)
    assert_true(_type_is_eq[first2[0], Int]())
    assert_true(_type_is_eq[first2[1], String]())

    # Slice to end (using default)
    comptime last2 = TL.slice[start=2]()
    assert_equal(last2.size, 2)
    assert_true(_type_is_eq[last2[0], Float64]())
    assert_true(_type_is_eq[last2[1], Bool]())

    # Full slice (defaults)
    comptime full = TL.slice[]
    assert_equal(full.size, 4)


def test_typelist_filter() raises:
    comptime TL = TypeList[type=AnyType, Int, String, Float64, Bool]()

    comptime IsNotInt[Type: AnyType] = not _type_is_eq[Type, Int]()
    comptime filtered = TL.filter[IsNotInt]()
    assert_equal(filtered.size, 3)
    assert_true(_type_is_eq[filtered[0], String]())
    assert_true(_type_is_eq[filtered[1], Float64]())
    assert_true(_type_is_eq[filtered[2], Bool]())


def test_typelist_filter_empty_result() raises:
    comptime TL = TypeList[type=AnyType, Int, String]

    comptime AlwaysFalse[Type: AnyType] = False
    comptime empty = TL.filter[AlwaysFalse]
    assert_equal(empty.size, 0)


def test_typelist_map() raises:
    comptime TL = TypeList[type=Copyable, Int, String, Float64]()

    comptime ToList[T: Copyable] = List[T]
    comptime mapped = TL.map[To=Copyable, Mapper=ToList]()
    assert_equal(mapped.size, 3)
    comptime assert _type_is_eq[mapped[0], List[Int]]()
    comptime assert _type_is_eq[mapped[1], List[String]]()
    comptime assert _type_is_eq[mapped[2], List[Float64]]()


def test_typelist_map_identity() raises:
    comptime TL = TypeList[type=AnyType, Int, Bool]()

    comptime Identity[T: AnyType] = T
    comptime mapped = TL.map[To=AnyType, Mapper=Identity]()
    assert_equal(mapped.size, 2)
    comptime assert _type_is_eq[mapped[0], Int]()
    comptime assert _type_is_eq[mapped[1], Bool]()


def test_typelist_map_empty() raises:
    comptime TL = TypeList[type=Copyable]()

    comptime ToList[T: Copyable] = List[T]
    comptime mapped = TL.map[To=Copyable, Mapper=ToList]()
    assert_equal(mapped.size, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
