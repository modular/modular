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

from sys.info import CompilationTarget, _current_target

from compile.reflection import (
    get_linkage_name,
    get_type_name,
    get_function_name,
    get_struct_field_count,
    get_struct_field_names,
    get_struct_field_types,
)
from testing import assert_equal
from testing import TestSuite


fn my_func() -> Int:
    return 0


def test_get_linkage_name():
    var name = get_linkage_name[my_func]()
    assert_equal(name, "test_reflection::my_func()")


def test_get_linkage_name_nested():
    fn nested_func(x: Int) -> Int:
        return x

    var name = get_linkage_name[nested_func]()
    assert_equal(
        name,
        "test_reflection::test_get_linkage_name_nested()_nested_func(::Int)",
    )


fn your_func[x: Int]() raises -> Int:
    return x


def test_get_linkage_name_parameterized():
    var name = get_linkage_name[your_func[7]]()
    assert_equal(name, "test_reflection::your_func[::Int](),x=7")


def test_get_linkage_name_on_itself():
    var name = get_linkage_name[_current_target]()
    assert_equal(name, "std::sys::info::_current_target()")


def test_get_function_name():
    var name = get_function_name[my_func]()
    assert_equal(name, "my_func")


def test_get_function_name_nested():
    fn nested_func(x: Int) -> Int:
        return x

    var name2 = get_function_name[nested_func]()
    assert_equal(name2, "nested_func")


def test_get_function_name_parameterized():
    var name = get_function_name[your_func]()
    assert_equal(name, "your_func")

    var name2 = get_function_name[your_func[7]]()
    assert_equal(name2, "your_func")


def test_get_type_name():
    var name = get_type_name[Int]()
    assert_equal(name, "Int")

    name = get_type_name[Int, qualified_builtins=True]()
    assert_equal(name, "std.builtin.int.Int")


def test_get_type_name_nested():
    fn nested_func[T: AnyType]() -> StaticString:
        return get_type_name[T]()

    var name = nested_func[String]()
    assert_equal(name, "String")


def test_get_type_name_simd():
    var name = get_type_name[Float32]()
    assert_equal(name, "SIMD[DType.float32, 1]")

    name = get_type_name[SIMD[DType.uint16, 4], qualified_builtins=True]()
    assert_equal(
        name, "std.builtin.simd.SIMD[std.builtin.dtype.DType.uint16, 4]"
    )


@fieldwise_init
struct Bar[x: Int, f: Float32 = 1.3](Intable):
    fn __int__(self) -> Int:
        return self.x

    var y: Int
    var z: Float64


@fieldwise_init
struct Foo[
    T: Intable, //, b: T, c: Bool, d: NoneType = None, e: StaticString = "hello"
]:
    pass


def test_get_type_name_non_scalar_simd_value():
    var name = get_type_name[
        Foo[SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0), True]
    ]()
    assert_equal(
        name,
        (
            "test_reflection.Foo[SIMD[DType.float32, 4], "
            '[1, 2, 3, 4] : SIMD[DType.float32, 4], True, None, {"hello\0", 5}]'
        ),
    )

    name = get_type_name[
        Foo[SIMD[DType.bool, 4](True, False, True, False), True]
    ]()
    assert_equal(
        name,
        (
            "test_reflection.Foo[SIMD[DType.bool, 4], "
            "[True, False, True, False] : SIMD[DType.bool, 4], "
            'True, None, {"hello\0", 5}]'
        ),
    )


def test_get_type_name_struct():
    var name = get_type_name[Foo[Bar[2](y=3, z=4.1), True]]()
    assert_equal(
        name,
        (
            "test_reflection.Foo["
            "test_reflection.Bar[2, 1.29999995 : SIMD[DType.float32, 1]], "
            "{3, 4.0999999999999996 : SIMD[DType.float64, 1]}, "
            'True, None, {"hello\0", 5}]'
        ),
    )


def test_get_type_name_partially_bound_type():
    var name = get_type_name[Foo[Bar[2](y=3, z=0.125)]]()
    assert_equal(
        name,
        (
            "test_reflection.Foo["
            "test_reflection.Bar[2, 1.29999995 : SIMD[DType.float32, 1]], "
            '{3, 0.125 : SIMD[DType.float64, 1]}, ?, None, {"hello\0", 5}]'
        ),
    )


def test_get_type_name_unprintable():
    var name = get_type_name[CompilationTarget[_current_target()]]()
    assert_equal(name, "std.sys.info.CompilationTarget[<unprintable>]")


def test_get_type_name_alias():
    comptime T = Bar[5]
    var name = get_type_name[T]()
    assert_equal(
        name, "test_reflection.Bar[5, 1.29999995 : SIMD[DType.float32, 1]]"
    )

    # Also test parametric aliases (i.e. unbound parameters).
    comptime R = Bar[_]
    name = get_type_name[R]()
    assert_equal(
        name, "test_reflection.Bar[?, 1.29999995 : SIMD[DType.float32, 1]]"
    )


# ===----------------------------------------------------------------------=== #
# Struct Field Reflection Tests
# ===----------------------------------------------------------------------=== #


# Simple test struct for field reflection
struct SimpleStruct:
    var x: Int
    var y: Float64


# Struct with nested struct field
struct Inner:
    var a: Int
    var b: Int


struct Outer:
    var name: String
    var inner: Inner
    var count: Int


# Empty struct
struct EmptyStruct:
    pass


# Struct with private-like naming (all fields included)
struct MixedVisibility:
    var public_field: Int
    var _private_field: Float64
    var __dunder_field: Bool


def test_struct_field_count_simple():
    comptime count = get_struct_field_count[SimpleStruct]()
    assert_equal(count, 2)


def test_struct_field_count_nested():
    comptime outer_count = get_struct_field_count[Outer]()
    assert_equal(outer_count, 3)

    comptime inner_count = get_struct_field_count[Inner]()
    assert_equal(inner_count, 2)


def test_struct_field_count_empty():
    comptime count = get_struct_field_count[EmptyStruct]()
    assert_equal(count, 0)


def test_struct_field_count_mixed_visibility():
    comptime count = get_struct_field_count[MixedVisibility]()
    assert_equal(count, 3)


def test_struct_field_names():
    # Test field names via indexing - returns InlineArray[StaticString, N]
    comptime names = get_struct_field_names[SimpleStruct]()
    assert_equal(names[0], "x")
    assert_equal(names[1], "y")


def test_struct_field_types():
    # Test field types via indexing (detailed verification in test_struct_field_types_are_correct)
    comptime types = get_struct_field_types[SimpleStruct]()
    _ = types


def test_nested_struct_returns_struct_type():
    # When a struct contains another struct, we get the struct type
    # not flattened fields. Users can recursively introspect.
    comptime outer_count = get_struct_field_count[Outer]()
    assert_equal(outer_count, 3)  # name, inner, count - not flattened


def test_struct_field_iteration():
    # Test iterating over struct fields with @parameter for
    var count = 0

    @parameter
    for i in range(get_struct_field_count[SimpleStruct]()):
        comptime field_type = get_struct_field_types[SimpleStruct]()[i]
        comptime field_name = get_struct_field_names[SimpleStruct]()[i]
        _ = field_type
        _ = field_name
        count += 1
    assert_equal(count, 2)


def test_struct_field_types_are_correct():
    # Verify that field types match expected types using type names.
    # Note: _type_is_eq doesn't work across !kgen.type / Mojo type boundary.
    comptime types = get_struct_field_types[SimpleStruct]()
    assert_equal(get_type_name[types[0]](), "Int")
    assert_equal(get_type_name[types[1]](), "SIMD[DType.float64, 1]")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
