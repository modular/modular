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

from reflection import (
    get_linkage_name,
    get_type_name,
    get_function_name,
    struct_field_index_by_name,
    struct_field_type_by_name,
    struct_field_count,
    struct_field_names,
    struct_field_types,
)
from testing import assert_equal, assert_true
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


# Generic struct for testing types with constructor calls in parameters
struct WrapperWithValue[T: AnyType, //, value: T]:
    pass


@fieldwise_init
@register_passable("trivial")
struct SimpleParam:
    var b: Bool


@fieldwise_init
struct MemoryOnlyParam:
    var b: Bool


# Struct with field that has a constructor call in type parameter
struct StructWithCtorParam:
    var field: WrapperWithValue[SimpleParam(True)]


# Struct with memory-only type constructor in parameter
struct StructWithMemoryOnlyCtorParam:
    var field: WrapperWithValue[MemoryOnlyParam(True)]


def test_get_type_name_ctor_param_from_field_types():
    """Ensure get_type_name works with constructor calls in type params.

    Issue #5732: using get_type_name on types with constructor calls (like
    A[B(True)]) extracted via struct_field_types would crash the compiler.
    """
    comptime types = struct_field_types[StructWithCtorParam]()
    var name = get_type_name[types[0]]()
    assert_equal(
        name,
        "test_reflection.WrapperWithValue[test_reflection.SimpleParam, True]",
    )


def test_get_type_name_memory_only_ctor_param_from_field_types():
    """Ensure memory-only types with constructor calls work.

    Memory-only types use `apply_result_slot` instead of `apply`, which was also
    affected by issue #5732.
    """
    comptime types = struct_field_types[StructWithMemoryOnlyCtorParam]()
    var name = get_type_name[types[0]]()
    assert_equal(
        name,
        (
            "test_reflection.WrapperWithValue[test_reflection.MemoryOnlyParam,"
            " {True}]"
        ),
    )


def test_get_type_name_ctor_param_direct():
    """Test that direct usage of types with constructor calls works.

    This demonstrates that the issue is specifically with types extracted via
    struct_field_types, not with constructor call parameters in general.
    """
    # Direct usage works fine - the constructor is evaluated
    var name = get_type_name[WrapperWithValue[SimpleParam(True)]]()
    assert_equal(
        name,
        "test_reflection.WrapperWithValue[test_reflection.SimpleParam, True]",
    )


# ===----------------------------------------------------------------------=== #
# Nested Parametric Type Tests (Issue #5723)
# ===----------------------------------------------------------------------=== #


# Generic struct for testing nested parametric types
struct GenericWrapper[T: AnyType]:
    pass


# Struct with nested parametric type fields for issue #5723 regression tests
struct NestedParametricStruct:
    var simple: GenericWrapper[String]
    var nested: GenericWrapper[List[String]]


# More deeply nested parametric types
struct DeeplyNestedStruct:
    var single: GenericWrapper[Int]
    var double: GenericWrapper[GenericWrapper[Int]]


def test_get_type_name_nested_parametric_from_field_types():
    """Test that get_type_name works with nested parametric types from struct_field_types.
    """
    # Both fields should work
    comptime types = struct_field_types[NestedParametricStruct]()
    var name0 = get_type_name[types[0]]()
    assert_equal(name0, "test_reflection.GenericWrapper[String]")

    # Nested parametric type should work too
    var name1 = get_type_name[types[1]]()
    assert_equal(name1, "test_reflection.GenericWrapper[List[String]]")


def test_get_type_name_deeply_nested_parametric_from_field_types():
    """Test deeply nested parametric types from struct_field_types works.

    This tests the case where we have GenericWrapper[GenericWrapper[T]] - both
    single and double nested types should work.
    """
    comptime types = struct_field_types[DeeplyNestedStruct]()

    # Single level parametric from struct_field_types
    var name0 = get_type_name[types[0]]()
    assert_equal(name0, "test_reflection.GenericWrapper[Int]")

    # Double nested should work too
    var name1 = get_type_name[types[1]]()
    assert_equal(
        name1,
        "test_reflection.GenericWrapper[test_reflection.GenericWrapper[Int]]",
    )


def test_get_type_name_nested_parametric_direct():
    """Test that directly using nested parametric types works (not from struct_field_types).

    This demonstrates that the issue is specifically with types extracted via
    struct_field_types, not with nested parametric types in general.
    """
    # Direct usage works fine
    var name = get_type_name[GenericWrapper[List[String]]]()
    assert_equal(name, "test_reflection.GenericWrapper[List[String]]")

    # Deeply nested direct usage also works
    name = get_type_name[GenericWrapper[GenericWrapper[Int]]]()
    assert_equal(
        name,
        "test_reflection.GenericWrapper[test_reflection.GenericWrapper[Int]]",
    )


# Additional edge case structs for issue #5723
struct Pair[T: AnyType, U: AnyType]:
    """Struct with multiple type parameters."""

    pass


struct StructWithMultipleParametricFields:
    """Struct with multiple different parametric field types."""

    var list_field: List[Int]
    var optional_field: Optional[String]
    var simd_field: SIMD[DType.float32, 4]


struct StructWithPair:
    """Struct with a field that has multiple type parameters."""

    var pair: Pair[String, List[Int]]


struct TripleNested:
    """Struct with three levels of nesting."""

    var triple: GenericWrapper[GenericWrapper[GenericWrapper[Int]]]


def test_get_type_name_multiple_type_params_from_field_types():
    """Test parametric types with multiple type parameters from struct_field_types.
    """
    comptime types = struct_field_types[StructWithPair]()
    var name = get_type_name[types[0]]()
    assert_equal(name, "test_reflection.Pair[String, List[Int]]")


def test_get_type_name_various_stdlib_parametric_from_field_types():
    """Test various stdlib parametric types (List, Optional, SIMD) from struct_field_types.
    """
    comptime types = struct_field_types[StructWithMultipleParametricFields]()

    var name0 = get_type_name[types[0]]()
    assert_equal(name0, "List[Int]")

    var name1 = get_type_name[types[1]]()
    assert_equal(name1, "std.collections.optional.Optional[String]")

    # SIMD[DType.float32, 4] - value parameters (not type parameters) print correctly
    var name2 = get_type_name[types[2]]()
    assert_equal(name2, "SIMD[DType.float32, 4]")


def test_get_type_name_triple_nested_from_field_types():
    """Test three levels of nesting from struct_field_types."""
    comptime types = struct_field_types[TripleNested]()
    var name = get_type_name[types[0]]()
    assert_equal(
        name,
        "test_reflection.GenericWrapper[test_reflection.GenericWrapper[test_reflection.GenericWrapper[Int]]]",
    )


def test_iterate_parametric_field_types_no_crash():
    """Test that iterating over parametric field types in a loop doesn't crash.

    This tests the common pattern of using struct_field_types in a @parameter for loop.
    """
    var count = 0

    @parameter
    for i in range(struct_field_count[StructWithMultipleParametricFields]()):
        comptime field_type = struct_field_types[
            StructWithMultipleParametricFields
        ]()[i]
        # Getting type name in a loop should not crash
        var name = get_type_name[field_type]()
        _ = name
        count += 1

    assert_equal(count, 3)


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


fn _count_fields_generic[T: AnyType]() -> Int:
    """Helper function to test struct_field_count through generic parameter."""
    return struct_field_count[T]()


fn _get_field_names_generic[T: AnyType]() -> StaticString:
    """Helper function to test struct_field_names through generic parameter."""
    return struct_field_names[T]()[0]


fn _get_type_name_generic[T: AnyType]() -> StaticString:
    """Helper function to test get_type_name through generic parameter."""
    return get_type_name[T]()


def test_reflection_on_int():
    """Test that reflection functions work on Int (issue #5731).

    Int is a struct with one field (_mlir_value). Previously passing Int
    through a generic parameter to reflection functions would crash the
    compiler.
    """
    assert_equal(struct_field_count[Int](), 1)
    assert_equal(_count_fields_generic[Int](), 1)

    assert_equal(struct_field_names[Int]()[0], "_mlir_value")
    assert_equal(_get_field_names_generic[Int](), "_mlir_value")

    # get_type_name through generic (direct is tested elsewhere)
    assert_equal(_get_type_name_generic[Int](), "Int")


def test_reflection_on_origin():
    """Test that reflection functions work on Origin (issue #5731).

    Origin is a struct with one field (_mlir_origin). Previously this would
    crash the compiler when passed through generic parameters.
    """
    assert_equal(_count_fields_generic[Origin[mut=True]](), 1)
    assert_equal(_count_fields_generic[Origin[mut=False]](), 1)

    assert_equal(_get_field_names_generic[Origin[mut=True]](), "_mlir_origin")

    assert_equal(
        _get_type_name_generic[Origin[mut=True]](),
        "std.builtin.type_aliases.Origin[True]",
    )


def test_reflection_on_nonetype():
    """Test that reflection functions work on NoneType (issue #5731).

    NoneType is a struct with one field (_value). Previously this would crash
    the compiler when passed through generic parameters.
    """
    assert_equal(_count_fields_generic[NoneType](), 1)

    assert_equal(_get_field_names_generic[NoneType](), "_value")

    assert_equal(
        _get_type_name_generic[NoneType](), "std.builtin.none.NoneType"
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
    comptime count = struct_field_count[SimpleStruct]()
    assert_equal(count, 2)


def test_struct_field_count_nested():
    comptime outer_count = struct_field_count[Outer]()
    assert_equal(outer_count, 3)

    comptime inner_count = struct_field_count[Inner]()
    assert_equal(inner_count, 2)


def test_struct_field_count_empty():
    comptime count = struct_field_count[EmptyStruct]()
    assert_equal(count, 0)


def test_struct_field_count_mixed_visibility():
    comptime count = struct_field_count[MixedVisibility]()
    assert_equal(count, 3)


def test_struct_field_names():
    # Test field names via indexing - returns InlineArray[StaticString, N]
    comptime names = struct_field_names[SimpleStruct]()
    assert_equal(names[0], "x")
    assert_equal(names[1], "y")


def test_struct_field_types():
    # Test field types via indexing (detailed verification in test_struct_field_types_are_correct)
    comptime types = struct_field_types[SimpleStruct]()
    _ = types


def test_nested_struct_returns_struct_type():
    # When a struct contains another struct, we get the struct type
    # not flattened fields. Users can recursively introspect.
    comptime outer_count = struct_field_count[Outer]()
    assert_equal(outer_count, 3)  # name, inner, count - not flattened


def test_struct_field_iteration():
    # Test iterating over struct fields with @parameter for
    var count = 0

    @parameter
    for i in range(struct_field_count[SimpleStruct]()):
        comptime field_type = struct_field_types[SimpleStruct]()[i]
        comptime field_name = struct_field_names[SimpleStruct]()[i]
        _ = field_type
        _ = field_name
        count += 1
    assert_equal(count, 2)


def test_struct_field_types_are_correct():
    # Verify that field types match expected types using type names.
    # Note: _type_is_eq doesn't work across !kgen.type / Mojo type boundary.
    comptime types = struct_field_types[SimpleStruct]()
    assert_equal(get_type_name[types[0]](), "Int")
    assert_equal(get_type_name[types[1]](), "SIMD[DType.float64, 1]")


# ===----------------------------------------------------------------------=== #
# Field Index and Type by Name Tests
# ===----------------------------------------------------------------------=== #


def test_struct_field_index_simple():
    # Test getting field index by name for SimpleStruct
    comptime x_idx = struct_field_index_by_name[SimpleStruct, "x"]()
    assert_equal(x_idx, 0)

    comptime y_idx = struct_field_index_by_name[SimpleStruct, "y"]()
    assert_equal(y_idx, 1)


def test_struct_field_index_nested():
    # Test getting field index for nested struct
    comptime name_idx = struct_field_index_by_name[Outer, "name"]()
    assert_equal(name_idx, 0)

    comptime inner_idx = struct_field_index_by_name[Outer, "inner"]()
    assert_equal(inner_idx, 1)

    comptime count_idx = struct_field_index_by_name[Outer, "count"]()
    assert_equal(count_idx, 2)


def test_struct_field_index_mixed_visibility():
    # Test that all fields are accessible, including private-like names
    comptime public_idx = struct_field_index_by_name[
        MixedVisibility, "public_field"
    ]()
    assert_equal(public_idx, 0)

    comptime private_idx = struct_field_index_by_name[
        MixedVisibility, "_private_field"
    ]()
    assert_equal(private_idx, 1)

    comptime dunder_idx = struct_field_index_by_name[
        MixedVisibility, "__dunder_field"
    ]()
    assert_equal(dunder_idx, 2)


def test_struct_field_type_by_index_simple():
    # Test getting field type by index (using composition of index lookup and types)
    comptime x_idx = struct_field_index_by_name[SimpleStruct, "x"]()
    comptime x_type = struct_field_types[SimpleStruct]()[x_idx]
    assert_equal(get_type_name[x_type](), "Int")

    comptime y_idx = struct_field_index_by_name[SimpleStruct, "y"]()
    comptime y_type = struct_field_types[SimpleStruct]()[y_idx]
    assert_equal(get_type_name[y_type](), "SIMD[DType.float64, 1]")


def test_struct_field_type_by_index_nested():
    # Test that nested struct fields return the struct type, not flattened
    comptime inner_idx = struct_field_index_by_name[Outer, "inner"]()
    comptime inner_type = struct_field_types[Outer]()[inner_idx]
    assert_equal(get_type_name[inner_type](), "test_reflection.Inner")


def test_struct_field_index_consistent():
    # Verify that struct_field_index_by_name returns consistent indices
    # for field names - test with literal strings
    comptime x_idx = struct_field_index_by_name[SimpleStruct, "x"]()
    comptime y_idx = struct_field_index_by_name[SimpleStruct, "y"]()
    # x is the first field, y is the second
    assert_equal(x_idx, 0)
    assert_equal(y_idx, 1)
    # They should be different
    assert_equal(x_idx != y_idx, True)


def test_struct_field_type_matches_index():
    # Verify that field index lookup correctly corresponds to field types
    comptime x_idx = struct_field_index_by_name[SimpleStruct, "x"]()
    comptime x_type_by_idx = struct_field_types[SimpleStruct]()[x_idx]
    assert_equal(get_type_name[x_type_by_idx](), "Int")

    comptime y_idx = struct_field_index_by_name[SimpleStruct, "y"]()
    comptime y_type_by_idx = struct_field_types[SimpleStruct]()[y_idx]
    assert_equal(get_type_name[y_type_by_idx](), "SIMD[DType.float64, 1]")


# ===----------------------------------------------------------------------=== #
# Field Type by Name Tests (using ReflectedType wrapper)
# ===----------------------------------------------------------------------=== #


def test_struct_field_type_by_name_simple():
    # Test getting field type by name for SimpleStruct
    comptime x_type = struct_field_type_by_name[SimpleStruct, "x"]()
    assert_equal(get_type_name[x_type.T](), "Int")

    comptime y_type = struct_field_type_by_name[SimpleStruct, "y"]()
    assert_equal(get_type_name[y_type.T](), "SIMD[DType.float64, 1]")


def test_struct_field_type_by_name_nested():
    # Test that nested struct fields return the struct type, not flattened
    comptime inner_type = struct_field_type_by_name[Outer, "inner"]()
    assert_equal(get_type_name[inner_type.T](), "test_reflection.Inner")


def test_struct_field_type_by_name_matches_index():
    # Verify that struct_field_type_by_name returns the same type
    # as get_struct_field_types with the corresponding index
    comptime x_idx = struct_field_index_by_name[SimpleStruct, "x"]()
    comptime x_type_by_name = struct_field_type_by_name[SimpleStruct, "x"]()
    comptime x_type_by_idx = struct_field_types[SimpleStruct]()[x_idx]
    assert_equal(
        get_type_name[x_type_by_name.T](), get_type_name[x_type_by_idx]()
    )


# ===----------------------------------------------------------------------=== #
# Magic Function Tests (for generic type support)
# ===----------------------------------------------------------------------=== #


# Test struct for generic iteration test
struct GenericTestPoint:
    var x: Int
    var y: Int
    var z: Float64


fn generic_field_info_printer[T: AnyType]():
    """Generic function that uses magic functions to introspect any struct."""

    @parameter
    for i in range(struct_field_count[T]()):
        comptime field_name = struct_field_names[T]()[i]
        comptime field_type = struct_field_types[T]()[i]
        # Just verify we can access them - the types are correct if this compiles
        _ = field_name
        _ = field_type


def test_generic_iteration():
    # Test that we can iterate over struct fields generically
    generic_field_info_printer[GenericTestPoint]()
    generic_field_info_printer[SimpleStruct]()
    generic_field_info_printer[Outer]()


fn count_fields_generically[T: AnyType]() -> Int:
    """Counts fields generically - works with any struct type."""
    return struct_field_count[T]()


def test_count_through_generic_function():
    # Verify generic function returns correct counts for different types
    assert_equal(count_fields_generically[SimpleStruct](), 2)
    assert_equal(count_fields_generically[Outer](), 3)
    assert_equal(count_fields_generically[Inner](), 2)
    assert_equal(count_fields_generically[EmptyStruct](), 0)
    assert_equal(count_fields_generically[MixedVisibility](), 3)


fn get_first_field_name_generically[T: AnyType]() -> StaticString:
    """Gets first field name - works with any struct type."""
    # Only call this for structs with at least one field
    return struct_field_names[T]()[0]


def test_magic_field_name_through_generic_function():
    # Verify generic function returns correct first field name
    assert_equal(get_first_field_name_generically[SimpleStruct](), "x")
    assert_equal(get_first_field_name_generically[Outer](), "name")
    assert_equal(get_first_field_name_generically[Inner](), "a")
    assert_equal(
        get_first_field_name_generically[MixedVisibility](), "public_field"
    )


# ===----------------------------------------------------------------------=== #
# Parametric Struct Tests
# ===----------------------------------------------------------------------=== #


def test_parametric_struct_field_count():
    # Test that parametric structs work correctly with reflection.
    # SIMD is a parametric struct with type and size parameters.
    comptime simd_count = struct_field_count[SIMD[DType.float32, 4]]()
    # SIMD has a single 'value' field internally
    assert_equal(
        simd_count >= 0, True
    )  # Just verify it works, don't depend on internals

    # InlineArray is another parametric struct
    comptime array_count = struct_field_count[InlineArray[Int, 3]]()
    assert_equal(array_count >= 0, True)


def test_parametric_struct_in_generic_function():
    # Verify parametric structs work through generic functions
    _ = count_fields_generically[SIMD[DType.float32, 4]]()
    _ = count_fields_generically[InlineArray[Int, 3]]()
    # If this compiles and runs, parametric structs work with generics


struct ParametricTestStruct[T: Copyable & ImplicitlyDestructible, size: Int]:
    """A user-defined parametric struct for testing."""

    var data: Self.T
    var count: Int

    fn __init__(out self, var data: Self.T, count: Int):
        self.data = data^
        self.count = count


def test_user_defined_parametric_struct():
    # Test user-defined parametric structs
    comptime count = struct_field_count[ParametricTestStruct[Float64, 10]]()
    assert_equal(count, 2)

    comptime names = struct_field_names[ParametricTestStruct[Float64, 10]]()
    assert_equal(names[0], "data")
    assert_equal(names[1], "count")


fn generic_parametric_inspector[
    T: Copyable & ImplicitlyDestructible, size: Int
]() -> Int:
    """Generic function that inspects a parametric struct."""
    return struct_field_count[ParametricTestStruct[T, size]]()


def test_generic_with_parametric_struct():
    # Test generic function instantiated with different parameter values
    assert_equal(generic_parametric_inspector[Int, 5](), 2)
    assert_equal(generic_parametric_inspector[Float64, 100](), 2)


# ===----------------------------------------------------------------------=== #
# conforms_to with Reflection APIs
# ===----------------------------------------------------------------------=== #


# Test struct with various trait-conforming types
struct TraitTestStruct:
    var copyable_field: Int  # Int is Copyable
    var stringable_field: String  # String is Stringable


def test_conforms_to_with_field_types():
    """Test that conforms_to works with types from struct_field_types."""
    comptime types = struct_field_types[TraitTestStruct]()

    # Int conforms to Copyable
    assert_true(comptime (conforms_to(types[0], Copyable)))

    # String conforms to Stringable
    assert_true(comptime (conforms_to(types[1], Stringable)))


def test_conforms_to_field_iteration():
    """Test iterating over field types and checking trait conformance."""
    var copyable_count = 0

    @parameter
    for i in range(struct_field_count[SimpleStruct]()):
        comptime field_type = struct_field_types[SimpleStruct]()[i]

        @parameter
        if conforms_to(field_type, Copyable):
            copyable_count += 1

    # Both Int and Float64 are Copyable
    assert_equal(copyable_count, 2)


fn count_copyable_fields[T: AnyType]() -> Int:
    """Generic function that counts fields conforming to Copyable."""
    var count = 0

    @parameter
    for i in range(struct_field_count[T]()):
        comptime field_type = struct_field_types[T]()[i]

        @parameter
        if conforms_to(field_type, Copyable):
            count += 1

    return count


def test_conforms_to_generic_function():
    """Test conforms_to with field types in a generic context."""
    # SimpleStruct has 2 Copyable fields (Int, Float64)
    assert_equal(count_copyable_fields[SimpleStruct](), 2)

    # Inner has 2 Copyable fields (Int, Int)
    assert_equal(count_copyable_fields[Inner](), 2)


# ===----------------------------------------------------------------------=== #
# Struct Field Reference Tests (__struct_field_ref)
# ===----------------------------------------------------------------------=== #


# Non-copyable type for testing reference semantics
struct NonCopyableValue:
    """A type that cannot be copied - simulates complex resources."""

    var data: Int

    fn __init__(out self, data: Int):
        self.data = data

    fn __copyinit__(out self, other: Self):
        # If this is called, we have a bug!
        print("ERROR: NonCopyableValue was copied!")
        self.data = other.data


struct ContainerWithNonCopyable:
    """A struct containing a non-copyable field."""

    var id: Int
    var resource: NonCopyableValue
    var count: Int

    fn __init__(out self, id: Int, value: Int, count: Int):
        self.id = id
        self.resource = NonCopyableValue(value)
        self.count = count


# Simple test struct for field reference tests
struct PointForRef:
    var x: Int
    var y: Int

    fn __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y


def test___struct_field_ref_basic_read():
    """Test reading struct fields through __struct_field_ref."""
    var p = PointForRef(10, 20)

    # Get references to fields by index
    ref x_ref = __struct_field_ref(0, p)
    ref y_ref = __struct_field_ref(1, p)

    # Verify we can read the correct values
    assert_equal(x_ref, 10)
    assert_equal(y_ref, 20)


def test___struct_field_ref_mutation():
    """Test mutating struct fields through __struct_field_ref."""
    var p = PointForRef(1, 2)

    # Modify through __struct_field_ref directly
    __struct_field_ref(0, p) = 100
    __struct_field_ref(1, p) = 200

    # Verify the original struct was modified
    assert_equal(p.x, 100)
    assert_equal(p.y, 200)

    # Also test mutation through a locally-bound ref
    var p2 = PointForRef(0, 0)
    ref x = __struct_field_ref(0, p2)
    ref y = __struct_field_ref(1, p2)
    x = 42
    y = 99
    assert_equal(p2.x, 42)
    assert_equal(p2.y, 99)


def test___struct_field_ref_non_copyable():
    """Test that __struct_field_ref doesn't copy non-copyable fields.

    This is the key use case: accessing fields without copying them,
    enabling reflection-based utilities for types with non-copyable fields.
    """
    var c = ContainerWithNonCopyable(42, 100, 5)

    # Get references to fields - this should NOT trigger copies
    ref id_ref = __struct_field_ref(0, c)
    ref resource_ref = __struct_field_ref(1, c)
    ref count_ref = __struct_field_ref(2, c)

    # Read through references without copying
    assert_equal(id_ref, 42)
    assert_equal(resource_ref.data, 100)
    assert_equal(count_ref, 5)

    # Modify through reference
    __struct_field_ref(1, c).data = 999
    assert_equal(c.resource.data, 999)


def test___struct_field_ref_with_names():
    """Test combining __struct_field_ref with struct_field_names."""
    var p = PointForRef(30, 40)
    comptime names = struct_field_names[PointForRef]()

    # Get field references and verify names match values
    ref f0 = __struct_field_ref(0, p)
    ref f1 = __struct_field_ref(1, p)

    # Verify the names are correct
    assert_equal(names[0], "x")
    assert_equal(names[1], "y")

    # Verify the values are correct
    assert_equal(f0, 30)
    assert_equal(f1, 40)


fn print_struct_debug[T: AnyType](ref s: T):
    """Example: A generic debug-print function using struct field reflection.

    This demonstrates a common use case: implementing a Debug-like
    trait that can iterate over struct fields without copying them.

    Uses __struct_field_ref with parametric indices (the loop variable i).
    """
    comptime names = struct_field_names[T]()
    comptime count = struct_field_count[T]()

    # Test that __struct_field_ref works with parametric indices
    @parameter
    for i in range(count):
        _ = names[i]
        # Access the field by parametric index - this tests support for
        # parametric indices in struct field reflection
        _ = __struct_field_ref(i, s)


def test___struct_field_ref_parametric_index():
    """Test __struct_field_ref with parametric indices (loop variables)."""
    var p = PointForRef(10, 20)
    var c = ContainerWithNonCopyable(1, 2, 3)

    # These should compile and work without copying
    print_struct_debug(p)
    print_struct_debug(c)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
