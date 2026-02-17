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
"""Tests for struct field reflection and introspection APIs."""

from reflection import (
    get_type_name,
    is_struct_type,
    struct_field_index_by_name,
    struct_field_type_by_name,
    struct_field_count,
    struct_field_names,
    struct_field_types,
    offset_of,
)
from testing import assert_equal, assert_true, assert_false
from testing import TestSuite


# ===----------------------------------------------------------------------=== #
# Helper structs for testing
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


# Dedicated struct with explicit MLIR-typed field for testing is_struct_type.
struct StructWithMLIRField(TrivialRegisterPassable):
    """A struct with an explicit MLIR-typed field for testing purposes."""

    var mojo_field: Int
    var mlir_field: __mlir_type.index


# Generic struct for testing types with constructor calls in parameters
struct WrapperWithValue[T: AnyType, //, value: T]:
    pass


@fieldwise_init
struct SimpleParam(TrivialRegisterPassable):
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


# Test struct for offset calculations - designed with predictable layout
@fieldwise_init
struct OffsetTestStruct(TrivialRegisterPassable):
    """Struct with predictable field layout for offset testing."""

    var a: Int8  # 1 byte at offset 0
    var b: Int64  # 8 bytes, needs 8-byte alignment -> offset 8
    var c: Int8  # 1 byte at offset 16
    var d: Int32  # 4 bytes, needs 4-byte alignment -> offset 20


@fieldwise_init
struct SimpleOffsetStruct(TrivialRegisterPassable):
    """Simple struct for basic offset tests."""

    var x: Int64  # 8 bytes at offset 0
    var y: Int64  # 8 bytes at offset 8


struct SingleFieldStruct(TrivialRegisterPassable):
    """Struct with a single field."""

    var value: Int32


struct AllSameTypeStruct(TrivialRegisterPassable):
    """Struct with all fields of the same type."""

    var a: Int64
    var b: Int64
    var c: Int64


# Parametric struct for testing offset calculation with varying field types.
# The offset of `footer` depends on the size of `data`, which varies with T.
struct GenericOffsetContainer[T: Copyable & Movable & ImplicitlyDestructible]:
    """Parametric struct where field offsets depend on the type parameter."""

    var header: Int8  # 1 byte at offset 0
    var data: Self.T  # Size and alignment vary with T
    var footer: Int8  # Offset depends on T's size and alignment


# ===----------------------------------------------------------------------=== #
# is_struct_type Tests
# ===----------------------------------------------------------------------=== #


def test_is_struct_type_with_user_structs():
    assert_true(is_struct_type[Bar[2]]())
    assert_true(is_struct_type[Foo[Bar[2](y=3, z=0.125), True]]())
    assert_true(is_struct_type[StructWithMLIRField]())


def test_is_struct_type_with_stdlib_structs():
    assert_true(is_struct_type[Int]())
    assert_true(is_struct_type[String]())
    assert_true(is_struct_type[Float64]())
    assert_true(is_struct_type[List[Int]]())
    assert_true(is_struct_type[Optional[String]]())


def test_is_struct_type_with_empty_struct():
    assert_true(is_struct_type[EmptyStruct]())


fn _is_struct_generic[T: AnyType]() -> Bool:
    """Helper function to test is_struct_type through generic parameter."""
    return is_struct_type[T]()


def test_is_struct_type_through_generic_function():
    """Test is_struct_type works correctly through a generic function."""
    # Mojo struct types should return True
    assert_true(_is_struct_generic[Int]())
    assert_true(_is_struct_generic[String]())
    assert_true(_is_struct_generic[Bar[2]]())
    assert_true(_is_struct_generic[SIMD[DType.float32, 4]]())
    assert_true(_is_struct_generic[EmptyStruct]())

    # Note: When __mlir_type.index is passed directly as a type parameter,
    # it returns True. However, when the same MLIR type is obtained via
    # struct_field_types (as in test_is_struct_type_with_mlir_primitive_types),
    # it returns False. This difference is because direct instantiation wraps
    # the MLIR type differently than reflection retrieval.
    assert_true(_is_struct_generic[__mlir_type.index]())


def test_is_struct_type_field_types_through_generic():
    """Test is_struct_type with field types passed through a generic function.

    This tests the scenario where you get field types from a struct using
    struct_field_types and then pass those types to a generic function that
    calls is_struct_type. This is a key use case to ensure is_struct_type
    correctly identifies MLIR types obtained via reflection.
    """
    comptime field_types = struct_field_types[StructWithMLIRField]()

    # First field (Int) should be a struct type even through a generic function
    assert_true(_is_struct_generic[field_types[0]]())

    # Second field (__mlir_type.index) should NOT be a struct type through a
    # generic function. This is the critical case that ensures is_struct_type
    # works correctly when guarding reflection APIs in generic contexts.
    assert_false(_is_struct_generic[field_types[1]]())


fn safe_field_count[T: AnyType]() -> Int:
    """Safe field count that returns -1 for non-struct types.

    Note: Use `@parameter if` (not runtime `if`) since `is_struct_type` must
    be evaluated at compile time to guard compile-time reflection APIs.
    """

    @parameter
    if is_struct_type[T]():
        return struct_field_count[T]()
    else:
        return -1


def test_is_struct_type_as_guard():
    """Test using is_struct_type as a guard before calling reflection APIs."""
    # User-defined struct should have field count > 0
    assert_equal(safe_field_count[Bar[2]]() >= 0, True)

    # Stdlib structs should also work
    assert_equal(safe_field_count[Int]() >= 0, True)

    # Empty structs should return 0
    assert_equal(safe_field_count[EmptyStruct](), 0)


def test_is_struct_type_with_mlir_primitive_types():
    """Test is_struct_type returns False for MLIR primitive types.

    This is the key use case from issue #5734: when iterating over struct fields,
    we may encounter MLIR primitive types that are not Mojo structs and would
    cause errors if passed to struct reflection APIs.
    """
    comptime field_types = struct_field_types[StructWithMLIRField]()
    assert_true(is_struct_type[field_types[0]]())
    assert_false(is_struct_type[field_types[1]]())


def test_is_struct_type_guard_with_mlir_types():
    """Test using is_struct_type to safely iterate over mixed field types.

    This demonstrates the use case from issue #5734: safely iterating over
    struct fields where some fields may be MLIR primitive types.
    """
    var struct_field_count_found = 0
    var non_struct_field_count_found = 0

    # StructWithMLIRField has one Mojo struct field and one MLIR type field
    @parameter
    for i in range(struct_field_count[StructWithMLIRField]()):
        comptime field_type = struct_field_types[StructWithMLIRField]()[i]

        @parameter
        if is_struct_type[field_type]():
            struct_field_count_found += 1
        else:
            non_struct_field_count_found += 1

    # One struct field (Int), one MLIR type field
    assert_equal(struct_field_count_found, 1)
    assert_equal(non_struct_field_count_found, 1)


# ===----------------------------------------------------------------------=== #
# Struct Field Reflection Tests
# ===----------------------------------------------------------------------=== #


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
    var names = struct_field_names[SimpleStruct]()
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
    assert_equal(get_type_name[inner_type](), "test_struct_fields.Inner")


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
    assert_equal(get_type_name[inner_type.T](), "test_struct_fields.Inner")


def test_struct_field_type_by_name_matches_index():
    # Verify that struct_field_type_by_name returns the same type
    # as get_struct_field_types with the corresponding index
    comptime x_idx = struct_field_index_by_name[SimpleStruct, "x"]()
    comptime x_type_by_name = struct_field_type_by_name[SimpleStruct, "x"]()
    comptime x_type_by_idx = struct_field_types[SimpleStruct]()[x_idx]
    assert_equal(
        get_type_name[x_type_by_name.T](), get_type_name[x_type_by_idx]()
    )


def test_struct_field_type_by_name_as_type_annotation():
    # Test that ReflectedType.T can be used as a type annotation.
    # This tests the fix for: https://github.com/modularml/modular/issues/5754
    comptime x_type = struct_field_type_by_name[SimpleStruct, "x"]()
    var value: x_type.T = 42
    assert_equal(value, 42)

    comptime y_type = struct_field_type_by_name[SimpleStruct, "y"]()
    var float_value: y_type.T = 3.14
    assert_true(float_value > 3.0)


def test_struct_field_type_by_name_nested_struct():
    # Test that ReflectedType.T works with nested struct types.
    comptime inner_type = struct_field_type_by_name[Outer, "inner"]()
    assert_equal(get_type_name[inner_type.T](), "test_struct_fields.Inner")

    # Also verify we can use the count field type as an annotation
    comptime count_type = struct_field_type_by_name[Outer, "count"]()
    var count_value: count_type.T = 42
    assert_equal(count_value, 42)


def test_struct_field_type_by_name_parametric_struct():
    # Test that ReflectedType.T works with parametric struct fields.
    comptime list_type = struct_field_type_by_name[
        StructWithMultipleParametricFields, "list_field"
    ]()
    # Verify we can use the type as an annotation and it works as List[Int]
    var list_value: list_type.T = List[Int]()
    list_value.append(1)
    list_value.append(2)
    list_value.append(3)
    assert_equal(len(list_value), 3)


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

    var names = struct_field_names[ParametricTestStruct[Float64, 10]]()
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
# get_type_name with struct_field_types Integration Tests
# ===----------------------------------------------------------------------=== #


def test_get_type_name_ctor_param_from_field_types():
    """Ensure get_type_name works with constructor calls in type params.

    Issue #5732: using get_type_name on types with constructor calls (like
    A[B(True)]) extracted via struct_field_types would crash the compiler.
    """
    comptime types = struct_field_types[StructWithCtorParam]()
    var name = get_type_name[types[0]]()
    assert_equal(
        name,
        (
            "test_struct_fields.WrapperWithValue[test_struct_fields.SimpleParam,"
            " True]"
        ),
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
            "test_struct_fields.WrapperWithValue[test_struct_fields.MemoryOnlyParam,"
            " {True}]"
        ),
    )


def test_get_type_name_nested_parametric_from_field_types():
    """Test that get_type_name works with nested parametric types from struct_field_types.
    """
    # Both fields should work
    comptime types = struct_field_types[NestedParametricStruct]()
    var name0 = get_type_name[types[0]]()
    assert_equal(name0, "test_struct_fields.GenericWrapper[String]")

    # Nested parametric type should work too
    var name1 = get_type_name[types[1]]()
    assert_equal(name1, "test_struct_fields.GenericWrapper[List[String]]")


def test_get_type_name_deeply_nested_parametric_from_field_types():
    """Test deeply nested parametric types from struct_field_types works.

    This tests the case where we have GenericWrapper[GenericWrapper[T]] - both
    single and double nested types should work.
    """
    comptime types = struct_field_types[DeeplyNestedStruct]()

    # Single level parametric from struct_field_types
    var name0 = get_type_name[types[0]]()
    assert_equal(name0, "test_struct_fields.GenericWrapper[Int]")

    # Double nested should work too
    var name1 = get_type_name[types[1]]()
    assert_equal(
        name1,
        "test_struct_fields.GenericWrapper[test_struct_fields.GenericWrapper[Int]]",
    )


def test_get_type_name_multiple_type_params_from_field_types():
    """Test parametric types with multiple type parameters from struct_field_types.
    """
    comptime types = struct_field_types[StructWithPair]()
    var name = get_type_name[types[0]]()
    assert_equal(name, "test_struct_fields.Pair[String, List[Int]]")


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
        "test_struct_fields.GenericWrapper[test_struct_fields.GenericWrapper[test_struct_fields.GenericWrapper[Int]]]",
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


# ===----------------------------------------------------------------------=== #
# Reflection on Builtin Types Tests
# ===----------------------------------------------------------------------=== #


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
    var int = 0
    assert_equal(_count_fields_generic[type_of(origin_of(int))](), 0)

    assert_equal(
        _get_type_name_generic[type_of(origin_of(int))](),
        "std.builtin.type_aliases.Origin[True, {}]",
    )

    _ = int  # silence unused warning.


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

    fn __copyinit__(out self, copy: Self):
        # If this is called, we have a bug!
        print("ERROR: NonCopyableValue was copied!")
        self.data = copy.data


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
    var names = struct_field_names[PointForRef]()

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
    var names = struct_field_names[T]()
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


# ===----------------------------------------------------------------------=== #
# Struct Field Offset Tests (offset_of)
# ===----------------------------------------------------------------------=== #


def test_offset_of_first_field_is_zero():
    """Test that the first field always has offset 0."""
    comptime off = offset_of[SimpleOffsetStruct, name="x"]()
    assert_equal(off, 0)

    comptime off2 = offset_of[OffsetTestStruct, name="a"]()
    assert_equal(off2, 0)

    comptime off3 = offset_of[SingleFieldStruct, name="value"]()
    assert_equal(off3, 0)


def test_offset_of_sequential_fields():
    """Test offsets of sequential same-type fields."""
    comptime x_off = offset_of[SimpleOffsetStruct, name="x"]()
    comptime y_off = offset_of[SimpleOffsetStruct, name="y"]()

    # x is at 0, y is at 8 (since Int64 is 8 bytes)
    assert_equal(x_off, 0)
    assert_equal(y_off, 8)


def test_offset_of_with_alignment_padding():
    """Test that alignment padding is correctly accounted for."""
    # OffsetTestStruct layout:
    # a: Int8 at offset 0 (1 byte)
    # padding: 7 bytes to align b to 8
    # b: Int64 at offset 8 (8 bytes)
    # c: Int8 at offset 16 (1 byte)
    # padding: 3 bytes to align d to 4
    # d: Int32 at offset 20 (4 bytes)

    comptime a_off = offset_of[OffsetTestStruct, name="a"]()
    comptime b_off = offset_of[OffsetTestStruct, name="b"]()
    comptime c_off = offset_of[OffsetTestStruct, name="c"]()
    comptime d_off = offset_of[OffsetTestStruct, name="d"]()

    assert_equal(a_off, 0)
    assert_equal(b_off, 8)  # Aligned to 8 bytes
    assert_equal(c_off, 16)  # After b (8 + 8)
    assert_equal(d_off, 20)  # Aligned to 4 bytes (16 + 1 + 3 padding)


def test_offset_of_by_index_basic():
    """Test offset_of[index=] with basic structs."""
    comptime off0 = offset_of[SimpleOffsetStruct, index=0]()
    comptime off1 = offset_of[SimpleOffsetStruct, index=1]()

    assert_equal(off0, 0)
    assert_equal(off1, 8)


def test_offset_of_by_index_with_padding():
    """Test offset_of[index=] accounts for alignment padding."""
    comptime off0 = offset_of[OffsetTestStruct, index=0]()
    comptime off1 = offset_of[OffsetTestStruct, index=1]()
    comptime off2 = offset_of[OffsetTestStruct, index=2]()
    comptime off3 = offset_of[OffsetTestStruct, index=3]()

    assert_equal(off0, 0)
    assert_equal(off1, 8)
    assert_equal(off2, 16)
    assert_equal(off3, 20)


def test_offset_of_by_name_basic():
    """Test offset_of[name=] with basic structs."""
    comptime x_off = offset_of[SimpleOffsetStruct, name="x"]()
    comptime y_off = offset_of[SimpleOffsetStruct, name="y"]()

    assert_equal(x_off, 0)
    assert_equal(y_off, 8)


def test_offset_of_index_and_name_are_consistent():
    """Test that offset_of[index=] and offset_of[name=] return consistent values.
    """
    # index and name should be consistent for same field
    comptime idx = struct_field_index_by_name[OffsetTestStruct, "c"]()
    comptime off_by_name = offset_of[OffsetTestStruct, name="c"]()
    comptime off_by_idx = offset_of[OffsetTestStruct, index=idx]()
    assert_equal(off_by_name, off_by_idx)

    # Test all fields of OffsetTestStruct
    assert_equal(
        offset_of[OffsetTestStruct, index=0](),
        offset_of[OffsetTestStruct, name="a"](),
    )
    assert_equal(
        offset_of[OffsetTestStruct, index=1](),
        offset_of[OffsetTestStruct, name="b"](),
    )
    assert_equal(
        offset_of[OffsetTestStruct, index=2](),
        offset_of[OffsetTestStruct, name="c"](),
    )
    assert_equal(
        offset_of[OffsetTestStruct, index=3](),
        offset_of[OffsetTestStruct, name="d"](),
    )


fn get_offset_generically[T: AnyType, idx: Int]() -> Int:
    """Helper to test offset calculation through generic function."""
    return offset_of[T, index=idx]()


def test_offset_through_generic_function():
    """Test that offset calculation works through generic functions."""
    comptime off0 = get_offset_generically[SimpleOffsetStruct, 0]()
    comptime off1 = get_offset_generically[SimpleOffsetStruct, 1]()

    assert_equal(off0, 0)
    assert_equal(off1, 8)


def test_offset_iteration_with_parameter_for():
    """Test iterating over field offsets with @parameter for."""
    var offsets = InlineArray[Int, 4](uninitialized=True)

    @parameter
    for i in range(struct_field_count[OffsetTestStruct]()):
        offsets[i] = offset_of[OffsetTestStruct, index=i]()

    assert_equal(offsets[0], 0)
    assert_equal(offsets[1], 8)
    assert_equal(offsets[2], 16)
    assert_equal(offsets[3], 20)


def test_offset_with_nested_struct():
    """Test offset calculation with nested struct fields."""
    # Inner has two Int fields (2 * 8 = 16 bytes on 64-bit)
    # Outer has: name (String), inner (Inner), count (Int)
    comptime name_off = offset_of[Outer, name="name"]()
    comptime inner_off = offset_of[Outer, name="inner"]()
    comptime count_off = offset_of[Outer, name="count"]()

    # First field is always at 0
    assert_equal(name_off, 0)

    # inner and count offsets depend on String's size, but should be > 0
    assert_true(inner_off > 0)
    assert_true(count_off > inner_off)


def test_offset_with_size_of_verification():
    """Test that field offsets are consistent with size_of for packed fields."""
    from sys.info import size_of

    # For SimpleOffsetStruct with two Int64 fields:
    # offset[y] should equal offset[x] + size_of[Int64]()
    # since Int64 is naturally aligned and there's no padding
    comptime x_off = offset_of[SimpleOffsetStruct, name="x"]()
    comptime y_off = offset_of[SimpleOffsetStruct, name="y"]()
    comptime int64_size = size_of[Int64]()

    assert_equal(x_off, 0)
    assert_equal(y_off, x_off + int64_size)


def test_offset_by_name_with_generic_type():
    """Test that offset_of[name=] works with generic type parameters."""

    # Test offset_of through a generic function
    # The field name must be a string literal, but T can be a generic type parameter
    fn get_x_offset[T: AnyType]() -> Int:
        """Helper that uses offset_of with a generic type parameter."""
        return offset_of[T, name="x"]()

    fn get_a_offset[T: AnyType]() -> Int:
        """Helper that uses offset_of with a generic type parameter."""
        return offset_of[T, name="a"]()

    # Test with different struct types that have "x" field
    comptime off1 = get_x_offset[SimpleOffsetStruct]()
    assert_equal(off1, 0)

    # Test with struct that has "a" field
    comptime off2 = get_a_offset[OffsetTestStruct]()
    assert_equal(off2, 0)


fn get_all_offsets_generically[T: AnyType]() -> InlineArray[Int, 4]:
    """Helper to get all field offsets for a struct with up to 4 fields."""
    var offsets = InlineArray[Int, 4](fill=0)
    comptime count = struct_field_count[T]()

    @parameter
    for i in range(count):
        offsets[i] = offset_of[T, index=i]()

    return offsets^


def test_generic_offset_iteration():
    """Test getting all offsets through a generic function."""
    var offsets = get_all_offsets_generically[OffsetTestStruct]()

    assert_equal(offsets[0], 0)
    assert_equal(offsets[1], 8)
    assert_equal(offsets[2], 16)
    assert_equal(offsets[3], 20)


def test_offset_single_field_struct():
    """Test offset calculation for struct with a single field."""
    comptime off = offset_of[SingleFieldStruct, index=0]()
    assert_equal(off, 0)

    comptime off_by_name = offset_of[SingleFieldStruct, name="value"]()
    assert_equal(off_by_name, 0)


def test_offset_all_same_type_fields():
    """Test offset calculation when all fields have the same type."""
    from sys.info import size_of

    comptime a_off = offset_of[AllSameTypeStruct, name="a"]()
    comptime b_off = offset_of[AllSameTypeStruct, name="b"]()
    comptime c_off = offset_of[AllSameTypeStruct, name="c"]()
    comptime int64_size = size_of[Int64]()

    # All fields are Int64, so offsets should be multiples of Int64 size
    assert_equal(a_off, 0)
    assert_equal(b_off, int64_size)
    assert_equal(c_off, 2 * int64_size)


def test_offset_consistency_by_index_and_by_name():
    """Verify index= and name= overloads return identical values for all fields.
    """
    # OffsetTestStruct has fields: a, b, c, d
    # Verify index= and name= produce the same results for each
    comptime off_a_idx = offset_of[OffsetTestStruct, index=0]()
    comptime off_a_name = offset_of[OffsetTestStruct, name="a"]()
    assert_equal(off_a_idx, off_a_name)

    comptime off_b_idx = offset_of[OffsetTestStruct, index=1]()
    comptime off_b_name = offset_of[OffsetTestStruct, name="b"]()
    assert_equal(off_b_idx, off_b_name)

    comptime off_c_idx = offset_of[OffsetTestStruct, index=2]()
    comptime off_c_name = offset_of[OffsetTestStruct, name="c"]()
    assert_equal(off_c_idx, off_c_name)

    comptime off_d_idx = offset_of[OffsetTestStruct, index=3]()
    comptime off_d_name = offset_of[OffsetTestStruct, name="d"]()
    assert_equal(off_d_idx, off_d_name)


def test_offset_with_parametric_struct_varying_field_type():
    """Test offset calculation for parametric struct where field type varies.

    This tests that offset_of correctly handles generic type parameters where
    the field type (and thus its size/alignment) varies with the type parameter.
    The footer field offset depends on the size and alignment of the data field.
    """

    # With Int8 data: header (1 byte) + data (1 byte) = 2 bytes, footer at 2
    comptime footer_off_int8 = offset_of[
        GenericOffsetContainer[Int8], name="footer"
    ]()
    assert_equal(footer_off_int8, 2)

    # With Int64 data: header (1 byte) + padding (7 bytes) + data (8 bytes) = 16
    # footer needs alignment, but Int8 only needs 1-byte alignment, so at 16
    comptime footer_off_int64 = offset_of[
        GenericOffsetContainer[Int64], name="footer"
    ]()
    assert_equal(footer_off_int64, 16)

    # Verify data field offsets differ based on T's alignment
    comptime data_off_int8 = offset_of[
        GenericOffsetContainer[Int8], name="data"
    ]()
    comptime data_off_int64 = offset_of[
        GenericOffsetContainer[Int64], name="data"
    ]()

    # Int8 only needs 1-byte alignment, so data is at offset 1
    assert_equal(data_off_int8, 1)
    # Int64 needs 8-byte alignment, so data is at offset 8 (with 7 bytes padding)
    assert_equal(data_off_int64, 8)

    # Verify that the difference between instantiations is correct
    # footer_off_int64 - footer_off_int8 should reflect the size difference
    assert_true(footer_off_int64 > footer_off_int8)


def test_offset_matches_runtime_memory_layout():
    """Verify that offset_of matches actual memory layout using pointer arithmetic.

    This test creates struct instances and verifies that the compile-time
    offsets match the actual byte offsets in memory at runtime.
    """
    # Test with OffsetTestStruct which has alignment padding
    var s = OffsetTestStruct(a=1, b=2, c=3, d=4)

    # Get address of struct and each field
    var struct_addr = Int(UnsafePointer(to=s).bitcast[Int8]())
    var a_addr = Int(UnsafePointer(to=s.a))
    var b_addr = Int(UnsafePointer(to=s.b))
    var c_addr = Int(UnsafePointer(to=s.c))
    var d_addr = Int(UnsafePointer(to=s.d))

    # Verify offsets match
    assert_equal(a_addr - struct_addr, offset_of[OffsetTestStruct, name="a"]())
    assert_equal(b_addr - struct_addr, offset_of[OffsetTestStruct, name="b"]())
    assert_equal(c_addr - struct_addr, offset_of[OffsetTestStruct, name="c"]())
    assert_equal(d_addr - struct_addr, offset_of[OffsetTestStruct, name="d"]())

    # Also test with SimpleOffsetStruct (no padding between fields)
    var simple = SimpleOffsetStruct(x=10, y=20)
    var simple_addr = Int(UnsafePointer(to=simple).bitcast[Int8]())
    var x_addr = Int(UnsafePointer(to=simple.x))
    var y_addr = Int(UnsafePointer(to=simple.y))

    assert_equal(
        x_addr - simple_addr, offset_of[SimpleOffsetStruct, name="x"]()
    )
    assert_equal(
        y_addr - simple_addr, offset_of[SimpleOffsetStruct, name="y"]()
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
