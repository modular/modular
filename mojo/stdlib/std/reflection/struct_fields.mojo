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
"""Provides struct field reflection and introspection utilities.

This module provides compile-time introspection of struct fields:

- `struct_field_count[T]()` - returns the number of fields
- `struct_field_names[T]()` - returns an `InlineArray[StaticString, N]` of field names
- `struct_field_types[T]()` - returns a variadic of field types

These APIs work with both concrete types and generic type parameters, enabling
generic serialization, comparison, and other reflection-based utilities.

For field lookup by name (concrete types only):

- `struct_field_index_by_name[T, name]()` - returns the index of a field by name
- `struct_field_type_by_name[T, name]()` - returns the type of a field by name

Example iterating over all fields (works with generics):

```mojo
fn print_fields[T: AnyType]():
    comptime names = struct_field_names[T]()
    comptime for i in range(struct_field_count[T]()):
        print(names[i])

fn main():
    print_fields[Point]()  # Works with any struct!
```

Example looking up a field by name:

```mojo
comptime idx = struct_field_index_by_name[Point, "x"]()  # 0
comptime field_type = struct_field_type_by_name[Point, "y"]()
var value: field_type.T = 3.14  # field_type.T is Float64
```

For accessing struct field values by index (returns a reference, not a copy):

- `__struct_field_ref(idx, ref s)` - returns a reference to the field at index

The `__struct_field_ref` magic function enables reflection-based utilities to work
with non-copyable types by returning references instead of copies. It works with
both literal indices and parametric indices (such as loop variables in
`@parameter for` loops):

```mojo
struct Container:
    var id: Int
    var resource: NonCopyableResource

fn inspect(ref c: Container):
    # Get references to fields without copying
    ref id_ref = __struct_field_ref(0, c)
    ref resource_ref = __struct_field_ref(1, c)
    print("id:", id_ref)
    print("resource:", resource_ref.data)

    # Mutation through reference also works
    __struct_field_ref(0, c) = 42

# Works in generic contexts with parameter indices
fn print_all_fields[T: AnyType](ref s: T):
    comptime names = struct_field_names[T]()
    comptime for i in range(struct_field_count[T]()):
        print(names[i], "=", __struct_field_ref(i, s))
```

For struct field byte offsets (useful for low-level memory operations):

- `offset_of[T, name="field_name"]()` - get offset by field name
- `offset_of[T, index=0]()` - get offset by field index

Example:

```mojo
struct Point:
    var x: Int      # offset 0
    var y: Float64  # offset 8 (aligned to 8 bytes)

fn main():
    comptime x_off = offset_of[Point, name="x"]()  # 0
    comptime y_off = offset_of[Point, index=1]()   # 8
```
"""

from sys.info import _current_target, _TargetType


fn struct_field_index_by_name[
    T: AnyType,
    name: StringLiteral,
]() -> Int:
    """Returns the index of the field with the given name in struct `T`.

    This function provides compile-time lookup of a struct field's index by name.
    It produces a compile error if the field name does not exist in the struct.

    Note: `T` must be a concrete type, not a generic type parameter.
    See the module documentation for details on this limitation.

    Parameters:
        T: A concrete struct type.
        name: The name of the field to look up.

    Returns:
        The zero-based index of the field in the struct.
    """
    # Access the StringLiteral's `value` type parameter to get the raw string
    comptime str_value = name.value
    return Int(
        mlir_value=__mlir_attr[
            `#kgen.struct_field_index_by_name<`,
            T,
            `, `,
            str_value,
            `> : index`,
        ]
    )


struct ReflectedType[T: __mlir_type.`!kgen.type`](TrivialRegisterPassable):
    """Wrapper struct for compile-time type values from reflection.

    This struct wraps a `!kgen.type` value as a type parameter, allowing
    type values to be returned from functions and passed around at compile time.

    Parameters:
        T: The wrapped type value.

    Example:
        ```mojo
        # Get the type of field "x" in MyStruct
        comptime field_type = struct_field_type_by_name[MyStruct, "x"]()
        # Access the underlying type via the T parameter
        var value: field_type.T = 42
        ```
    """

    @always_inline("nodebug")
    fn __init__(out self):
        """Create a ReflectedType instance."""
        pass


fn struct_field_type_by_name[
    StructT: AnyType,
    name: StringLiteral,
]() -> ReflectedType[
    __mlir_attr[
        `#kgen.struct_field_type_by_name<`,
        StructT,
        `, `,
        name.value,
        `> : !kgen.type`,
    ]
]:
    """Returns the type of the field with the given name in struct `StructT`.

    This function provides compile-time lookup of a struct field's type by name.
    It produces a compile error if the field name does not exist in the struct.

    The returned `ReflectedType` wrapper contains the field type as its `T`
    parameter, which can be used as a type in declarations.

    Note: `StructT` must be a concrete type, not a generic type parameter.
    See the module documentation for details on this limitation.

    Parameters:
        StructT: A concrete struct type to introspect.
        name: The name of the field to look up.

    Returns:
        A ReflectedType wrapper containing the field's type.

    Example:
        ```mojo
        struct Point:
            var x: Int
            var y: Float64

        fn example():
            # Get the type of field "x"
            comptime x_type = struct_field_type_by_name[Point, "x"]()
            # x_type.T is Int
            var value: x_type.T = 42
        ```
    """
    return {}


# ===----------------------------------------------------------------------=== #
# Struct Field Reflection APIs
# ===----------------------------------------------------------------------=== #
#
# Implementation Note: KGEN Attributes with ContextuallyEvaluatedAttrInterface
#
# The struct field reflection APIs use KGEN attributes that implement
# ContextuallyEvaluatedAttrInterface. This interface allows attributes to be
# evaluated during elaboration AFTER generic type parameters have been
# specialized to concrete types. This enables reflection to work with generic
# code:
#
#   fn foo[T: AnyType]():
#       # Works - T is concrete when the attribute is evaluated
#       comptime count = struct_field_count[T]()
#
# The implementation approach varies by API:
# - struct_field_count: Uses #kgen.variadic.size<#kgen.struct_field_types<T>>
# - struct_field_types/names: Use magic functions for type validation
# - struct_field_index/type_by_name: Use KGEN attributes directly (require
#   compile-time string literal for field name, only available with concrete
#   types anyway)
# ===----------------------------------------------------------------------=== #


fn struct_field_count[T: AnyType]() -> Int:
    """Returns the number of fields in struct `T`.

    This function works with both concrete types and generic type parameters.

    Note: For best performance, assign the result to a `comptime` variable to
    ensure compile-time evaluation:
        `comptime count = struct_field_count[T]()`

    Parameters:
        T: A struct type.

    Constraints:
        T must be a struct type. Passing a non-struct type results in a
        compile-time error.

    Returns:
        The number of fields in the struct.

    Example:
        ```mojo
        fn count_fields[T: AnyType]() -> Int:
            return struct_field_count[T]()

        fn main():
            print(count_fields[MyStruct]())  # Prints field count
        ```
    """
    # Use variadic.size on struct_field_types to get the count.
    # This avoids needing a dedicated struct_field_count attribute.
    return Int(
        mlir_value=__mlir_attr[
            `#kgen.variadic.size<#kgen.struct_field_types<`,
            T,
            `> : !kgen.variadic<!kgen.type>> : index`,
        ]
    )


fn struct_field_types[
    T: AnyType,
]() -> __mlir_type[`!kgen.variadic<!kgen.type>`]:
    """Returns the types of all fields in struct `T` as a variadic.

    This function works with both concrete types and generic type parameters.

    For nested structs, this returns the struct type itself, not its flattened
    fields. Use recursive calls to introspect nested types.

    Note: For best performance, assign the result to a `comptime` variable to
    ensure compile-time evaluation:
        `comptime types = struct_field_types[T]()`

    Parameters:
        T: A struct type.

    Constraints:
        T must be a struct type. Passing a non-struct type results in a
        compile-time error.

    Returns:
        A variadic of types, one for each field in the struct. Access individual
        types by indexing: `types[i]`.

    Example:
        ```mojo
        from reflection import get_type_name

        fn print_field_types[T: AnyType]():
            comptime types = struct_field_types[T]()
            comptime for i in range(struct_field_count[T]()):
                print(get_type_name[types[i]]())

        fn main():
            print_field_types[MyStruct]()  # Works with any struct!
        ```
    """
    return __struct_field_types(T)


fn _struct_field_names_raw[
    T: AnyType,
]() -> __mlir_type[`!kgen.variadic<!kgen.string>`]:
    """Returns the names of all fields in struct `T` as a raw variadic.

    This is an internal helper. Use `struct_field_names` for a more
    ergonomic API that returns `InlineArray[StaticString, N]`.
    """
    return __struct_field_names(T)


fn struct_field_names[
    T: AnyType,
]() -> InlineArray[StaticString, struct_field_count[T]()]:
    """Returns the names of all fields in struct `T` as an InlineArray.

    This function works with both concrete types and generic type parameters.

    Note: For best performance, assign the result to a `comptime` variable to
    ensure compile-time evaluation:
        `comptime names = struct_field_names[T]()`

    Parameters:
        T: A struct type.

    Constraints:
        T must be a struct type. Passing a non-struct type results in a
        compile-time error.

    Returns:
        An InlineArray of StaticStrings, one for each field name in the struct.

    Example:
        ```mojo
        fn print_field_names[T: AnyType]():
            comptime names = struct_field_names[T]()
            comptime for i in range(struct_field_count[T]()):
                print(names[i])

        fn main():
            print_field_names[MyStruct]()  # Works with any struct!
        ```
    """
    comptime count = struct_field_count[T]()
    comptime raw = _struct_field_names_raw[T]()

    # Safety: uninitialized=True is safe here because the @parameter for loop
    # guarantees complete initialization of all elements at compile time.
    var result = InlineArray[StaticString, count](uninitialized=True)

    comptime for i in range(count):
        result[i] = StaticString(raw[i])

    return result^


fn is_struct_type[T: AnyType]() -> Bool:
    """Returns `True` if `T` is a Mojo struct type, `False` otherwise.

    This function distinguishes between Mojo struct types and MLIR primitive
    types (such as `__mlir_type.index` or `__mlir_type.i64`). This is useful
    when iterating over struct fields that may contain MLIR types, allowing
    you to guard calls to struct-specific reflection APIs like
    `struct_field_count`, `struct_field_names`, or `struct_field_types`
    which produce compile errors when used on MLIR types.

    Note: Since all reflection functions take `[T: AnyType]` parameters, you
    can only pass types to them. Attempting to pass a trait, function, or
    comptime value would result in a compiler error regardless of this check.

    Note: When using this function as a guard, you must use `@parameter if`
    (not a runtime `if` statement) because the guarded reflection APIs are
    evaluated at compile time. A runtime `if` would still cause a compile
    error since the compiler evaluates both branches.

    Note: When an MLIR type like `__mlir_type.index` is passed directly as a
    type parameter, it returns `True` because it gets wrapped as a Mojo type.
    However, when the same MLIR type is obtained via `struct_field_types`
    (e.g., from a struct field declared as `var x: __mlir_type.index`), it
    returns `False`. This is the expected behavior for the primary use case of
    guarding reflection APIs when iterating over struct fields. When you obtain
    types via `struct_field_types` and pass them to a generic function, this
    function correctly identifies the MLIR types.

    Parameters:
        T: A type to check (either a Mojo struct type or an MLIR type).

    Returns:
        `True` if `T` is a Mojo struct type, `False` if it is an MLIR type.

    Example:
        ```mojo
        from reflection import get_type_name

        fn process_type[T: AnyType]():
            comptime if is_struct_type[T]():
                # Safe to use struct reflection APIs
                comptime count = struct_field_count[T]()
                print("Struct with", count, "fields")
            else:
                print("Non-struct type:", get_type_name[T]())
        ```
    """
    return __mlir_attr[
        `#kgen.is_struct_type<`,
        T,
        `> : i1`,
    ]


# ===----------------------------------------------------------------------=== #
# Struct Field Offset APIs
# ===----------------------------------------------------------------------=== #


fn _struct_field_offset_by_index[
    T: AnyType, idx: Int, target: _TargetType = _current_target()
]() -> Int:
    """Internal: returns byte offset of field at given index. Use `offset_of`.
    """
    return Int(
        mlir_value=__mlir_attr[
            `#kgen.struct_field_offset_by_index<`,
            T,
            `, `,
            idx,
            `, `,
            target,
            `> : index`,
        ]
    )


fn _struct_field_offset_by_name[
    T: AnyType, name: StringLiteral, target: _TargetType = _current_target()
]() -> Int:
    """Internal: returns byte offset of field with given name. Use `offset_of`.
    """
    # Access the StringLiteral's `value` type parameter to get the raw string
    comptime str_value = name.value
    return Int(
        mlir_value=__mlir_attr[
            `#kgen.struct_field_offset_by_name<`,
            T,
            `, `,
            str_value,
            `, `,
            target,
            `> : index`,
        ]
    )


fn offset_of[
    T: AnyType, *, name: StringLiteral, target: _TargetType = _current_target()
]() -> Int:
    """Returns the byte offset of a field within a struct by name.

    This function computes the byte offset from the start of the struct to the
    named field, accounting for alignment padding between fields. The offset
    is computed using the target's data layout.

    This is useful for low-level memory operations like no-copy serialization,
    memory-mapped I/O, or interfacing with C structs.

    Note: This function works with both concrete types and generic type parameters.

    Parameters:
        T: A struct type.
        name: The name of the field.
        target: The target architecture (defaults to current target).

    Constraints:
        T must be a struct type. The field name must exist in the struct.

    Returns:
        The byte offset of the field from the start of the struct.

    Example:
        ```mojo
        struct Point:
            var x: Int      # offset 0
            var y: Float64  # offset 8 (aligned to 8 bytes)

        fn main():
            comptime x_off = offset_of[Point, name="x"]()  # 0
            comptime y_off = offset_of[Point, name="y"]()  # 8
        ```
    """
    return _struct_field_offset_by_name[T, name, target]()


fn offset_of[
    T: AnyType, *, index: Int, target: _TargetType = _current_target()
]() -> Int:
    """Returns the byte offset of a field within a struct by index.

    This function computes the byte offset from the start of the struct to the
    specified field, accounting for alignment padding between fields. The offset
    is computed using the target's data layout.

    This is useful for low-level memory operations like no-copy serialization,
    memory-mapped I/O, or interfacing with C structs.

    Note: This function works with both concrete types and generic type parameters.

    Parameters:
        T: A struct type.
        index: The zero-based index of the field.
        target: The target architecture (defaults to current target).

    Constraints:
        T must be a struct type. The index must be valid (0 <= index < field_count).

    Returns:
        The byte offset of the field from the start of the struct.

    Example:
        ```mojo
        struct Point:
            var x: Int      # offset 0
            var y: Float64  # offset 8 (aligned to 8 bytes)

        fn main():
            comptime x_off = offset_of[Point, index=0]()  # 0
            comptime y_off = offset_of[Point, index=1]()  # 8
        ```
    """
    return _struct_field_offset_by_index[T, index, target]()
