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
"""Deprecated free-function reflection API.

The free functions and `ReflectedType[T]` wrapper in this module are
**deprecated** in favor of the unified `reflect[T]() -> Reflected[T]` API.
New code should use `reflect[T]()` from `std.reflection` directly:

| Deprecated                                | Replacement                              |
|-------------------------------------------|------------------------------------------|
| `struct_field_count[T]()`                 | `reflect[T]().field_count()`             |
| `struct_field_names[T]()`                 | `reflect[T]().field_names()`             |
| `struct_field_types[T]()`                 | `reflect[T]().field_types()`             |
| `struct_field_index_by_name[T, name]()`   | `reflect[T]().field_index[name]()`       |
| `struct_field_type_by_name[T, name]()`    | `reflect[T]().field_type[name]()`        |
| `struct_field_ref[idx](s)`                | `reflect[T]().field_ref[idx](s)`         |
| `is_struct_type[T]()`                     | `reflect[T]().is_struct()`               |
| `offset_of[T, name=...]()`                | `reflect[T]().field_offset[name=...]()` |
| `offset_of[T, index=...]()`               | `reflect[T]().field_offset[index=...]()`|
| `ReflectedType[T]`                        | `Reflected[T]`                           |

Each wrapper here delegates to the new API; they remain to give external
callers time to migrate and will be removed in a future release.
"""

from std.sys.info import _current_target, _TargetType

from .reflect import Reflected, reflect


@deprecated("Use `reflect[T]().field_index[name]()` instead.")
def struct_field_index_by_name[
    T: AnyType,
    name: StringLiteral,
]() -> Int:
    """Deprecated: use `reflect[T]().field_index[name]()` instead.

    Returns the index of the field with the given name in struct `T`.

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


@deprecated("Use `Reflected[T]` from `std.reflection` instead.")
struct ReflectedType[T: AnyType](TrivialRegisterPassable):
    """Deprecated: use `Reflected[T]` from `std.reflection` instead.

    Wrapper struct for compile-time type values from reflection.

    This struct wraps a `!kgen.non_struct_type` value as a type parameter, allowing
    type values to be returned from functions and passed around at compile time.

    Parameters:
        T: The wrapped type value.

    Example:
        ```mojo
        from std.reflection import struct_field_type_by_name

        struct MyStruct:
            var x: Int
            var y: Float64

        def main():
            # Get the type of field "x" in MyStruct
            comptime field_type = struct_field_type_by_name[MyStruct, "x"]()
            # Access the underlying type via the T parameter
            var value: field_type.T = 42
        ```
    """

    @always_inline("nodebug")
    def __init__(out self):
        """Create a ReflectedType instance."""
        pass


@deprecated(
    "Use `reflect[StructT]().field_type[name]()` instead. The returned"
    " `Reflected[FieldT]` exposes the field type via its `T` parameter."
)
def struct_field_type_by_name[
    StructT: AnyType,
    name: StringLiteral,
]() -> Reflected[
    __mlir_attr[
        `#kgen.struct_field_type_by_name<`,
        StructT,
        `, `,
        name.value,
        `> : `,
        AnyType,
    ]
]:
    """Deprecated: use `reflect[StructT]().field_type[name]()` instead.

    Returns the type of the field with the given name in struct `StructT`.

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
        from std.reflection import struct_field_type_by_name

        struct Point:
            var x: Int
            var y: Float64

        def example():
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
#   def foo[T: AnyType]():
#       # Works - T is concrete when the attribute is evaluated
#       comptime count = struct_field_count[T]()
#
# The implementation approach varies by API:
# - struct_field_count: Uses #kgen.param_list.size<#kgen.struct_field_types<T>>
# - struct_field_types/names: Use magic functions for type validation
# - struct_field_index/type_by_name: Use KGEN attributes directly (require
#   compile-time string literal for field name, only available with concrete
#   types anyway)
# ===----------------------------------------------------------------------=== #


@deprecated("Use `reflect[T]().field_count()` instead.")
@always_inline("builtin")
def struct_field_count[T: AnyType]() -> Int:
    """Deprecated: use `reflect[T]().field_count()` instead.

    Returns the number of fields in struct `T`.

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
        from std.reflection import struct_field_count

        struct MyStruct:
            var x: Int
            var y: Float64

        def count_fields[T: AnyType]() -> Int:
            return struct_field_count[T]()

        def main():
            print(count_fields[MyStruct]())  # Prints field count
        ```
    """
    return reflect[T]().field_count()


@deprecated("Use `reflect[T]().field_types()` instead.")
comptime struct_field_types[
    T: AnyType,
] = TypeList[
    __mlir_attr[
        `#kgen.struct_field_types<`, T, `> : !kgen.param_list<`, AnyType, `>`
    ]
]
"""Deprecated: use `reflect[T]().field_types()` instead.

Returns the types of all fields in struct `T` as a TypeList.

This function works with both concrete types and generic type parameters.

For nested structs, this returns the struct type itself, not its flattened
fields. Use recursive calls to introspect nested types.

Parameters:
    T: A struct type.

Returns:
    A list of types, one for each field in the struct.

Example:
    ```mojo
    from std.reflection import get_type_name, struct_field_types, struct_field_count

    struct MyStruct:
        var x: Int
        var y: Float64

    def print_field_types[T: AnyType]():
        comptime types = struct_field_types[T]()
        comptime for i in range(struct_field_count[T]()):
            print(get_type_name[types[i]]())

    def main():
        print_field_types[MyStruct]()  # Works with any struct!
    ```
"""

comptime _struct_field_names_raw[
    T: AnyType,
] = ParameterList[
    __mlir_attr[
        `#kgen.struct_field_names<`, T, `> : !kgen.param_list<!kgen.string>`
    ]
]


@deprecated("Use `reflect[T]().field_names()` instead.")
def struct_field_names[
    T: AnyType,
]() -> InlineArray[StaticString, reflect[T]().field_count()]:
    """Deprecated: use `reflect[T]().field_names()` instead.

    Returns the names of all fields in struct `T` as an InlineArray.

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
        from std.reflection import struct_field_names, struct_field_count

        struct MyStruct:
            var x: Int
            var y: Float64

        def print_field_names[T: AnyType]():
            comptime names = struct_field_names[T]()
            comptime for i in range(struct_field_count[T]()):
                print(names[i])

        def main():
            print_field_names[MyStruct]()  # Works with any struct!
        ```
    """
    comptime count = reflect[T]().field_count()
    comptime raw = _struct_field_names_raw[T]()

    # Safety: uninitialized=True is safe here because the comptime for loop
    # guarantees complete initialization of all elements at compile time.
    var result = InlineArray[StaticString, count](uninitialized=True)

    comptime for i in range(raw.size):
        result[i] = comptime (StaticString(raw[i]))

    return result^


@deprecated("Use `reflect[T]().is_struct()` instead.")
def is_struct_type[T: AnyType]() -> Bool:
    """Deprecated: use `reflect[T]().is_struct()` instead.

    Returns `True` if `T` is a Mojo struct type, `False` otherwise.

    This function distinguishes between Mojo struct types and MLIR primitive
    types (such as `__mlir_type.index` or `__mlir_type.i64`). This is useful
    when iterating over struct fields that may contain MLIR types, allowing
    you to guard calls to struct-specific reflection APIs like
    `struct_field_count`, `struct_field_names`, or `struct_field_types`
    which produce compile errors when used on MLIR types.

    Note: Since all reflection functions take `[T: AnyType]` parameters, you
    can only pass types to them. Attempting to pass a trait, function, or
    comptime value would result in a compiler error regardless of this check.

    Note: When using this function as a guard, you must use `comptime if`
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
        from std.reflection import get_type_name, is_struct_type, struct_field_count

        def process_type[T: AnyType]():
            comptime if is_struct_type[T]():
                # Safe to use struct reflection APIs
                comptime count = struct_field_count[T]()
                print("Struct with", count, "fields")
            else:
                print("Non-struct type:", get_type_name[T]())
        ```
    """
    return reflect[T]().is_struct()


# ===----------------------------------------------------------------------=== #
# Struct Field Reference API
# ===----------------------------------------------------------------------=== #


@deprecated("Use `reflect[T]().field_ref[idx](s)` instead.")
@always_inline("nodebug")
def struct_field_ref[
    idx: Int, T: AnyType
](ref s: T) -> ref[s] reflect[T]().field_types()[idx]:
    """Deprecated: use `reflect[T]().field_ref[idx](s)` instead.

    Returns a reference to the struct field at the given index.

    This function provides reference-based access to struct fields by index,
    enabling reflection-based utilities to work with non-copyable types by
    returning references instead of copies. It works with both literal indices
    and parametric indices (such as loop variables in `comptime for` loops),
    and with both concrete struct types and generic type parameters.

    Parameters:
        idx: The zero-based index of the field.
        T: A struct type.

    Args:
        s: The struct value to access.

    Constraints:
        `T` must be a struct type. The index must be in range
        `[0, struct_field_count[T]())`.

    Returns:
        A reference to the field at the specified index, with the same
        mutability as `s`.

    Example:
        ```mojo
        from std.reflection import struct_field_ref

        @fieldwise_init
        struct Container:
            var id: Int
            var name: String

        def inspect(mut c: Container):
            ref id_ref = struct_field_ref[0](c)
            ref name_ref = struct_field_ref[1](c)

            # Mutation through reference
            struct_field_ref[0](c) = 42

        def main():
            var c = Container(id=1, name="test")
            inspect(c)
        ```
    """
    return reflect[T]().field_ref[idx](s)


# ===----------------------------------------------------------------------=== #
# Struct Field Offset APIs
# ===----------------------------------------------------------------------=== #


def _struct_field_offset_by_index[
    T: AnyType, idx: Int, target: _TargetType = _current_target()
]() -> Int:
    """Internal: returns byte offset of field at given index. Use `offset_of`.
    """
    return Int(
        mlir_value=__mlir_attr[
            `#kgen.struct_field_offset_by_index<`,
            T,
            `, `,
            idx._int_mlir_index(),
            `, `,
            target,
            `> : index`,
        ]
    )


def _struct_field_offset_by_name[
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


@deprecated("Use `reflect[T]().field_offset[name=name]()` instead.")
def offset_of[
    T: AnyType, *, name: StringLiteral, target: _TargetType = _current_target()
]() -> Int:
    """Deprecated: use `reflect[T]().field_offset[name=name]()` instead.

    Returns the byte offset of a field within a struct by name.

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
        from std.reflection import offset_of

        struct Point:
            var x: Int      # offset 0
            var y: Float64  # offset 8 (aligned to 8 bytes)

        def main():
            comptime x_off = offset_of[Point, name="x"]()  # 0
            comptime y_off = offset_of[Point, name="y"]()  # 8
        ```
    """
    return _struct_field_offset_by_name[T, name, target]()


@deprecated("Use `reflect[T]().field_offset[index=index]()` instead.")
def offset_of[
    T: AnyType, *, index: Int, target: _TargetType = _current_target()
]() -> Int:
    """Deprecated: use `reflect[T]().field_offset[index=index]()` instead.

    Returns the byte offset of a field within a struct by index.

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
        from std.reflection import offset_of

        struct Point:
            var x: Int      # offset 0
            var y: Float64  # offset 8 (aligned to 8 bytes)

        def main():
            comptime x_off = offset_of[Point, index=0]()  # 0
            comptime y_off = offset_of[Point, index=1]()  # 8
        ```
    """
    return _struct_field_offset_by_index[T, index, target]()
