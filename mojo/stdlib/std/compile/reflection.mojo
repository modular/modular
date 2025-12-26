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
"""Compile-time reflection utilities.

This module provides compile-time reflection capabilities for introspecting
Mojo types, functions, and struct fields.

Struct Field Reflection:

The struct field reflection APIs (`get_struct_field_types`, `get_struct_field_names`,
`get_struct_field_count`, `struct_field_index_by_name`, `struct_field_type_by_name`)
allow compile-time introspection of struct fields.

Note: These APIs require concrete types. They do not work with generic type
parameters. For example:

```mojo
# This works - Point is a concrete type
comptime names = get_struct_field_names[Point]()

# This does NOT work - T is a generic parameter
fn serialize[T: AnyType](value: T):
    # Error: T is not yet specialized
    comptime names = get_struct_field_names[T]()
```

This limitation exists because the reflection attributes are evaluated before
generic type parameters are specialized to concrete types.
"""

from sys.info import _current_target, _TargetType


fn get_linkage_name[
    func_type: AnyType,
    //,
    func: func_type,
    *,
    target: _TargetType = _current_target(),
]() -> StaticString:
    """Returns `func`'s symbol name.

    Parameters:
        func_type: Type of func.
        func: A mojo function.
        target: The compilation target, defaults to the current target.

    Returns:
        Symbol name.
    """
    var res = __mlir_attr[
        `#kgen.get_linkage_name<`,
        target,
        `,`,
        func,
        `> : !kgen.string`,
    ]
    return StaticString(res)


fn get_function_name[func_type: AnyType, //, func: func_type]() -> StaticString:
    """Returns `func`'s name as declared in the source code.

    The returned name does not include any information about the function's
    parameters, arguments, or return type, just the name as declared in the
    source code.

    Parameters:
        func_type: Type of func.
        func: A mojo function.

    Returns:
        The function's name as declared in the source code.
    """
    var res = __mlir_attr[`#kgen.get_source_name<`, func, `> : !kgen.string`]
    return StaticString(res)


fn get_type_name[
    type_type: AnyTrivialRegType,
    //,
    type: type_type,
    *,
    qualified_builtins: Bool = False,
]() -> StaticString:
    """Returns the struct name of the given type parameter.

    Parameters:
        type_type: Type of type.
        type: A mojo type.
        qualified_builtins: Whether to print fully qualified builtin type names
            (e.g. `std.builtin.int.Int`) or shorten them (e.g. `Int`).

    Returns:
        Type name.
    """
    var res = __mlir_attr[
        `#kgen.get_type_name<`,
        type,
        `, `,
        qualified_builtins._mlir_value,
        `> : !kgen.string`,
    ]
    return StaticString(res)


fn get_struct_field_types[
    T: AnyType,
]() -> __mlir_type[`!kgen.variadic<!kgen.type>`]:
    """Returns the types of all fields in struct `T` as a variadic.

    This function provides compile-time reflection over struct fields. It returns
    the types of all fields (including private fields) in declaration order.

    For nested structs, this returns the struct type itself, not its flattened
    fields. Use recursive calls to introspect nested types.

    Note: `T` must be a concrete type, not a generic type parameter.
    See the module documentation for details on this limitation.

    Parameters:
        T: A concrete struct type.

    Returns:
        A variadic of types, one for each field in the struct.
    """
    return __mlir_attr[
        `#kgen.struct_field_types<`,
        T,
        `> : !kgen.variadic<!kgen.type>`,
    ]


fn _get_struct_field_names_raw[
    T: AnyType,
]() -> __mlir_type[`!kgen.variadic<!kgen.string>`]:
    """Returns the names of all fields in struct `T` as a raw variadic.

    This is an internal helper. Use `get_struct_field_names` for a more
    ergonomic API that returns `InlineArray[StaticString, N]`.
    """
    return __mlir_attr[
        `#kgen.struct_field_names<`,
        T,
        `> : !kgen.variadic<!kgen.string>`,
    ]


fn get_struct_field_names[
    T: AnyType,
]() -> InlineArray[StaticString, get_struct_field_count[T]()]:
    """Returns the names of all fields in struct `T` as an InlineArray.

    This function provides compile-time reflection over struct field names.
    It returns the names of all fields (including private fields) in declaration
    order.

    Note: `T` must be a concrete type, not a generic type parameter.
    See the module documentation for details on this limitation.

    Parameters:
        T: A concrete struct type.

    Returns:
        An InlineArray of StaticStrings, one for each field name in the struct.
    """
    comptime count = get_struct_field_count[T]()
    comptime raw = _get_struct_field_names_raw[T]()
    var result = InlineArray[StaticString, count](uninitialized=True)

    @parameter
    for i in range(count):
        result[i] = StaticString(raw[i])

    return result


fn get_struct_field_count[
    T: AnyType,
]() -> Int:
    """Returns the number of fields in struct `T`.

    Note: `T` must be a concrete type, not a generic type parameter.
    See the module documentation for details on this limitation.

    Parameters:
        T: A concrete struct type.

    Returns:
        The number of fields in the struct.
    """
    return std.builtin.Variadic.size(get_struct_field_types[T]())


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


@register_passable("trivial")
struct ReflectedType[T: __mlir_type.`!kgen.type`]:
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
