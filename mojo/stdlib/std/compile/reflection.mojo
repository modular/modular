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

    Parameters:
        T: A struct type.

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

    Parameters:
        T: A struct type.

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

    Parameters:
        T: A struct type.

    Returns:
        The number of fields in the struct.
    """
    return std.builtin.Variadic.size(get_struct_field_types[T]())
