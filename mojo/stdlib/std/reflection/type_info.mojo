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
"""Provides type and function name introspection utilities.

This module provides compile-time introspection of type and function names:

- `get_type_name[T]()` - returns the name of a type
- `get_function_name[func]()` - returns the source name of a function
- `get_linkage_name[func]()` - returns the symbol/linkage name of a function

Example:

```mojo
from reflection import get_type_name, get_function_name

struct Point:
    var x: Int
    var y: Float64

fn my_function():
    pass

fn main():
    print(get_type_name[Point]())        # "Point"
    print(get_function_name[my_function]())  # "my_function"
```
"""

from sys.info import _current_target, _TargetType
from collections.string.string_slice import get_static_string


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
    type_type: __TypeOfAllTypes,
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


# TODO: This currently does not strip the module name from the inner type name.
# For example, Generic[Foo] should return "Generic[Foo]" but currently returns
# "Generic[module_name.Foo]".
fn _unqualified_type_name[type: AnyType]() -> StaticString:
    comptime name = get_type_name[type]()
    comptime parameter_list_start = name.find("[")
    if parameter_list_start == -1:
        return name.split(".")[-1]
    else:
        comptime base_name = name[:parameter_list_start].split(".")[-1]
        comptime parameters = name[parameter_list_start:]
        return get_static_string[base_name, parameters]()
