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
    func_type: AnyTrivialRegType, //,
    target: _TargetType,
    func: func_type,
]() -> StaticString:
    """Returns `func` symbol name.

    Parameters:
        func_type: Type of func.
        target: The compilation target.
        func: A mojo function.

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


fn get_linkage_name[
    func_type: AnyTrivialRegType, //,
    func: func_type,
]() -> StaticString:
    """Returns `func` symbol name.

    Parameters:
        func_type: Type of func.
        func: A mojo function.

    Returns:
        Symbol name.
    """
    return get_linkage_name[_current_target(), func]()


fn get_type_name[
    type_type: AnyTrivialRegType, //,
    type: type_type,
    *,
    qualified_builtins: Bool = False,
]() -> StaticString:
    """Returns the struct name of the given type parameter.

    Parameters:
        type_type: Type of type.
        type: A mojo type.
        qualified_builtins: Whether to print fully qualified builtin type names
            (e.g. `stdlib.builtin.int.Int`) or shorten them (e.g. `Int`).

    Returns:
        Type name.
    """
    var res = __mlir_attr[
        `#kgen.get_type_name<`,
        type,
        `, `,
        qualified_builtins.value,
        `> : !kgen.string`,
    ]
    return StaticString(res)
