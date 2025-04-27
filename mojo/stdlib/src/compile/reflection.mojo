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

from collections.string import StaticString
from sys.info import _current_target


fn get_linkage_name[
    func_type: AnyTrivialRegType, //,
    target: __mlir_type.`!kgen.target`,
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
    return __mlir_attr[
        `#kgen.param.expr<get_linkage_name,`,
        target,
        `,`,
        func,
        `> : !kgen.string`,
    ]


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
