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
"""This module includes the inlined_assembly function."""

from builtin.string_literal import get_string_literal_slice
from collections.string.string_slice import StringSlice

from .intrinsics import _mlirtype_is_eq


@always_inline("nodebug")
fn inlined_assembly[
    asm: StringSlice,
    result_type: AnyTrivialRegType,
    *types: AnyType,
    constraints: StringSlice,
    has_side_effect: Bool = True,
](*args: *types) -> result_type:
    """Generates assembly via inline assembly."""
    var loaded_pack = args.get_loaded_kgen_pack()

    alias asm_literal = get_string_literal_slice[asm]().value
    alias constraints_literal = get_string_literal_slice[constraints]().value

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly=asm_literal,
                constraints=constraints_literal,
                hasSideEffects = __mlir_attr.unit,
            ](loaded_pack)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly=asm_literal,
                constraints=constraints_literal,
            ](loaded_pack)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly=asm_literal,
                constraints=constraints_literal,
                hasSideEffects = __mlir_attr.unit,
            ](loaded_pack)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly=asm_literal,
                constraints=constraints_literal,
            ](loaded_pack)
