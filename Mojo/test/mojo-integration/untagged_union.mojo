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
# A raw `!kgen.untagged_union` field in a user struct:
# 1. Check the layout and member projection of a raw union in a user struct.
# 2. Copy the union, switch its active member and preserve the original.
# RUN: %mojo %s | FileCheck %s

from std.sys import align_of, size_of


struct Storage(Copyable):
    comptime Cases = TypeList.of[Trait=AnyType, Int, Float64]
    var raw: __mlir_type[
        `!kgen.untagged_union<[rebind(:`,
        type_of(Self.Cases.values),
        ` `,
        Self.Cases.values,
        `)]>`,
    ]

    def __init__(out self, value: Int):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        self.member[0]() = value

    def __init__(out self, *, copy: Self):
        self.raw = copy.raw

    def member[id: Int](ref self) -> ref[self] Self.Cases()[id]:
        var member = __mlir_op.`lit.ref.union.cast`[index=id.__mlir_index__()](
            __get_mvalue_as_litref(self.raw)
        )
        var typed_member = rebind[
            Pointer[Self.Cases()[id], origin_of(self.raw)]._mlir_lit_ref
        ](member)
        return Pointer[Self.Cases()[id], origin_of(self)](
            _mlir_value=__mlir_op.`lit.ref.to_pointer`(typed_member)
        )[]


def main():
    # CHECK: layout 8 8
    print("layout", size_of[Storage](), align_of[Storage]())
    var value = Storage(7)
    # CHECK-NEXT: initialized 7
    print("initialized", value.member[0]())
    value.member[0]() = 11
    var copied = value.copy()
    # CHECK-NEXT: copied 11
    print("copied", copied.member[0]())
    copied.member[1]() = 2.5
    # CHECK-NEXT: switched 2.5
    print("switched", copied.member[1]())
    # CHECK-NEXT: original 11
    print("original", value.member[0]())
