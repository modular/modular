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
# RUN: %mojo %s | FileCheck %s

# The hand-written form of the members the compiler generates for
#
#     __enum Token(ImplicitlyCopyable, Writable):
#         case eof
#         case ident(String)
#         case span(Int, Int)
#
# `std.builtin._enum` is written against exactly these members.

from std.builtin._enum import __Enum, __EnumStorage


struct Token(ImplicitlyCopyable, Writable, __Enum):
    comptime _enum_case_names = ParameterList.of[
        "eof".value, "ident".value, "span".value
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, String, Tuple[Int, Int]
    ].values
    var __enum_storage: __EnumStorage[Self._enum_case_types]

    comptime eof: Self = Self.__enum_init[0, NoneType]()

    @staticmethod
    def ident(var payload: String) -> Self:
        return Self.__enum_init[1, String](payload^)

    @staticmethod
    def span(var payload0: Int, var payload1: Int) -> Self:
        return Self.__enum_init[2, Tuple[Int, Int]](payload0, payload1)


def describe(token: Token) -> String:
    match token:
        case .eof:
            return "eof"
        case .ident(ref name):
            return "ident " + name
        case .span(ref range):
            return "span " + String(range[0]) + ".." + String(range[1])


def main():
    # CHECK: eof
    print(describe(Token.eof))
    # CHECK-NEXT: ident x
    print(describe(Token.ident("x")))
    # CHECK-NEXT: span 1..2
    print(describe(Token.span(1, 2)))

    # The `EnumLike` methods are `__Enum`'s defaults over the storage field.
    # CHECK-NEXT: 0 1 2
    print(
        Token.eof._get_enum_discriminant(),
        Token.ident("x")._get_enum_discriminant(),
        Token.span(1, 2)._get_enum_discriminant(),
    )

    # Copy, move and the derived `write_to` go through the storage.
    var original = Token.ident("copied")
    var copy = original
    var moved = original^
    # CHECK-NEXT: Token.ident(copied) Token.ident(copied)
    print(copy, moved)
    # CHECK-NEXT: Token.eof Token.span((1, 2))
    print(Token.eof, Token.span(1, 2))
