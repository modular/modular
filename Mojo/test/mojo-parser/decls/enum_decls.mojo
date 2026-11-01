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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# An `__enum` is a struct marked `isEnum` whose cases are `lit.enum.case` ops
# in its body, in declaration order, each with the payload types written in its
# parentheses (none for a case without a payload) and its doc string, if any.

# CHECK-DAG: [[RED_DOC:#doc_string[0-9]*]] = #lit.doc.string<"Red.",
# CHECK-DAG: [[RGB_DOC:#doc_string[0-9]*]] = #lit.doc.string<"\0A    A color by its components.\0A    ",
# CHECK-DAG: [[ONLY_DOC:#doc_string[0-9]*]] = #lit.doc.string<"Only.",


# CHECK-LABEL: lit.struct.decl @Color(
# CHECK-SAME: attributes {isEnum,
# CHECK: lit.enum.case red : []{{$}}
# CHECK-NEXT: lit.enum.case green : []{{$}}
# CHECK-NEXT: lit.enum.case rgb : {{\[}}[[INT:!alias_Int[0-9]*]], [[INT]], [[INT]]]{{$}}
# CHECK: lit.fn @"is_red(
# CHECK: lit.fn @"make()"(){{.*}}isStatic
# CHECK: lit.alias.decl *"N`"
__enum Color:
    case red
    case green
    case rgb(Int, Int, Int)

    def is_red(self) -> Bool:
        return True

    @staticmethod
    def make() -> Int:
        return 1

    comptime N = 3


# CHECK-LABEL: lit.struct.decl @Result<
# CHECK-SAME: T: [[BOUND:!AnyType_Deinitable_Movable]], E: [[BOUND]]>
# CHECK-SAME: isEnum
# CHECK: lit.enum.case success : [!kgen.param<:[[BOUND]] T>]
# CHECK-NEXT: lit.enum.case failure : [!kgen.param<:[[BOUND]] E>]
__enum Result[T: Movable & Deinitable, E: Movable & Deinitable](Movable):
    case success(Self.T)
    case failure(Self.E)


# A payload can name a member declared after the case.
# CHECK-LABEL: lit.struct.decl @Forward(
# CHECK: lit.enum.case a : [!kgen.param<:!mt_Bool sugar_member_alias(!Forward, "Payload", !Bool)>]
# CHECK-NEXT: lit.alias.decl *"Payload`"
__enum Forward:
    case a(Self.Payload)

    comptime Payload = Bool


# The header takes what a struct's does: decorators, conformances and a
# `where` clause.
# CHECK-LABEL: lit.struct.decl @Bounded<n: !Int, {<
# CHECK-SAME: deprecationInfo = #lit.deprecation<"use Color">, isEnum
# CHECK: lit.enum.case a : {{\[}}[[INT]]]{{$}}
# CHECK-NEXT: lit.enum.case b : []{{$}}
@deprecated("use Color")
__enum Bounded[n: Int](Movable) where n > 0:
    case a(Int)
    case b


# CHECK-LABEL: lit.struct.decl @OneLine(
# CHECK: lit.enum.case only : []{{$}}
__enum OneLine: case only


# An enum may declare no cases.
# CHECK-LABEL: lit.struct.decl @Caseless(
# CHECK-SAME: attributes {isEnum,
# CHECK-NOT: lit.enum.case
# CHECK: lit.fn @"method(
__enum Caseless:
    def method(self):
        pass


# Doc strings sit where a field's does: on the lines right after the case.
# CHECK-LABEL: lit.struct.decl @Documented(
# CHECK: lit.enum.case red : [] {docString = [[RED_DOC]]}
# CHECK-NEXT: lit.enum.case plain : []{{$}}
# CHECK-NEXT: lit.enum.case rgb : {{\[}}[[INT]], [[INT]], [[INT]]] {docString = [[RGB_DOC]]}
__enum Documented:
    case red
    """Red."""
    case plain
    case rgb(Int, Int, Int)
    """
    A color by its components.
    """


# CHECK-LABEL: lit.struct.decl @AllDocumented(
# CHECK: lit.enum.case only : [] {docString = [[ONLY_DOC]]}
__enum AllDocumented:
    case only
    """Only."""


# CHECK-LABEL: lit.struct.decl @Spellings(
# CHECK: lit.enum.case a : []{{$}}
# CHECK-NEXT: lit.enum.case b : {{\[}}[[INT]]]{{$}}
# CHECK-NEXT: lit.enum.case trailing_comma : {{\[}}[[INT]], !Bool]{{$}}
# CHECK-NEXT: lit.enum.case multi_line : {{\[}}[[INT]], !Bool]{{$}}
# CHECK-NEXT: lit.enum.case tuple : [!lit.struct<#Tuple <:param_list<!AnyType_Movable> [!Int, !Int]{{.*}}>>]{{$}}
# CHECK-NEXT: lit.enum.case def : []{{$}}
__enum Spellings:
    case a; case b(Int)
    case trailing_comma(Int, Bool,)
    case multi_line(
        Int,
        Bool,
    )
    case tuple(Tuple[Int, Int])
    case `def`


# CHECK-LABEL: lit.fn @"take(enum_decls::Color)"
def take(c: Color) -> Bool:
    return c.is_red()
