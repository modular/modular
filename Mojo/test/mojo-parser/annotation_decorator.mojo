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

# `@__annotation(...)` stores its comptime operands on the declaration it
# decorates. Reading them back is the reflection library's job; what the parser
# owes is the `annotations` attribute on `lit.struct.decl` and
# `lit.struct.field`.


# Repeated decorators accumulate into one list in source order, top to bottom
# and left to right within a decorator.
#
# A literal is stored in its materialized form, so `1` is an `Int` struct value
# rather than a bare `scalar<index>`. That is what makes `type_of` report `Int`
# on the way back out. Each entry's trailing type is what pins that down: a
# `StringLiteral` still appears inside the third entry, as the argument the
# `String` initializer converts away, so the payload text alone does not
# distinguish a materialized value from an unmaterialized one.
#
# CHECK:      lit.struct.decl @Target
# CHECK-SAME:   annotations = #kgen<exprs[
# CHECK-SAME:     _mlir_value: scalar<index> = 1}> : !Int
# CHECK-SAME:     _mlir_value: scalar<index> = 2}> : !Int
# CHECK-SAME:     @String::@"__init__
# CHECK-SAME:     <:string "tag">
# CHECK-SAME:     : !String]
@__annotation(1)
@__annotation(2, "tag")
struct Target:
    # CHECK: lit.struct.field first {annotations = #kgen<exprs[
    # CHECK-SAME: _mlir_value: scalar<index> = 3}> : !Int
    @__annotation(3)
    var first: Int

    # An undecorated field carries no annotation attribute at all.
    # CHECK: lit.struct.field second : !{{.*}}Int
    # CHECK-NOT: annotations
    var second: Int


# Annotations are emitted in the decorated struct's own scope, so a value can
# name the struct's parameters and stays unevaluated until the struct is bound.
#
# CHECK:      lit.struct.decl @Generic
# CHECK-SAME:   annotations = #kgen<exprs[
# CHECK-SAME:     #kgen.param.decl.ref<"N">
#
# A field's annotations are emitted in the struct's scope too.
#
# CHECK:      lit.struct.field value {annotations = #kgen<exprs[
# CHECK-SAME:     #kgen.param.decl.ref<"N">
@__annotation(Self.N)
struct Generic[N: Int]:
    @__annotation(Self.N)
    var value: Int


# A field's annotations resolve with its signature, while the struct body is
# still being resolved, so they can name comptime aliases at module scope, in
# the struct (including one declared after the field), and on a trait bound.
comptime module_tag = 7


trait Tagged:
    comptime tag: Int


# CHECK:      lit.struct.decl @UsesAliases
# CHECK:      lit.struct.field value {annotations = #kgen<exprs[
# CHECK-SAME:     _mlir_value: scalar<index> = 7}> : !Int
# CHECK-SAME:     _mlir_value: scalar<index> = 5}> : !Int
struct UsesAliases:
    comptime member_tag = 5

    @__annotation(module_tag, Self.member_tag)
    var value: Int


# CHECK:      lit.struct.decl @ForwardAlias
# CHECK:      lit.struct.field value {annotations = #kgen<exprs[
# CHECK-SAME:     _mlir_value: scalar<index> = 9}> : !Int
struct ForwardAlias:
    @__annotation(Self.later_tag)
    var value: Int

    comptime later_tag = 9


# CHECK:      lit.struct.decl @TraitBoundValue
# CHECK:      lit.struct.field value {annotations = #kgen<exprs[
# CHECK-SAME:     #kgen.param.decl.ref<"v">
struct TraitBoundValue[T: Copyable & Deinitable, v: T]:
    @__annotation(Self.v)
    var value: Int


# CHECK:      lit.struct.decl @TraitAlias
# CHECK:      lit.struct.field value {annotations = #kgen<exprs[
# CHECK-SAME:     #kgen.get_witness<{{.*}}@Tagged, "tag">
struct TraitAlias[T: Tagged]:
    @__annotation(Self.T.tag)
    var value: Int
