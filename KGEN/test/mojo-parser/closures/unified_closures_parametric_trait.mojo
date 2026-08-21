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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# A closure trait declares its call method as a function generator type builder
# whose four components - parameter declarations, argument types, result type
# and function metadata - are references to the trait's own parameters, so they
# stay symbolic until the trait is bound.

# CHECK:      lit.trait.decl @"##__mojo_closure__##"<*"P#0": param_list<type>, *"A#1": param_list<type>, *"R#2": type, *"M#3": non_struct_type, *"O#4": param_list<string>
# CHECK:        lit.alias.decl __call__: !kgen.func_gen_type_builder<
#
# CHECK-SAME:     #kgen.param_list.concat<#kgen.param_list<[#kgen.fn_gen_builder.param.decl<"_Self`",
# CHECK-SAME:     *"P#0"> : !kgen.param_list<param_list<type>>> : !kgen.param_list<type>,
#
# CHECK-SAME:     #kgen.param_list.concat<#kgen.param_list<[!lit.ref<:trait<@"##__mojo_closure__##"
# CHECK-SAME:     *"_Self`0x", mut #kgen.fn_gen_builder.param.decl.ref<"_self_origin`", !lit.origin<true>>>],
# CHECK-SAME:     *"A#1"> : !kgen.param_list<param_list<type>>> : !kgen.param_list<type>,
#
# CHECK-SAME:     #kgen.param.decl.ref<"R#2"> : !kgen.type,
#
# CHECK-SAME:     #kgen.param.decl.ref<"M#3"> : !kgen.non_struct_type,
#
# CHECK-SAME:     #kgen.param_list.concat<#kgen.param_list<["_self_origin`"], *"O#4"> : !kgen.param_list<param_list<string>>> : !kgen.param_list<string>>

# CHECK:        lit.alias.decl *"ClosureTraitP`0x"
# CHECK-SAME:     @"##__mojo_closure__##"<
# CHECK-SAME:     :param_list<type> [#kgen.fn_gen_builder.param.decl<"Fn_P#0`0", !AnyType_Copyable_Movable>],
# CHECK-SAME:     :param_list<type> [!lit.ref<!lit.struct<#List <:!AnyType_Copyable_Movable #kgen.fn_gen_builder.param.decl.ref<"Fn_P#0`0", !AnyType_Copyable_Movable>>>, imm #kgen.fn_gen_builder.param.decl.ref<"0_unnamed`0", !lit.origin<false>>>]
# CHECK-SAME:     :type !NoneType,
# CHECK-SAME:     :non_struct_type #kgen.fn_metadata<[mut, read_mem], "none"
# CHECK-SAME:     :param_list<string> ["0_unnamed`0"]
comptime ClosureTraitP = def[T: Copyable](List[T]) __param_trait__ -> NoneType


# Should be able to verify and build conformance table.
# CHECK-LABEL: lit.struct.decl @Foo
struct Foo[T: AnyType](def(T) __param_trait__):
    # CHECK:      kgen.conformance @"##__mojo_closure__##"
    # CHECK-NEXT:   kgen.witness "__call__" : {{.*}} @unified_closures_parametric_trait::@Foo::@"__call__(unified_closures_parametric_trait::Foo[$0],$0)"<:!AnyType T>)
    def __call__(mut self, arg: Self.T):
        pass
