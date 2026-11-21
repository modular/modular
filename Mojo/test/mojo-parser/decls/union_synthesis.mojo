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

# Checks:
# 1. Synthesize tuple metadata, hidden storage and constructor argument packs.
# 2. Forward default methods and generic payloads through the support trait.
# 3. Keep empty enum cases as value aliases.

# CHECK-LABEL: lit.struct.decl @Token(!Union_AnyType_Copyable_Deinitable_EnumLike_Movable)
# CHECK: lit.alias.decl _enum_case_names: param_list<string> = <["eof", "ident", "span"]> <hasDocHiddenDecorator>
# CHECK: lit.alias.decl _enum_case_types: param_list<!AnyType> = <[[[EMPTY:@std::@builtin::@stubs::@Tuple<:param_list<!AnyType_Movable> \[\],[^}]*}>]], [[ONE:@std::@builtin::@stubs::@Tuple<:param_list<!AnyType_Movable> \[!String\],[^}]*}>]], [[TUPLE:@std::@builtin::@stubs::@Tuple<:param_list<!AnyType_Movable> \[!Int, !Int\],[^}]*}>]]]> <hasDocHiddenDecorator>
# CHECK: lit.struct.field __union_storage : !lit.struct<#UnionStorage <:param_list<!AnyType> [[[EMPTY]], [[ONE]], [[TUPLE]]]>> <isDocHidden>

# A case without a payload is a static method too, which passes an empty pack.
# CHECK: lit.fn @"eof()"[mut *"__result__`"](?, %__result__: !lit.ref<!Token, {{.*}}> byref_result) {{.*}} attributes {isStatic, sourceName = "eof"
# The callee reads the pack, which lives in this frame, so this is not a tail
# call.
# CHECK: lit.call @union_synthesis::@Token::@"__enum_init[{{.*}}__Union"{{.*}}<:param_list<!AnyType_Movable> [], :!Int {:scalar<index> 0}, {{.*}}>(%anonymous{{.*}}, %__result__)

# A case with a payload is a static method taking it by `var`, positionally,
# and passing it on in a pack.
# CHECK: lit.fn @"ident(::String$)"{{.*}}(%payload0: !lit.ref<!String, {{.*}}> owned_in_mem, |, ?, %__result__: !lit.ref<!Token, {{.*}}> byref_result) {{.*}} attributes {isStatic, sourceName = "ident"
# CHECK-NEXT: lit.ref.pack.create(%payload0) : !lit.ref.pack<:param_list<!AnyType_Movable> [!String], {{.*}}>
# CHECK: lit.call @union_synthesis::@Token::@"__enum_init[{{.*}}__Union"{{.*}}<:param_list<!AnyType_Movable> [!String], :!Int {:scalar<index> 1}, {{.*}}>(%anonymous{{.*}}, %__result__)

# Each payload type is passed by var in the argument pack.
# CHECK: lit.fn @"span(::SIMD[DType.int, 1]$,::SIMD[DType.int, 1]$)"{{.*}}(%payload0: !lit.ref<:meta<!Int> #alias_Int, mut *"payload0`"> owned_in_mem, %payload1: !lit.ref<:meta<!Int> #alias_Int, mut *"payload1`"> owned_in_mem, |, ?, %__result__: !lit.ref<!Token, {{.*}}> byref_result) {{.*}} attributes {isStatic, sourceName = "span"
# CHECK: lit.ref.pack.create(%{{.*}}, %{{.*}}) : !lit.ref.pack<:param_list<!AnyType_Movable> [!Int, !Int], {{.*}}>
# CHECK: lit.call @union_synthesis::@Token::@"__enum_init[{{.*}}__Union"{{.*}}<:param_list<!AnyType_Movable> [!Int, !Int], :!Int {:scalar<index> 2}, {{.*}}>(%anonymous{{.*}}, %__result__)

# The `__Union` defaults, `__get_enum_storage`, `__enum_init` and the two
# `EnumLike` methods, come in as always-inlined wrappers that call them.
# CHECK: lit.fn @"__enum_init[{{.*}}__Union"<{{.*}}, +, id: !Int, Ts: {{.*}} pos_vararg, ?, {{.*}}>{{.*}}(%payload: !lit.ref<!lit.struct<#VariadicPack {{.*}}> owned_in_mem|pack_vararg, ?, %result: !lit.ref<!Token, {{.*}}> byref_result) -> !kgen.none always_inline_no_debug attributes {defaultFnRef = @std::@builtin::@_enum::@__Union::@"__enum_init[{{.*}}", inheritedFrom = {{.*}}, isStatic, sourceName = "__enum_init"
# CHECK-NEXT: lit.call @std::@builtin::@_enum::@__Union::@"__enum_init[{{.*}}<:!Union_AnyType_EnumLike !Token, :param_list<!AnyType_Movable> {{.*}}, :!Int id, {{.*}}>(%payload, %result)
# CHECK: lit.fn @"__get_enum_storage[{{.*}}](union_synthesis::Token%)__Union"{{.*}}(%self: !lit.ref<!Token, {{.*}}> ref) refresult {{.*}} always_inline_no_debug attributes {defaultFnRef = @std::@builtin::@_enum::@__Union::@"__get_enum_storage[{{.*}}", inheritedFrom
# CHECK-NEXT: lit.call @std::@builtin::@_enum::@__Union::@"__get_enum_storage[{{.*}}<:!Union_AnyType_EnumLike !Token, {{.*}}>(%self)
# CHECK: lit.fn @"_get_enum_discriminant(union_synthesis::Token)__Union"{{.*}}(%self: !lit.ref<!Token, {{.*}}> imm_mem) -> !alias_Int1 always_inline_no_debug attributes {defaultFnRef = @std::@builtin::@_enum::@__Union::@"_get_enum_discriminant($0)", inheritedFrom = #kgen.trait_symbol<@std::@builtin::@_enum::@__Union>
# CHECK-NEXT: lit.call @std::@builtin::@_enum::@__Union::@"_get_enum_discriminant($0)"{{.*}}<:!Union_AnyType_EnumLike !Token>(%self)
# CHECK: lit.fn @"_unsafe_get_enum_payload[{{.*}}](union_synthesis::Token%)__Union"<id: !Int, ?, {{.*}}>(%self: !lit.ref<!Token, {{.*}}> ref) refresult {{.*}} always_inline_no_debug attributes {defaultFnRef = @std::@builtin::@_enum::@__Union::@"_unsafe_get_enum_payload[{{.*}}", inheritedFrom = #kgen.trait_symbol<@std::@builtin::@_enum::@__Union>
# CHECK-NEXT: lit.call @std::@builtin::@_enum::@__Union::@"_unsafe_get_enum_payload[{{.*}}<:!Union_AnyType_EnumLike !Token, :!Int id, {{.*}}>(%self)

# CHECK: kgen.conformance @std::@builtin::@_enum::@__Union {
# CHECK-NEXT: kgen.witness "__get_enum_storage[{{.*}} = @union_synthesis::@Token::@"__get_enum_storage[{{.*}}__Union"
# CHECK-NEXT: kgen.witness "__enum_init[{{.*}} = @union_synthesis::@Token::@"__enum_init[{{.*}}__Union"
# CHECK-NEXT: kgen.witness "_get_enum_discriminant($0)" {{.*}} = @union_synthesis::@Token::@"_get_enum_discriminant(union_synthesis::Token)__Union"
# CHECK-NEXT: kgen.witness "_unsafe_get_enum_payload[{{.*}} = @union_synthesis::@Token::@"_unsafe_get_enum_payload[{{.*}}__Union"
# CHECK: kgen.conformance @std::@builtin::@stubs::@EnumLike {
# CHECK-NEXT: kgen.witness "_enum_case_names"
# CHECK-NEXT: kgen.witness "_enum_case_types"
# CHECK-NEXT: kgen.witness "_enum_is_exhaustive"
# CHECK-NEXT: kgen.witness "_enum_elt_type_for_case"
# CHECK-NEXT: kgen.witness "_is_empty_payload"
# CHECK-NEXT: kgen.witness "_enum_has_no_payloads"
# CHECK-NEXT: kgen.witness "_get_enum_discriminant($0)" {{.*}} = @union_synthesis::@Token::@"_get_enum_discriminant(union_synthesis::Token)__Union"
# CHECK-NEXT: kgen.witness "_unsafe_get_enum_payload[{{.*}} = @union_synthesis::@Token::@"_unsafe_get_enum_payload[{{.*}}__Union"
__union Token(Copyable):
    case eof()
    case ident(String)
    case span(Int, Int)


# An enum without cases gets empty case lists and a scalar tag.
# CHECK-LABEL: lit.struct.decl @Uninhabited(!Enum_AnyType_Copyable_Deinitable_EnumLike_Equatable_Hashable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable) register_passable_trivial
# CHECK: lit.alias.decl _enum_case_names: param_list<string> = <[]> <hasDocHiddenDecorator>
# CHECK: lit.alias.decl _enum_case_types: param_list<!AnyType> = <[]> <hasDocHiddenDecorator>
# CHECK: lit.struct.field __enum_storage : !Scalar_ui8 <isDocHidden>
# CHECK: kgen.conformance @std::@builtin::@stubs::@EnumLike {
# CHECK-NEXT: kgen.witness "_enum_case_names" : param_list<string> = []
# CHECK-NEXT: kgen.witness "_enum_case_types" : param_list<!AnyType> = []
__enum Uninhabited:
    pass


# A payload can be a parameter of the union.
# CHECK-LABEL: lit.struct.decl @Maybe<
# CHECK-SAME: T: [[BOUND:!AnyType_Copyable_Deinitable_Movable]]>
# CHECK: lit.alias.decl _enum_case_types: param_list<!AnyType> = <[[[EMPTY]], [[JUST:@std::@builtin::@stubs::@Tuple<:param_list<!AnyType_Movable> \[upcast\(:.* T\)\],[^}]*}>]]]>
# CHECK: lit.struct.field __union_storage : !lit.struct<#UnionStorage <:param_list<!AnyType> [[[EMPTY]], [[JUST]]]>> <isDocHidden>
# CHECK: lit.fn @"nothing()"{{.*}}(?, %__result__: !lit.ref<!lit.struct<#Maybe <:[[BOUND]] T>>, {{.*}}> byref_result)
# CHECK: lit.fn @"just($0$)"{{.*}}(%payload0: !lit.ref<:[[BOUND]] T, {{.*}}> owned_in_mem, |, ?, %__result__: !lit.ref<!lit.struct<#Maybe <:[[BOUND]] T>>, {{.*}}> byref_result)
__union Maybe[T: Copyable & Deinitable](Copyable):
    case nothing()
    case just(Self.T)


# A custom copy constructor lifts only the payload's Copyable requirement.
# Generated move and destruction still require their payload conformances.
struct NoCopy(Movable):
    var x: Int

    def __init__(out self):
        self.x = 0


# CHECK-LABEL: lit.struct.decl @Handwritten(!Union_AnyType_Copyable_Deinitable_EnumLike_Movable)
# CHECK: lit.fn @"__init__(copy:union_synthesis::Handwritten)"{{.*}}specialFnKind = 3 : i8}
# CHECK: lit.fn @"__init__(move:union_synthesis::Handwritten$)"{{.*}}specialFnKind = 4 : i8, synthetic}
__union Handwritten(Copyable):
    case keep(NoCopy)

    def __init__(out self, *, copy: Self):
        self = Self.keep(NoCopy())


# CHECK-LABEL: lit.struct.decl @EnumValue(
# CHECK: lit.alias.decl empty: !EnumValue = <apply_result_slot(
# CHECK-NOT: lit.fn @"empty()"
# CHECK-NOT: kgen.conformance @std::@builtin::@_enum::@__Union
# CHECK: } <{{.*}}isEnum>
__enum EnumValue(ImplicitlyCopyable):
    case empty
