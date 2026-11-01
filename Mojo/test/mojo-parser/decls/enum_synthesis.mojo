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

# An `__enum` conforms to `__Enum`, which refines `EnumLike`, and
# gets one hidden storage field, the `EnumLike` case lists, a `comptime` member
# for each case without a payload and a static method for each case with one.
# TODO(MOCO-5080): the `!NoneType` in the case lists below is the "no payload"
# marker; it becomes `Tuple[]`.


# CHECK-LABEL: lit.struct.decl @Token(!Enum_AnyType_Copyable_Deinitable_EnumLike_Movable)
# CHECK: lit.alias.decl _enum_case_names: param_list<string> = <["eof", "ident", "span"]> {hasDocHiddenDecorator}
# CHECK: lit.alias.decl _enum_case_types: param_list<!AnyType> = <[!NoneType, !String, [[TUPLE:@std::@builtin::@stubs::@Tuple<:param_list<!AnyType_Movable> \[!Int, !Int\].*}>]]]> {hasDocHiddenDecorator}
# CHECK: lit.struct.field __enum_storage {isDocHidden} : !lit.struct<#EnumStorage <:param_list<!AnyType> [!NoneType, !String, [[TUPLE]]]>>

# A case without a payload is a `comptime` member:
# `Token.__enum_init[0, NoneType]()`, called with an empty pack.
# CHECK: lit.alias.decl eof: !Token = <apply_result_slot({{.*}}@enum_synthesis::@Token::@"__enum_init[{{.*}}__Enum"<:param_list<!AnyType_Movable> [], :!Int {:scalar<index> 0}, :!AnyType !NoneType, {{.*}}@std::@builtin::@stubs::@VariadicPack::@"__init__({{.*}}

# A case with a payload is a static method taking it by `var`, positionally,
# and passing it on in a pack.
# CHECK: lit.fn @"ident(::String$)"{{.*}}(%payload: !lit.ref<!String, {{.*}}> owned_in_mem, |, ?, %__result__: !lit.ref<!Token, {{.*}}> byref_result) {{.*}} attributes {isStatic, sourceName = "ident"
# CHECK-NEXT: lit.ref.pack.create(%payload) : !lit.ref.pack<:param_list<!AnyType_Movable> [!String], {{.*}}>
# CHECK: lit.call tail @enum_synthesis::@Token::@"__enum_init[{{.*}}__Enum"{{.*}}<:param_list<!AnyType_Movable> [!String], :!Int {:scalar<index> 1}, :!AnyType !String, {{.*}}>(%anonymous{{.*}}, %__result__)

# Several payload types are passed on together; each is taken by `var`.
# CHECK: lit.fn @"span(::SIMD[DType.int, 1]$,::SIMD[DType.int, 1]$)"{{.*}}(%payload0: !lit.ref<:meta<!Int> #alias_Int, mut *"payload0`"> owned_in_mem, %payload1: !lit.ref<:meta<!Int> #alias_Int, mut *"payload1`"> owned_in_mem, |, ?, %__result__: !lit.ref<!Token, {{.*}}> byref_result) {{.*}} attributes {isStatic, sourceName = "span"
# CHECK: lit.ref.pack.create(%{{.*}}, %{{.*}}) : !lit.ref.pack<:param_list<!AnyType_Movable> [!Int, !Int], {{.*}}>
# CHECK: lit.call tail @enum_synthesis::@Token::@"__enum_init[{{.*}}__Enum"{{.*}}<:param_list<!AnyType_Movable> [!Int, !Int], :!Int {:scalar<index> 2}, :!AnyType @std::@builtin::@stubs::@Tuple<{{.*}}>(%anonymous{{.*}}, %__result__)

# The `__Enum` defaults, `__get_enum_storage`, `__enum_init` and the two
# `EnumLike` methods, come in as always-inlined wrappers that call them.
# CHECK: lit.fn @"__get_enum_storage[{{.*}}](enum_synthesis::Token%)__Enum"{{.*}}(%self: !lit.ref<!Token, {{.*}}> ref) refresult {{.*}} always_inline_no_debug attributes {defaultFnRef = @std::@builtin::@_enum::@__Enum::@"__get_enum_storage[{{.*}}", inheritedFrom
# CHECK-NEXT: lit.call @std::@builtin::@_enum::@__Enum::@"__get_enum_storage[{{.*}}<:!Enum_AnyType_EnumLike !Token, {{.*}}>(%self)
# CHECK: lit.fn @"__enum_init[{{.*}}__Enum"<{{.*}}, +, id: !Int, Case: !AnyType, Ts: {{.*}} pos_vararg, ?, {{.*}}>{{.*}}(%payload: !lit.ref<!lit.struct<#VariadicPack {{.*}}> owned_in_mem|pack_vararg, ?, %result: !lit.ref<!Token, {{.*}}> byref_result) -> !kgen.none always_inline_no_debug attributes {defaultFnRef = @std::@builtin::@_enum::@__Enum::@"__enum_init[{{.*}}", inheritedFrom = {{.*}}, isStatic, sourceName = "__enum_init"
# CHECK-NEXT: lit.call @std::@builtin::@_enum::@__Enum::@"__enum_init[{{.*}}<:!Enum_AnyType_EnumLike !Token, :param_list<!AnyType_Movable> {{.*}}, :!Int id, :!AnyType Case, {{.*}}>(%payload, %result)
# CHECK: lit.fn @"_get_enum_discriminant(enum_synthesis::Token)__Enum"{{.*}}(%self: !lit.ref<!Token, {{.*}}> imm_mem) -> !alias_Int1 always_inline_no_debug attributes {defaultFnRef = @std::@builtin::@_enum::@__Enum::@"_get_enum_discriminant($0)", inheritedFrom = #kgen.trait_symbol<@std::@builtin::@_enum::@__Enum>
# CHECK-NEXT: lit.call @std::@builtin::@_enum::@__Enum::@"_get_enum_discriminant($0)"{{.*}}<:!Enum_AnyType_EnumLike !Token>(%self)
# CHECK: lit.fn @"_unsafe_get_enum_payload[{{.*}}](enum_synthesis::Token%)__Enum"<id: !Int, ?, {{.*}}>(%self: !lit.ref<!Token, {{.*}}> ref) refresult {{.*}} always_inline_no_debug attributes {defaultFnRef = @std::@builtin::@_enum::@__Enum::@"_unsafe_get_enum_payload[{{.*}}", inheritedFrom = #kgen.trait_symbol<@std::@builtin::@_enum::@__Enum>
# CHECK-NEXT: lit.call @std::@builtin::@_enum::@__Enum::@"_unsafe_get_enum_payload[{{.*}}<:!Enum_AnyType_EnumLike !Token, :!Int id, {{.*}}>(%self)

# CHECK: kgen.conformance @std::@builtin::@_enum::@__Enum {
# CHECK-NEXT: kgen.witness "__get_enum_storage[{{.*}} = @enum_synthesis::@Token::@"__get_enum_storage[{{.*}}__Enum"
# CHECK-NEXT: kgen.witness "__enum_init[{{.*}} = @enum_synthesis::@Token::@"__enum_init[{{.*}}__Enum"
# CHECK-NEXT: kgen.witness "_get_enum_discriminant($0)" {{.*}} = @enum_synthesis::@Token::@"_get_enum_discriminant(enum_synthesis::Token)__Enum"
# CHECK-NEXT: kgen.witness "_unsafe_get_enum_payload[{{.*}} = @enum_synthesis::@Token::@"_unsafe_get_enum_payload[{{.*}}__Enum"
# CHECK: kgen.conformance @std::@builtin::@stubs::@EnumLike {
# CHECK-NEXT: kgen.witness "_enum_case_names"
# CHECK-NEXT: kgen.witness "_enum_case_types"
# CHECK-NEXT: kgen.witness "_enum_is_exhaustive"
# CHECK-NEXT: kgen.witness "_enum_elt_type_for_case"
# CHECK-NEXT: kgen.witness "_get_enum_discriminant($0)" {{.*}} = @enum_synthesis::@Token::@"_get_enum_discriminant(enum_synthesis::Token)__Enum"
# CHECK-NEXT: kgen.witness "_unsafe_get_enum_payload[{{.*}} = @enum_synthesis::@Token::@"_unsafe_get_enum_payload[{{.*}}__Enum"
__enum Token(Copyable):
    case eof
    case ident(String)
    case span(Int, Int)


# An enum without cases gets empty case lists and storage with no alternatives.
# CHECK-LABEL: lit.struct.decl @Uninhabited(!Enum_AnyType_Deinitable_EnumLike_Movable)
# CHECK: lit.alias.decl _enum_case_names: param_list<string> = <[]> {hasDocHiddenDecorator}
# CHECK: lit.alias.decl _enum_case_types: param_list<!AnyType> = <[]> {hasDocHiddenDecorator}
# CHECK: lit.struct.field __enum_storage {isDocHidden} : !lit.struct<#EnumStorage <:param_list<!AnyType> []>>
# CHECK: kgen.conformance @std::@builtin::@stubs::@EnumLike {
# CHECK-NEXT: kgen.witness "_enum_case_names" : param_list<string> = []
# CHECK-NEXT: kgen.witness "_enum_case_types" : param_list<!AnyType> = []
__enum Uninhabited:
    pass


# A payload can be a parameter of the enum.
# CHECK-LABEL: lit.struct.decl @Maybe<
# CHECK-SAME: T: [[BOUND:!AnyType_Copyable_Deinitable_Movable]]>
# CHECK: lit.alias.decl _enum_case_types: param_list<!AnyType> = <[!NoneType, upcast(:[[BOUND]] T)]>
# CHECK: lit.struct.field __enum_storage {isDocHidden} : !lit.struct<#EnumStorage <:param_list<!AnyType> [!NoneType, upcast(:[[BOUND]] T)]>>
# CHECK: lit.alias.decl nothing: !lit.struct<#Maybe <:[[BOUND]] T>> = <apply_result_slot(
# CHECK: lit.fn @"just($0$)"{{.*}}(%payload: !lit.ref<:[[BOUND]] T, {{.*}}> owned_in_mem, |, ?, %__result__: !lit.ref<!lit.struct<#Maybe <:[[BOUND]] T>>, {{.*}}> byref_result)
__enum Maybe[T: Copyable & Deinitable](Copyable):
    case nothing
    case just(Self.T)


# A copy constructor written in the enum lifts the `Copyable` requirement on
# the payloads, as it does for a struct's fields; the move constructor and the
# destructor are still generated, so their requirements stay.
struct NoCopy(Movable):
    var x: Int

    def __init__(out self):
        self.x = 0


# CHECK-LABEL: lit.struct.decl @Handwritten(!Enum_AnyType_Copyable_Deinitable_EnumLike_Movable)
# CHECK: lit.fn @"__init__(copy:enum_synthesis::Handwritten)"{{.*}}specialFnKind = 3 : i8}
# CHECK: lit.fn @"__init__(move:enum_synthesis::Handwritten$)"{{.*}}specialFnKind = 4 : i8, synthetic}
__enum Handwritten(Copyable):
    case keep(NoCopy)

    def __init__(out self, *, copy: Self):
        self = Self.keep(NoCopy())
