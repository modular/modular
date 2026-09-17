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
# RUN: env MOJO_ENABLE_PARAMETRIC_CLOSURE_TRAIT=1 %parse-mojo-isolated %s \
# RUN:   --kgen-print-inline-type-values -split-input-file | FileCheck %s

# A closure trait declares its call method as a function generator type builder
# whose four components - parameter declarations, argument types, result type
# and function metadata - are references to the trait's own parameters, so they
# stay symbolic until the trait is bound.

# CHECK:      lit.trait.decl @"##__mojo_closure__##"<*"P#0": param_list<type>, *"A#1": param_list<type>, *"R#2": type, *"M#3": non_struct_type
# CHECK:        lit.alias.decl __call__: !kgen.func_gen_type_builder<
#
# CHECK-SAME:     #kgen.param_list.concat<#kgen.param_list<[#kgen.quote<trait<@"##__mojo_closure__##"
# CHECK-SAME:     *"P#0"> : !kgen.param_list<param_list<type>>> : !kgen.param_list<type>,
#
# CHECK-SAME:     #kgen.param_list.concat<#kgen.param_list<[#kgen.quote<!lit.ref<:trait<@"##__mojo_closure__##"
# CHECK-SAME:     *(0,0), mut *[0,0]>>], *"A#1"> : !kgen.param_list<param_list<type>>> : !kgen.param_list<type>,
#
# CHECK-SAME:     #kgen.param.decl.ref<"R#2"> : !kgen.type,
#
# CHECK-SAME:     #kgen.param.decl.ref<"M#3"> : !kgen.non_struct_type

# The bound components are quoted; the closure's own references are shifted up
# by one to make room for `_Self` (so `T` becomes `*(0,1)`).
# CHECK:        lit.alias.decl *"ClosureTraitP`0x"
# CHECK-SAME:     @"##__mojo_closure__##"<
# CHECK-SAME:     :param_list<type> [#kgen.quote<!AnyType_Copyable_Movable>],
# CHECK-SAME:     :param_list<type> [#kgen.quote<!lit.ref<!lit.struct<#List <:!AnyType_Copyable_Movable *(0,1)>>, imm *[0,1]>>],
# CHECK-SAME:     :type #kgen.quote<!NoneType>,
# CHECK-SAME:     :non_struct_type #kgen.fn_metadata<[mut, imm_mem], "none", #lit.fn_meta_origin_data<2>>
comptime ClosureTraitP = def[T: Copyable](List[T]) -> NoneType


# Should be able to verify and build conformance table.
# CHECK-LABEL: lit.struct.decl @Foo
struct Foo[T: AnyType](def(T)):
    # CHECK:      kgen.conformance @"##__mojo_closure__##"
    # CHECK-NEXT:   kgen.witness "__call__" : {{.*}} @unified_closures_parametric_trait::@Foo::@"__call__(unified_closures_parametric_trait::Foo[$0],$0)"<:!AnyType T>)
    def __call__(mut self, arg: Self.T):
        pass


# A quote is itself a parameter scope, so the inner closure reaches the outer
# `T` one level up: `*(1,1)` (index 1 again because of the outer `_Self`).
# CHECK: lit.alias.decl *"NestedClosure
# CHECK-SAME: "##__mojo_closure__##"<:param_list<type> [
# CHECK-SAME:   #kgen.quote<!AnyType>,
# CHECK-SAME:   #kgen.quote<trait<@"##__mojo_closure__##"<:param_list<type> [
# CHECK-SAME:     #kgen.quote<!kgen.param<:!AnyType *(1,1)>>
comptime NestedClosure = def[
    T: AnyType,
    InnerClosure: def[T](),
](t: InnerClosure)


# // -----


struct MemOnly:
    pass


@fieldwise_init
struct Foo[T: AnyType](def(T)):
    def __call__(mut self, arg: Self.T):
        pass


def call_int[T: def(Int)](closure: T):
    # TODO: closure.__call__(1)
    pass


def call_mem_only[T: def(MemOnly)](closure: T):
    pass


def main():
    # Parametric trait enables matching between parametric closure and instantiated one.

    var fi = Foo[Int]()
    # `Foo.__call__` takes its argument by reference while `def(Int)` passes an
    # `Int` in registers, so the two instances bridge through a thunk: `T`
    # binds to an extension anchored at `Foo[Int]`, whose only parameter is
    # that anchor.
    # CHECK:      lit.call {{.*}}@"call_int[##__mojo_closure__## & ::AnyType & ::Deinitable & ::Movable]($0)"
    # CHECK-SAME:   <:trait<@"##__mojo_closure__##"<
    # CHECK-SAME:     :param_list<type> [#kgen.quote<!Int>],
    # CHECK-SAME:   #kgen.extension<{{.*}}@Foo<:!AnyType !Int>, [
    # CHECK-SAME:     @"extension${{.*}}"<{{.*}}@Foo<:!AnyType !Int>>]>
    call_int(fi)

    var fm = Foo[MemOnly]()
    # A memory-only argument is passed by reference either way, so this one
    # needs no extension.
    # CHECK:      lit.call {{.*}}@"call_mem_only[##__mojo_closure__## & ::AnyType & ::Deinitable & ::Movable]($0)"
    # CHECK-SAME:   <:trait<@"##__mojo_closure__##"<
    # CHECK-SAME:     :param_list<type> [#kgen.quote<!lit.ref<!MemOnly, imm *[0,1]>>],
    # CHECK-SAME:     @Foo<:!AnyType !MemOnly>>
    call_mem_only(fm)


# // -----


trait MyStrategy(Movable):
    comptime Value: Copyable & Deinitable


@fieldwise_init
struct IntStrategy(MyStrategy):
    comptime Value = Int


@fieldwise_init
struct Runner(Movable):
    def test[
        StrategyType: MyStrategy, //
    ](
        self,
        var strategy: StrategyType,
        f: Some[def(var StrategyType.Value)],
    ):
        pass


def foo[C: def(var Int)](c: C):
    # Parametric trait enables matching between foldable expression during
    # binding: `StrategyType.Value` folds to `Int` once `StrategyType` is bound,
    # and quoting canonicalizes it, so the member-alias sugar is gone.

    # CHECK:      lit.call {{.*}}@Runner::@"test[
    # CHECK-SAME:   <:!AnyType_Movable_MyStrategy !IntStrategy,
    # CHECK-SAME:     :trait<@"##__mojo_closure__##"<
    # CHECK-SAME:     :param_list<type> [], :param_list<type> [
    # CHECK-SAME:     #kgen.quote<!lit.ref<!Int, mut *[0,1]>>]
    Runner().test(IntStrategy(), c)


# // -----


struct MemType:
    pass


def infer_arg_type[T: AnyType, C: def(y: T)](c: C):
    pass


def main():
    var x = 1

    def abc(y: MemType) {mut x}:
        pass

    # A closure's argument types are matched against the trait's symbolic
    # argument list, so binding `C` also infers `T` from the closure's own
    # parameter type.
    # CHECK:      lit.call {{.*}}@"infer_arg_type[
    # CHECK-SAME:   <:!AnyType !MemType,
    # CHECK-SAME:     :trait<@"##__mojo_closure__##"<
    # CHECK-SAME:     :param_list<type> [], :param_list<type> [
    # CHECK-SAME:     #kgen.quote<!lit.ref<!MemType, imm *[0,1]>>]
    infer_arg_type(abc)


# // -----


# Two instances of the parametric closure trait that are convertible only
# through a thunk bridge with a stateless extension struct: the thunk becomes
# its `__call__` witness, and what the thunk captured from the conversion site
# -- the anchor closure and the `T` its signature mentions -- is hoisted onto
# the struct's own parameters.

# CHECK:      lit.struct.decl @"extension${{.*}}"<T: {{.*}}, *"#Closure_Ext#": trait<@"##__mojo_closure__##"
# CHECK:        kgen.conformance @"##__mojo_closure__##"
# CHECK-NEXT:     kgen.witness "__call__" : {{.*}} = rebind(
# CHECK-SAME:       #kgen.get_witness<{{.*}} *"#Closure_Ext#", @"##__mojo_closure__##"{{.*}}"__call__">


def sink[T: AnyType, C: def(T)](mut c: C, arg: T):
    pass


def forward[T: TrivialRegisterPassable, C: def(t : T)](mut c: C, arg: T):
    # The extension leaves `c`'s physical type alone: `sink`'s `C` is bound to
    # a `#kgen.extension` anchored at `C`, with the struct above -- its
    # parameters bound to this scope's `T` and `C` -- supplying the conformance.
    # CHECK:      lit.call {{.*}}@"sink[
    # CHECK-SAME:   #kgen.extension<{{.*}}> C, [
    # CHECK-SAME:     @"extension${{.*}}"<:{{.*}} T, :{{.*}} C>]>
    sink(c, arg)
