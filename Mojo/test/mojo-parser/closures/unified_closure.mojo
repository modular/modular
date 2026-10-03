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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -o %t.mlir

# RUN: FileCheck %s --enable-var-scope --check-prefixes=S0 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S1 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S2 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S3 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S4 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S5 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S6 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S7 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S8 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S9 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S10 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S11 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S12 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S13 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S14 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S15 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S17 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S18 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S19 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=KWARGS < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=STAR_ARGS < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=KWARGS_FN_PTR < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=STAR_ARGS_KWARGS < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=MIXED_KWARGS < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=MIXED_KWARGS_FN_PTR < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=STAR_ARGS_KWARGS_FN_PTR < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S20 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S21 < %t.mlir
# COM: Verify generated storage-struct structure. The closure's signature lives
# COM: in the bindings of the one parametric closure trait, not in a
# COM: per-signature trait, and no bridging wrapper is emitted.
# S0: lit.struct.decl @"closure$s0_make_closure{{.*}}::my_closure::__storage"(trait<@"##__mojo_closure__##"<
# S0-SAME: :param_list<type> [], :param_list<type> [#kgen.quote<!Int{{[0-9]*}}>], :type #kgen.quote<!Int{{[0-9]*}}>
# S0-SAME: #kgen.fn_metadata<[imm_mem, imm], "capturing"
# S0-SAME: >, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@Deinitable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@Movable>) attributes {{{.*}}synthetic}
# S0-NEXT: move :{{.*}}@{{.*}}::@"{{.*}}::my_closure::__storage"::@"__init__(move:
# S0-NEXT: copy :{{.*}}@{{.*}}::@"{{.*}}::my_closure::__storage"::@"__init__(copy:
# S0: lit.fn @"my_closure{{.*}}"[mut {{.*}}](%{{.*}}: !lit.ref<!storage{{.*}}, mut {{.*}}> imm_mem, |, %y: {{.*}}) capturing -> {{.*}}
# S0: lit.fn @"__init__(move:{{.*}}::my_closure::__storage$)"
# S0: lit.fn @"__deinit__({{.*}}::my_closure::__storage$)"
# S0: kgen.witness "__call__" : {{.*}} = @{{.*}}::@"{{.*}}::my_closure::__storage"::@"__call__{{.*}}"
# S0: kgen.witness "__init__(move:$0$)" : {{.*}} = @{{.*}}::@"{{.*}}::my_closure::__storage"::@"__init__(move:
# S0: kgen.witness "__deinit__{{.*}}" : {{.*}} = @{{.*}}::@"{{.*}}::my_closure::__storage"::@"__deinit__(
# S0-NOT: lit.struct.decl @"extension${{.*}}"


def s0_make_closure(x: Int, mem: String):
    def my_closure(y: Int) {var x, var mem} -> Int:
        return x + y


# COM: Verify Nested closures are supported. Each closure gets its own storage
# COM: struct, named by its nesting path and bound to the closure trait with its
# COM: own signature -- the inner one takes `z`, the outer `y`.
# S1-DAG: lit.struct.decl @"closure$s1_make_closure{{.*}}::my_closure::__storage"(trait<@"##__mojo_closure__##"<{{.*}}<"y", pos_or_kw, not_vararg>
# S1-DAG: lit.struct.decl @"closure$s1_make_closure{{.*}}::my_closure{{.*}}::my_nested_closure::__storage"(trait<@"##__mojo_closure__##"<{{.*}}<"z", pos_or_kw, not_vararg>


def s1_make_closure(x: Int, mem: String):
    def my_closure(y: Int) {var x, var mem} -> Int:
        def my_nested_closure(z: Int) {var x, var mem} -> Int:
            return x

        return x + y


# COM: One closure trait serves the whole module -- two closures with the same
# COM: signature share it -- and no bridging wrapper struct is emitted.
# S2-COUNT-1: lit.trait.decl @"##__mojo_closure__##"
# S2-NOT: lit.trait.decl @"##__mojo_closure__##"
# S2-NOT: lit.struct.decl @"extension$


def s2_make_closure(x: Int):
    def my_closure(y: Int) {var} -> Int:
        return y


def make_identical_closure(x: Int):
    def my_closure(y: Int) {var} -> Int:
        return y


# COM: Test that parametric functions in traits are handled correctly. The
# COM: closure's own parameters `[T, b, c]` become the closure trait's parameter
# COM: list, with `b` and `c` referring back to `T` positionally, and the
# COM: argument `a: T` lands in the argument list.
# S3: lit.struct.decl @"closure$s3_make_closure{{.*}}::parametric::__storage"(trait<@"##__mojo_closure__##"<
# S3-SAME: :param_list<type> [#kgen.quote<!AnyType_Movable_MyInterface>, #kgen.quote<!kgen.param<:!AnyType_Movable_MyInterface *(0,1)>>, #kgen.quote<!lit.struct<#Foo <:!AnyType_Movable upcast(:!AnyType_Movable_MyInterface *(0,1)), :!kgen.param<:!AnyType_Movable_MyInterface *(0,1)> *(0,2)>>>]
# S3-SAME: :param_list<type> [#kgen.quote<!lit.ref<:!AnyType_Movable_MyInterface *(0,1), imm *[0,1]>>], :type #kgen.quote<none>
# S3-SAME: #kgen.pog_list<[<"_Self", inferred, not_vararg>, <"T", pos_or_kw, not_vararg>, <"b", pos_or_kw, not_vararg>, <"c", pos_or_kw, not_vararg>]>


trait s3_MyInterface(Movable):
    def thing(self):
        ...


struct Foo[T: Movable, b: T](Movable where False):
    pass


def s3_make_closure(x: Int, mem: String) -> Int:
    def parametric[T: s3_MyInterface, b: T, c: Foo[T, b]](a: T) {var}:
        _ = mem

    return x


# COM: Explicit origins are handled on the storage struct (no parametric
# COM: wrapper). The declared origin `lt` and its `Origin` value bind into the
# COM: closure trait's parameter list, and `a` keeps the mutable origin.
# S4: lit.struct.decl @"closure$s4_make_closure{{.*}}::mutate::__storage"(trait<@"##__mojo_closure__##"<
# S4-SAME: :param_list<type> [#kgen.quote<origin<true>>, #kgen.quote<!lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *(0,1)>>>]
# S4-SAME: :param_list<type> [#kgen.quote<!lit.ref<!String, mut *(0,1)>>, #kgen.quote<!lit.ref<!String, imm *[0,1]>>], :type #kgen.quote<none>
# S4-SAME: >) attributes {{{.*}}synthetic}
# S4: kgen.conformance @"##__mojo_closure__##"<:param_list<type> [#kgen.quote<origin<true>>
# S4-DAG: kgen.witness "__call__" : {{.*}} = @{{.*}}::@"{{.*}}::mutate::__storage"::@"__call__{{.*}}"
# S4-NOT: lit.struct.decl @"extension$


def s4_make_closure(x: Int, mem: String) -> Int:
    def mutate[
        lt: Origin[mut=True]
    ](a: Pointer[String, lt]._mlir_lit_ref, b: String) {var}:
        _ = mem

    return x


# COM: Verify storage constructor takes the captured value (no wrapper impl arg).
# S5: lit.struct.decl @"closure$s5_make_closure{{.*}}::parametric::__storage"(trait<@"##__mojo_closure__##"<
# S5-SAME: :param_list<type> [#kgen.quote<!AnyType_MyInterface>], :param_list<type> [#kgen.quote<!lit.ref<:!AnyType_MyInterface *(0,1), imm *[0,1]>>], :type #kgen.quote<none>
# S5-SAME: >, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@Deinitable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@Movable>) attributes {{{.*}}synthetic}
# S5: lit.fn @"__init__(::String)"[imm *"mem`", mut *"self`"]
# S5-NOT: lit.fn @"__init__($0$)"[mut *"impl`", mut *"self`"]


trait s5_MyInterface:
    def thing(self):
        ...


def s5_make_closure(x: Int, mem: String) -> Int:
    def parametric[T: s5_MyInterface](a: T) {var}:
        _ = mem

    return x


# COM: Verify the closure instance is created as the storage struct alone
# COM: (no parametric wrapper around it).
# S6-DAG: lit.var.decl "my_closure" var
# S6-DAG: lit.call {{.*}}s6_make_closure{{.*}}::my_closure::__storage"::@"__init__
# S6-NOT: lit.var.decl "my_closure.storage" var
# S6-NOT: lit.call {{.*}}::@"def(y: Int) -> Int_{{.*}}::@"__init__($0$)"


def s6_make_closure(x: Int, mem: String):
    def my_closure(y: Int) {var x, var mem} -> Int:
        return x + y


# COM: Check that the argument is augmented at the definition site: the closure
# COM: parameter `f` is bound to the closure trait carrying the signature, and
# COM: the call goes through that trait's `__call__` witness.
# S7: lit.fn @"s7_take_closure{{.*}}"<f: trait<@"##__mojo_closure__##"<:param_list<type> [], :param_list<type> [#kgen.quote<!Int{{[0-9]*}}>], :type #kgen.quote<!Int{{[0-9]*}}>
# S7-SAME: @{{.*}}::@AnyType, @{{.*}}::@Deinitable, @{{.*}}::@Movable>>[imm *"myFunc`"](%myFunc:
# S7-NEXT: lit.call tail[!lit.generator<[1](!lit.ref<:trait<@"##__mojo_closure__##"
# S7-SAME: capturing -> !Int{{[0-9]*}}>: #kgen.get_witness<:trait<@"##__mojo_closure__##"
# S7-NEXT: lit.ownership.use %0
# S7-NEXT: %none = kgen.param.constant: none = <#kgen.none>


def s7_take_closure[f: def(y: Int) -> Int](myFunc: f, x: Int):
    _ = myFunc(x)


# COM: Ensure the transformed parameters are propagated into the underlying
# COM: closure trait: the nested closure's own closure parameter `closure2` is
# COM: mangled into its symbol and declared with the bound trait.
# S8: lit.fn *"nested[##__mojo_closure__##[{{.*}}] & ::AnyType & ::Deinitable & ::Movable]($0,::SIMD[DType.int, 1])"
# S8-SAME: <closure2: trait<@"##__mojo_closure__##"<:param_list<type> [], :param_list<type> [#kgen.quote<!Int{{[0-9]*}}>], :type #kgen.quote<!Int{{[0-9]*}}>
# S8-SAME: @{{.*}}::@AnyType, @{{.*}}::@Deinitable, @{{.*}}::@Movable> closure2, imm *"impl`{{.*}}"> imm_mem, %y: !Int{{[0-9]*}}) capturing -> {{.*}}sourceName = "nested"


def s8_take_closure[closure1: def(y: Int) -> Int](x: Int):
    def nested[
        closure2: def(y: Int) -> Int
    ](impl: closure2, y: Int) {var x} -> Int:
        return x


# COM: ensure many closure parameters are handled. Each closure parameter keeps
# COM: its own trait binding -- `closure1` takes one argument, `closure2` two --
# COM: and they interleave with the plain parameters `T` and `U`.
# S9: lit.fn @"take_closures{{.*}})"
# S9-SAME: <closure1: trait<@"##__mojo_closure__##"<:param_list<type> [], :param_list<type> [#kgen.quote<!Int{{[0-9]*}}>], :type #kgen.quote<!Int{{[0-9]*}}>
# S9-SAME: @{{.*}}::@Movable>, T: !Int{{[0-9]*}}, closure2: trait<@"##__mojo_closure__##"<:param_list<type> [], :param_list<type> [#kgen.quote<!Int{{[0-9]*}}>, #kgen.quote<!Int{{[0-9]*}}>], :type #kgen.quote<!Int{{[0-9]*}}>
# S9-SAME: @{{.*}}::@Movable>, U: !Int{{[0-9]*}}>[imm *"[[S9_L0:.*]]`", imm *"[[S9_L1:.*]]`1"](%impl1: !lit.ref<:trait<@"##__mojo_closure__##"
# S9-SAME: > closure1, imm *"[[S9_L0]]`"> imm_mem, %impl2: !lit.ref<:trait<@"##__mojo_closure__##"
# S9-SAME: > closure2, imm *"[[S9_L1]]`1"> imm_mem, %x: !Int{{[0-9]*}}) capturing -> !kgen.none


def take_closures[
    closure1: def(y: Int) -> Int,
    T: Int,
    closure2: def(y: Int, z: Int) -> Int,
    U: Int,
](impl1: closure1, impl2: closure2, x: Int):
    pass


# COM: Unified Closure Parameters compose: a closure whose own parameter `y` is
# COM: itself a closure nests one closure trait inside the other's parameter
# COM: list, both in the mangled symbol and in the declared bound.
# S10: lit.fn @"nested[##__mojo_closure__##[#kgen.quote<trait<_\22##__mojo_closure__##\22<
# S10-SAME: <"impl", pos_or_kw, not_vararg>, <"u", pos_or_kw, not_vararg>]>>>, @{{.*}}::@AnyType, @{{.*}}::@Deinitable, @{{.*}}::@Movable> x, imm *"impl`"> imm_mem, %do_not_dce_int: !Int{{[0-9]*}}) capturing -> !kgen.none attributes {{{.*}}sourceName = "nested"


# TODO: remove the 'do_not_dce_int' argument (MOCO 2461)
def nested[
    x: def[y: def(z: Int) -> Int](impl: y, u: Int) -> Int, //
](impl: x, do_not_dce_int: Int):
    pass


# COM: Check that the closure storage struct is generated correctly.
# S11-DAG: lit.struct.decl @"closure$s11_bindIt(::SIMD[DType.int, 1],::SIMD[DType.int, 1],::String)::myclosure::__storage"
# S11-DAG: kgen.conformance @{{.*}}::@AnyType {
# S11-DAG: kgen.conformance @{{.*}}::@Deinitable {
# S11-DAG: kgen.witness "__deinit__{{.*}}" : {{.*}} = @{{.*}}::@"closure$s11_bindIt{{.*}}::myclosure::__storage"::@"__deinit__
# S11-DAG: kgen.conformance @{{.*}}::@Movable {
# S11-DAG: kgen.witness "__init__(move:$0$)" : {{.*}} = @{{.*}}::@"closure$s11_bindIt{{.*}}::myclosure::__storage"::@"__init__(move:
# S11-DAG: kgen.conformance @"##__mojo_closure__##"<{{.*}}<"", pos, not_vararg>, <"z", pos_or_kw, not_vararg>]>>> {
# S11-DAG: kgen.witness "__call__" : {{.*}} = @{{.*}}::@"closure$s11_bindIt{{.*}}::myclosure::__storage"::@"__call__


def s11_bindIt(x: Int, y: Int, mem: String) -> Int:
    def myclosure(z: Int) {var x, var y, var mem} -> Int:
        return x + y + z


# COM: Check that parameters are emitted correctly

# S12: lit.struct.decl @"closure$s12_bindIt({{.*}})::myclosure::__storage"
# S12: kgen.witness "__call__{{.*}}" : !lit.generator<<"my_param": !AnyType>
# S12-SAME: [1](!lit.ref<{{.*}}, mut *[0,0]> imm_mem, |, "z": !Int{{.*}}) capturing -> !kgen.none>
# S12-SAME: = @{{.*}}::@"closure$s12_bindIt({{.*}})::myclosure::__storage"::@"__call__{{.*}}"

# S12-DAG: lit.file_module


def s12_bindIt(mem: String) -> Int:
    def myclosure[my_param: AnyType](z: Int) {var}:
        _ = mem


# COM: Captured mutable reference contributes byRefMut's origin to storage.
# S13-LABEL: lit.fn @"nonemptyOriginSet(::String)"
# S13: lit.call {{.*}}::myclosure::__storage"::@"__init__
# S13-SAME: <:origin<true> *"byRefMut
# S13-NOT: lit.call @unified_closure::@"def() -> None_{{.*}}"::@"__init__


def nonemptyOriginSet(mut byRefMut: String):
    def myclosure() {mut byRefMut}:
        pass


# COM: Verify that closures can be rebound to compatible traits. The closure
# COM: conforms under its own argument name `x`, while `s14_takeIt` states the
# COM: same signature positionally -- the two bindings differ, and conformance
# COM: still holds.
# S14-DAG: lit.struct.decl @"closure$s14_bindIt{{.*}}::myclosure::__storage"
# S14-DAG: kgen.conformance @"##__mojo_closure__##"<{{.*}}<"", pos, not_vararg>, <"x", pos_or_kw, not_vararg>]>>> {
# S14-DAG: kgen.witness "__call__" : {{.*}} = @{{.*}}::@"closure$s14_bindIt{{.*}}::myclosure::__storage"::@"__call__$trait
# S14-DAG: lit.fn @"s14_takeIt{{.*}}"<C: trait<@"##__mojo_closure__##"{{.*}}<"", pos, not_vararg>, <"", pos, not_vararg>]>>


def s14_takeIt[C: def(Int) -> Int](closure: C):
    _ = closure(3)


def s14_bindIt(z: Int, mem: String):
    def myclosure(x: Int) {var} -> Int:
        _ = mem
        return z

    s14_takeIt[type_of(myclosure)](myclosure)


# COM: Verify that closures can be rebound even when traits are combined: the
# COM: closure conforms under its own `x` while `s15_takeIt` asks for `y` as
# COM: part of a `Copyable & ...` composition.
# S15-DAG: lit.struct.decl @"closure$s15_bindIt{{.*}}::myclosure::__storage"
# S15-DAG: kgen.conformance @"##__mojo_closure__##"<{{.*}}<"", pos, not_vararg>, <"x", pos_or_kw, not_vararg>]>>> {
# S15-DAG: kgen.witness "__call__" : {{.*}} = @{{.*}}::@"closure$s15_bindIt{{.*}}::myclosure::__storage"::@"__call__$trait
# S15-DAG: lit.fn @"s15_takeIt{{.*}}"<C: trait<@"##__mojo_closure__##"{{.*}}<"", pos, not_vararg>, <"y", pos_or_kw, not_vararg>]>>


def s15_takeIt[C: Copyable & def(y: Int) -> Int](closure: C):
    _ = closure(3)


def s15_bindIt(z: Int, mem: String):
    def myclosure(x: Int) {var} -> Int:
        _ = mem
        return z

    s15_takeIt[type_of(myclosure)](myclosure)


# COM: Verify that closures can be rebound with differing parameter names: the
# COM: closure declares `[a](b: Int)` and `s17_takeIt` asks for `[x](y: Int)`.
# COM: The witness keeps the closure's own names; the caller's binding renames.
# S17-DAG: lit.struct.decl @"closure$s17_bindIt{{.*}}::myclosure::__storage"
# S17-DAG: kgen.conformance @"##__mojo_closure__##"<{{.*}}<"", pos, not_vararg>, <"b", pos_or_kw, not_vararg>]>>> {
# S17-DAG: kgen.witness "__call__" : !lit.generator<<"a": !Int{{[0-9]*}}>{{.*}}, "b": !Int{{[0-9]*}}) capturing -> {{.*}} = @{{.*}}::@"closure$s17_bindIt{{.*}}::myclosure::__storage"::@"__call__$trait
# S17-DAG: lit.fn @"s17_takeIt{{.*}}"<C: trait<@"##__mojo_closure__##"{{.*}}<"", pos, not_vararg>, <"y", pos_or_kw, not_vararg>]>>


def s17_takeIt[C: def[x: Int](y: Int) -> Int](closure: C):
    # see MOCO-2606
    _ = closure.__call__[2](3)


def s17_bindIt(z: Int, mem: String):
    def myclosure[a: Int](b: Int) {var} -> Int:
        _ = mem
        return z

    s17_takeIt[type_of(myclosure)](myclosure)


# COM: Ensure that structs can conform to the closure trait: a hand-written
# COM: struct names the closure trait in its conformance list and picks up the
# COM: same binding a synthesized storage struct would.
# S18-DAG: lit.struct.decl @custom(trait<@"##__mojo_closure__##"<{{.*}}<"", pos, not_vararg>, <"x", pos_or_kw, not_vararg>]>>>, @{{.*}}::@AnyType, @{{.*}}::@Deinitable, @{{.*}}::@Movable>)


struct custom(def(x: Int) -> Int):
    def __call__(self, x: Int) capturing -> Int:
        return x


# COM: Storage conforms to Copyable and ImplicitlyCopyable in its own right, so
# COM: it is passed directly to `takeItImplicit` and `s19_takeIt` with no
# COM: bridging wrapper in between.
# S19-DAG: lit.struct.decl @"closure$giveIt{{.*}}::aThing::__storage"(trait<@"##__mojo_closure__##"<{{.*}}>, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@Deinitable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@Movable>)
# S19-DAG: lit.call @{{.*}}::@"takeItImplicit{{.*}}"{{.*}}<:!AnyType_Copyable_ImplicitlyCopyable_Movable !storage{{[0-9]*}}>
# S19-DAG: lit.call @{{.*}}::@"s19_takeIt{{.*}}"{{.*}}<:!AnyType_Copyable_Movable !storage{{[0-9]*}}>
# S19-NOT: lit.struct.decl @"extension$


def takeItImplicit[T: ImplicitlyCopyable](impl: T):
    pass


def s19_takeIt[T: Copyable](impl: T):
    pass


@fieldwise_init
struct CopyMe(ImplicitlyCopyable):
    var x: Int
    var y: Int


@fieldwise_init
struct OneOfAKind(Movable):
    var x: Int
    var y: Int


def useIt(var x: OneOfAKind):
    pass


@no_inline
def giveIt(z: Int, cm: CopyMe, var one: OneOfAKind):
    def aThing(x: Int) {var z, var cm} -> Int:
        return z + x

    takeItImplicit(aThing)
    s19_takeIt(aThing)

    def anotherThing(x: Int) {var^} -> Int:
        useIt(one^)
        return x


# COM: KWARGS: a `**kwargs` argument is accepted directly on the storage
# COM: struct's `__call__` (no parametric wrapper hop). The packed dict is
# COM: passed as a single `**` operand at the call site.

# Storage promoted method / canonical __call__ takes the dict kwargs...
# KWARGS: lit.fn @"g(kwargs:::SIMD[DType.int, 1]**)`"
# KWARGS: lit.fn @"kwargs_throughWrapper()"
# ...and the call site invokes the canonical `__call__` with the packed dict.
# KWARGS: lit.call{{.*}}::g::__storage"::@"__call__{{.*}}({{.*}}kwargs:::SIMD{{.*}}(%{{.*}}, %__call_result_tmp__)


def kwargs_throughWrapper() -> Int:
    var z = 1

    def g(var **kwargs: Int) {imm z} -> Int:
        return z

    return g(a=1, b=2)


# COM: STAR_ARGS: `*args` is accepted directly on the storage struct's
# COM: `__call__` (no parametric wrapper hop).

# STAR_ARGS: lit.fn @"h[{{.*}}(::SIMD[DType.int, 1]*)`"
# STAR_ARGS: lit.fn @"star_args_throughWrapper()"
# STAR_ARGS: lit.call{{.*}}::h::__storage"::@"__call__{{.*}}({{.*}}(%{{.*}}, %{{.*}})


def star_args_throughWrapper() -> Int:
    var z = 1

    def h(*args: Int) {imm z} -> Int:
        return z

    return h(1, 2)


# COM: KWARGS_FN_PTR: binding a plain `**kwargs` function into a closure-typed
# COM: value mints its own (fn-pointer) wrapper; its forwarding is pinned
# COM: separately.

# KWARGS_FN_PTR: lit.fn @"__call__(inflated$def(var **kwargs: ::SIMD[DType.int, 1]) thin -> ::SIMD[DType.int, 1]|{{.*}}[$0],kwargs:::SIMD[DType.int, 1]**)"
# KWARGS_FN_PTR: lit.call tail{{.*}}: *"#__CALL__#"]{{.*}}(%{{[0-9]+}})
# KWARGS_FN_PTR: lit.fn @"kwargs_fn_ptr_useFnWrapper()"


def kwargs_fn_ptr_top(var **kwargs: Int) -> Int:
    return 1


def kwargs_fn_ptr_takeClosure(f: Some[def(var ** kwargs: Int) -> Int]) -> Int:
    return f(a=1)


def kwargs_fn_ptr_useFnWrapper() -> Int:
    return kwargs_fn_ptr_takeClosure(kwargs_fn_ptr_top)


# COM: STAR_ARGS_KWARGS: `*args` and `**kwargs` together are accepted
# COM: directly on the storage struct's `__call__`.

# STAR_ARGS_KWARGS: lit.fn @"b[{{.*}}(::SIMD[DType.int, 1]*,kwargs:::SIMD[DType.int, 1]**)`"
# STAR_ARGS_KWARGS: lit.fn @"star_args_kwargs_throughWrapper()"
# STAR_ARGS_KWARGS: lit.call{{.*}}::b::__storage"::@"__call__{{.*}}({{.*}}(%{{.*}}, %{{.*}}, %__call_result_tmp__)


def star_args_kwargs_throughWrapper() -> Int:
    var z = 1

    def b(*args: Int, var **kwargs: Int) {imm z} -> Int:
        return z

    return b(1, 2, a=3)


# COM: MIXED_KWARGS: a named keyword-only argument is accepted alongside
# COM: `**kwargs` directly on the storage struct's `__call__`.

# MIXED_KWARGS: lit.fn @"m(::SIMD[DType.int, 1],named:::SIMD[DType.int, 1],kwargs:::SIMD[DType.int, 1]**)`"
# MIXED_KWARGS: lit.fn @"mixed_kwargs_throughWrapper()"
# MIXED_KWARGS: lit.call{{.*}}::m::__storage"::@"__call__{{.*}}({{.*}}(%{{.*}}, %{{.*}}, %{{.*}}, %__call_result_tmp__)


def mixed_kwargs_throughWrapper() -> Int:
    var z = 1

    def m(x: Int, *, named: Int, var **kwargs: Int) {imm z} -> Int:
        return z + x + named

    return m(1, named=2, a=3, b=4)


# A defaulted keyword-only argument is accepted the same way on storage
# `__call__`. The extra `y` keeps this closure's trait name distinct from
# mixed_kwargs_throughWrapper's -- the trait name omits defaults, and a
# collision is an "invalid redefinition".
def mixed_kwargs_defaultedThroughWrapper() -> Int:
    var z = 1

    def m(x: Int, y: Int, *, named: Int = 7, var **kwargs: Int) {imm z} -> Int:
        return z + named

    return m(1, 2, a=3)


# A second closure with the same signature reuses the cached trait /
# storage shape (no parametric wrapper).
def mixed_kwargs_duplicateSignature() -> Int:
    var z = 2

    def m(x: Int, *, named: Int, var **kwargs: Int) {imm z} -> Int:
        return z

    return m(1, named=2, a=3)


# COM: MIXED_KWARGS_FN_PTR: the same mixed signature forwards through the
# COM: fn-pointer wrapper minted when a plain function is bound into a
# COM: closure-typed value.

# MIXED_KWARGS_FN_PTR: lit.fn @"__call__(inflated$def(x: ::SIMD[DType.int, 1], *, named: ::SIMD[DType.int, 1], var **kwargs: ::SIMD[DType.int, 1]) thin -> ::SIMD[DType.int, 1]|{{.*}}[$0],::SIMD[DType.int, 1],named:::SIMD[DType.int, 1],kwargs:::SIMD[DType.int, 1]**)"
# MIXED_KWARGS_FN_PTR: lit.call tail{{.*}}: *"#__CALL__#"]{{.*}}(%x, %named, %{{[0-9]+}})
# MIXED_KWARGS_FN_PTR: lit.fn @"mixed_kwargs_fn_ptr_useFnBinding()"


def mixed_kwargs_fn_ptr_top(x: Int, *, named: Int, var **kwargs: Int) -> Int:
    return x + named


def mixed_kwargs_fn_ptr_takeClosure(
    f: Some[def(x: Int, *, named: Int, var ** kwargs: Int) -> Int]
) -> Int:
    return f(1, named=2, a=3)


def mixed_kwargs_fn_ptr_useFnBinding() -> Int:
    return mixed_kwargs_fn_ptr_takeClosure(mixed_kwargs_fn_ptr_top)


# COM: STAR_ARGS_KWARGS_FN_PTR: the both-variadics signature forwards through
# COM: the fn-pointer wrapper as well.

# STAR_ARGS_KWARGS_FN_PTR: lit.fn @"__call__{{.*}}(inflated$def{{.*}}(*args: ::SIMD[DType.int, 1], var **kwargs: ::SIMD[DType.int, 1]) thin -> ::SIMD[DType.int, 1]|{{.*}}[$0],::SIMD[DType.int, 1]*,kwargs:::SIMD[DType.int, 1]**)"
# STAR_ARGS_KWARGS_FN_PTR: lit.call tail{{.*}}(%{{[0-9]+}}, %{{[0-9]+}})
# STAR_ARGS_KWARGS_FN_PTR: lit.fn @"star_args_kwargs_fn_ptr_useFnWrapper()"


def star_args_kwargs_fn_ptr_top(*args: Int, var **kwargs: Int) -> Int:
    return 1


def star_args_kwargs_fn_ptr_takeClosure(
    f: Some[def(* args: Int, var ** kwargs: Int) -> Int]
) -> Int:
    return f(1, 2, a=3)


def star_args_kwargs_fn_ptr_useFnWrapper() -> Int:
    return star_args_kwargs_fn_ptr_takeClosure(star_args_kwargs_fn_ptr_top)


# COM: Ensure index replacement is asserted on name not attribute identity since
# COM: replacement operates over uncanonical form: the closure trait bound in
# COM: `s20_apply`'s symbol refers to `X` by index through the sugared bound.
# S20: lit.fn @"s20_apply[::AnyType & ::Copyable & ::Deinitable & ::Movable,##__mojo_closure__##[, #kgen.quote<!lit.ref<:trait<_std::_builtin::_stubs::_AnyType, _std::_builtin::_stubs::_Copyable, _std::_builtin::_stubs::_Deinitable, _std::_builtin::_stubs::_Movable> *(1,0), imm *[0,1]>>


# COM: S20_Bound has sugared type
comptime S20_Bound = Copyable & Deinitable


def s20_apply[X: S20_Bound, //, F: def(X)](f: F):
    pass


def s20_bindIt[X: S20_Bound]():
    # COM: Capture of sugared type
    def reap(y: X):
        pass

    # COM: To inflate reap into a closure type we need to bind the parameters of its wrapper type.
    #      That means we must match a parameter to a value. We asserted that there is a parameter
    #      to bind that value to but we assumed that the types were equal when in fact they only
    #      need to be canonically equal.
    s20_apply(reap)


# COM: A parameter default that names an earlier parameter on a unified
# COM: closure. The closure trait's pog list prepends `_Self`, so the default
# COM: of `b` must be shifted to index `a` at (0,1) rather than (0,0), and
# COM: `f[1]()` binds both `a` and `b` to 1.
# S21: lit.struct.decl @"closure$s21_dependent_default{{.*}}::f::__storage"(trait<@"##__mojo_closure__##"<{{.*}}#kgen.pog_list<[<"_Self", inferred, not_vararg>, <"a", pos_or_kw, not_vararg>, <"b", pos_or_kw, not_vararg, default :!Int *(0,1)>]>>
# S21: kgen.witness "__call__" : !lit.generator<<"a": !Int, "b": !Int = *(0,0)>
# S21: lit.fn @"s21_dependent_default
# S21: lit.call @{{.*}}::@"__call__$trait{{.*}}<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 1}>


def s21_dependent_default(x: Int) -> Int:
    def f[a: Int, b: Int = a]() {imm} -> Int:
        return x + a + b

    return f[1]()
