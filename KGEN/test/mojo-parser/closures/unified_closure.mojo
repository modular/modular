# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S16 < %t.mlir
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
# COM: Verify generated trait and struct structure.
# S0-DAG: [[S0_PARENT:!Int_AnyType_ImplicitlyDeletable_Movable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@ImplicitlyDeletable, @{{.*}}::@Movable>
# S0-DAG: [[S0_IMPL_PARENT:!Int_AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDeletable_Movable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@ImplicitlyDeletable, @{{.*}}::@Movable>
# S0-DAG: [[S0_INT:!.*]] = !lit.struct<#SIMD <{{.*}}>>
# S0-DAG: lit.trait.decl @"def(y: Int) -> Int"<?, *"_Self`{{.*}}": [[S0_PARENT]]>([[S0_PARENT]])
# S0-DAG: lit.fn @"__call__($0,::SIMD[::DType(int), ::SIMDLength(1)])"[mut *"self`"](%{{.*}}: !lit.ref<:{{.*}}, mut *"self`"> read_mem, |, %y: {{.*}}) capturing -> {{.*}} attributes {sourceName = "__call__", specialFnKind = 0 : i8, synthetic} {
# S0: lit.struct.decl @"def(y: Int) -> Int_{{[^"]*}}"<impl: [[S0_IMPL_PARENT]], origin_set: origin.set, |>([[S0_IMPL_PARENT]]) attributes {definesClosure,{{.*}}synthetic}
# S0-NEXT: move :
# S0-NEXT: copy :
# S0:  lit.struct.field field0 : !kgen.param<:[[S0_IMPL_PARENT]] impl>
# S0: lit.fn @"__call__({{.*}})"[mut *"[[S0_L0:.*]]`"](%0[*""]: !lit.ref<!lit.struct<[[S0_T:#.*]] <:[[S0_IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[S0_L0]]`"> read_mem, |, %y: [[S0_INT]]) capturing -> [[S0_INT]]
# S0-SAME: kgen.transparent_thunk_callee_expr = #kgen.get_witness<{{.*}}, "def(y: Int) -> Int", "__call__{{.*}}">
# S0-NEXT:  [[S0_FIELD:%.*]] = lit.ref.struct.ger %{{.*}}[field0]
# S0-NEXT:  [[S0_CLOSURE:%.*]] = lit.ref.immut [[S0_FIELD]]
# S0-NEXT:  [[S0_RES:%.*]] = lit.call[!lit.generator<[1](!lit.ref<:[[S0_IMPL_PARENT]] impl, mut *[0,0]> read_mem, |, "y": [[S0_INT]]) capturing -> [[S0_INT]]>: #kgen.get_witness<:[[S0_IMPL_PARENT]] impl, "def(y: Int) -> Int", "__call__{{.*}}">][muttoimm *"[[S0_L0]]`"->field0]([[S0_CLOSURE]], %y)
# S0-NEXT:  lit.return [[S0_RES]]
# S0-NEXT:  lit.end_fn
# S0-NEXT: }
# S0: lit.fn @"__init__{{.*}}"[mut *"[[S0_L2:.*]]`", mut *"[[S0_L3:.*]]`"](*, %move: !lit.ref<{{.*}}<:[[S0_IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[S0_L2]]`"> deinit_mem, ?, %self: !lit.ref<{{.*}} <:[[S0_IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[S0_L3]]`"> byref_result) -> !kgen.none
# S0: lit.ownership.mark_destroyed %move
# S0: lit.fn @"__del__({{.*}})"[mut *"[[S0_L1:.*]]`"](%self: !lit.ref<{{.*}}<:[[S0_IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[S0_L1]]`"> deinit_mem, |) -> !kgen.none
# S0: lit.ownership.mark_destroyed %self



# With -split-input-file and --kgen-print-inline-type-values, the closure trait may be printed as _Self: !Int or *"_Self`0x": !Int.





def s0_make_closure(x: Int, mem: String):
    def my_closure(y: Int) {var x, var mem} -> Int:
        return x + y

# COM: Verify Nested closures are supported
# S1-DAG: lit.trait.decl @"def[y: def(z: Int) -> Int]{{.*}}"
# S1-DAG: lit.trait.decl @"def(z: Int) -> Int"
# S1-DAG: lit.struct.decl @"{{.*}}s1_make_closure{{.*}}::my_closure::__storage"
# S1-DAG: lit.struct.decl @"{{.*}}my_nested_closure::__storage"






def s1_make_closure(x: Int, mem: String):
    def my_closure(y: Int) {var x, var mem} -> Int:
        def my_nested_closure(z: Int) {var x, var mem} -> Int:
            return x

        return x + y

# COM: Ensure identical closure traits are reused
# S2-COUNT-1: lit.trait.decl @"def(y: Int) {{.*}} -> Int"
# S2-COUNT-1: lit.struct.decl @"def(y: Int) {{.*}} -> Int




def s2_make_closure(x: Int):
    def my_closure(y: Int) {var} -> Int:
        return y


def make_identical_closure(x: Int):
    def my_closure(y: Int) {var} -> Int:
        return y

# COM: Test that parametric functions in traits are handled correctly
# S3: [[S3_TRAIT:!None_AnyType_ImplicitlyDeletable_Movable.*]] = !lit.trait<@"def[T: s3_MyInterface, b: T, c: Foo[T, b]](a: T) -> None", @{{.*}}::@AnyType, @{{.*}}::@ImplicitlyDeletable, @{{.*}}::@Movable>
# S3: lit.trait.decl @"def[T: s3_MyInterface, b: T, c: Foo[T, b]](a: T) -> None"<?, *"_Self`{{.*}}": [[S3_TRAIT]]>(!{{.*}}) unspecified attributes {{{.*}}} {
# S3: lit.fn @"__call__{{.*}}"<T: !AnyType_Movable_MyInterface, b: !kgen.param<:!AnyType_Movable_MyInterface T>, c: {{.*}}Foo <:!AnyType_Movable {{.*}}, :!kgen.param<:!AnyType_Movable_MyInterface T> b>>
# S3-SAME: [mut *"self`", imm *"[[S3_L1:.*]]`"](%0[*""]: !lit.ref<:[[S3_TRAIT]] *"_Self`{{.*}}", mut *"self`"> read_mem, |, %a: !lit.ref<:!AnyType_Movable_MyInterface T, imm *"[[S3_L1]]`"> read_mem) capturing -> !kgen.none



trait s3_MyInterface(Movable):
    def thing(self):
        ...


struct Foo[T: Movable, b: T](Movable where False):
    pass




def s3_make_closure(x: Int, mem: String) -> Int:
    def parametric[T: s3_MyInterface, b: T, c: Foo[T, b]](a: T) {var}:
        _ = mem

    return x

# COM: Test that explicit origins are handled correctly alongside implicit origins.
# S4: [[S4_TRAIT:!None_AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDeletable_Movable.*]] = !lit.trait<@"def[{{.*}}](a: ref[lt] String, b: String) -> None",
# S4: lit.struct.decl @"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{[^"]*}}"<impl: {{.*}}, origin_set: origin.set, |>({{.*}}) attributes {definesClosure,{{.*}}synthetic}
# S4: lit.struct.field field0 : !kgen.param<:[[S4_TRAIT]] impl>
# S4-NEXT: lit.fn @"__call__{{.*}}"<{{.*}}, lt: !lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *"lt._mlir_origin`2x">>>[
# S4-NEXT: [[S4_FIELD:%.*]] = lit.ref.struct.ger %0[field0]
# S4-NEXT: [[S4_V1:%.*]] = lit.ref.immut [[S4_FIELD]]
# S4-NEXT: [[S4_V2:%.*]] = lit.call[!lit.generator<[2](!lit.ref<:[[S4_TRAIT]] impl, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none>:
# S4-SAME: bind_params(:!lit.generator<<{{.*}}"lt": !lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *(0,0)>>>[2](!lit.ref<:[[S4_TRAIT]] impl, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none
# S4-SAME:> #kgen.get_witness<:[[S4_TRAIT]] impl, "def[{{.*}}](a: ref[lt] String, b: String) -> None", "__call__{{.*}}">{{.*}}][muttoimm *{{.*}}->field0, {{.*}}]([[S4_V1]], %a, %b)
# S4-NEXT: lit.return [[S4_V2]] : !kgen.none
# S4-NEXT: lit.end_fn
# S4: kgen.conformance @"def[{{.*}}](a: ref[lt] String, b: String) -> None" {
# S4-NEXT: kgen.witness "__call__{{.*}}" : !lit.generator<<{{.*}}"lt": !lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *(0,0)>>>[2](!lit.ref<!lit.struct<[[S4_T:#.*]] <:[[S4_TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none
# S4-SAME: > = @{{.*}}::@"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{.*}}"::@"__call__{{.*}}"<:[[S4_TRAIT]] impl, :origin.set origin_set,
# S4: kgen.conformance @{{.*}}::Movable" {
# S4-NEXT: kgen.witness "__init__{{.*}}" : !lit.generator<[2](*, "move": !lit.ref<!lit.struct<[[S4_T]] <:[[S4_TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> deinit_mem, ?, "self": !lit.ref<!lit.struct<[[S4_T]] <:[[S4_TRAIT]] impl, :origin.set origin_set>>, mut *[0,1]> byref_result) -> !kgen.none
# S4-SAME: > = @{{.*}}::@"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{.*}}"::@"__init__(move:
# S4: kgen.conformance @"{{.*}}::ImplicitlyDeletable" {
# S4-NEXT:  kgen.witness "__del__{{.*}}" : !lit.generator<[1]("self": !lit.ref<!lit.struct<[[S4_T]] <:[[S4_TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> deinit_mem, |) -> !kgen.none
# S4-SAME: > = @{{.*}}::@"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{.*}}"::@"__del__{{.*}}"<{{.*}}>
# S4: kgen.conformance @"{{.*}}::AnyType" {
# S4-NEXT: }











def s4_make_closure(x: Int, mem: String) -> Int:
    def mutate[
        lt: Origin[mut=True]
    ](a: Pointer[String, lt]._mlir_type, b: String) {var}:
        _ = mem

    return x

# COM: Verify that the constructor is assembled correctly
# S5: [[S5_TRAIT:!None_AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDeletable_Movable.*]] = !lit.trait<@"def[T: s5_MyInterface](a: T) -> None", @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@ImplicitlyDeletable, @{{.*}}::@Movable>
# S5: lit.fn @"__init__($0$)"[mut *"impl`", mut *"self`"](%impl: !lit.ref<:[[S5_TRAIT]] impl, mut *"impl`"> owned_in_mem, |, ?, %self: !lit.ref<!lit.struct<[[S5_T:#.*]] <:[[S5_TRAIT]] impl, :origin.set origin_set>>, mut *"self`"> byref_result)
# S5-NEXT: [[S5_V0:%.*]] = lit.ref.struct.ger %self[field0] : <!lit.struct<[[S5_T]] <:[[S5_TRAIT]] impl, :origin.set origin_set>>, mut *"self`"> -> :[[S5_TRAIT]] impl
# S5-NEXT: [[S5_V1:%.*]] = lit.call[!lit.generator<[2](*, "move": !lit.ref<:[[S5_TRAIT]] impl, mut *[0,0]> deinit_mem, ?, "self": !lit.ref<:[[S5_TRAIT]] impl, mut *[0,1]> byref_result) -> !kgen.none>: #kgen.get_witness<:[[S5_TRAIT]] impl, "{{.*}}::Movable", "__init__(move:$0$)">][mut *"impl`", mut *"self`"->field0](%impl, [[S5_V0]])
# S5-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# S5-NEXT: lit.return %none : !kgen.none
# S5-NEXT: lit.end_fn



trait s5_MyInterface:
    def thing(self):
        ...






def s5_make_closure(x: Int, mem: String) -> Int:
    def parametric[T: s5_MyInterface](a: T) {var}:
        _ = mem

    return x

# COM: Verify the closure instance is created correctly: the captured values are
# COM: copied into the closure storage struct, then the closure wrapper is
# COM: constructed from that storage.
# S6-DAG: lit.var.decl "my_closure.storage" var
# S6-DAG: lit.call {{.*}}s6_make_closure{{.*}}::my_closure::__storage"::@"__init__
# S6-DAG: lit.var.decl "my_closure" var
# S6-DAG: lit.call {{.*}}::@"def(y: Int) -> Int_{{.*}}::@"__init__($0$)"





def s6_make_closure(x: Int, mem: String):

    def my_closure(y: Int) {var x, var mem} -> Int:
        return x + y

# COM: Check that the argument is augmented at the definition site.
# S7-DAG: [[S7_TRAIT:!Int_AnyType_ImplicitlyDeletable_Movable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@ImplicitlyDeletable, @{{.*}}::@Movable>

# S7: lit.fn @"s7_take_closure{{.*}}"<f: [[S7_TRAIT]]>[imm *"myFunc`"](%myFunc: !lit.ref<:[[S7_TRAIT]] f, imm *"myFunc`"> read_mem, %x: !Int1) capturing -> !kgen.none
# S7-NEXT: %0 = lit.call tail[!lit.generator<[1](!lit.ref<:[[S7_TRAIT]] f, mut *[0,0]> read_mem, |, "y": !Int1) capturing -> !Int1>: #kgen.get_witness<:[[S7_TRAIT]] f, "def(y: Int) -> Int", "__call__{{.*}}">][imm *"myFunc`"](%myFunc, %x)
# S7-NEXT: lit.ownership.use %0
# S7-NEXT: %none = kgen.param.constant: none = <#kgen.none>





def s7_take_closure[f: def(y: Int) -> Int](myFunc: f, x: Int):
    _ = myFunc(x)

# COM: Ensure the transformed parameters are propagated into the underlying closure trait.
# S8-DAG: [[S8_TRAIT:!Int_AnyType_ImplicitlyDeletable_Movable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@ImplicitlyDeletable, @{{.*}}::@Movable>
# S8-DAG: [[S8_INT:!Int.*]] = !lit.struct<#SIMD <{{.*}}>>
# S8-DAG: lit.trait.decl @"def(y: Int) -> Int"
# S8-DAG: lit.fn *"nested[def(y: Int) -> Int & ::AnyType & ::ImplicitlyDeletable & ::Movable]($0,::SIMD[::DType(int), ::SIMDLength(1)])"<closure2: [[S8_TRAIT]]>





def s8_take_closure[closure1: def(y: Int) -> Int](x: Int):
    def nested[
        closure2: def(y: Int) -> Int
    ](impl: closure2, y: Int) {var x} -> Int:
        return x

# COM: ensure many closure parameters are handled.
# S9: lit.fn @"take_closures{{.*}})"
# S9-SAME: <closure1: !Int_AnyType_ImplicitlyDeletable_Movable{{.*}}, T: !Int1, closure2: !Int_AnyType_ImplicitlyDeletable_Movable{{.*}}, U: !Int1>
# S9-SAME: [imm *"[[S9_L0:.*]]`", imm *"[[S9_L1:.*]]`1"]
# S9-SAME: (%impl1: !lit.ref<:!Int_AnyType_ImplicitlyDeletable_Movable{{.*}} closure1, imm *"[[S9_L0]]`"> read_mem
# S9-SAME: , %impl2: !lit.ref<:!Int_AnyType_ImplicitlyDeletable_Movable{{.*}} closure2, imm *"[[S9_L1]]`1"> read_mem, %x: !Int1) capturing -> !kgen.none





def take_closures[
    closure1: def(y: Int) -> Int,
    T: Int,
    closure2: def(y: Int, z: Int) -> Int,
    U: Int,
](impl1: closure1, impl2: closure2, x: Int):
    pass

# COM: Unified Closure Parameters compose
# S10-DAG: [[S10_INNER:!Int_AnyType_ImplicitlyDeletable_Movable.*]] = !lit.trait<@"def(z: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@ImplicitlyDeletable, @{{.*}}::@Movable>
# S10-DAG: lit.fn @"__call__[def(z: Int) -> Int{{.*}}"<y: [[S10_INNER]]>
# S10-DAG: lit.fn @"nested[def[y: def(z: Int) -> Int](impl: y, u: Int) -> Int & ::AnyType & ::ImplicitlyDeletable & ::Movable]($0,::SIMD[::DType(int), ::SIMDLength(1)])"
# S10-DAG: %impl: !lit.ref<:!Int_AnyType_ImplicitlyDeletable_Movable{{.*}} x, imm *{{.*}} read_mem
# S10-DAG: %do_not_dce_int: !Int1) capturing -> !kgen.none attributes {{.*}}sourceName = "nested"





# TODO: remove the 'do_not_dce_int' argument (MOCO 2461)
def nested[
    x: def[y: def(z: Int) -> Int](impl: y, u: Int) -> Int, //
](impl: x, do_not_dce_int: Int):
    pass

# COM: Check that the closure storage struct is generated correctly.
# S11-DAG: [[S11_TRAIT:!Int_AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDeletable_Movable.*]] = !lit.trait<@"def(z: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@ImplicitlyDeletable, @{{.*}}::@Movable>
# S11-DAG: lit.struct.decl @"s11_bindIt(::SIMD[::DType(int), ::SIMDLength(1)],::SIMD[::DType(int), ::SIMDLength(1)],::String)::myclosure::__storage"
# S11-DAG: kgen.conformance @"{{.*}}::AnyType" {
# S11-DAG: kgen.conformance @"{{.*}}::ImplicitlyDeletable" {
# S11-DAG: kgen.witness "__del__{{.*}}"
# S11-DAG: kgen.conformance @"{{.*}}::Movable" {
# S11-DAG: kgen.witness "__init__(move:$0$)"
# S11-DAG: kgen.conformance @"def(z: Int) -> Int" {
# S11-DAG: kgen.witness "__call__{{.*}}"





def s11_bindIt(x: Int, y: Int, mem: String) -> Int:
    def myclosure(z: Int) {var x, var y, var mem} -> Int:
        return x + y + z

# COM: Check that parameters are emitted correctly

# S12: lit.struct.decl @"s12_bindIt({{.*}})::myclosure::__storage"
# S12: kgen.witness "__call__{{.*}}" : !lit.generator<<"my_param": !AnyType>
# S12-SAME: [1](!lit.ref<{{.*}}, mut *[0,0]> read_mem, |, "z": !Int1) capturing -> !kgen.none>
# S12-SAME: = {{.*}}<:!AnyType ?>

# S12-DAG: lit.file_module






def s12_bindIt(mem: String) -> Int:
    def myclosure[my_param: AnyType](z: Int) {var}:
        _ = mem

# COM: Check that the origin set is bound to the wrapper
# S13-LABEL: lit.fn @"nonemptyOriginSet(::String&)"
# COM: The captured mutable reference contributes byRefMut's origin to the
# COM: closure storage struct.
# S13: lit.call {{.*}}::myclosure::__storage"::@"__init__
# S13-SAME: <:origin<true> *"byRefMut
# COM: The origin set is bound to the wrapper.
# S13: lit.call @unified_closure::@"def() -> None_{{.*}}"::@"__init__({{.*}})"
# S13-SAME: :origin.set {}




def nonemptyOriginSet(mut byRefMut: String):
    def myclosure() {mut byRefMut}:
        pass

# COM: Verify that closures can be rebound to compatible traits
# S14-DAG: lit.struct.decl @"s14_bindIt{{.*}}::myclosure::__storage"
# S14-DAG: kgen.witness "__call__($0,::SIMD[::DType(int), ::SIMDLength(1)])"
# S14-DAG: read_mem, !Int1, |) capturing -> !Int1> = rebind(:!lit.generator<[1]({{.*}}read_mem, |, "x": !Int1) capturing -> !Int1>
# S14-DAG: @{{.*}}::@"def(x: Int) -> Int_3"::@"__call__(unified_closure::def(x: Int) -> Int_3






def s14_takeIt[C: def(Int) -> Int](closure: C):
    _ = closure(3)


def s14_bindIt(z: Int, mem: String):
    def myclosure(x: Int) {var} -> Int:
        _ = mem
        return z

    s14_takeIt[type_of(myclosure)](myclosure)

# COM: Verify that closures can be rebound even when traits are combined
# S15-DAG: lit.struct.decl @"s15_bindIt{{.*}}::myclosure::__storage"
# S15-DAG: kgen.witness "__call__($0,::SIMD[::DType(int), ::SIMDLength(1)])"
# S15-DAG: read_mem, |, "y": !Int1) capturing -> !Int1> = rebind(:!lit.generator<[1]({{.*}}read_mem, |, "x": !Int1) capturing -> !Int1>
# S15-DAG: @{{.*}}::@"def(x: Int) -> Int_3"::@"__call__(unified_closure::def(x: Int) -> Int_3






def s15_takeIt[C: Copyable & def(y: Int) -> Int](closure: C):
    _ = closure(3)


def s15_bindIt(z: Int, mem: String):
    def myclosure(x: Int) {var} -> Int:
        _ = mem
        return z

    s15_takeIt[type_of(myclosure)](myclosure)

# COM: Verify that all closures are rebound when closure traits are combined or inherited

# S16-DAG: lit.struct.decl @MultipleClosure
# S16-DAG: kgen.conformance @"def(Bool) -> Int"
# S16-DAG: kgen.witness "__call__($0,::Bool)"
# S16-DAG: read_mem, !Bool, |) capturing -> !Int1> = rebind(:!lit.generator<[1]("self": !lit.ref<!MultipleClosure, imm *[0,0]> read_mem, "x": !Bool) capturing -> !alias_Int1>
# S16-DAG: @{{.*}}::@MultipleClosure::@"__call__(unified_closure::MultipleClosure,::Bool)"
# S16-DAG: kgen.conformance @"def(Int) -> Int"
# S16-DAG: kgen.witness "__call__($0,::SIMD[::DType(int), ::SIMDLength(1)])"
# S16-DAG: read_mem, !Int1, |) capturing -> !Int1> = rebind(:!lit.generator<[1]("self": !lit.ref<!MultipleClosure, imm *[0,0]> read_mem, "x": !Int1) capturing -> !alias_Int1>
# S16-DAG: @{{.*}}::@MultipleClosure::@"__call__(unified_closure::MultipleClosure,::SIMD[::DType(int), ::SIMDLength(1)])"




def s16_takeIt[C: (def(Bool) -> Int) & def(Int) -> Int](closure: C):
    _ = closure(3)


trait BoolWrapper(def(Bool) -> Int):
    pass





struct MultipleClosure(BoolWrapper, Movable, def(Int) -> Int):
    def __init__(out self):
        pass

    def __call__(self, x: Bool) -> Int:
        return 1

    def __call__(self, x: Int) -> Int:
        return 2


def s16_bindIt(z: Int):
    var fakeclosure = MultipleClosure()

    s16_takeIt[type_of(fakeclosure)](fakeclosure)

# COM: Verify that closures can be rebound with differing parameter names
# S17-DAG: lit.struct.decl @"s17_bindIt{{.*}}::myclosure::__storage"
# S17-DAG: kgen.conformance @"def[x: Int](y: Int) -> Int"
# S17-DAG: kgen.witness "__call__[::SIMD[::DType(int), ::SIMDLength(1)]]($0,::SIMD[::DType(int), ::SIMDLength(1)])"
# S17-DAG: read_mem, |, "y": !Int1) capturing -> !Int1> = rebind(:!lit.generator<<"a": !Int1>[1]({{.*}}read_mem, |, "b": !Int1) capturing -> !Int1>
# S17-DAG: @{{.*}}::@"def[a: Int](b: Int) -> Int_3"::@"__call__[::SIMD[::DType(int), ::SIMDLength(1)]](unified_closure::def[a: Int](b: Int) -> Int_3






def s17_takeIt[C: def[x: Int](y: Int) -> Int](closure: C):
    # see MOCO-2606
    _ = closure.__call__[2](3)


def s17_bindIt(z: Int, mem: String):
    def myclosure[a: Int](b: Int) {var} -> Int:
        _ = mem
        return z

    s17_takeIt[type_of(myclosure)](myclosure)

# COM: Ensure that structs can conform to the closure trait

# S18-DAG: lit.struct.decl @custom(!Int_AnyType_ImplicitlyDeletable_Movable{{.*}})




struct custom(def(x: Int) -> Int):
    def __call__(self, x: Int) capturing -> Int:
        return x

# COM: The wrapper conforms to copyable
# S19-DAG: !lit.trait<@"def(x: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@ImplicitlyDeletable, @{{.*}}::@Movable>
# S19-DAG: lit.struct.decl @"def(x: Int) -> Int_{{.*}}"<impl: !Int_AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDeletable_Movable{{.*}}, origin_set: origin.set, |>







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


# COM: KWARGS: a `**kwargs` argument forwards through the closure
# COM: wrapper's `__call__` as a `**` splat, the dict passed whole. The
# COM: wrapper is synthesized when the closure's signature is resolved.

# The wrapper's __call__ takes the dict `kw_vararg`...
# KWARGS: lit.fn @"__call__({{.*}}def(**kwargs: Int) -> Int{{.*}},kwargs:::SIMD[::DType(int), ::SIMDLength(1)]**)"
# ...and forwards it to the impl as a single `**` dict operand.
# KWARGS: lit.call{{.*}}(%{{.*}}, %kwargs)
# KWARGS: lit.fn @"kwargs_throughWrapper()"


def kwargs_throughWrapper() -> Int:
    var z = 1

    def g(**kwargs: Int) {imm z} -> Int:
        return z

    return g(a=1, b=2)


# COM: STAR_ARGS: `*args` forwards as a `*` unpack through the same wrapper
# COM: hop.

# STAR_ARGS: lit.fn @"__call__{{.*}}def(*args: Int) -> Int{{.*}},::SIMD[::DType(int), ::SIMDLength(1)]*)"
# STAR_ARGS: lit.call{{.*}}(%{{.*}}, %args)
# STAR_ARGS: lit.fn @"star_args_throughWrapper()"


def star_args_throughWrapper() -> Int:
    var z = 1

    def h(*args: Int) {imm z} -> Int:
        return z

    return h(1, 2)


# COM: KWARGS_FN_PTR: binding a plain `**kwargs` function into a closure-typed
# COM: value mints its own (fn-pointer) wrapper; its forwarding is pinned
# COM: separately.

# KWARGS_FN_PTR: lit.fn @"__call__({{.*}}_PtrWrapper[$0],kwargs:::SIMD[::DType(int), ::SIMDLength(1)]**)"
# KWARGS_FN_PTR: lit.call{{.*}}: Impl]{{.*}}(%kwargs)
# KWARGS_FN_PTR: lit.fn @"kwargs_fn_ptr_useFnWrapper()"


def kwargs_fn_ptr_top(**kwargs: Int) -> Int:
    return 1


def kwargs_fn_ptr_takeClosure(f: Some[def(**kwargs: Int) -> Int]) -> Int:
    return f(a=1)


def kwargs_fn_ptr_useFnWrapper() -> Int:
    return kwargs_fn_ptr_takeClosure(kwargs_fn_ptr_top)


# COM: STAR_ARGS_KWARGS: `*args` and `**kwargs` together forward through the
# COM: same wrapper hop -- the packed list as a `*` unpack and the packed dict
# COM: as a `**` splat in one forwarding call. Depends on the call machinery
# COM: accepting a `*` unpack followed by a `**` splat.

# The wrapper's __call__ takes both the list `pos_vararg` and the dict
# `kw_vararg`...
# STAR_ARGS_KWARGS: lit.fn @"__call__{{.*}}def(*args: Int, **kwargs: Int) -> Int{{.*}}*,kwargs:::SIMD[::DType(int), ::SIMDLength(1)]**)"
# ...and forwards both packed values in one call.
# STAR_ARGS_KWARGS: lit.call{{.*}}(%{{.*}}, %args, %kwargs)
# STAR_ARGS_KWARGS: lit.fn @"star_args_kwargs_throughWrapper()"


def star_args_kwargs_throughWrapper() -> Int:
    var z = 1

    def b(*args: Int, **kwargs: Int) {imm z} -> Int:
        return z

    return b(1, 2, a=3)


# COM: MIXED_KWARGS: a named keyword-only argument forwards alongside the
# COM: `**kwargs` splat through the wrapper -- the literal keyword operand
# COM: binds its own parameter, the dict its `**kwargs`, in one call.

# MIXED_KWARGS: lit.fn @"__call__{{.*}}def(x: Int, *, named: Int, **kwargs: Int) -> Int{{.*}},named:::SIMD[::DType(int), ::SIMDLength(1)],kwargs:::SIMD[::DType(int), ::SIMDLength(1)]**)"
# MIXED_KWARGS: lit.call{{.*}}(%{{.*}}, %x, %named, %kwargs)
# MIXED_KWARGS: lit.fn @"mixed_kwargs_throughWrapper()"


def mixed_kwargs_throughWrapper() -> Int:
    var z = 1

    def m(x: Int, *, named: Int, **kwargs: Int) {imm z} -> Int:
        return z + x + named

    return m(1, named=2, a=3, b=4)


# A defaulted keyword-only argument forwards the same way (the wrapper always
# passes its own `named` argument through, defaulted or not). The extra `y`
# keeps this closure's trait name distinct from mixed_kwargs_throughWrapper's --
# the trait name omits defaults, and a collision is an "invalid redefinition".
def mixed_kwargs_defaultedThroughWrapper() -> Int:
    var z = 1

    def m(x: Int, y: Int, *, named: Int = 7, **kwargs: Int) {imm z} -> Int:
        return z + named

    return m(1, 2, a=3)


# A second closure with the same signature reuses the cached wrapper.
def mixed_kwargs_duplicateSignature() -> Int:
    var z = 2

    def m(x: Int, *, named: Int, **kwargs: Int) {imm z} -> Int:
        return z

    return m(1, named=2, a=3)


# COM: MIXED_KWARGS_FN_PTR: the same mixed signature forwards through the
# COM: fn-pointer wrapper minted when a plain function is bound into a
# COM: closure-typed value.

# MIXED_KWARGS_FN_PTR: lit.fn @"__call__{{.*}}_PtrWrapper[$0],::SIMD[::DType(int), ::SIMDLength(1)],named:::SIMD[::DType(int), ::SIMDLength(1)],kwargs:::SIMD[::DType(int), ::SIMDLength(1)]**)"
# MIXED_KWARGS_FN_PTR: lit.call{{.*}}: Impl]{{.*}}(%x, %named, %kwargs)
# MIXED_KWARGS_FN_PTR: lit.fn @"mixed_kwargs_fn_ptr_useFnBinding()"


def mixed_kwargs_fn_ptr_top(x: Int, *, named: Int, **kwargs: Int) -> Int:
    return x + named


def mixed_kwargs_fn_ptr_takeClosure(
    f: Some[def(x: Int, *, named: Int, **kwargs: Int) -> Int]
) -> Int:
    return f(1, named=2, a=3)


def mixed_kwargs_fn_ptr_useFnBinding() -> Int:
    return mixed_kwargs_fn_ptr_takeClosure(mixed_kwargs_fn_ptr_top)


# COM: STAR_ARGS_KWARGS_FN_PTR: the both-variadics signature forwards through
# COM: the fn-pointer wrapper as well.

# STAR_ARGS_KWARGS_FN_PTR: lit.fn @"__call__{{.*}}def(*args: Int, **kwargs: Int) thin -> Int_PtrWrapper[$0],::SIMD[::DType(int), ::SIMDLength(1)]*,kwargs:::SIMD[::DType(int), ::SIMDLength(1)]**)"
# STAR_ARGS_KWARGS_FN_PTR: lit.call{{.*}}(%args, %kwargs)
# STAR_ARGS_KWARGS_FN_PTR: lit.fn @"star_args_kwargs_fn_ptr_useFnWrapper()"


def star_args_kwargs_fn_ptr_top(*args: Int, **kwargs: Int) -> Int:
    return 1


def star_args_kwargs_fn_ptr_takeClosure(f: Some[def(*args: Int, **kwargs: Int) -> Int]) -> Int:
    return f(1, 2, a=3)


def star_args_kwargs_fn_ptr_useFnWrapper() -> Int:
    return star_args_kwargs_fn_ptr_takeClosure(star_args_kwargs_fn_ptr_top)
