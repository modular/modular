# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s


# COM: Verify generated trait and struct structure.

# CHECK-DAG: [[PARENT:!Int_AnyType_ImplicitlyDestructible_Movable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@Movable>
# CHECK-DAG: [[IMPL_PARENT:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@Movable, @{{.*}}@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>

# CHECK-DAG: [[TRAIT:!.*]] = !lit.trait<@"def(y: Int) -> Int">
# CHECK-DAG: [[INT:!.*]] = !lit.struct<@{{.*}}::@Int>

# With -split-input-file and --kgen-print-inline-type-values, the closure trait may be printed as _Self: !Int or *"_Self`0x": !Int.
# CHECK-DAG: lit.trait.decl @"def(y: Int) -> Int"<?, *"_Self`{{.*}}": [[TRAIT]]>([[PARENT]])
# CHECK-DAG: lit.fn @"__call__($0,::Int)"[mut *"self`"](%{{.*}}: !lit.ref<:{{.*}}, mut *"self`"> read_mem, |, %y: {{.*}}) capturing -> {{.*}} attributes {sourceName = "__call__", specialFnKind = 0 : i8, synthetic} {

# CHECK: lit.struct.decl @"def(y: Int) -> Int_{{[^"]*}}"<impl: [[IMPL_PARENT]], origin_set: origin.set, |>([[IMPL_PARENT]]) attributes {definesClosure,{{.*}}synthetic}
# CHECK-NEXT: move :
# CHECK-NEXT: copy :
# CHECK:  lit.struct.field field0 : !kgen.param<:[[IMPL_PARENT]] impl>
# CHECK: lit.fn @"__call__({{.*}})"[mut *"[[L0:.*]]`"](%0[*""]: !lit.ref<!lit.struct<[[T:#.*]] <:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L0]]`"> read_mem, |, %y: [[INT]]) capturing -> [[INT]]
# CHECK-SAME: kgen.transparent_thunk_callee_expr = #kgen.get_witness<{{.*}}, "def(y: Int) -> Int", "__call__{{.*}}">
# CHECK-NEXT:  [[FIELD:%.*]] = lit.ref.struct.ger %{{.*}}[field0]
# CHECK-NEXT:  [[CLOSURE:%.*]] = lit.ref.immut [[FIELD]]
# CHECK-NEXT:  [[RES:%.*]] = lit.call[!lit.generator<[1](!lit.ref<:[[IMPL_PARENT]] impl, mut *[0,0]> read_mem, |, "y": [[INT]]) capturing -> [[INT]]>: #kgen.get_witness<:[[IMPL_PARENT]] impl, "def(y: Int) -> Int", "__call__{{.*}}">][muttoimm *"[[L0]]`"->field0]([[CLOSURE]], %y)
# CHECK-NEXT:  lit.return [[RES]]
# CHECK-NEXT:  lit.end_fn
# CHECK-NEXT: }

# CHECK: lit.fn @"__init__{{.*}}"[mut *"[[L2:.*]]`", mut *"[[L3:.*]]`"](*, %take: !lit.ref<{{.*}}<:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L2]]`"> deinit_mem, ?, %self: !lit.ref<{{.*}} <:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L3]]`"> byref_result) -> !kgen.none
# CHECK: lit.ownership.mark_destroyed %take

# CHECK: lit.fn @"__del__({{.*}})"[mut *"[[L1:.*]]`"](%self: !lit.ref<{{.*}}<:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L1]]`"> deinit_mem, |) -> !kgen.none
# CHECK: lit.ownership.mark_destroyed %self


def make_closure(x: Int, mem: String):
    def my_closure(y: Int) {var x, var mem} -> Int:
        return x + y


# // -----

# COM: Verify Nested closures are supported


# CHECK: lit.trait.decl @"def(y: Int) -> Int"
# CHECK: lit.trait.decl @"def(z: Int) -> Int"
# CHECK: lit.struct.decl @"def(y: Int) -> Int_{{.*}}"
# CHECK: lit.struct.decl @"def(z: Int) -> Int_{{.*}}"


def make_closure(x: Int, mem:String):
    def my_closure(y: Int) {var x, var mem} -> Int:
        def my_nested_closure(z: Int) {var x, var mem} -> Int:
            return x

        return x + y


# // -----

# COM: Ensure identical closure traits are reused

# CHECK-COUNT-1: lit.trait.decl @"def(y: Int) {{.*}} -> Int"
# CHECK-COUNT-1: lit.struct.decl @"def(y: Int) {{.*}} -> Int


def make_closure(x: Int):
    def my_closure(y: Int) {var} -> Int:
        return y


def make_identical_closure(x: Int):
    def my_closure(y: Int) {var} -> Int:
        return y


# // -----

# COM: Test that parametric functions in traits are handled correctly


trait MyInterface(Movable):
    def thing(self):
        ...


struct Foo[T: Movable, b: T]:
    pass


# CHECK: [[TRAIT:!None.*]] = !lit.trait<@"def[T: MyInterface, b: T, c: Foo[T, b]](a: T) -> None">
# CHECK: lit.trait.decl @"def[T: MyInterface, b: T, c: Foo[T, b]](a: T) -> None"<?, *"_Self`{{.*}}": [[TRAIT]]>(!{{.*}}) unspecified attributes {{{.*}}} {
# CHECK: lit.fn @"__call__{{.*}}"<T: !MyInterface, b: !kgen.param<:!MyInterface T>, c: {{.*}}Foo <:!Movable {{.*}}, :!kgen.param<:!MyInterface T> b>>
# CHECK-SAME: [mut *"self`", imm *"[[L1:.*]]`"](%0[*""]: !lit.ref<:[[TRAIT]] *"_Self`{{.*}}", mut *"self`"> read_mem, |, %a: !lit.ref<:!MyInterface T, imm *"[[L1]]`"> read_mem) capturing -> !kgen.none


def make_closure(x: Int, mem: String) -> Int:
    def parametric[T: MyInterface, b: T, c: Foo[T, b]](a: T) {var}:
        _ = mem

    return x


# // -----

# COM: Test that explicit origins are handled correctly alongside implicit origins.

# CHECK: [[TRAIT:!None_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def[{{.*}}](a: ref[lt] String, b: String) -> None",


# CHECK: lit.struct.decl @"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{[^"]*}}"<impl: {{.*}}, origin_set: origin.set, |>({{.*}}) attributes {definesClosure,{{.*}}synthetic}
# CHECK: lit.struct.field field0 : !kgen.param<:[[TRAIT]] impl>

# CHECK-NEXT: lit.fn @"__call__{{.*}}"<{{.*}}, lt: !lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *"lt._mlir_origin`2x">>>[
# CHECK-NEXT: [[FIELD:%.*]] = lit.ref.struct.ger %0[field0]
# CHECK-NEXT: [[V1:%.*]] = lit.ref.immut [[FIELD]]
# CHECK-NEXT: [[V2:%.*]] = lit.call[!lit.generator<[2](!lit.ref<:[[TRAIT]] impl, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none>:
# CHECK-SAME: bind_params(:!lit.generator<<{{.*}}"lt": !lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *(0,0)>>>[2](!lit.ref<:[[TRAIT]] impl, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none
# CHECK-SAME:> #kgen.get_witness<:[[TRAIT]] impl, "def[{{.*}}](a: ref[lt] String, b: String) -> None", "__call__{{.*}}">{{.*}}][muttoimm *{{.*}}->field0, {{.*}}]([[V1]], %a, %b)
# CHECK-NEXT: lit.return [[V2]] : !kgen.none
# CHECK-NEXT: lit.end_fn


def make_closure(x: Int, mem: String) -> Int:
    def mutate[
        lt: Origin[mut=True]
    ](a: Pointer[String, lt]._mlir_type, b: String) {var}:
        _ = mem

    return x


# // -----

# COM: Verify that the constructor is assembled correctly


trait MyInterface:
    def thing(self):
        ...


# CHECK: [[TRAIT:!None_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def[T: MyInterface](a: T) -> None", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


# CHECK: lit.fn @"__init__($0$)"[mut *"impl`", mut *"self`"](%impl: !lit.ref<:[[TRAIT]] impl, mut *"impl`"> owned_in_mem, |, ?, %self: !lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *"self`"> byref_result)
# CHECK-NEXT: [[V0:%.*]] = lit.ref.struct.ger %self[field0] : <!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *"self`"> -> :[[TRAIT]] impl
# CHECK-NEXT: [[V1:%.*]] = lit.call[!lit.generator<[2](*, "take": !lit.ref<:[[TRAIT]] impl, mut *[0,0]> deinit_mem, ?, "self": !lit.ref<:[[TRAIT]] impl, mut *[0,1]> byref_result) -> !kgen.none>: #kgen.get_witness<:[[TRAIT]] impl, "{{.*}}::Movable", "__init__(take:$0$)">][mut *"impl`", mut *"self`"->field0](%impl, [[V0]])
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: lit.return %none : !kgen.none
# CHECK-NEXT: lit.end_fn


def make_closure(x: Int, mem:String) -> Int:
    def parametric[T: MyInterface](a: T) {var}:
        _ = mem

    return x


# // -----


# COM: Verify the closure instance is created correctly.

# CHECK: [[INT:!Int.*]] = !lit.struct<@{{.*}}::@Int>
# CHECK: [[TRAIT:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


def make_closure(x: Int, mem:String):
    # CHECK: [[RAW_CLOSURE:%.*]] = lit.closure.init[{{.*}}](%x, {{.*}})(%arg0[y]: [[INT]]) capturing -> [[INT]] {
    # CHECK-NEXT: [[BODY_OP:%.*]] = lit.call {{.*}}@Int::@"__add__{{.*}}"(%x, %arg0) : !lit.generator<("lhs": [[INT]], "rhs": [[INT]]) -> [[INT]]>
    # CHECK-NEXT: lit.return [[BODY_OP]] : [[INT]]
    # CHECK-NEXT: lit.end_fn
    # CHECK-NEXT: } : ([[INT]], {{.*}}), !lit.ref<!kgen.closure<@{{.*}}::make_closure{{.*}}", "my_closure" nonescaping>, mut *"[[L0:.*]]">

    # CHECK-NEXT: [[REBOUND_CLOSURE:%.*]] = kgen.rebind [[RAW_CLOSURE]] : !lit.ref<!kgen.closure<@{{.*}}::make_closure{{.*}}", "my_closure" nonescaping>, mut *"[[L0:.*]]"> to !lit.ref<struct<(!Int1, !String) memoryOnly>, mut *"[[L0]]">
    # CHECK-NEXT: lit.ownership.use [[RAW_CLOSURE]]
    # CHECK-NEXT: [[WRAPPER:%.*]] = lit.var.decl "my_closure" var : !lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT]] {{.*}}, :origin.set {}>>, mut *"[[L1:.*]]">
    # CHECK-NEXT: lit.call {{.*}}::@"def(y: Int) -> Int_{{.*}}"::@"__init__($0$)"[mut *"[[L0]]", mut *"[[L1]]"]<:[[TRAIT]] {{.*}}, :origin.set {}>([[REBOUND_CLOSURE]], [[WRAPPER]]) : !lit.generator<[2]("impl": !lit.ref<struct<(!Int1, !String) memoryOnly>, mut *[0,0]> owned_in_mem, |, ?, "self": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] {{.*}}, :origin.set {}>>, mut *[0,1]> byref_result) -> !kgen.none>

    def my_closure(y: Int) {var x, var mem} -> Int:
        return x + y


# // -----

# COM: Check that the argument is augmented at the definition site.

# CHECK-DAG: [[TRAIT:!Int.*]] = !lit.trait<@"def(y: Int) -> Int">


# CHECK: lit.fn @"take_closure{{.*}}"<f: [[TRAIT]]>[imm *"myFunc`"](%myFunc: !lit.ref<:[[TRAIT]] f, imm *"myFunc`"> read_mem, %x: !Int1) capturing -> !kgen.none
# CHECK-NEXT: %0 = lit.call tail[!lit.generator<[1](!lit.ref<:!Int f, mut *[0,0]> read_mem, |, "y": !Int1) capturing -> !Int1>: #kgen.get_witness<:!Int f, "def(y: Int) -> Int", "__call__{{.*}}">][imm *"myFunc`"](%myFunc, %x)
# CHECK-NEXT: lit.ownership.use %0
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
def take_closure[f: def(y: Int) -> Int](myFunc: f, x: Int):
    _ = myFunc(x)


# // -----

# COM: Ensure the transformed parameters are propagated into the underlying closure trait.


# CHECK-DAG: [[TRAIT:!Int_AnyType_ImplicitlyDestructible_Movable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@Movable>
# CHECK-DAG: [[TRAIT2:!Int.*]] = !lit.trait<@"def(y: Int) -> Int">
# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<@{{.*}}::@Int>

# CHECK: lit.trait.decl @"def(y: Int) -> Int"
# CHECK: lit.fn *"nested[def(y: Int) -> Int]($0,::Int)"<closure2: [[TRAIT2]]>
def take_closure[closure1: def(y: Int) -> Int](x: Int):
    def nested[
        closure2: def(y: Int) -> Int
    ](impl: closure2, y: Int) {var x} -> Int:
        return x


# // -----

# COM: ensure many closure parameters are handled.

# CHECK: lit.fn @"take_closures{{.*}})"
# CHECK-SAME: <closure1: !Int2, T: !Int1, closure2: !Int, U: !Int1>
# CHECK-SAME: [imm *"[[L0:.*]]`", imm *"[[L1:.*]]`1"]
# CHECK-SAME: (%impl1: !lit.ref<:!Int2 closure1, imm *"[[L0]]`"> read_mem
# CHECK-SAME:, %impl2: !lit.ref<:!Int closure2, imm *"[[L1]]`1"> read_mem, %x: !Int1) capturing -> !kgen.none


def take_closures[
    closure1: def(y: Int) -> Int,
    T: Int,
    closure2: def(y: Int, z: Int) -> Int,
    U: Int,
](impl1: closure1, impl2: closure2, x: Int):
    pass


# // -----

# COM: Unified Closure Parameters compose

# CHECK: [[INNER:!Int1.*]] = !lit.trait<@"def(z: Int) -> Int">
# CHECK: lit.fn @"__call__{{.*}}"<y: [[INNER]]>


# CHECK: lit.fn @"nested[{{.*}})"
# CHECK-SAME: <x: !Int, +>[imm *"[[L0:.*]]"]
# CHECK-SAME: (%impl: !lit.ref<:!Int x, imm *"[[L0]]"> read_mem
# TODO: remove the 'do_not_dce_int' argument (MOCO 2461)
def nested[
    x: def[y: def(z: Int) -> Int](impl: y, u: Int) -> Int, //
](impl: x, do_not_dce_int: Int):
    pass


# // -----

# COM: Check that the struct generator of the lit op is generated correctly.


# CHECK: [[TRAIT:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def(z: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


# CHECK: kgen.struct.generator @"bindIt(::Int,::Int,::String)::myclosure": [[TRAIT]] = struct_inst<"bindIt(::Int,::Int,::String)::myclosure"{{.*}}memoryOnly>{
# CHECK: kgen.conformance @"{{.*}}::AnyType" {
# CHECK-NEXT: }
# CHECK: kgen.conformance @"{{.*}}::ImplicitlyDestructible" {
# CHECK-NEXT: kgen.witness "__del__{{.*}}"
# CHECK: kgen.conformance @"{{.*}}::Movable" {
# CHECK-NEXT: kgen.witness "__init__(take:$0$)"
# CHECK: kgen.conformance @"def(z: Int) -> Int" {
# CHECK-NEXT: kgen.witness "__call__{{.*}}"
def bindIt(x: Int, y: Int, mem:String) -> Int:
    def myclosure(z: Int) {var x, var y, var mem} -> Int:
        return x + y + z


# // -----

# COM: Check that parameters are emitted correctly


# CHECK: kgen.struct.generator @"bindIt({{.*}})::myclosure"
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator<<"my_param": !AnyType>
# CHECK-SAME: [1](!lit.ref<struct_inst<"bindIt(::String)::myclosure"{{.*}}memoryOnly>, mut *[0,0]> read_mem, |, "z": !Int) capturing -> !kgen.none
# CHECK-SAME:> = #kgen.closure.symbol<@"bindIt(::String)", "myclosure", #kgen.closure_method<call>
# CHECK-SAME:, <:!AnyType ?>>


# CHECK: lit.file_module
def bindIt(mem: String) -> Int:
    def myclosure[my_param: AnyType](z: Int) {var}:
        _ = mem


# // -----

# COM: Verify Conformance tables of the Wrapper are generated correctly

# CHECK: [[TRAIT:!None_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def[{{.*}}](a: ref[lt] String, b: String) -> None",


# CHECK: lit.struct.decl @"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{[^"]*}}"
# CHECK-SAME: <impl: [[TRAIT]], origin_set: origin.set, |>({{.*}}) attributes {definesClosure,{{.*}}synthetic}


# CHECK: kgen.conformance @"def[{{.*}}](a: ref[lt] String, b: String) -> None" {
# CHECK-NEXT: kgen.witness "__call__{{.*}}" : !lit.generator<<{{.*}}"lt": !lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *(0,0)>>>[2](!lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none
# CHECK-SAME: > = @{{.*}}::@"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{.*}}"::@"__call__{{.*}}"<:[[TRAIT]] impl, :origin.set origin_set,

# CHECK: kgen.conformance @{{.*}}::Movable" {
# CHECK-NEXT: kgen.witness "__init__{{.*}}" : !lit.generator<[2](*, "take": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> deinit_mem, ?, "self": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,1]> byref_result) -> !kgen.none
# CHECK-SAME: > = @{{.*}}::@"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{.*}}"::@"__init__(take:

# CHECK: kgen.conformance @"{{.*}}::ImplicitlyDestructible" {
# CHECK-NEXT:  kgen.witness "__del__{{.*}}" : !lit.generator<[1]("self": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> deinit_mem, |) -> !kgen.none
# CHECK-SAME: > = @{{.*}}::@"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{.*}}"::@"__del__{{.*}}"<{{.*}}>

# CHECK: kgen.conformance @"{{.*}}::AnyType" {
# CHECK-NEXT: }


def make_closure(x: Int, mem:String) -> Int:
    def mutate[
        lt: Origin[mut=True]
    ](a: Pointer[String, lt]._mlir_type, b: String) {var}:
        _ = mem

    return x


# // -----

# COM: Check that the origin set is bound to the wrapper

# CHECK-LABEL: lit.fn @"nonemptyOriginSet(::String&)"
# CHECK: lit.closure.init[#kgen.type<typevalue<:[[TRAIT:!None_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]]


def nonemptyOriginSet(mut byRefMut: String):
    # CHECK: lit.call @unified_closure::@"def() -> None_{{.*}}"::@"__init__({{.*}})"
    # CHECK-SAME: :origin.set {}
    def myclosure() {mut byRefMut}:
        pass


# // -----

# COM: Verify that closures can be rebound to compatible traits

# CHECK-DAG: [[TRAIT1:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def(x: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>
# CHECK-DAG: [[TRAIT:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable_Int.*]] = !lit.trait<@"def(x: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable,
# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<{{.*}}::@Int>

# CHECK: lit.struct.decl @"def(x: Int) -> Int_{{.*}}"<impl: [[TRAIT1]], origin_set: origin.set, |>([[TRAIT]])
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator<[1](!lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, [[INT]], |) capturing -> [[INT]]> =
# CHECK-SAME: rebind(:!lit.generator<[1](!lit.ref<!lit.struct<[[T]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "x": [[INT]]) capturing -> [[INT]]> @{{.*}}::@"def(x: Int) -> Int_{{.*}}"::@"__call__{{.*}}"<:[[TRAIT1]] impl, :origin.set origin_set>)


def takeIt[C: def(Int) -> Int](closure: C):
    _ = closure(3)


def bindIt(z: Int, mem:String):
    def myclosure(x: Int) {var} -> Int:
        _ = mem
        return z

    takeIt[type_of(myclosure)](myclosure)


# // -----

# COM: Verify that closures can be rebound even when traits are combined

# CHECK-DAG: [[TRAIT1:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def(x: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>
# CHECK-DAG: [[TRAIT:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable_Int.*]] = !lit.trait<@"def(x: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable,
# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<{{.*}}::@Int>

# CHECK: lit.struct.decl @"def(x: Int) -> Int_{{.*}}"<impl: [[TRAIT1]], origin_set: origin.set, |>([[TRAIT]])
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator<[1](!lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "y": [[INT]]) capturing -> [[INT]]> =
# CHECK-SAME: rebind(:!lit.generator<[1](!lit.ref<!lit.struct<[[T]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "x": [[INT]]) capturing -> [[INT]]> @{{.*}}::@"def(x: Int) -> Int_{{.*}}"::@"__call__{{.*}}"<:[[TRAIT1]] impl, :origin.set origin_set>)


def takeIt[C: Copyable & def(y: Int) -> Int](closure: C):
    _ = closure(3)


def bindIt(z: Int, mem: String):
    def myclosure(x: Int) {var} -> Int:
        _ = mem
        return z

    takeIt[type_of(myclosure)](myclosure)


# // -----

# COM: Verify that all closures are rebound when closure traits are combined or inherited


def takeIt[C: (def(Bool) -> Int) & def(Int) -> Int](closure: C):
    _ = closure(3)


trait BoolWrapper(def(Bool) -> Int):
    pass


# CHECK: lit.struct.decl @MultipleClosure

# CHECK: kgen.conformance @"def(Bool) -> Int"
# CHECK: kgen.witness "__call__($0,::Bool)" : !lit.generator<[1](!lit.ref<!MultipleClosure, mut *[0,0]> read_mem, !Bool, |) capturing -> !Int1> = rebind(:!lit.generator<[1]("self": !lit.ref<!MultipleClosure, imm *[0,0]> read_mem, "x": !Bool) capturing -> !Int1> @{{.*}}::@MultipleClosure::@"__call__({{.*}}::MultipleClosure,::Bool)")


# CHECK: kgen.conformance @"def(Int) -> Int"
# CHECK:kgen.witness "__call__($0,::Int)" : !lit.generator<[1](!lit.ref<!MultipleClosure, mut *[0,0]> read_mem, !Int1, |) capturing -> !Int1> = rebind(:!lit.generator<[1]("self": !lit.ref<!MultipleClosure, imm *[0,0]> read_mem, "x": !Int1) capturing -> !Int1> @{{.*}}::@MultipleClosure::@"__call__({{.*}}::MultipleClosure,::Int)")
struct MultipleClosure(BoolWrapper, Movable, def(Int) -> Int):
    def __init__(out self):
        pass

    def __call__(self, x: Bool) -> Int:
        return 1

    def __call__(self, x: Int) -> Int:
        return 2


def bindIt(z: Int):
    var fakeclosure = MultipleClosure()

    takeIt[type_of(fakeclosure)](fakeclosure)


# // -----

# COM: Verify that closures can be rebound with differing parameter names

# CHECK-DAG: [[TRAIT1:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def[a: Int](b: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>
# CHECK-DAG: [[TRAIT:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable_Int.*]] = !lit.trait<@"def[a: Int](b: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable,
# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<{{.*}}::@Int>

# CHECK: lit.struct.decl @"def[a: Int](b: Int) -> Int_{{.*}}"<impl: [[TRAIT1]], origin_set: origin.set, |>([[TRAIT]])
# CHECK: kgen.conformance @"def[x: Int](y: Int) -> Int"
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator<<"x": [[INT]]>[1](!lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "y": [[INT]]) capturing -> [[INT]]> =
# CHECK-SAME: rebind(:!lit.generator<<"a": [[INT]]>[1](!lit.ref<!lit.struct<[[T]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "b": [[INT]]) capturing -> [[INT]]> @{{.*}}::@"def[a: Int](b: Int) -> Int_{{.*}}"::@"__call__[::Int]({{.*}}::def[a: Int](b: Int) -> Int_{{.*}}"<:[[TRAIT1]] impl, :origin.set origin_set, :[[INT]] ?>)


def takeIt[C: def[x: Int](y: Int) -> Int](closure: C):
    # see MOCO-2606
    _ = closure.__call__[2](3)


def bindIt(z: Int, mem: String):
    def myclosure[a: Int](b: Int) {var} -> Int:
        _ = mem
        return z

    takeIt[type_of(myclosure)](myclosure)


# // -----

# COM: Ensure that structs can conform to the closure trait


# CHECK: [[TRAIT:!Int_AnyType.*]] = !lit.trait<@"def(x: Int) -> Int"
# CHECK: lit.struct.decl @custom([[TRAIT]])
struct custom(def(x: Int) -> Int):
    def __call__(self, x: Int) capturing -> Int:
        return x


# // -----

# COM: The wrapper conforms to copyable

# CHECK: [[CANONICAL_TRAIT:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable*.]] = !lit.trait<
# CHECK-SAME: @"def(x: Int) -> Int"
# CHECK-SAME:, @{{.*}}::@Movable
# CHECK-SAME:, @{{.*}}::@ImplicitlyDestructible
# CHECK-SAME:, @{{.*}}::@AnyType
# CHECK-SAME:, @{{.*}}::@Copyable
# CHECK-SAME:, @{{.*}}::@ImplicitlyCopyable>


# CHECK: lit.struct.decl @"def(x: Int) -> Int_{{.*}}"
# CHECK: lit.struct.decl @"def(x: Int) -> Int_{{.*}}"<impl: [[CANONICAL_TRAIT]], origin_set: origin.set, |>([[CANONICAL_TRAIT]])


def takeItImplicit[T: ImplicitlyCopyable](impl: T):
    pass


def takeIt[T: Copyable](impl: T):
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
    takeIt(aThing)

    # COM: uncopyable version can still implement the def(x:Int) -> Int trait
    def anotherThing(x: Int) {var ^} -> Int:
        useIt(one^)
        return x


# // -----

# COM: "U" cannot be called "T" until MOCO-4028 is fixed

# COM: The captured parameter becomes an alias on the trait
# CHECK: lit.trait.decl @"def{{.*}} -> U"
# CHECK-NEXT: lit.alias.decl U: !TrivialRegisterPassable

# COM: The captured parameter becomes a parameter of the struct generator
# CHECK: kgen.struct.generator @"makeIt{{.*}}::parametric"<U: !TrivialRegisterPassable>
# CHECK: kgen.witness "U" : !TrivialRegisterPassable = U


# COM: The alias is set to the alias of the impl in the struct wrapper
# CHECK: lit.struct.decl @"def{{.*}} -> U_{{.*}}"
# CHECK: kgen.witness "U" : !TrivialRegisterPassable = #kgen.get_witness<:!{{.*}} impl, "def{{.*}} -> U", "U">
def makeIt[U: TrivialRegisterPassable](a: U):
    def parametric() {var a} -> U:
        return a


# // -----

# COM: Check that device passable conformance is emitted properly


def conditionallyDevicePassable(x: Int):
    # CHECK: kgen.conformance @"{{.*}}::DevicePassable" {
    # CHECK-NEXT: kgen.witness "device_type" : type =
    # CHECK-NEXT: kgen.witness "_is_convertible_to_device_type{{.*}}" : !lit.generator
    # CHECK-NEXT: kgen.witness "_to_device_type{{.*}}" : !lit.generator
    # CHECK-NEXT: kgen.witness "get_type_name{{.*}}" : !lit.generator
    def device_passable() {var} -> Int:
        return x


# // -----

# COM: Ensure external parameter references are pulled into alias decls


trait DoIt:
    def thing(self):
        ...


# CHECK: lit.trait.decl @"def{{.*}} -> None"
# CHECK-NEXT: lit.alias.decl T: !DoIt
struct House[T: DoIt]:
    def aMethod[C: def(x: Self.T)](self, impl: C):
        pass


# CHECK: lit.trait.decl @"def{{.*}} -> None"
# CHECK-NEXT: lit.alias.decl TT: !DoIt
def useIt[TT: DoIt, C: def(x: TT)](impl: C):
    pass


# // -----

# CHECK: kgen.conformance @"{{.*}}::RegisterPassable" {

def takesRegisterPassable[T: RegisterPassable](impl: T):
    pass


def addTrivialRegisterPassable(x: Int):
    def closure() {var} -> Int:
        return x
    takesRegisterPassable(closure)


# // -----

# COM: Verify top-level function symbols get conformance for count's closure
# COM: trait.


@fieldwise_init
struct ToyBool:
    var value: Int


@fieldwise_init
struct ToyMask[dtype_tag: Int, w: Int]:
    var value: Int


struct ToySIMD[dtype_tag: Int, w: Int]:
    pass


struct ToyScalar[dtype_tag: Int]:
    pass


@fieldwise_init
struct MiniSpan[dtype_tag: Int]:
    var value: Int

    def count[
        F: def[w: Int](vec: ToySIMD[Self.dtype_tag, w]) -> ToyMask[
            Self.dtype_tag, w
        ]
    ](self, func: F) -> Int:
        return 0


# CHECK: lit.struct.decl @"def[w: Int](vec: ToySIMD[1, w]) -> ToyMask[1, w]_{{.*}}"
# CHECK: kgen.conformance @"def[{{.*}}w: Int](vec: ToySIMD[dtype_tag, w]) -> ToyMask[dtype_tag, w]" {
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator
# CHECK: kgen.witness "dtype_tag" : !Int = {1}


def is_vec_a[w: Int](vec: ToySIMD[1, w]) -> ToyMask[1, w]:
    _ = vec
    return ToyMask[1, w](0)


def repro_top_level():
    var s = MiniSpan[1](0)
    _ = s.count(is_vec_a)


# // -----

# COM: Verify nested captured closures get conformance for count's
# COM: closure trait.


@fieldwise_init
struct ToyBool:
    var value: Int


@fieldwise_init
struct ToyMask[dtype_tag: Int, u: Int]:
    var value: Int


struct ToySIMD[dtype_tag: Int, u: Int]:
    pass


struct ToyScalar[dtype_tag: Int]:
    pass


@fieldwise_init
struct MiniSpan[dtype_tag: Int]:
    var value: Int

    def count[
        F: def[u: Int](vec: ToySIMD[Self.dtype_tag, u]) -> ToyMask[
            Self.dtype_tag, u
        ]
    ](self, func: F) -> Int:
        return 0


# CHECK: lit.struct.decl @"def[u: Int](vec: ToySIMD[1, u]) -> ToyMask[1, u]_{{.*}}"
# CHECK: kgen.conformance @"def[{{.*}}u: Int](vec: ToySIMD[dtype_tag, u]) -> ToyMask[dtype_tag, u]" {
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator
# CHECK: kgen.witness "dtype_tag" : !Int = {1}
def repro_capturing(mem:String):
    var capture = 0

    def is_vec_a_capturing[
        u: Int
    ](vec: ToySIMD[1, u]) {var capture, var mem} -> ToyMask[1, u]:
        _ = vec
        _ = capture
        return ToyMask[1, u](0)

    var s = MiniSpan[1](0)
    _ = s.count(is_vec_a_capturing)


# // -----

# COM: Verify nested type parameters constrained by a trait (not just Int
# COM: parameters) get conformance resolved from nested struct type arguments.


trait ElemLike:
    pass


struct ConcreteElem(ElemLike):
    pass


@fieldwise_init
struct Box[E: ElemLike, n: Int]:
    var value: Int


@fieldwise_init
struct Store[E: ElemLike]:
    var value: Int

    def apply[
        F: def[n: Int](item: Box[Self.E, n]) -> Box[Self.E, n]
    ](self, func: F) -> Int:
        return 0


# CHECK: lit.struct.decl @"def[n: Int](item: Box[ConcreteElem, n]) -> Box[ConcreteElem, n]_{{.*}}"
# CHECK: kgen.conformance @"def[{{.*}}n: Int](item: Box[E, n]) -> Box[E, n]" {
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator
# CHECK: kgen.witness "E" : !ElemLike = !ConcreteElem
def repro_nested_type_param(mem: String):
    var capture = 0

    def apply_concrete[
        n: Int
    ](item: Box[ConcreteElem, n]) {var capture, var mem} -> Box[ConcreteElem, n]:
        _ = item
        _ = capture
        return Box[ConcreteElem, n](0)

    var s = Store[ConcreteElem](0)
    _ = s.apply(apply_concrete)


# // -----

# COM: Verify that custom types (the result type !kgen.none in this case) are compared using equality


def print(x: Int):
    pass


def callee[
    func: def[width: Int, rank: Int, alignment: Int = 1]() -> None,
    //,
    simd_width: Int,
](shape: Int, ctx: Int, closure: func):
    closure[simd_width, 2]()


# CHECK: lit.struct.decl @"def[simd_width: Int, rank: Int, alignment: Int]() -> None_{{.*}}"
# CHECK: kgen.conformance @"def[width: Int, rank: Int, alignment: Int]() -> None" {
# CHECK:   kgen.witness "__call__{{.*}}" : !lit.generator
def main() raises:
    var x = 42
    var mem: String = "hello"

    @always_inline
    def my_func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ]() {read x, var mem}:
        print(x)

    callee[simd_width=4](10, 11, my_func)


# // -----

# COM: Verify the result is properly rebound in the struct wrapper when a closure
# COM: lazily conforms to a trait whose return type contains an alias parameter.


@fieldwise_init
struct V[dtype: Int, width: Int](RegisterPassable):
    var _v: Int


# CHECK: lit.struct.decl @"def[width: Int]() -> V[42, width]_PtrWrapper"

# CHECK: lit.fn @"__call__$def{{.*}} -> V{{.*}}"
# CHECK: kgen.rebind %{{.*}} : {{.*}}{42}{{.*}} to {{.*}}_dtype{{.*}}
# CHECK-NEXT: lit.return


# CHECK: kgen.conformance @"def[dtype: Int, #, width: Int]() -> V[dtype, width]" {
# CHECK-NEXT: kgen.witness "__call__{{.*}}" : !lit.generator
# CHECK-NEXT: kgen.witness "dtype" :{{.*}} = {42}
def callee[
    dtype: Int,
    F: RegisterPassable & def[width: Int]() -> V[dtype, width],
](closure: F):
    var result = closure[4]()


def rebindResult():
    def my_closure[width: Int]() {} -> V[42, width]:
        return V[42, width](0)

    callee[42](my_closure)


# // -----

# COM: Verify ParamListAttr matching: closure returning Tuple with parameterized
# COM: elements requires recursive matching through #kgen.param_list param values.


struct ToyIndex[size: Int](RegisterPassable):
    var _v: Int

    def __init__(out self):
        self._v = 0


def variadic_callee[
    rank: Int,
    map_fn: def(ToyIndex[rank]) -> Tuple[
        ToyIndex[rank],
        ToyIndex[rank],
    ],
](closure: map_fn):
    var point = ToyIndex[rank]()
    var result = closure(point)


# CHECK: lit.struct.decl @"def(point: ToyIndex[2]) -> Tuple[ToyIndex[2], ToyIndex[2]]_{{.*}}"
# CHECK: @"def[rank: Int, #](ToyIndex[rank]) -> Tuple[ToyIndex[rank], ToyIndex[rank]]" {
# CHECK:   kgen.witness "__call__{{.*}}" : !lit.generator
# CHECK:   kgen.witness "rank" : !Int = {2}
def repro_variadic_attr():
    var x = 10
    var mem: String = "hello"
    def my_map_fn(
        point: ToyIndex[2],
    ) {read x, var mem} -> Tuple[ToyIndex[2], ToyIndex[2]]:
        return ToyIndex[2](), ToyIndex[2]()

    variadic_callee[2, type_of(my_map_fn)](my_map_fn)


# // -----

# COM: Verify ParamOperatorAttr and LITStructAttr matching: Pair(tag, 0) lowers
# COM: to #kgen.param.expr<apply, ...> containing #lit.struct constants, which
# COM: requires recursive matching through both composite attr types.


@fieldwise_init
struct Pair(RegisterPassable):
    var a: Int
    var b: Int


@fieldwise_init
struct Container[p: Pair](RegisterPassable):
    var value: Int


def struct_callee[
    tag: Int,
    F: def() -> Container[Pair(tag, 0)],
](closure: F):
    var result = closure()


# CHECK-LABEL: lit.fn @"repro_struct_attr()"
# CHECK: lit.closure.init[#kgen.type<typevalue<:trait<@"def() -> Container[Pair(2, 0)]"
# CHECK: lit.call @unified_closure::@"struct_callee[::Int,def[tag: Int, #]() -> Container[Pair(tag, 0)]]($1){(eq $1.tag, $0)}"
# CHECK-SAME: <:!Int {2}
def repro_struct_attr():
    var x = 10

    def my_fn() {read x} -> Container[Pair(2, 0)]:
        return Container[Pair(2, 0)](x)

    struct_callee[2, type_of(my_fn)](my_fn)


# // -----

# COM: Verify SymbolConstantAttr matching: closure returning a type
# COM: parameterized by a function reference (exercises symbol recursion).


struct Dispatch[F: def(Int) thin -> Int]:
    var data: Int

    def __init__(out self, data: Int):
        self.data = data


def identity(x: Int) -> Int:
    return x


def symbol_callee[
    tag: Int,
    C: def() -> Dispatch[identity],
](closure: C):
    var result = closure()


# CHECK-LABEL: lit.fn @"repro_symbol_attr()"
# CHECK: lit.closure.init[#kgen.type<typevalue<:trait<@"def() -> Dispatch[identity]"
# CHECK: lit.call @unified_closure::@"symbol_callee[::Int,def() -> Dispatch[identity]]($1)"{{.*}}<:!Int {1}
def repro_symbol_attr():
    var x = 10

    def my_fn() {read x} -> Dispatch[identity]:
        return Dispatch[identity](x)

    symbol_callee[1, type_of(my_fn)](my_fn)


# // -----

# COM: Ensure non-ref closure call operands are transformed/rebound in wrapper
# COM: __call__ before dispatching to impl witness call.


struct Width(TrivialRegisterPassable):
    var _mlir_value: __mlir_type.index

    @always_inline("builtin")
    def __mlir_index__(self) -> __mlir_type.index:
        return self._mlir_value

    @implicit
    @always_inline
    def __init__[T: AnyType](out self, value: T):
        pass


struct Vec[tag: Int, size: Width](TrivialRegisterPassable):
    var _dummy: __mlir_type.i1


def repro_rebind_nonref_operand[
    tag: Int,
    F: def[w: Width](v: Vec[tag, w]) -> Bool,
](func: F):
    # CHECK-LABEL: lit.fn @"__call__[::Int,::Int](unified_closure::def[tag: Int, #, w: Int](val: Vec[tag, w]) -> Bool_{{.*}}"
    # CHECK: [[REBIND:%.*]] = kgen.rebind %val : !lit.struct<#Vec <:!Int _tag
    # CHECK-SAME: to !lit.struct<#Vec <:!Int #kgen.get_witness<:!{{.*}} impl, "def[tag: Int, #, w: Int](val: Vec[tag, w]) -> Bool", "tag">
    # CHECK: lit.call[{{.*}}"val": !lit.struct<#Vec <:!Int #kgen.get_witness<:!{{.*}} impl, "def[tag: Int, #, w: Int](val: Vec[tag, w]) -> Bool", "tag">
    # CHECK-SAME: ]{{.*}}(%{{.*}}, [[REBIND]])
    def body[w: Int](val: Vec[tag, w]) {read func} -> Bool:
        return func[w=w](val)

    _ = body


# // -----

trait Coord(ImplicitlyCopyable):
    comptime Dim: Int

    def prettyPrint(self):
        pass


@fieldwise_init
struct Cartesian(Coord):
    comptime Dim = 2
    var x: Int
    var y: Int

    def prettyPrint(self):
        pass


struct Sphere(Coord):
    comptime Dim = 2
    var theta: Int
    var phi: Int


@fieldwise_init
struct HasParam[T: Coord](ImplicitlyCopyable):
    comptime P = Self.T.Dim
    var x: Self.T


@fieldwise_init
struct Container[N: Int]:
    pass


def takes[T: Int, F: def(x: Container[T])](impl: F):
    impl(Container[T]())


def takes2[
    T: Int, U: Int, F: def(x: Container[T], y: Container[U])
](impl: F):
    impl(Container[T](), Container[U]())


def takes_w[T: Int, F: def(w: Container[T])](impl: F):
    impl(Container[T]())


# // -----

struct Foo(ImplicitlyCopyable, Movable):
    var x: Int
    var y: Int


def copyIt[X: Copyable](x: X):
    var copy = X.__init__(copy=x)


# CHECK: lit.struct.decl @"def() -> None_{{.*}}"
def thing(foo: Foo):
    # CHECK: kgen.conformance @"std::builtin::{{.*}}::Copyable"
    # CHECK-NEXT: kgen.witness "__init__(copy:$0)" : !lit.generator<[2](*, "copy":
    def thing() {var}:
        _ = foo


# // -----

# COM: Overload resolution with a closure overload must not crash when the
# COM: non-closure argument's struct is not yet body-resolved.


@always_inline
def dispatch[
    FuncType: TrivialRegisterPassable & def() -> None, //
](func: FuncType):
    pass


@always_inline
def dispatch[T: AnyType](val: T):
    pass


def test(x: Foo):
    dispatch(x)


struct Foo:
    var x: Int


# // -----

# COM: Verify generic map where the actual closure returns in-register but the
# COM: trait signature expects a memory-only ByRefResult slot.

# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<@{{.*}}::@Int>

# CHECK: kgen.conformance @"def{{.*}}(x: T) -> U" {
# CHECK:   kgen.witness "__call__{{.*}}" : !lit.generator
# CHECK:   kgen.witness "T" : {{.*}} = [[INT]]
# CHECK:   kgen.witness "U" : {{.*}} = [[INT]]

comptime CollectionElement = ImplicitlyDestructible & ImplicitlyCopyable


def foo(x: Int):
    def map[
        T: CollectionElement,
        U: CollectionElement,
        func: def(x: T) -> U,
    ](item: T, closure: func,) -> U:
        return closure(item)

    def double(x: Int) {mut} -> Int:
        return x * 2

    _ = map[Int, Int, type_of(double)](x, double)


# // -----

# COM: Verify names match cache keys to avoid collisions.


trait DoA:
    def doA(self):
        ...


trait DoB:
    def doB(self):
        ...


# CHECK-DAG: !lit.trait<@"def[U: DoB, #](y: U) -> None">
# CHECK-DAG: !lit.trait<@"def[T: DoA](y: T) -> None">
# CHECK-DAG: !lit.trait<@"def[T: DoA, #](y: T) -> None">
def foo[T: DoA](x: T):
    def closure(y: T) {read x}:
        _ = x

    def closure2[T: DoA](y: T) {var}:
        pass


def bar[U: DoB](x: U):
    def closure(y: U) {var}:
        pass


# // -----

# COM: Verify that @__llvm_metadata on a closure is preserved on the op

# CHECK: LLVMMetadataArray = ["nvvm.maxntid", #pop.array<256> : !pop.array<1, i32>]


def metadata_closure(x: Int):
    @__llvm_metadata(
        `nvvm.maxntid`=__mlir_attr.`#pop.array<256> : !pop.array<1, i32>`
    )
    def _kernel() {var x} -> Int:
        return x

    _ = _kernel()


# // -----

# COM: Verify that @__llvm_arg_metadata on a closure is preserved

# CHECK: LLVMArgMetadataArray
# CHECK-SAME: "nvvm.grid_constant", unit


def arg_metadata_closure(x: Int):
    @__llvm_arg_metadata(x, `nvvm.grid_constant`)
    def _kernel(x: Int) {var} -> Int:
        return x

    _ = _kernel(x)


# // -----

# COM: Verify that a register_passable closure capturing a generic
# COM: register_passable closure and a concrete register_passable struct gets
# COM: convention register_passable (not trivial)

# CHECK: lit.struct.decl @"def(y: Int) -> Int_{{.*}}"{{.*}} register_passable attributes


struct NonTrivialPayload(ImplicitlyCopyable, RegisterPassable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value


def call_inner[
    F: ImplicitlyCopyable & RegisterPassable & def(Int) -> Int
](f: F, x: Int) -> Int:
    var payload = NonTrivialPayload(1)

    def outer(y: Int) {var f, var payload} -> Int:
        return f(y) + payload.value

    return outer(x)


# // -----

# COM: Verify that a register_passable closure capturing a trivially
# COM: register_passable callback and a trivial struct gets convention
# COM: register_passable_trivial.

# CHECK: lit.struct.decl @"def(y: Int) -> Int_{{.*}}"{{.*}} register_passable_trivial attributes


struct TrivialPayload(TrivialRegisterPassable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value


def call_inner[
    F: TrivialRegisterPassable & def(Int) -> Int
](f: F, x: Int) -> Int:
    var payload = TrivialPayload(1)

    def outer(y: Int) {var f, var payload} -> Int:
        return f(y) + payload.value

    return outer(x)


# // -----

def captures_with_default_convention():
    var a, b, c, d = ("a", "b", "c", "d")
    # COM: a
    # CHECK: lit.closure.init[{{.*}}](%{{.*}}[ref: mut *"a
    # COM: b
    # CHECK-SAME: %{{.*}}[ref: muttoimm *"b
    # COM: c
    # CHECK-SAME: %{{.*}}[{{.*}} move])
    # COM: d is omitted because it uses default convention.
    def my_fn() {mut a, b, c^, read}:
        pass

# // -----
#
# COM: Verify stateless promoted closures are registered for apply attributes.


def trigger_dtype():
    comptime k = 64

    def nonsense(n: Int) {} -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    # CHECK: lit.alias.decl *"dtype{{.*}}": !DType = <apply(:!lit.generator<("n": !Int) -> !DType> @{{.*}}::@"nonsense(::Int)`{{.*}}", {64})>
    comptime dtype = nonsense(k)
    var x = SIMD[dtype, 1]()
    _ = x

# // -----

# COM: Verify async closures use an async trait name and async call op.

# CHECK-DAG: lit.trait.decl @"def() async -> None"
# CHECK-DAG: sourceName = #debuginfo.source_name<"def() async -> None">
# CHECK-LABEL: lit.fn @"async_unified_closure()"
# CHECK: lit.async.call[!lit.generator<


def async_unified_closure():
    var value = 0

    async def inc() {mut value}:
        value += 1

    _ = inc()


# // -----

# COM: Verify promoted closures keep captured params before implicit origins.

# CHECK-LABEL: lit.fn @"trigger_dtype_implicit_origin{{.*}}"<n: !Int>() -> !kgen.none
# CHECK: lit.alias.decl *"dtype{{.*}}": !DType = <apply(:!lit.generator<[1]("impl": !lit.ref<!String, imm #lit.comptime.origin> read_mem) -> !DType> rebind(:!lit.generator<[1]("impl": !lit.ref<!String, imm *[0,0]> read_mem) -> !DType> @{{.*}}::@"nonsense{{.*}}"<:!Int n, :!AnyType !String>)
# CHECK-LABEL: lit.fn @"nonsense{{.*}}"<n: !Int, U: !AnyType, +>[imm *{{.*}}](%impl:
def trigger_dtype_implicit_origin[n: Int]():
    def nonsense[U: AnyType, //](impl: U) {} -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    comptime dtype = nonsense("here")
    var x = SIMD[dtype, 1]()
    _ = x

# // -----
#
# COM: Verify promoted stateless closures can bind captured params
# COM: before satisfying thin function generic constraints.

# CHECK-LABEL: lit.fn @"trigger_promoted_params{{.*}}"<n: !Int>() -> !kgen.none
# CHECK: lit.call tail @{{.*}}::@"takesThin{{.*}}"<:!lit.generator<<"U": !AnyType, +>[1]("impl": !lit.ref<:!AnyType *(0,0), imm *[0,0]> read_mem) -> !DType> @{{.*}}::@"nonsense{{.*}}"<:!Int n, :!AnyType ?>
# CHECK-LABEL: lit.fn @"nonsense{{.*}}"<n: !Int, U: !AnyType, +>[imm *{{.*}}](%impl:
def takesThin[FuncType: def[U: AnyType, //](impl: U) thin -> DType]():
    _ = FuncType("here")


def trigger_promoted_params[n: Int]():
    def nonsense[U: AnyType, //](impl: U) {} -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    takesThin[nonsense]()

# // -----

# COM: Verify promoted stateless closures create a function wrapper
# COM: when passed as a value to a thin-compatible parameter.

# CHECK-LABEL: lit.fn @"trigger_promoted_param_wrapper{{.*}}"<n: !Int>() -> !kgen.none
# CHECK: %[[WRAP:.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!lit.struct<#PtrWrapper
# CHECK: lit.call @{{.*}}::@"def[U: AnyType{{.*}}_PtrWrapper"{{.*}}(%[[WRAP]])
# CHECK: %[[WRAP_IMM:.*]] = lit.ref.immut %[[WRAP]]
# CHECK: lit.call @{{.*}}::@"takesFatVale{{.*}}"{{.*}}(%[[WRAP_IMM]])
# CHECK-LABEL: lit.fn @"nonsense{{.*}}"<n: !Int, U: !AnyType, +>[imm *{{.*}}](%impl:
def takesFatVale[FuncType: def[U: AnyType, //](impl: U) -> DType](
    impl: FuncType
):
    _ = impl("here")


def trigger_promoted_param_wrapper[n: Int]():
    def nonsense[U: AnyType, //](impl: U) -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    takesFatVale(nonsense)

# // -----

# COM: Verify promoted closures keep captured params before implicit origins.

# CHECK-LABEL: lit.fn @"trigger_dtype_implicit_origin{{.*}}"<n: !Int>() -> !kgen.none
# CHECK: lit.alias.decl *"dtype{{.*}}": !DType = <apply(:!lit.generator<[1]("impl": !lit.ref<!String, imm #lit.comptime.origin> read_mem) -> !DType> rebind(:!lit.generator<[1]("impl": !lit.ref<!String, imm *[0,0]> read_mem) -> !DType> @{{.*}}::@"nonsense{{.*}}"<:!Int n, :!AnyType !String>)
# CHECK-LABEL: lit.fn @"nonsense{{.*}}"<n: !Int, U: !AnyType, +>[imm *{{.*}}](%impl:
def trigger_dtype_implicit_origin[n: Int]():
    def nonsense[U: AnyType, //](impl: U) -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    comptime dtype = nonsense("here")
    var x = SIMD[dtype, 1]()
    _ = x

# // -----
#
# COM: Verify promoted stateless unified closures can bind captured params
# COM: before satisfying thin function generic constraints.

# CHECK-LABEL: lit.fn @"trigger_promoted_params{{.*}}"<n: !Int>() -> !kgen.none
# CHECK: lit.call tail @{{.*}}::@"takesThin{{.*}}"<:!lit.generator<<"U": !AnyType, +>[1]("impl": !lit.ref<:!AnyType *(0,0), imm *[0,0]> read_mem) -> !DType> @{{.*}}::@"nonsense{{.*}}"<:!Int n, :!AnyType ?>
# CHECK-LABEL: lit.fn @"nonsense{{.*}}"<n: !Int, U: !AnyType, +>[imm *{{.*}}](%impl:
def takesThin[FuncType: def[U: AnyType, //](impl: U) thin -> DType]():
    _ = FuncType("here")


def trigger_promoted_params[n: Int]():
    def nonsense[U: AnyType, //](impl: U) -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    takesThin[nonsense]()

# // -----

# COM: Verify promoted stateless unified closures create a function wrapper
# COM: when passed as a value to a thin-compatible parameter.

# CHECK-LABEL: lit.fn @"trigger_promoted_param_wrapper{{.*}}"<n: !Int>() -> !kgen.none
# CHECK: %[[WRAP:.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!lit.struct<#PtrWrapper
# CHECK: lit.call @{{.*}}::@"def[U: AnyType{{.*}}_PtrWrapper"{{.*}}(%[[WRAP]])
# CHECK: %[[WRAP_IMM:.*]] = lit.ref.immut %[[WRAP]]
# CHECK: lit.call @{{.*}}::@"takesFatVale{{.*}}"{{.*}}(%[[WRAP_IMM]])
# CHECK-LABEL: lit.fn @"nonsense{{.*}}"<n: !Int, U: !AnyType, +>[imm *{{.*}}](%impl:
def takesFatVale[FuncType: def[U: AnyType, //](impl: U) -> DType](
    impl: FuncType
):
    _ = impl("here")


def trigger_promoted_param_wrapper[n: Int]():
    def nonsense[U: AnyType, //](impl: U) -> DType:
        if n >= 64:
            return DType.int32
        elif n >= 32:
            return DType.uint32
        else:
            return DType.float32

    takesFatVale(nonsense)

# // -----

# COM: Verify comptime conversion of a promoted wrapper value constructs the
# COM: concrete PtrWrapper via apply_result_slot before calling the closure
# COM: parameter.

def take_closure_param[
    C: def[n: Int](arg: Int) -> Int
](impl: C) -> Int:
    return impl[3](4)


@parameter
def legacy(arg0: Int) -> Int:
    return arg0 + 3


def trigger[xx: Int, func: def(Int) capturing -> Int]() -> Int:
    def wrapped_ok[n: Int](arg: Int) -> Int:
        return func(arg) + xx

    # CHECK-LABEL: lit.fn @"trigger
    # CHECK: lit.alias.decl *"X`": !Int1 = <apply(
    # CHECK-SAME: @"take_closure_param[def[n: Int](arg: Int) -> Int]($0)"
    # CHECK-SAME: store_to_mem(apply_result_slot(
    # CHECK-SAME: @"def[n: Int](arg: Int) capturing -> Int_PtrWrapper"::@"__init__()"
    comptime X = take_closure_param[type_of(wrapped_ok)](wrapped_ok)
    # CHECK: lit.call @{{.*}}::@"take_closure_param
    var Y = take_closure_param[type_of(wrapped_ok)](wrapped_ok)

# // -----

# COM: Verify promoted top-level functions with captured parameters
# COM: build a wrapper whose Impl type is self-contained while preserving the
# COM: promoted function symbol's native parameter ordering.

# CHECK-LABEL: lit.struct.decl @"def[dtype: DType, //, simd_width: Int]() -> SIMD[dtype, simd_width]_PtrWrapper"
# CHECK: lit.alias.decl dtype: !DType = <__capture_dtype>
# CHECK: lit.fn @"__call__[::DType,::Int](unified_closure::def[dtype: DType, //, simd_width: Int]() -> SIMD[dtype, simd_width]_PtrWrapper[$0, $1])"
# CHECK: {{.*}} = lit.call tail[!lit.generator<() -> !lit.struct<#SIMD <:!DType _dtype, :!Int simd_width>>>: bind_params(:!lit.generator<<"dtype": !DType, +, "simd_width": !Int>() -> !lit.struct<#SIMD <:!DType *(0,0), :!Int *(0,1)>>> Impl, :!DType _dtype, :!Int simd_width)]()
# CHECK: kgen.witness "dtype" : !DType = __capture_dtype

# CHECK-LABEL: lit.fn @"trigger[::Int,::DType]()"
# CHECK: %[[WRAP:.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!lit.struct
# CHECK-SAME: @{{.*}}::@"compute_init2[::Int]()`0x"<:!DType ?, :!Int ?>
# CHECK: %[[INIT:.*]] = lit.call @{{.*}}::@"def[dtype: DType, //, simd_width: Int]() -> SIMD[dtype, simd_width]_PtrWrapper"::@"__init__()"{{.*}}(%[[WRAP]])
# CHECK: %[[IMM:.*]] = lit.ref.immut %[[WRAP]]
# CHECK: lit.call @{{.*}}::@"local_higher_order

def local_higher_order[
    rank: Int,
    dtype: DType,
    compute_init: def[simd_width: Int]() -> SIMD[dtype, simd_width],
](
    compute_init_closure: compute_init,
):
    pass


def trigger[rank: Int, dtype: DType]():
    def compute_init2[
        simd_width: Int
    ]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    local_higher_order[rank, dtype, type_of(compute_init2)](compute_init2)

# // -----

# COM: Ensure Proper Ordering Of Parameters In Promoted Functions

struct MyList[T:AnyType]:
    pass

# CHECK: lit.fn @"thinClosure{{.*}}"<T: !AnyType, +, *"list`2x": !lit.struct<#MyList <:!AnyType T>>>() -> !Int
def callIt[T:AnyType, list: MyList[T]]():
    def thinClosure[list: MyList[T]]() -> Int:
        return 1
    comptime x = thinClosure[list]()


# // -----


# COM: Thin Closures With Concrete Captures Are Properly Lifted

comptime SIMDSize = Int

struct IndexList[r:Int]:
    pass

def target[
    rank: Int,
    dtype: DType,
    ComputeFnType: def[simd_width: SIMDSize](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width],
](
    compute_func: ComputeFnType,
) raises:
    pass


# CHECK: lit.fn @"compute_gpu

def repro_stencil_indirect_call[
    dtype: DType,
    num_channels: Int,
]() raises:
    comptime rank = 4

    def compute_gpu[
        simd_width: SIMDSize
    ](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        _ = point
        return val + result

    target[rank, dtype, type_of(compute_gpu)](compute_gpu)

# // -----

# COM: Stateless nested functions whose signature references both a captured
# COM: parameter (dtype) and a free wildcard (alignment) are promoted to a
# COM: top-level fn whose call site binds the captured params directly.

def _current_target() -> __mlir_type.`!kgen.target`:
    return __mlir_attr.`#kgen.param.expr<current_target> : !kgen.target`


def _align_of[dtype: DType, target: __mlir_type.`!kgen.target` = _current_target()]() -> Int:
    return 1

@fieldwise_init
struct LayoutTensor[dtype:DType, alignment: Int = _align_of[dtype, _current_target()]()](TrivialRegisterPassable):
   pass

# CHECK-LABEL: lit.fn @"outer
def outer[
    dtype: DType, valid: Bool
](
    a: LayoutTensor[dtype, ...]
) raises:
    comptime assert valid, "need float"

    def inner(
        buf: LayoutTensor[dtype, ...]
    ) -> LayoutTensor[dtype]:
        return LayoutTensor[dtype]()

    # CHECK: lit.call tail @{{.*}}::@"inner{{.*}}"<:!DType dtype, :!Int *"a.alignment{{.*}}">(%a)
    var x = inner(a)


# // -----

# COM: Verify Captures Are Prepended

def bind[D:Copyable, E:Copyable, FuncType: def[F:Copyable](a:D, b:E, c:F)](impl:FuncType):
    pass


# CHECK-LABEL: lit.struct.decl @"def[A: Copyable, B: Copyable, #, C: Copyable](a: A, b: B, c: C) -> None_{{.*}}"
# CHECK: lit.fn @"__call__{{.*}}"<_A: !Copyable, _B: !Copyable, C: !Copyable, +>

def top[A:Copyable, B:Copyable](aa:A, bb:B):
    def closure[C: Copyable, //](a:A, b:B, c:C) {read}:
        pass

    closure(aa, bb, 3)
    bind[A,B,type_of(closure)](closure)


# // -----

# COM: Verify Lazy Conformance

def bind[D:Copyable, E:Copyable, FuncType: def[F:Copyable](a:D, b:E, c:F)](impl:FuncType):
    pass

# CHECK-LABEL: lit.struct.decl @"def[{{.*}}C: Copyable](a: {{.*}}, b: {{.*}}, c: C) -> None_{{.*}}"
# CHECK: lit.fn @"__call__$def
# CHECK-NEXT: kgen.rebind %a : !lit.ref<:!Copyable _D, imm *"1_unnamed`"> to !lit.ref<!String, imm *"1_unnamed`">
# CHECK-NEXT: kgen.rebind %b : !lit.ref<:!Copyable _E, imm *"2_unnamed`"> to !lit.ref<!String, imm *"2_unnamed`">

def top():
    def closureConcrete[C: Copyable, //](a:String, b:String, c:C) {read}:
        pass

    bind[String,String,type_of(closureConcrete)](closureConcrete)

# // -----

# COM: Origins are properly captured and lifted into the struct generator

def can_mutate[FuncType: def() -> None](impl: FuncType):
   impl()

def demo[o: Origin[mut=True]](
   ptr: UnsafePointer[
       Int,
       o,
       address_space=AddressSpace.GENERIC,
   ],
):
   # CHECK: kgen.struct.generator @"demo{{.*}}::write"<*"o._mlir_origin`": origin<true>, o: !lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *"o._mlir_origin`">>>
   # CHECK-SAME: struct_inst<"demo{{.*}}::write"[*"o._mlir_origin`", o]<:origin<true> *"o._mlir_origin`", :!lit.struct<#Origin <:!Bool {:scalar<bool> true}, :origin<true> *"o._mlir_origin`">> o>
   def write() {read ptr}:
       ptr.store(0, 3)
   can_mutate(write)


# // -----

# COM: If a mutable origin is captured but only in the context of a cast to immutable, do not lift and bind a mutable origin to the closure struct

def must_be_read_only[Mut: Bool, //, o: Origin[mut=Mut], FuncType: def() -> None](impl: FuncType, ptr: UnsafePointer[Int, o, address_space=AddressSpace.GENERIC]):
   impl()

def demo[o: Origin[mut=True]](
   ptr: UnsafePointer[
       Int,
       o,
       address_space=AddressSpace.GENERIC,
   ],
):
   var immut_ptr = ptr.as_immutable()

   # CHECK: kgen.struct.generator @"demo{{.*}}::read"<*"o._mlir_origin`": origin<false>, *"immut_ptr{{.*}}": origin<false>>
   # CHECK-SAME: struct_inst<"demo{{.*}}::read"[*"o._mlir_origin`", *"immut_ptr{{.*}}"]<:origin<false> *"o._mlir_origin`", :origin<false> *"immut_ptr{{.*}}">

   def read() {read immut_ptr}:
       _ = immut_ptr[0]

   must_be_read_only(read, immut_ptr)
