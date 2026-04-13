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
# CHECK-NEXT: destructor :
# CHECK-NEXT: move :
# CHECK-NEXT: copy :
# CHECK:  lit.struct.field field0 : !kgen.param<:[[IMPL_PARENT]] impl>
# CHECK: lit.fn @"__call__({{.*}})"[mut *"[[L0:.*]]`"](%0[*""]: !lit.ref<!lit.struct<[[T:#.*]] <:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L0]]`"> read_mem, |, %y: [[INT]]) capturing -> [[INT]]
# CHECK-NEXT:  [[CLOSURE:%.*]] = lit.ref.struct.ger %{{.*}}[field0]
# CHECK-NEXT:  [[RES:%.*]] = lit.call[!lit.generator<[1](!lit.ref<:[[IMPL_PARENT]] impl, mut *[0,0]> read_mem, |, "y": [[INT]]) capturing -> [[INT]]>: #kgen.get_witness<:[[IMPL_PARENT]] impl, "def(y: Int) -> Int", "__call__{{.*}}">][mut *"[[L0]]`"->field0]([[CLOSURE]], %y)
# CHECK-NEXT:  lit.return [[RES]]
# CHECK-NEXT:  lit.end_fn
# CHECK-NEXT: }

# CHECK: lit.fn @"__init__{{.*}}"[mut *"[[L2:.*]]`", mut *"[[L3:.*]]`"](*, %take: !lit.ref<{{.*}}<:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L2]]`"> deinit_mem, ?, %self: !lit.ref<{{.*}} <:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L3]]`"> byref_result) -> !kgen.none
# CHECK: lit.ownership.mark_destroyed %take

# CHECK: lit.fn @"__del__({{.*}})"[mut *"[[L1:.*]]`"](%self: !lit.ref<{{.*}}<:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L1]]`"> deinit_mem, |) -> !kgen.none
# CHECK: lit.ownership.mark_destroyed %self


def make_closure(x: Int):
    def my_closure(y: Int) unified {var x} -> Int:
        return x + y


# // -----

# COM: Verify Nested unified closures are supported


# CHECK: lit.trait.decl @"def(y: Int) -> Int"
# CHECK: lit.trait.decl @"def(z: Int) -> Int"
# CHECK: lit.struct.decl @"def(y: Int) -> Int_{{.*}}"
# CHECK: lit.struct.decl @"def(z: Int) -> Int_{{.*}}"


def make_closure(x: Int):
    def my_closure(y: Int) unified {var x} -> Int:
        def my_nested_closure(z: Int) unified {var x} -> Int:
            return x

        return x + y


# // -----

# COM: Ensure identical closure traits are reused

# CHECK-COUNT-1: lit.trait.decl @"def(y: Int) -> Int"
# CHECK-COUNT-1: lit.struct.decl @"def(y: Int) -> Int


def make_closure(x: Int):
    def my_closure(y: Int) unified {var} -> Int:
        return y


def make_identical_closure(x: Int):
    def my_closure(y: Int) unified {var} -> Int:
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


def make_closure(x: Int) -> Int:
    def parametric[T: MyInterface, b: T, c: Foo[T, b]](a: T) unified {var}:
        pass

    return x


# // -----

# COM: Test that explicit origins are handled correctly alongside implicit origins.

# CHECK: [[TRAIT:!None_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def[{{.*}}](a: ref[lt] String, b: String) -> None",


# CHECK: lit.struct.decl @"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{[^"]*}}"<impl: {{.*}}, origin_set: origin.set, |>({{.*}}) attributes {definesClosure,{{.*}}synthetic}
# CHECK: lit.struct.field field0 : !kgen.param<:[[TRAIT]] impl>

# CHECK-NEXT: lit.fn @"__call__{{.*}}"<{{.*}}, lt: !lit.struct<#Origin <:!Bool {:i1 1}, :origin<1> *"lt._mlir_origin`2x">>>[
# CHECK-NEXT: [[V1:%.*]] = lit.ref.struct.ger %0[field0]
# CHECK-NEXT: [[V2:%.*]] = lit.call[!lit.generator<[2](!lit.ref<:[[TRAIT]] impl, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none>:
# CHECK-SAME: bind_params(:!lit.generator<<{{.*}}"lt": !lit.struct<#Origin <:!Bool {:i1 1}, :origin<1> *(0,0)>>>[2](!lit.ref<:[[TRAIT]] impl, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none
# CHECK-SAME:> #kgen.get_witness<:[[TRAIT]] impl, "def[{{.*}}](a: ref[lt] String, b: String) -> None", "__call__{{.*}}">{{.*}}]([[V1]], %a, %b)
# CHECK-NEXT: lit.return [[V2]] : !kgen.none
# CHECK-NEXT: lit.end_fn


def make_closure(x: Int) -> Int:
    def mutate[
        lt: Origin[mut=True]
    ](a: Pointer[String, lt]._mlir_type, b: String) unified {var}:
        pass

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


def make_closure(x: Int) -> Int:
    def parametric[T: MyInterface](a: T) unified {var}:
        pass

    return x


# // -----


# COM: Verify the closure instance is created correctly.

# CHECK: [[INT:!Int.*]] = !lit.struct<@{{.*}}::@Int>
# CHECK: [[TRAIT:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


def make_closure(x: Int):
    # CHECK: [[RAW_CLOSURE:%.*]] = lit.closure.init[{{.*}}](%x)(%arg0[y]: [[INT]]) capturing -> [[INT]] {
    # CHECK-NEXT: [[BODY_OP:%.*]] = lit.call {{.*}}@Int::@"__add__{{.*}}"(%x, %arg0) : !lit.generator<("lhs": [[INT]], "rhs": [[INT]]) -> [[INT]]>
    # CHECK-NEXT: lit.return [[BODY_OP]] : [[INT]]
    # CHECK-NEXT: lit.end_fn
    # CHECK-NEXT: } : ([[INT]]), !lit.ref<!kgen.closure<@{{.*}}::make_closure{{.*}}", "my_closure" nonescaping>, mut *"[[L0:.*]]">

    # CHECK-NEXT: lit.ownership.use [[RAW_CLOSURE]]
    # CHECK-NEXT: [[WRAPPER:%.*]] = lit.var.decl "my_closure" var : !lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT]] {{.*}}, :origin.set {}>>, mut *"[[L1:.*]]">
    # CHECK-NEXT: lit.call {{.*}}::@"def(y: Int) -> Int_{{.*}}"::@"__init__($0$)"[mut *"[[L0]]", mut *"[[L1]]"]<:[[TRAIT]] {{.*}}, :origin.set {}>([[RAW_CLOSURE]], [[WRAPPER]]) : !lit.generator<[2]("impl": !lit.ref<!kgen.closure<@{{.*}}::make_closure{{.*}}", "my_closure" nonescaping>, mut *[0,0]> owned_in_mem, |, ?, "self": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] {{.*}}, :origin.set {}>>, mut *[0,1]> byref_result) -> !kgen.none>

    def my_closure(y: Int) unified {var x} -> Int:
        return x + y


# // -----

# COM: Check that the argument is augmented at the definition site.

# CHECK-DAG: [[TRAIT:!Int.*]] = !lit.trait<@"def(y: Int) -> Int">


# CHECK: lit.fn @"take_closure{{.*}}"<f: [[TRAIT]]>[imm *"myFunc`"](%myFunc: !lit.ref<:[[TRAIT]] f, imm *"myFunc`"> read_mem, %x: !Int1) capturing -> !kgen.none
# CHECK-NEXT: %0 = lit.call tail[!lit.generator<[1](!lit.ref<:!Int f, mut *[0,0]> read_mem, |, "y": !Int1) capturing -> !Int1>: #kgen.get_witness<:!Int f, "def(y: Int) -> Int", "__call__{{.*}}">][imm *"myFunc`"](%myFunc, %x)
# CHECK-NEXT: lit.ownership.use %0
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
def take_closure[f: def(y: Int) unified -> Int](myFunc: f, x: Int):
    _ = myFunc(x)


# // -----

# COM: Ensure the transformed parameters are propagated into the underlying closure trait.


# CHECK-DAG: [[TRAIT:!Int_AnyType_ImplicitlyDestructible_Movable.*]] = !lit.trait<@"def[closure2: def(y: Int) -> Int](impl: closure2, y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@Movable>
# CHECK-DAG: [[TRAIT2:!Int.*]] = !lit.trait<@"def(y: Int) -> Int">
# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<@{{.*}}::@Int>
# CHECK-DAG: [[TRAIT3:!Int.*]] = !lit.trait<@"def[closure2: def(y: Int) -> Int](impl: closure2, y: Int) -> Int">


# CHECK: lit.trait.decl @"def[closure2: def(y: Int) -> Int](impl: closure2, y: Int) -> Int"
# CHECK-NEXT: lit.fn @"__call__{{.*}}"<closure2: [[TRAIT2]]>
# CHECK-SAME: [mut *"self`", imm *"[[L0:.*]]`"]
# CHECK-SAME: (%0[*""]: !lit.ref<:[[TRAIT3]] *"_Self{{.*}}, mut *"self`"> read_mem, |
# CHECK-SAME:, %impl: !lit.ref<:[[TRAIT2]] closure2, imm *"[[L0]]`"> read_mem, %y: [[INT]]) capturing -> [[INT]]
def take_closure[closure1: def(y: Int) unified -> Int](x: Int):
    def nested[
        closure2: def(y: Int) unified -> Int
    ](impl: closure2, y: Int) unified {var x} -> Int:
        return x


# // -----

# COM: ensure many closure parameters are handled.

# CHECK: lit.fn @"take_closures{{.*}})"
# CHECK-SAME: <closure1: !Int2, T: !Int1, closure2: !Int, U: !Int1>
# CHECK-SAME: [imm *"[[L0:.*]]`", imm *"[[L1:.*]]`1"]
# CHECK-SAME: (%impl1: !lit.ref<:!Int2 closure1, imm *"[[L0]]`"> read_mem
# CHECK-SAME:, %impl2: !lit.ref<:!Int closure2, imm *"[[L1]]`1"> read_mem, %x: !Int1) capturing -> !kgen.none


def take_closures[
    closure1: def(y: Int) unified -> Int,
    T: Int,
    closure2: def(y: Int, z: Int) unified -> Int,
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
    x: def[y: def(z: Int) unified -> Int](impl: y, u: Int) unified -> Int, //
](impl: x, do_not_dce_int: Int):
    pass


# // -----

# COM: Check that the struct generator of the lit op is generated correctly.


# CHECK: [[TRAIT:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def(z: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


# CHECK: kgen.struct.generator @"bindIt({{.*}})::myclosure"
# CHECK-SAME: <CAPTURES: !kgen.param_closure<@{{.*}}::bindIt(::Int,::Int)" "myclosure">>:
# CHECK-SAME: [[TRAIT]] = !kgen.closure<@{{.*}}::bindIt({{.*}})", "myclosure" nonescaping>{
# CHECK: kgen.conformance @"{{.*}}::AnyType" {
# CHECK-NEXT: }
# CHECK: kgen.conformance @"{{.*}}::ImplicitlyDestructible" {
# CHECK-NEXT: kgen.witness "__del__{{.*}}"
# CHECK: kgen.conformance @"{{.*}}::Movable" {
# CHECK-NEXT: kgen.witness "__init__(take:$0$)"
# CHECK: kgen.conformance @"def(z: Int) -> Int" {
# CHECK-NEXT: kgen.witness "__call__{{.*}}"
def bindIt(x: Int, y: Int) -> Int:
    def myclosure(z: Int) unified {var x, var y} -> Int:
        return x + y + z


# // -----

# COM: Check that parameters are emitted correctly


# CHECK: kgen.struct.generator @"bindIt()::myclosure"
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator<<"my_param": !AnyType>
# CHECK-SAME: [1](!lit.ref<!kgen.closure<@{{.*}}::bindIt()", "myclosure" nonescaping>, mut *[0,0]> read_mem, |, "z": !Int) capturing -> !kgen.none
# CHECK-SAME:> = #kgen.closure.symbol<@{{.*}}::bindIt()", "myclosure", #kgen.closure_method<call>
# CHECK-SAME:, <:!AnyType ?, :!kgen.param_closure<@{{.*}}::bindIt()" "myclosure"> CAPTURES>>


# CHECK: lit.file_module
def bindIt() -> Int:
    def myclosure[my_param: AnyType](z: Int) unified {var}:
        pass


# // -----

# COM: Verify Conformance tables of the Wrapper are generated correctly

# CHECK: [[TRAIT:!None_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def[{{.*}}](a: ref[lt] String, b: String) -> None",


# CHECK: lit.struct.decl @"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{[^"]*}}"
# CHECK-SAME: <impl: [[TRAIT]], origin_set: origin.set, |>({{.*}}) attributes {definesClosure,{{.*}}synthetic}


# CHECK: kgen.conformance @"def[{{.*}}](a: ref[lt] String, b: String) -> None" {
# CHECK-NEXT: kgen.witness "__call__{{.*}}" : !lit.generator<<{{.*}}"lt": !lit.struct<#Origin <:!Bool {:i1 1}, :origin<1> *(0,0)>>>[2](!lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none
# CHECK-SAME: > = @{{.*}}::@"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{.*}}"::@"__call__{{.*}}"<:[[TRAIT]] impl, :origin.set origin_set,

# CHECK: kgen.conformance @{{.*}}::Movable" {
# CHECK-NEXT: kgen.witness "__init__{{.*}}" : !lit.generator<[2](*, "take": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> deinit_mem, ?, "self": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,1]> byref_result) -> !kgen.none
# CHECK-SAME: > = @{{.*}}::@"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{.*}}"::@"__init__(take:

# CHECK: kgen.conformance @"{{.*}}::ImplicitlyDestructible" {
# CHECK-NEXT:  kgen.witness "__del__{{.*}}" : !lit.generator<[1]("self": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> deinit_mem, |) -> !kgen.none
# CHECK-SAME: > = @{{.*}}::@"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{.*}}"::@"__del__{{.*}}"<{{.*}}>

# CHECK: kgen.conformance @"{{.*}}::AnyType" {
# CHECK-NEXT: }


def make_closure(x: Int) -> Int:
    def mutate[
        lt: Origin[mut=True]
    ](a: Pointer[String, lt]._mlir_type, b: String) unified {var}:
        pass

    return x


# // -----

# COM: Check that the origin set is bound to the wrapper

# CHECK: [[TRAIT:!None_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable*.]] = !lit.trait<@"def() -> None", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


def nonemptyOriginSet(mut byRefMut: String):
    # CHECK: lit.call {{.*}}::@"def() -> None_{{.*}}"::@"__init__({{.*}})"[{{.*}}]<:[[TRAIT]] {{.*}}, :origin.set {mut *"byRefMut`"}>
    def myclosure() unified {mut byRefMut}:
        pass


# // -----

# COM: Verify that closures can be rebound to compatible traits

# CHECK-DAG: [[TRAIT1:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"def(x: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>
# CHECK-DAG: [[TRAIT:!Int_Movable_ImplicitlyDestructible_AnyType_Copyable_ImplicitlyCopyable_Int.*]] = !lit.trait<@"def(x: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@ImplicitlyDestructible, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable,
# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<{{.*}}::@Int>

# CHECK: lit.struct.decl @"def(x: Int) -> Int_{{.*}}"<impl: [[TRAIT1]], origin_set: origin.set, |>([[TRAIT]])
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator<[1](!lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, [[INT]], |) capturing -> [[INT]]> =
# CHECK-SAME: rebind(:!lit.generator<[1](!lit.ref<!lit.struct<[[T]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "x": [[INT]]) capturing -> [[INT]]> @{{.*}}::@"def(x: Int) -> Int_{{.*}}"::@"__call__{{.*}}"<:[[TRAIT1]] impl, :origin.set origin_set>)


def takeIt[C: def(Int) unified -> Int](closure: C):
    _ = closure(3)


def bindIt(z: Int):
    def myclosure(x: Int) unified {var} -> Int:
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


def takeIt[C: Copyable & def(y: Int) unified -> Int](closure: C):
    _ = closure(3)


def bindIt(z: Int):
    def myclosure(x: Int) unified {var} -> Int:
        return z

    takeIt[type_of(myclosure)](myclosure)


# // -----

# COM: Verify that all closures are rebound when closure traits are combined or inherited


def takeIt[C: (def(Bool) unified -> Int) & def(Int) unified -> Int](closure: C):
    _ = closure(3)


trait BoolWrapper(def(Bool) unified -> Int):
    pass


# CHECK: lit.struct.decl @MultipleClosure

# CHECK: kgen.conformance @"def(Bool) -> Int"
# CHECK: kgen.witness "__call__($0,::Bool)" : !lit.generator<[1](!lit.ref<!MultipleClosure, mut *[0,0]> read_mem, !Bool, |) capturing -> !Int1> = rebind(:!lit.generator<[1]("self": !lit.ref<!MultipleClosure, imm *[0,0]> read_mem, "x": !Bool) capturing -> !Int1> @{{.*}}::@MultipleClosure::@"__call__({{.*}}::MultipleClosure,::Bool)")


# CHECK: kgen.conformance @"def(Int) -> Int"
# CHECK:kgen.witness "__call__($0,::Int)" : !lit.generator<[1](!lit.ref<!MultipleClosure, mut *[0,0]> read_mem, !Int1, |) capturing -> !Int1> = rebind(:!lit.generator<[1]("self": !lit.ref<!MultipleClosure, imm *[0,0]> read_mem, "x": !Int1) capturing -> !Int1> @{{.*}}::@MultipleClosure::@"__call__({{.*}}::MultipleClosure,::Int)")
struct MultipleClosure(BoolWrapper, Movable, def(Int) unified -> Int):
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


def takeIt[C: def[x: Int](y: Int) unified -> Int](closure: C):
    # see MOCO-2606
    _ = closure.__call__[2](3)


def bindIt(z: Int):
    def myclosure[a: Int](b: Int) unified {var} -> Int:
        return z

    takeIt[type_of(myclosure)](myclosure)


# // -----

# COM: Ensure that structs can conform to the closure trait


# CHECK: [[TRAIT:!Int_AnyType.*]] = !lit.trait<@"def(x: Int) -> Int"
# CHECK: lit.struct.decl @custom([[TRAIT]])
struct custom(def(x: Int) unified -> Int):
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
    def aThing(x: Int) unified {var z, var cm} -> Int:
        return z + x

    takeItImplicit(aThing)
    takeIt(aThing)

    # COM: uncopyable version can still implement the def(x:Int) -> Int trait
    def anotherThing(x: Int) unified {var ^} -> Int:
        useIt(one^)
        return x


# // -----


# COM: The captured parameter becomes an alias on the trait
# CHECK: lit.trait.decl @"def{{.*}} -> T"
# CHECK-NEXT: lit.alias.decl T: !TrivialRegisterPassable

# COM: The captured parameter becomes a parameter of the struct generator
# CHECK: kgen.struct.generator @"makeIt{{.*}}::parametric"<{{.*}}, T: !TrivialRegisterPassable>
# CHECK: kgen.witness "T" : !TrivialRegisterPassable = T


# COM: The alias is set to the alias of the impl in the struct wrapper
# CHECK: lit.struct.decl @"def{{.*}} -> T_{{.*}}"
# CHECK: kgen.witness "T" : !TrivialRegisterPassable = #kgen.get_witness<:!{{.*}} impl, "def{{.*}} -> T", "T">
def makeIt[T: TrivialRegisterPassable](a: T):
    def parametric() unified {var a} -> T:
        return a


# // -----

# COM: Check that device passable conformance is emitted properly


def conditionallyDevicePassable(x: Int):
    # CHECK: kgen.conformance @"{{.*}}::DevicePassable" {
    # CHECK-NEXT: kgen.witness "device_type" : type =
    # CHECK-NEXT: kgen.witness "_is_convertible_to_device_type{{.*}}" : !lit.generator
    # CHECK-NEXT: kgen.witness "_to_device_type{{.*}}" : !lit.generator
    # CHECK-NEXT: kgen.witness "get_type_name{{.*}}" : !lit.generator
    def device_passable() unified register_passable {var} -> Int:
        return x


# // -----

# COM: Ensure external parameter references are pulled into alias decls


trait DoIt:
    def thing(self):
        ...


# CHECK: lit.trait.decl @"def{{.*}} -> None"
# CHECK-NEXT: lit.alias.decl T: !DoIt
struct House[T: DoIt]:
    def aMethod[C: def(x: Self.T) unified](self, impl: C):
        pass


# CHECK: lit.trait.decl @"def{{.*}} -> None"
# CHECK-NEXT: lit.alias.decl TT: !DoIt
def useIt[TT: DoIt, C: def(x: TT) unified](impl: C):
    pass


# // -----

# Verify the trait alias includes TrivialRegisterPassable conformance.
# CHECK-DAG: [[TRAIT:![^=]*TrivialRegisterPassable.*]] = !lit.trait<@"def() register_passable -> Int",{{.*}}@std::@builtin::@stubs::@TrivialRegisterPassable>


# CHECK: lit.struct.decl @"def() register_passable -> Int_{{.*}}"<impl: [[TRAIT]]


def addTrivialRegisterPassable(x: Int):
    def closure() unified register_passable {var} -> Int:
        return x


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
        F: def[w: Int](vec: ToySIMD[Self.dtype_tag, w]) unified -> ToyMask[
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

# COM: Verify nested captured unified closures get conformance for count's
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
        F: def[u: Int](vec: ToySIMD[Self.dtype_tag, u]) unified -> ToyMask[
            Self.dtype_tag, u
        ]
    ](self, func: F) -> Int:
        return 0


# CHECK: lit.struct.decl @"def[u: Int](vec: ToySIMD[1, u]) -> ToyMask[1, u]_{{.*}}"
# CHECK: kgen.conformance @"def[{{.*}}u: Int](vec: ToySIMD[dtype_tag, u]) -> ToyMask[dtype_tag, u]" {
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator
# CHECK: kgen.witness "dtype_tag" : !Int = {1}
def repro_capturing():
    var capture = 0

    def is_vec_a_capturing[
        u: Int
    ](vec: ToySIMD[1, u]) unified {var capture} -> ToyMask[1, u]:
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
        F: def[n: Int](item: Box[Self.E, n]) unified -> Box[Self.E, n]
    ](self, func: F) -> Int:
        return 0


# CHECK: lit.struct.decl @"def[n: Int](item: Box[ConcreteElem, n]) -> Box[ConcreteElem, n]_{{.*}}"
# CHECK: kgen.conformance @"def[{{.*}}n: Int](item: Box[E, n]) -> Box[E, n]" {
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator
# CHECK: kgen.witness "E" : !ElemLike = !ConcreteElem
def repro_nested_type_param():
    var capture = 0

    def apply_concrete[
        n: Int
    ](item: Box[ConcreteElem, n]) unified {var capture} -> Box[ConcreteElem, n]:
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
    func: def[width: Int, rank: Int, alignment: Int = 1]() unified -> None,
    //,
    simd_width: Int,
](shape: Int, ctx: Int, closure: func):
    closure[simd_width, 2]()


# CHECK: lit.struct.decl @"def[simd_width: Int, rank: Int, alignment: Int]() -> None_{{.*}}"
# CHECK: kgen.conformance @"def[width: Int, rank: Int, alignment: Int]() -> None" {
# CHECK:   kgen.witness "__call__{{.*}}" : !lit.generator
def main() raises:
    var x = 42

    @always_inline
    def my_func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ]() unified {read x}:
        print(x)

    callee[simd_width=4](10, 11, my_func)


# // -----

# COM: Verify the result is properly rebound in the struct wrapper when a closure
# COM: lazily conforms to a trait whose return type contains an alias parameter.


@fieldwise_init
struct V[dtype: Int, width: Int](RegisterPassable):
    var _v: Int


# CHECK: lit.struct.decl @"def[width: Int]() -> V[42, width]_PtrWrapper"

# CHECK: lit.fn @"__call__$def{{.*}} register_passable -> V{{.*}}"
# CHECK: kgen.rebind %{{.*}} : {{.*}}{42}{{.*}} to {{.*}}_dtype{{.*}}
# CHECK-NEXT: lit.return


# CHECK: kgen.conformance @"def[dtype: Int, #, width: Int]() register_passable -> V[dtype, width]" {
# CHECK-NEXT: kgen.witness "__call__{{.*}}" : !lit.generator
# CHECK-NEXT: kgen.witness "dtype" :{{.*}} = {42}
def callee[
    dtype: Int,
    F: def[width: Int]() unified register_passable -> V[dtype, width],
](closure: F):
    var result = closure[4]()


def rebindResult():
    def my_closure[width: Int]() unified register_passable {} -> V[42, width]:
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
    map_fn: def(ToyIndex[rank]) unified -> Tuple[
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

    def my_map_fn(
        point: ToyIndex[2],
    ) unified {read x} -> Tuple[ToyIndex[2], ToyIndex[2]]:
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
    F: def() unified -> Container[Pair(tag, 0)],
](closure: F):
    var result = closure()


# CHECK: lit.struct.decl @"def() -> Container[Pair(2, 0)]_{{.*}}"
# CHECK: kgen.conformance @"def[tag: Int, #]() -> Container[Pair(tag, 0)]" {
# CHECK:   kgen.witness "__call__{{.*}}" : !lit.generator
# CHECK:   kgen.witness "tag" : !Int = {2}
def repro_struct_attr():
    var x = 10

    def my_fn() unified {read x} -> Container[Pair(2, 0)]:
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
    C: def() unified -> Dispatch[identity],
](closure: C):
    var result = closure()


# CHECK: lit.struct.decl @"def() -> Dispatch[identity]_{{.*}}"
# CHECK: kgen.conformance @"def() -> Dispatch[identity]" {
# CHECK:   kgen.witness "__call__{{.*}}" : !lit.generator
def repro_symbol_attr():
    var x = 10

    def my_fn() unified {read x} -> Dispatch[identity]:
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
    F: def[w: Width](v: Vec[tag, w]) unified -> Bool,
](func: F):
    # CHECK: lit.fn @"__call__[::Int,::Int](unified_closure::def[tag: Int, #, w: Int](val: Vec[tag, w]) -> Bool_{{.*}}%val: !lit.struct<#Vec <:!Int _tag,
    # CHECK: [[REBIND:%.*]] = kgen.rebind %val : !lit.struct<#Vec <:!Int _tag
    # CHECK-SAME: to !lit.struct<#Vec <:!Int #kgen.get_witness<:!{{.*}} impl, "def[tag: Int, #, w: Int](val: Vec[tag, w]) -> Bool", "tag">
    # CHECK: lit.call[{{.*}}"val": !lit.struct<#Vec <:!Int #kgen.get_witness<:!{{.*}} impl, "def[tag: Int, #, w: Int](val: Vec[tag, w]) -> Bool", "tag">
    # CHECK-SAME: ]{{.*}}(%{{.*}}, [[REBIND]])
    def body[w: Int](val: Vec[tag, w]) unified {read func} -> Bool:
        return func[w=w](val)

    _ = body


# // -----

# COM: Verify lazy conformance emission for captured param expression closures.

# COM: Verify nested closure conformance for 2-arg closure (emitted before 1-arg).
# CHECK: kgen.conformance @"def[T: Int, U: Int, #](x: Container[T], y: Container[U]) -> None" {
# CHECK: kgen.witness "T" : !Int = #kgen.get_witness<:!{{.*}} impl, "def[{{.*}}](x: Container[{{.*}}], y: Container[{{.*}}]) -> None", "{{.*}}">
# CHECK: kgen.witness "U" : !Int = #kgen.get_witness<:!{{.*}} impl, "def[{{.*}}](x: Container[{{.*}}], y: Container[{{.*}}]) -> None", "{{.*}}">

# CHECK: kgen.conformance @"def[T: Int, #](x: Container[T]) -> None" {
# CHECK: kgen.witness "T" : !Int = #kgen.get_witness<:!{{.*}} impl, "def[{{.*}}](x: Container[{{.*}}]) -> None", "{{.*}}">


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


def takes[T: Int, F: def(x: Container[T]) unified](impl: F):
    impl(Container[T]())


def takes2[
    T: Int, U: Int, F: def(x: Container[T], y: Container[U]) unified
](impl: F):
    impl(Container[T](), Container[U]())


def takes_w[T: Int, F: def(w: Container[T]) unified](impl: F):
    impl(Container[T]())


# CHECK-LABEL: lit.fn @"defines[
def defines[T: Coord](foo: HasParam[T]):
    # CHECK: kgen.param.declare E1: !Int
    # CHECK-NOT: kgen.param.declare E2
    comptime S = foo.P

    def closure(x: Container[S]) unified {var}:
        pass

    takes[S, type_of(closure)](closure)


# CHECK-LABEL: lit.fn @"defines_nested[
def defines_nested[T: Coord, U: Coord](foo: HasParam[T], bar: HasParam[U]):
    comptime S = foo.P

    def closure(w: Container[S]) unified {var}:
        comptime Q = bar.P

        def closure2(x: Container[Q], y: Container[S]) unified {var}:
            pass

        takes2[Q, S, type_of(closure2)](closure2)

    takes_w[S, type_of(closure)](closure)


# // -----

# COM: Parameter Expressions are Outlined properly


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


@fieldwise_init
struct HasParam[T: Coord](ImplicitlyCopyable):
    comptime P = Self.T.Dim
    var x: Self.T


@fieldwise_init
struct Container[N: Int]:
    pass


def takes[
    T: Int, F: def[R: Coord](x: Container[T], r: HasParam[R]) unified
](impl: F):
    impl[Cartesian](Container[T](), HasParam[Cartesian](Cartesian(1, 2)))


# CHECK-LABEL: lit.fn @"foo
# CHECK-NEXT: kgen.param.declare E1
def foo[T: Coord](foo: HasParam[T]):
    comptime S = foo.P

    # CHECK: lit.closure.init
    # CHECK-NEXT: kgen.param.declare E2
    def closure[R: Coord](x: Container[S], r: HasParam[R]) unified {var}:
        def closure2(x: Container[S]) unified {var}:
            pass

        comptime SS = r.P

        def closure3[
            R3: Coord
        ](x: Container[SS], r3: HasParam[R3]) unified {var}:
            pass

        takes[SS, type_of(closure3)](closure3)

    # CHECK-NOT: kgen.param.declare E3
    def closure4[R4: Coord](x: Container[S], r4: HasParam[R4]) unified {var}:
        pass

    takes[S, type_of(closure)](closure)
    takes[S, type_of(closure4)](closure4)


# // -----

# COM: ParamOperatorAttr expressions are lifted.


@fieldwise_init
struct Container[N: Int]:
    pass


def takes_w[T: Int, F: def(w: Container[T]) unified](impl: F):
    impl(Container[T]())


# CHECK-LABEL: lit.fn @"defines_expression[
def defines_expression[X: Int, Y: Int]():
    # CHECK: kgen.param.declare E1: !Int
    # CHECK-NOT: kgen.param.declare E1
    def closure(ww: Container[X + Y]) unified {var}:
        pass

    takes_w[X + Y, type_of(closure)](closure)


# // -----

# COM: When the whole expression can be hoisted, do not emit child hoists.


@fieldwise_init
struct Container[N: Int]:
    pass


def takes_w[
    F: def[X: Int, Y: Int](w: Container[(X + Y) + (X + Y)]) unified
](impl: F):
    pass


# CHECK-LABEL: lit.fn @"no_hoist
def no_hoist():
    # CHECK-NOT: kgen.param.declare E1
    def closure[X: Int, Y: Int](ww: Container[(X + Y) + (X + Y)]) unified {var}:
        pass

    takes_w[type_of(closure)](closure)


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
    def thing() unified {var}:
        _ = foo


# // -----

# COM: Overload resolution with a closure overload must not crash when the
# COM: non-closure argument's struct is not yet body-resolved.


@always_inline
def dispatch[
    FuncType: def() unified register_passable -> None, //
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
        func: def(x: T) unified -> U,
    ](item: T, closure: func,) -> U:
        return closure(item)

    def double(x: Int) unified {mut} -> Int:
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


# CHECK-DAG: !lit.trait<@"def[T: DoB, #](y: T) -> None">
# CHECK-DAG: !lit.trait<@"def[T: DoA](y: T) -> None">
# CHECK-DAG: !lit.trait<@"def[T: DoA, #](y: T) -> None">
def foo[T: DoA](x: T):
    def closure(y: T) unified {var}:
        pass

    def closure2[T: DoA](y: T) unified {var}:
        pass


def bar[T: DoB](x: T):
    def closure(y: T) unified {var}:
        pass


# // -----

# COM: Verify that @__llvm_metadata on a unified closure is preserved on the op

# CHECK: LLVMMetadataArray = ["nvvm.maxntid", #pop.array<256> : !pop.array<1, i32>]


def metadata_closure(x: Int):
    @__llvm_metadata(
        `nvvm.maxntid`=__mlir_attr.`#pop.array<256> : !pop.array<1, i32>`
    )
    def _kernel() unified register_passable {var x} -> Int:
        return x

    _ = _kernel()


# // -----

# COM: Verify that @__llvm_arg_metadata on a unified closure is preserved

# CHECK: LLVMArgMetadataArray
# CHECK-SAME: "nvvm.grid_constant", unit


def arg_metadata_closure(x: Int):
    @__llvm_arg_metadata(x, `nvvm.grid_constant`)
    def _kernel(x: Int) unified register_passable {var} -> Int:
        return x

    _ = _kernel(x)


# // -----

# COM: Verify that a register_passable closure capturing a generic
# COM: register_passable closure and a concrete register_passable struct gets
# COM: convention register_passable (not trivial)

# CHECK: lit.struct.decl @"def(y: Int) register_passable -> Int_{{.*}}"{{.*}} register_passable attributes


struct NonTrivialPayload(ImplicitlyCopyable, RegisterPassable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value


def call_inner[
    F: ImplicitlyCopyable & def(Int) unified register_passable -> Int
](f: F, x: Int) -> Int:
    var payload = NonTrivialPayload(1)

    def outer(y: Int) unified register_passable {var f, var payload} -> Int:
        return f(y) + payload.value

    return outer(x)


# // -----

# COM: Verify that a register_passable closure capturing a trivially
# COM: register_passable callback and a trivial struct gets convention
# COM: register_passable_trivial.

# CHECK: lit.struct.decl @"def(y: Int) register_passable -> Int_{{.*}}"{{.*}} register_passable_trivial attributes


struct TrivialPayload(TrivialRegisterPassable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value


def call_inner[
    F: TrivialRegisterPassable & def(Int) unified register_passable -> Int
](f: F, x: Int) -> Int:
    var payload = TrivialPayload(1)

    def outer(y: Int) unified register_passable {var f, var payload} -> Int:
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
    def my_fn() unified {mut a, b, c^, read}:
        pass


# // -----
#
# COM: Verify stateless promoted closures are registered for apply attributes.


def trigger_dtype():
    comptime k = 64

    def nonsense(n: Int) unified {} -> DType:
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
