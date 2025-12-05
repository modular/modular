# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s


# COM: Verify generated trait and struct structure.

# CHECK-DAG: [[PARENT:!Int_AnyType_Movable_UnknownDestructibility.*]] = !lit.trait<@"fn(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Movable, @{{.*}}::@UnknownDestructibility>
# CHECK-DAG: [[IMPL_PARENT:!Int_Movable_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"fn(y: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>

# CHECK-DAG: [[TRAIT:!.*]] = !lit.trait<@"fn(y: Int) -> Int">
# CHECK-DAG: [[INT:!.*]] = !lit.struct<@{{.*}}::@Int>

# CHECK: lit.trait.decl @"fn(y: Int) -> Int"<?, *"_Self`{{.*}}": [[TRAIT]]>([[PARENT]])
# CHECK-SAME: unspecified attributes {closureSignature = {{.*}}, definesClosure, dtorWitness = #kgen.get_witness<:[[TRAIT]] *"_Self`{{.*}}", "{{.*}}::AnyType", "__del__{{.*}}"> : !kgen.generator<!lit.generator<<"_Self`0x": [[TRAIT]], |>[1]("self": !lit.ref<:[[TRAIT]] *(0,0), mut *[0,0]> deinit_mem, |) -> !kgen.none>>
# CHECK-NEXT:  lit.fn @"__call__({{.*}})"
# CHECK-SAME: [mut *"self`"](%{{.*}}: !lit.ref<:[[TRAIT]] *"_Self`{{.*}}", mut *"self`"> read_mem, |, %y: [[INT]]) capturing -> [[INT]]
# CHECK-SAME: attributes {sourceName = "__call__", specialFnKind = 0 : i8, synthetic} {
# CHECK-NEXT: kgen.unreachable
# CHECK-NEXT: }
# CHECK-NEXT: lit.fn @"__del__($0$)"
# CHECK-NEXT: kgen.unreachable
# CHECK-NEXT: }
# CHECK-NEXT: lit.fn @"__moveinit__($0$)"
# CHECK-NEXT: kgen.unreachable
# CHECK-NEXT: }
# CHECK-NEXT: }

# CHECK: lit.struct.decl @"fn(y: Int) -> Int_{{.*}}"<impl: [[IMPL_PARENT]], origin_set: origin.set, |>([[IMPL_PARENT]]) attributes {synthetic}
# CHECK-NEXT: destructor :
# CHECK-NEXT: move :
# CHECK-NEXT: copy :
# CHECK:  lit.struct.field field0 : !kgen.param<:[[IMPL_PARENT]] impl>
# CHECK: lit.fn @"__call__({{.*}})"[mut *"[[L0:.*]]`"](%0[*""]: !lit.ref<!lit.struct<[[T:#.*]] <:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L0]]`"> read_mem, |, %y: [[INT]]) capturing -> [[INT]]
# CHECK-NEXT:  [[CLOSURE:%.*]] = lit.ref.struct.ger %{{.*}}[field0]
# CHECK-NEXT:  [[RES:%.*]] = lit.call[!lit.generator<[1](!lit.ref<:[[IMPL_PARENT]] impl, mut *[0,0]> read_mem, |, "y": [[INT]]) capturing -> [[INT]]>: #kgen.get_witness<:[[IMPL_PARENT]] impl, "fn(y: Int) -> Int", "__call__{{.*}}">][mut *"[[L0]]`"->field0]([[CLOSURE]], %y)
# CHECK-NEXT:  lit.return [[RES]]
# CHECK-NEXT:  lit.end_fn
# CHECK-NEXT: }

# CHECK: lit.fn @"__moveinit__({{.*}})"[mut *"[[L2:.*]]`", mut *"[[L3:.*]]`"](%existing: !lit.ref<{{.*}}<:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L2]]`"> deinit_mem, |, ?, %self: !lit.ref<{{.*}} <:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L3]]`"> byref_result) -> !kgen.none
# CHECK: lit.ownership.mark_destroyed %existing

# CHECK: lit.fn @"__del__({{.*}})"[mut *"[[L1:.*]]`"](%self: !lit.ref<{{.*}}<:[[IMPL_PARENT]] impl, :origin.set origin_set>>, mut *"[[L1]]`"> deinit_mem, |) -> !kgen.none
# CHECK: lit.ownership.mark_destroyed %self


fn make_closure(x: Int):
    fn my_closure(y: Int) unified {var x} -> Int:
        return x + y


# // -----

# COM: Verify Nested unified closures are supported


# CHECK: lit.trait.decl @"fn(y: Int) -> Int"
# CHECK: lit.trait.decl @"fn(z: Int) -> Int"
# CHECK: lit.struct.decl @"fn(y: Int) -> Int_{{.*}}"
# CHECK: lit.struct.decl @"fn(z: Int) -> Int_{{.*}}"


fn make_closure(x: Int):
    fn my_closure(y: Int) unified {var x} -> Int:
        fn my_nested_closure(z: Int) unified {var x} -> Int:
            return x

        return x + y


# // -----

# COM: Ensure identical closure traits are reused

# CHECK-COUNT-1: lit.trait.decl @"fn(y: Int) -> Int"
# CHECK-COUNT-1: lit.struct.decl @"fn(y: Int) -> Int


fn make_closure(x: Int):
    fn my_closure(y: Int) unified {} -> Int:
        return y


fn make_identical_closure(x: Int):
    fn my_closure(y: Int) unified {} -> Int:
        return y


# // -----

# COM: Test that parametric functions in traits are handled correctly


trait MyInterface(Movable):
    fn thing(self):
        ...


struct Foo[T: Movable, b: T]:
    pass


# CHECK: [[TRAIT:!None.*]] = !lit.trait<@"fn[T: MyInterface, b: T, c: Foo[T, b]](a: T) -> None">
# CHECK: lit.trait.decl @"fn[T: MyInterface, b: T, c: Foo[T, b]](a: T) -> None"<?, *"_Self`{{.*}}": [[TRAIT]]>(!{{.*}}) unspecified attributes {{{.*}}} {
# CHECK: lit.fn @"__call__{{.*}}"<T: !MyInterface, b: !kgen.param<:!MyInterface T>, c: {{.*}}Foo <:!Movable {{.*}}, :!kgen.param<:!MyInterface T> b>>
# CHECK-SAME: [mut *"self`", imm *"[[L1:.*]]`"](%0[*""]: !lit.ref<:[[TRAIT]] *"_Self`{{.*}}", mut *"self`"> read_mem, |, %a: !lit.ref<:!MyInterface T, imm *"[[L1]]`"> read_mem) capturing -> !kgen.none


fn make_closure(x: Int) -> Int:
    fn parametric[T: MyInterface, b: T, c: Foo[T, b]](a: T) unified {}:
        pass

    return x


# // -----

# COM: Test that explicit origins are handled correctly alongside implicit origins.

# CHECK: [[TRAIT:!None.*]] = !lit.trait<@"fn[lt: MutOrigin](a: ref [lt] String, b: String) -> None", @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


# CHECK: lit.struct.decl @"fn[lt: MutOrigin](a: ref [lt] String, b: String) -> None_{{.*}}"
# CHECK-SAME: <impl: [[TRAIT]], origin_set: origin.set, |>([[TRAIT]]) attributes {synthetic}
# CHECK: lit.struct.field field0 : !kgen.param<:[[TRAIT]] impl>

# CHECK-NEXT: lit.fn @"__call__{{.*}}"<lt: !lit.struct<#Origin <:!Bool {:i1 1}>>>[mut *"[[L1:.*]]`", imm *"[[L2:.*]]`"](%0[*""]: !lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *"[[L1]]`"> read_mem, |, %a: !lit.ref<!String, {{.*}}>, %b: !lit.ref<!String, imm *"[[L2]]`"> read_mem)
# CHECK-NEXT: [[V1:%.*]] = lit.ref.struct.ger %0[field0] : <!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *"[[L1]]`"> -> :[[TRAIT]] impl
# CHECK-NEXT: [[V2:%.*]] = lit.call[!lit.generator<[2](!lit.ref<:[[TRAIT]] impl, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none>:
# CHECK-SAME: bind_params(:!lit.generator<<"lt": !lit.struct<#Origin <:!Bool {:i1 1}>>>[2](!lit.ref<:[[TRAIT]] impl, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none
# CHECK-SAME:> #kgen.get_witness<:[[TRAIT]] impl, "fn[lt: MutOrigin](a: ref [lt] String, b: String) -> None", "__call__{{.*}}">, :!lit.struct<#Origin <:!Bool {:i1 1}>> lt)][mut *"[[L1]]`"->field0, imm *"[[L2]]`"]([[V1]], %a, %b)
# CHECK-NEXT: lit.return [[V2]] : !kgen.none
# CHECK-NEXT: lit.end_fn


fn make_closure(x: Int) -> Int:
    fn mutate[
        lt: MutOrigin
    ](a: Pointer[String, lt]._mlir_type, b: String) unified {}:
        pass

    return x


# // -----

# COM: Verify that the constructor is assembled correctly


trait MyInterface:
    fn thing(self):
        ...


# CHECK: [[TRAIT:!None_Movable_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"fn[T: MyInterface](a: T) -> None", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


# CHECK: lit.fn @"__init__($0$)"[mut *"impl`", mut *"self`"](%impl: !lit.ref<:[[TRAIT]] impl, mut *"impl`"> owned_in_mem, |, ?, %self: !lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *"self`"> byref_result)
# CHECK-NEXT: [[V0:%.*]] = lit.ref.struct.ger %self[field0] : <!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *"self`"> -> :[[TRAIT]] impl
# CHECK-NEXT: [[V1:%.*]] = lit.call[!lit.generator<[2]("existing": !lit.ref<:[[TRAIT]] impl, mut *[0,0]> deinit_mem, |, ?, "self": !lit.ref<:[[TRAIT]] impl, mut *[0,1]> byref_result) -> !kgen.none>: #kgen.get_witness<:[[TRAIT]] impl, "{{.*}}::Movable", "__moveinit__{{.*}}">][mut *"impl`", mut *"self`"->field0](%impl, [[V0]])
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: lit.return %none : !kgen.none
# CHECK-NEXT: lit.end_fn


fn make_closure(x: Int) -> Int:
    fn parametric[T: MyInterface](a: T) unified {}:
        pass

    return x


# // -----


# COM: Verify the closure instance is created correctly.

# CHECK: [[INT:!Int.*]] = !lit.struct<@{{.*}}::@Int>
# CHECK: [[TRAIT:!Int_Movable_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"fn(y: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


fn make_closure(x: Int):
    # CHECK: [[RAW_CLOSURE:%.*]] = lit.closure.init[{{.*}}](%x)(%arg0[y]: [[INT]]) capturing -> [[INT]] {
    # CHECK-NEXT: [[BODY_OP:%.*]] = lit.call @{{.*}}@Int::@"__add__{{.*}}"(%x, %arg0) : !lit.generator<("lhs": [[INT]], "rhs": [[INT]]) -> [[INT]]>
    # CHECK-NEXT: lit.return [[BODY_OP]] : [[INT]]
    # CHECK-NEXT: lit.end_fn
    # CHECK-NEXT: } : ([[INT]]), !lit.ref<!kgen.closure<@{{.*}}::make_closure{{.*}}", "my_closure" nonescaping>, mut *"[[L0:.*]]">

    # CHECK-NEXT: lit.ownership.use [[RAW_CLOSURE]]
    # CHECK-NEXT: [[WRAPPER:%.*]] = lit.var.decl "my_closure" var : !lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT]] {{.*}}, :origin.set {}>>, mut *"[[L1:.*]]">
    # CHECK-NEXT: lit.call @{{.*}}::@"fn(y: Int) -> Int_{{.*}}"::@"__init__($0$)"[mut *"[[L0]]", mut *"[[L1]]"]<:[[TRAIT]] {{.*}}, :origin.set {}>([[RAW_CLOSURE]], [[WRAPPER]]) : !lit.generator<[2]("impl": !lit.ref<!kgen.closure<@{{.*}}::make_closure{{.*}}", "my_closure" nonescaping>, mut *[0,0]> owned_in_mem, |, ?, "self": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] {{.*}}, :origin.set {}>>, mut *[0,1]> byref_result) -> !kgen.none>

    fn my_closure(y: Int) unified {var x} -> Int:
        return x + y


# // -----

# COM: Check that the argument is augmented at the definition site.

# CHECK-DAG: [[TRAIT:!Int.*]] = !lit.trait<@"fn(y: Int) -> Int">


# CHECK: lit.fn @"take_closure{{.*}}"<f: [[TRAIT]]>[imm *"myFunc`"](%myFunc: !lit.ref<:[[TRAIT]] f, imm *"myFunc`"> read_mem, %x: !Int1) capturing -> !kgen.none
# CHECK-NEXT: %0 = lit.call[!lit.generator<[1](!lit.ref<:!Int f, mut *[0,0]> read_mem, |, "y": !Int1) capturing -> !Int1>: #kgen.get_witness<:!Int f, "fn(y: Int) -> Int", "__call__{{.*}}">][imm *"myFunc`"](%myFunc, %x)
# CHECK-NEXT: lit.ownership.use %0
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
fn take_closure[f: fn (y: Int) unified -> Int](myFunc: f, x: Int):
    _ = myFunc(x)


# // -----

# COM: Ensure the transformed parameters are propagated into the underlying closure trait.


# CHECK-DAG: [[TRAIT:!Int_AnyType_Movable_UnknownDestructibility.*]] = !lit.trait<@"fn[closure2: fn(y: Int) -> Int](impl: closure2, y: Int) capturing -> Int", @{{.*}}::@AnyType, @{{.*}}::@Movable, @{{.*}}::@UnknownDestructibility>
# CHECK-DAG: [[TRAIT2:!Int.*]] = !lit.trait<@"fn(y: Int) -> Int">
# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<@{{.*}}::@Int>
# CHECK-DAG: [[TRAIT3:!Int.*]] = !lit.trait<@"fn[closure2: fn(y: Int) -> Int](impl: closure2, y: Int) capturing -> Int">


# CHECK: lit.trait.decl @"fn[closure2: fn(y: Int) -> Int](impl: closure2, y: Int) capturing -> Int"
# CHECK-NEXT: lit.fn @"__call__{{.*}}"<closure2: [[TRAIT2]]>
# CHECK-SAME: [mut *"self`", imm *"[[L0:.*]]`"]
# CHECK-SAME: (%0[*""]: !lit.ref<:[[TRAIT3]] *"_Self`{{.*}}", mut *"self`"> read_mem, |
# CHECK-SAME:, %impl: !lit.ref<:[[TRAIT2]] closure2, imm *"[[L0]]`"> read_mem, %y: [[INT]]) capturing -> [[INT]]
fn take_closure[closure1: fn (y: Int) unified -> Int](x: Int):
    fn nested[
        closure2: fn (y: Int) unified -> Int
    ](impl: closure2, y: Int) unified {var x} -> Int:
        return x


# // -----

# COM: ensure many closure parameters are handled.

# CHECK: lit.fn @"take_closures{{.*}})"
# CHECK-SAME: <closure1: !Int2, T: !Int1, closure2: !Int, U: !Int1>
# CHECK-SAME: [imm *"[[L0:.*]]`", imm *"[[L1:.*]]`1"]
# CHECK-SAME: (%impl1: !lit.ref<:!Int2 closure1, imm *"[[L0]]`"> read_mem
# CHECK-SAME:, %impl2: !lit.ref<:!Int closure2, imm *"[[L1]]`1"> read_mem, %x: !Int1) capturing -> !kgen.none


fn take_closures[
    closure1: fn (y: Int) unified -> Int,
    T: Int,
    closure2: fn (y: Int, z: Int) unified -> Int,
    U: Int,
](impl1: closure1, impl2: closure2, x: Int):
    pass


# // -----

# COM: Unified Closure Parameters compose

# CHECK: [[INNER:!Int1.*]] = !lit.trait<@"fn(z: Int) -> Int">
# CHECK: lit.fn @"__call__{{.*}}"<y: [[INNER]]>


# CHECK: lit.fn @"nested[{{.*}})"
# CHECK-SAME: <x: !Int, +>[imm *"[[L0:.*]]"]
# CHECK-SAME: (%impl: !lit.ref<:!Int x, imm *"[[L0]]"> read_mem
# TODO: remove the 'do_not_dce_int' argument (MOCO 2461)
fn nested[
    x: fn[y: fn (z: Int) unified -> Int] (impl: y, u: Int) unified -> Int, //
](impl: x, do_not_dce_int: Int):
    pass


# // -----

# COM: Check that the struct generator of the lit op is generated correctly.


# CHECK: [[TRAIT:!Int_Movable_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"fn(z: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


# CHECK: kgen.struct.generator @"bindIt({{.*}})::myclosure"
# CHECK-SAME: <CAPTURES: !kgen.param_closure<@{{.*}}::bindIt(::Int,::Int)" "myclosure">>:
# CHECK-SAME: [[TRAIT]] = !kgen.closure<@{{.*}}::bindIt({{.*}})", "myclosure" nonescaping>{
# CHECK: kgen.conformance @"{{.*}}::AnyType" {
# CHECK-NEXT: kgen.witness "__del__{{.*}}"
# CHECK: kgen.conformance @"{{.*}}::Movable" {
# CHECK-NEXT: kgen.witness "__moveinit__{{.*}}"
# CHECK: kgen.conformance @"fn(z: Int) -> Int" {
# CHECK-NEXT: kgen.witness "__call__{{.*}}"
fn bindIt(x: Int, y: Int) -> Int:
    fn myclosure(z: Int) unified {var x, var y} -> Int:
        return x + y + z


# // -----

# COM: Check that parameters are emitted correctly


# CHECK: kgen.struct.generator @"bindIt()::myclosure"
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator<<"my_param": !AnyType>
# CHECK-SAME: [1](!lit.ref<!kgen.closure<@{{.*}}::bindIt()", "myclosure" nonescaping>, mut *[0,0]> read_mem, |, "z": !Int) capturing -> !kgen.none
# CHECK-SAME:> = #kgen.closure.symbol<@{{.*}}::bindIt()", "myclosure", #kgen.closure_method<call>
# CHECK-SAME:, <:!AnyType ?, :!kgen.param_closure<@{{.*}}::bindIt()" "myclosure"> CAPTURES>>


# CHECK: lit.file_module
fn bindIt() -> Int:
    fn myclosure[my_param: AnyType](z: Int) unified {}:
        pass


# // -----

# COM: Verify Conformance tables of the Wrapper are generated correctly

# CHECK: [[TRAIT:!None_Movable_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"fn[lt: MutOrigin](a: ref [lt] String, b: String) -> None", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


# CHECK: lit.struct.decl @"fn[lt: MutOrigin](a: ref [lt] String, b: String) -> None_{{.*}}"
# CHECK-SAME: <impl: [[TRAIT]], origin_set: origin.set, |>([[TRAIT]]) attributes {synthetic}


# CHECK: kgen.conformance @"fn[lt: MutOrigin](a: ref [lt] String, b: String) -> None" {
# CHECK-NEXT: kgen.witness "__call__{{.*}}" : !lit.generator<<"lt": !lit.struct<#Origin <:!Bool {:i1 1}>>>[2](!lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, {{.*}}>, "b": !lit.ref<!String, imm *[0,1]> read_mem) capturing -> !kgen.none
# CHECK-SAME: > = @{{.*}}::@"fn[lt: MutOrigin](a: ref [lt] String, b: String) -> None_{{.*}}"::@"__call__{{.*}}"<:[[TRAIT]] impl, :origin.set origin_set, :!lit.struct<#Origin <:!Bool {:i1 1}>> ?>

# CHECK: kgen.conformance @{{.*}}::Movable" {
# CHECK-NEXT: kgen.witness "__moveinit__{{.*}}" : !lit.generator<[2]("existing": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> deinit_mem, |, ?, "self": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,1]> byref_result) -> !kgen.none
# CHECK-SAME: > = @{{.*}}::@"fn[lt: MutOrigin](a: ref [lt] String, b: String) -> None_{{.*}}"::@"__moveinit__({{.*}})"<{{.*}}>

# CHECK: kgen.conformance @"{{.*}}::AnyType" {
# CHECK-NEXT:  kgen.witness "__del__{{.*}}" : !lit.generator<[1]("self": !lit.ref<!lit.struct<[[T]] <:[[TRAIT]] impl, :origin.set origin_set>>, mut *[0,0]> deinit_mem, |) -> !kgen.none
# CHECK-SAME: > = @{{.*}}::@"fn[lt: MutOrigin](a: ref [lt] String, b: String) -> None_{{.*}}"::@"__del__{{.*}}"<{{.*}}>


fn make_closure(x: Int) -> Int:
    fn mutate[
        lt: MutOrigin
    ](a: Pointer[String, lt]._mlir_type, b: String) unified {}:
        pass

    return x


# // -----

# COM: Check that the origin set is bound to the wrapper

# CHECK: [[TRAIT:!None_Movable_AnyType_Copyable_ImplicitlyCopyable*.]] = !lit.trait<@"fn() -> None", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>


fn nonemptyOriginSet(mut byRefMut: String):
    # CHECK: lit.call @{{.*}}::@"fn() -> None_{{.*}}"::@"__init__({{.*}})"[{{.*}}]<:[[TRAIT]] {{.*}}, :origin.set {mut *"byRefMut`"}>
    fn myclosure() unified {mut byRefMut}:
        pass


# // -----

# COM: Verify that closures can be rebound to compatible traits

# CHECK-DAG: [[TRAIT1:!Int_Movable_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"fn(x: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>
# CHECK-DAG: [[TRAIT:!Int_Movable_AnyType_Copyable_ImplicitlyCopyable_Int.*]] = !lit.trait<@"fn(x: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable,
# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<{{.*}}::@Int>

# CHECK: lit.struct.decl @"fn(x: Int) -> Int_{{.*}}"<impl: [[TRAIT1]], origin_set: origin.set, |>([[TRAIT]])
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator<[1](!lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, [[INT]], |) capturing -> [[INT]]> =
# CHECK-SAME: rebind(:!lit.generator<[1](!lit.ref<!lit.struct<[[T]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "x": [[INT]]) capturing -> [[INT]]> @{{.*}}::@"fn(x: Int) -> Int_{{.*}}"::@"__call__{{.*}}"<:[[TRAIT1]] impl, :origin.set origin_set>)


fn takeIt[C: fn (Int) unified -> Int](closure: C):
    _ = closure(3)


fn bindIt(z: Int):
    fn myclosure(x: Int) unified {var} -> Int:
        return z

    takeIt[type_of(myclosure)](myclosure)


# // -----

# COM: Verify that closures can be rebound even when traits are combined

# CHECK-DAG: [[TRAIT1:!Int_Movable_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"fn(x: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>
# CHECK-DAG: [[TRAIT:!Int_Movable_AnyType_Copyable_ImplicitlyCopyable_Int.*]] = !lit.trait<@"fn(x: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable,
# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<{{.*}}::@Int>

# CHECK: lit.struct.decl @"fn(x: Int) -> Int_{{.*}}"<impl: [[TRAIT1]], origin_set: origin.set, |>([[TRAIT]])
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator<[1](!lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "y": [[INT]]) capturing -> [[INT]]> =
# CHECK-SAME: rebind(:!lit.generator<[1](!lit.ref<!lit.struct<[[T]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "x": [[INT]]) capturing -> [[INT]]> @{{.*}}::@"fn(x: Int) -> Int_{{.*}}"::@"__call__{{.*}}"<:[[TRAIT1]] impl, :origin.set origin_set>)


fn takeIt[C: Copyable & fn (y: Int) unified -> Int](closure: C):
    _ = closure(3)


fn bindIt(z: Int):
    fn myclosure(x: Int) unified {var} -> Int:
        return z

    takeIt[type_of(myclosure)](myclosure)


# // -----

# COM: Verify that all closures are rebound when closure traits are combined or inherited


fn takeIt[C: (fn (Bool) unified -> Int) & fn (Int) unified -> Int](closure: C):
    _ = closure(3)


trait BoolWrapper(fn (Bool) unified -> Int):
    pass


# CHECK: lit.struct.decl @MultipleClosure

# CHECK: kgen.conformance @"fn(Bool) -> Int"
# CHECK: kgen.witness "__call__($0,::Bool)" : !lit.generator<[1](!lit.ref<!MultipleClosure, mut *[0,0]> read_mem, !Bool, |) capturing -> !Int1> = rebind(:!lit.generator<[1]("self": !lit.ref<!MultipleClosure, imm *[0,0]> read_mem, "x": !Bool) capturing -> !Int1> @{{.*}}::@MultipleClosure::@"__call__({{.*}}::MultipleClosure,::Bool)")


# CHECK: kgen.conformance @"fn(Int) -> Int"
# CHECK:kgen.witness "__call__($0,::Int)" : !lit.generator<[1](!lit.ref<!MultipleClosure, mut *[0,0]> read_mem, !Int1, |) capturing -> !Int1> = rebind(:!lit.generator<[1]("self": !lit.ref<!MultipleClosure, imm *[0,0]> read_mem, "x": !Int1) capturing -> !Int1> @{{.*}}::@MultipleClosure::@"__call__({{.*}}::MultipleClosure,::Int)")
struct MultipleClosure(BoolWrapper, Movable, fn (Int) unified -> Int):
    fn __init__(out self):
        pass

    fn __call__(self, x: Bool) -> Int:
        return 1

    fn __call__(self, x: Int) -> Int:
        return 2


fn bindIt(z: Int):
    var fakeclosure = MultipleClosure()

    takeIt[type_of(fakeclosure)](fakeclosure)


# // -----

# COM: Verify that closures can be rebound with differing parameter names

# CHECK-DAG: [[TRAIT1:!Int_Movable_AnyType_Copyable_ImplicitlyCopyable.*]] = !lit.trait<@"fn[a: Int](b: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable>
# CHECK-DAG: [[TRAIT:!Int_Movable_AnyType_Copyable_ImplicitlyCopyable_Int.*]] = !lit.trait<@"fn[a: Int](b: Int) -> Int", @{{.*}}::@Movable, @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@ImplicitlyCopyable,
# CHECK-DAG: [[INT:!Int.*]] = !lit.struct<{{.*}}::@Int>

# CHECK: lit.struct.decl @"fn[a: Int](b: Int) -> Int_{{.*}}"<impl: [[TRAIT1]], origin_set: origin.set, |>([[TRAIT]])
# CHECK: kgen.conformance @"fn[x: Int](y: Int) -> Int"
# CHECK: kgen.witness "__call__{{.*}}" : !lit.generator<<"x": [[INT]]>[1](!lit.ref<!lit.struct<[[T:#.*]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "y": [[INT]]) capturing -> [[INT]]> =
# CHECK-SAME: rebind(:!lit.generator<<"a": [[INT]]>[1](!lit.ref<!lit.struct<[[T]] <:[[TRAIT1]] impl, :origin.set origin_set>>, mut *[0,0]> read_mem, |, "b": [[INT]]) capturing -> [[INT]]> @{{.*}}::@"fn[a: Int](b: Int) -> Int_{{.*}}"::@"__call__[::Int]({{.*}}::fn[a: Int](b: Int) -> Int_{{.*}}"<:[[TRAIT1]] impl, :origin.set origin_set, :[[INT]] ?>)


fn takeIt[C: fn[x: Int] (y: Int) unified -> Int](closure: C):
    # see MOCO-2606
    _ = closure.__call__[2](3)


fn bindIt(z: Int):
    fn myclosure[a: Int](b: Int) unified {var} -> Int:
        return z

    takeIt[type_of(myclosure)](myclosure)


# // -----

# COM: Ensure that structs can conform to the closure trait


# CHECK: [[TRAIT:!Int_AnyType.*]] = !lit.trait<@"fn(x: Int) -> Int"
# CHECK: lit.struct.decl @custom([[TRAIT]])
struct custom(fn (x: Int) unified -> Int):
    fn __call__(self, x: Int) capturing -> Int:
        return x


# // -----

# COM: The wrapper conforms to copyable

# CHECK: [[CANONICAL_TRAIT:!Int_Movable_AnyType_Copyable_ImplicitlyCopyable*.]] = !lit.trait<
# CHECK-SAME: @"fn(x: Int) -> Int"
# CHECK-SAME:, @{{.*}}::@Movable
# CHECK-SAME:, @{{.*}}::@AnyType
# CHECK-SAME:, @{{.*}}::@Copyable
# CHECK-SAME:, @{{.*}}::@ImplicitlyCopyable>


# CHECK: lit.struct.decl @"fn(x: Int) -> Int_{{.*}}"
# CHECK: lit.struct.decl @"fn(x: Int) -> Int_{{.*}}"<impl: [[CANONICAL_TRAIT]], origin_set: origin.set, |>([[CANONICAL_TRAIT]])


fn takeItImplicit[T: ImplicitlyCopyable](impl: T):
    pass


fn takeIt[T: Copyable](impl: T):
    pass


@fieldwise_init
struct CopyMe(ImplicitlyCopyable):
    var x: Int
    var y: Int


@fieldwise_init
struct OneOfAKind(Movable):
    var x: Int
    var y: Int


fn useIt(var x: OneOfAKind):
    pass


@no_inline
fn giveIt(z: Int, cm: CopyMe, var one: OneOfAKind):
    fn aThing(x: Int) unified {var z, var cm} -> Int:
        return z + x

    takeItImplicit(aThing)
    takeIt(aThing)

    # COM: uncopyable version can still implement the fn(x:Int) -> Int trait
    fn anotherThing(x: Int) unified {var ^} -> Int:
        useIt(one^)
        return x


# // -----


# COM: Ensure result type index ref has been replaced


# CHECK: lit.struct.decl @"fn[T: {{.*}}, /]() -> T_{{.*}}"
# CHECK: lit.struct.field field0
# CHECK-NEXT:      lit.fn @"__call__
# CHECK-NEXT:        %1 = lit.ref.struct.ger
# CHECK-NEXT:        %2 = lit.call[!lit.generator<[1]({{.*}}) capturing -> !kgen.param<{{.*}}T)
fn makeIt[T: AnyTrivialRegType](a: T):
    fn parametric() unified {var a} -> T:
        return a


# // -----

# COM: Check that device passable conformance is emitted properly


fn conditionallyDevicePassable(x: Int):
    # CHECK: kgen.conformance @"{{.*}}::DevicePassable" {
    # CHECK-NEXT: kgen.witness "device_type" : type =
    # CHECK-NEXT: kgen.witness "_to_device_type{{.*}}" : !lit.generator
    # CHECK-NEXT: kgen.witness "get_type_name{{.*}}" : !lit.generator
    # CHECK-NEXT: kgen.witness "get_device_type_name{{.*}}" : !lit.generator
    fn device_passable() unified register_passable {var} -> Int:
        return x
