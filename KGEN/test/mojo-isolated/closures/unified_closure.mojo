# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s

# COM: Verify generated trait and struct structure.

# CHECK-DAG: [[PARENT:!.*]] = !lit.trait<@{{.*}}::@AnyType, @{{.*}}::@Movable, @{{.*}}::@UnknownDestructibility, @{{.*}}:@"fn(y: Int) -> Int">
# CHECK-DAG: [[TRAIT:!.*]] = !lit.trait<@unified_closure::@"fn(y: Int) -> Int">
# CHECK-DAG: [[INT:!.*]] = !lit.struct<@{{.*}}::@Int>

# CHECK: lit.struct.decl @"fn(y: Int) -> Int_wrapper"<impl: [[TRAIT]], |>([[TRAIT]]) attributes {isSynthetic} {
# CHECK:  lit.struct.field field0 : !kgen.param<:[[TRAIT]] impl>
# CHECK: lit.fn @"__call__({{.*}})"[mut *"[[L0:.*]]`"](%0[*""]: !lit.ref<@{{.*}}::@"fn(y: Int) -> Int_wrapper"<:[[TRAIT]] impl>, mut *"[[L0]]`"> read_mem, |, %y: !Int1) -> [[INT]]
# CHECK-NEXT:  [[CLOSURE:%.*]] = lit.ref.struct.ger %{{.*}}[field0]
# CHECK-NEXT:  [[RES:%.*]] = lit.call[!lit.generator<[1](!lit.ref<:[[TRAIT]] impl, mut *[0,0]> read_mem, |, "y": [[INT]]) -> !Int1>: get_vtable_entry(:[[TRAIT]] impl, "__call__")][mut *"[[L0]]`"->field0]([[CLOSURE]], %y)
# CHECK-NEXT:  lit.return [[RES]]
# CHECK-NEXT:  lit.end_fn
# CHECK-NEXT: }
# CHECK: lit.fn @"__del__({{.*}})"[mut *"[[L1:.*]]`"](%self: !lit.ref<@{{.*}}::@"fn(y: Int) -> Int_wrapper"<:[[TRAIT]] impl>, mut *"[[L1]]`"> owned_in_mem, |) -> !kgen.none
# CHECK: lit.ownership.mark_destroyed %self
# CHECK: lit.fn @"__moveinit__({{.*}})"[mut *"[[L2:.*]]`", mut *"[[L3:.*]]`"](%existing: !lit.ref<@{{.*}}::@"fn(y: Int) -> Int_wrapper"<:[[TRAIT]] impl>, mut *"[[L2]]`"> owned_in_mem, |, ?, %self: !lit.ref<@{{.*}}::@"fn(y: Int) -> Int_wrapper"<:[[TRAIT]] impl>, mut *"[[L3]]`"> byref_result) -> !kgen.none
# CHECK: lit.ownership.mark_destroyed %existing

# CHECK: lit.trait.decl @"fn(y: Int) -> Int"<?, *"_Self`": [[TRAIT]]>([[PARENT]])  unspecified attributes {definesClosure, dtorSig = !kgen.generator<!lit.generator<<[[TRAIT]], |>[1]("self": !lit.ref<:[[TRAIT]] *(0,0), mut *[0,0]> owned_in_mem, |) -> !kgen.none>>
# CHECK-NEXT:  lit.fn @"__call__({{.*}})"
# CHECK-SAME: [mut *"self`"](%{{.*}}: !lit.ref<:!Int *"_Self`", mut *"self`"> read_mem, |, %y: [[INT]]) -> [[INT]]
# CHECK-SAME: attributes {isSynthetic, sourceName = "__call__", specialFnKind = 0 : i8} {
# CHECK-NEXT: kgen.unreachable
# CHECK-NEXT: }
# CHECK-NEXT: lit.fn @"__del__($0)"
# CHECK-NEXT: kgen.unreachable
# CHECK-NEXT: }
# CHECK-NEXT: lit.fn @"__moveinit__($0)"
# CHECK-NEXT: kgen.unreachable
# CHECK-NEXT: }
# CHECK-NEXT: }


fn make_closure(x: Int):
    fn my_closure(y: Int) unified -> Int:
        return x + y


# // -----

# COM: Verify Nested unified closures are supported


# CHECK: lit.struct.decl @"fn(y: Int) -> Int_wrapper"
# CHECK: lit.trait.decl @"fn(y: Int) -> Int"
# CHECK: lit.struct.decl @"fn(z: Int) -> Int_wrapper"
# CHECK: lit.trait.decl @"fn(z: Int) -> Int"
fn make_closure(x: Int):
    fn my_closure(y: Int) unified -> Int:
        fn my_nested_closure(z: Int) unified -> Int:
            return x

        return x + y


# // -----

# COM: Ensure identical closure traits are reused


# CHECK-COUNT-1: lit.struct.decl @"fn(y: Int) -> Int_wrapper"
# CHECK-COUNT-1: lit.trait.decl @"fn(y: Int) -> Int"
fn make_closure(x: Int):
    fn my_closure(y: Int) unified -> Int:
        return y


fn make_identical_closure(x: Int):
    fn my_closure(y: Int) unified -> Int:
        return y


# // -----

# COM: Test that parametric functions in traits are handled correctly


trait MyInterface(Movable):
    fn thing(self):
        ...


struct Foo[T: Movable, b: T]:
    pass


# CHECK-DAG: [[TRAIT:!None.*]] = !lit.trait<@{{.*}}::@"fn[MyInterface, $0, Foo[$0, $1]](a: $0) -> None">
# CHECK: lit.trait.decl @"fn[MyInterface, $0, Foo[$0, $1]](a: $0) -> None"<?, *"_Self`": [[TRAIT]]>(!{{.*}}) unspecified attributes {{{.*}}} {
# CHECK: lit.fn @"__call__{{.*}}"<T: !MyInterface, b: !kgen.param<:!MyInterface T>, c: @{{.*}}::@Foo<:!Movable {{.*}}, :!kgen.param<:!MyInterface T> b>>
# CHECK-SAME: [mut *"self`", imm *"[[L1:.*]]`"](%0[*""]: !lit.ref<:[[TRAIT]] *"_Self`", mut *"self`"> read_mem, |, %a: !lit.ref<:!MyInterface T, imm *"[[L1]]`"> read_mem) -> !kgen.none


fn make_closure(x: Int) -> Int:
    fn parametric[T: MyInterface, b: T, c: Foo[T, b]](a: T) unified:
        pass

    return x


# // -----

# COM: Test that explicit origins are handled correctly alongside implicit origins.

# CHECK-DAG: [[TRAIT:!None.*]] = !lit.trait<@{{.*}}::@"fn[MutableOrigin](a: ref [$0] String, b: String) -> None">

# CHECK: lit.struct.decl @"fn[MutableOrigin](a: ref [$0] String, b: String) -> None_wrapper"<impl: [[TRAIT]], |>([[TRAIT]]) attributes {isSynthetic} {
# CHECK-NEXT: lit.struct.field field0 : !kgen.param<:[[TRAIT]] impl>
# CHECK-NEXT: lit.fn @"__call__{{.*}}"<lt: origin<1>>[mut *"[[L1:.*]]`", imm *"[[L2:.*]]`"](%0[*""]: !lit.ref<@{{.*}}::@"fn[MutableOrigin](a: ref [$0] String, b: String) -> None_wrapper"
# CHECK-SAME: <:[[TRAIT]] impl>, mut *"[[L1]]`"> read_mem, |, %a: !lit.ref<!String, mut lt>, %b: !lit.ref<!String, imm *"[[L2]]`"> read_mem) -> !kgen.none attributes {isSynthetic, sourceName = "__call__", specialFnKind = 0 : i8} {
# CHECK-NEXT: [[V1:%.*]] = lit.ref.struct.ger %0[field0] : <@{{.*}}::@"fn[MutableOrigin](a: ref [$0] String, b: String) -> None_wrapper"<:[[TRAIT]] impl>, mut *"[[L1]]`"> -> :[[TRAIT]] impl
# CHECK-NEXT: [[V2:%.*]] = lit.call[!lit.generator<[2](!lit.ref<:[[TRAIT]] impl, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, mut lt>, "b": !lit.ref<!String, imm *[0,1]> read_mem) -> !kgen.none>
# CHECK-SAME:: bind_params(:!lit.generator<<"lt": origin<1>>[2](!lit.ref<:[[TRAIT]] impl, mut *[0,0]> read_mem, |, "a": !lit.ref<!String, mut *(0,0)>, "b": !lit.ref<!String, imm *[0,1]> read_mem) -> !kgen.none
# CHECK-SAME:> get_vtable_entry(:[[TRAIT]] impl, "__call__"), lt)][mut *"[[L1]]`"->field0, imm *"[[L2]]`"]([[V1]], %a, %b)
# CHECK-NEXT: lit.return [[V2]] : !kgen.none
# CHECK-NEXT: lit.end_fn


fn make_closure(x: Int) -> Int:
    fn mutate[
        lt: MutableOrigin
    ](a: Pointer[String, lt]._mlir_type, b: String) unified:
        pass

    return x


# // -----

# COM: Verify that the constructor is assembled correctly


trait MyInterface:
    fn thing(self):
        ...


# CHECK-DAG: [[TRAIT:!None.*]] = !lit.trait<@{{.*}}::@"fn[MyInterface](a: $0) -> None">


# CHECK: lit.fn @"__init__($0)"[mut *"impl`", mut *"self`"](%impl: !lit.ref<:[[TRAIT]] impl, mut *"impl`"> owned_in_mem, |, ?, %self: !lit.ref<@{{.*}}::@"fn[MyInterface](a: $0) -> None_wrapper"<:[[TRAIT]] impl>, mut *"self`"> byref_result) -> !kgen.none attributes {isStatic, isSynthetic, sourceName = "__init__", specialFnKind = 2 : i8} {
# CHECK-NEXT: [[V0:%.*]] = lit.ref.struct.ger %self[field0] : <@{{.*}}::@"fn[MyInterface](a: $0) -> None_wrapper"<:[[TRAIT]] impl>, mut *"self`"> -> :[[TRAIT]] impl
# CHECK-NEXT: [[V1:%.*]] = lit.call[!lit.generator<[2]("existing": !lit.ref<:[[TRAIT]] impl, mut *[0,0]> owned_in_mem, |, ?, "self": !lit.ref<:[[TRAIT]] impl, mut *[0,1]> byref_result) -> !kgen.none>: get_vtable_entry(:[[TRAIT]] impl, "__moveinit__")][mut *"impl`", mut *"self`"->field0](%impl, [[V0]])
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: lit.return %none : !kgen.none
# CHECK-NEXT: lit.end_fn


fn make_closure(x: Int) -> Int:
    fn parametric[T: MyInterface](a: T) unified:
        pass

    return x


# // -----


# COM: Verify the closure instance is created correctly.


fn make_closure(x: Int):
    # CHECK: [[RAW_CLOSURE:%.*]] = lit.closure.init[{{.*}}](%x)(%arg0[y]: !Int1) -> !Int1 {
    # CHECK-NEXT: [[BODY_OP:%.*]] = lit.call @{{.*}}@Int::@"__add__{{.*}}"(%x, %arg0) : !lit.generator<("lhs": !Int1, "rhs": !Int1) -> !Int1>
    # CHECK-NEXT: lit.return [[BODY_OP]] : !Int1
    # CHECK-NEXT: lit.end_fn
    # CHECK-NEXT: } : (!Int1), !lit.ref<!kgen.closure<@{{.*}}::@"make_closure{{.*}}", "my_closure" nonescaping>, mut *"[[L0:.*]]">

    # CHECK-NEXT: lit.ownership.use [[RAW_CLOSURE]]
    # CHECK-NEXT: [[WRAPPER:%.*]] = lit.var.decl "my_closure" var : !lit.ref<@{{.*}}::@"fn(y: Int) -> Int_wrapper"<:!Int {{.*}}, mut *"[[L1:.*]]">
    # CHECK-NEXT: lit.call @{{.*}}::@"fn(y: Int) -> Int_wrapper"::@"__init__($0)"[mut *"[[L0]]", mut *"[[L1]]"]<:!Int {{.*}}>([[RAW_CLOSURE]], [[WRAPPER]]) : !lit.generator<[2]("impl": !lit.ref<!kgen.closure<@{{.*}}::@"make_closure{{.*}}", "my_closure" nonescaping>, mut *[0,0]> owned_in_mem, |, ?, "self": !lit.ref<@{{.*}}::@"fn(y: Int) -> Int_wrapper"<:!Int {{.*}}>, mut *[0,1]> byref_result) -> !kgen.none>

    fn my_closure(y: Int) unified -> Int:
        return x + y


# // -----

# COM: Verify that the vtable entry is generated correctly


trait MyInterface:
    fn thing(self):
        ...


# CHECK:, {"__call__" : !lit.generator<<"T": !MyInterface>[2](!lit.ref<!kgen.closure<@{{.*}}::@"make_closure(::Int)", "parametric" nonescaping>, mut *[0,0]> read_mem
# CHECK-SAME:, |, "a": !lit.ref<:!MyInterface *(0,0), imm *[0,1]> read_mem) -> !kgen.none> = #kgen.closure.symbol<@{{.*}}::@"make_closure(::Int)", "parametric", #kgen.closure_method<call>>
# CHECK-SAME:, "__del__" : !lit.generator<[1]("self": !lit.ref<!kgen.closure<@{{.*}}::@"make_closure(::Int)", "parametric" nonescaping>, mut *[0,0]> owned_in_mem, |) -> !kgen.none
# CHECK-SAME:> = #kgen.closure.symbol<@{{.*}}::@"make_closure(::Int)", "parametric", #kgen.closure_method<del>>
# CHECK-SAME:, "__moveinit__" : !lit.generator<[2]("existing": !lit.ref<!kgen.closure<@{{.*}}::@"make_closure(::Int)", "parametric" nonescaping>, mut *[0,0]> owned_in_mem
# CHECK-SAME:, |, ?, "self": !lit.ref<!kgen.closure<@{{.*}}::@"make_closure(::Int)", "parametric" nonescaping>, mut *[0,1]> byref_result) -> !kgen.none> = #kgen.closure.symbol<@{{.*}}::@"make_closure(::Int)", "parametric", #kgen.closure_method<move>>}> : !None


fn make_closure(x: Int) -> Int:
    fn parametric[T: MyInterface](a: T) unified:
        pass

    return x


# // -----

# COM: Check that the argument is augmented at the definition site.

# CHECK-DAG: [[TRAIT:!Int.*]] = !lit.trait<@{{.*}}::@"fn(y: Int) -> Int">


# CHECK: lit.fn @"take_closure{{.*}}"<closure1: [[TRAIT]]>[imm *"closure1`"](%closure1: !lit.ref<:!Int closure1, imm *"closure1`"> read_mem, |, %x: !Int1) -> !kgen.none
# CHECK-NEXT: %0 = lit.call[!lit.generator<[1](!lit.ref<:!Int closure1, mut *[0,0]> read_mem, |, "y": !Int1) -> !Int1>: #kgen.get_witness<#kgen.param.decl.ref<"closure1"> : !Int, "unified_closure::fn(y: Int) -> Int", "__call__">][imm *"closure1`"](%closure1, %x)
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: lit.return %none : !kgen.none
# CHECK-NEXT: lit.end_fn
fn take_closure[closure1: fn (y: Int) unified -> Int](x: Int):
    _ = closure1(x)


# // -----

# COM: Ensure the transformed parameters are propagated into the underlying closure trait.


# CHECK: lit.trait.decl @"fn[fn(y: Int) -> Int](closure2: $0, /, y: Int) -> Int"
# CHECK-NEXT: lit.fn @"__call__{{.*}}"<closure2: !Int1>
# CHECK-SAME: [mut *"self`", imm *"[[L0:.*]]`"]
# CHECK-SAME: (%0[*""]: !lit.ref<:!Int *"_Self`", mut *"self`"> read_mem
# CHECK-SAME:, %closure2: !lit.ref<:!Int1 closure2, imm *"[[L0]]`"> read_mem, |, %y: !Int2) -> !Int2
fn take_closure[closure1: fn (y: Int) unified -> Int](x: Int):
    fn nested[closure2: fn (y: Int) unified -> Int](y: Int) unified -> Int:
        return x


# // -----

# COM: ensure many closure parameters are handled.

# CHECK: lit.fn @"take_closures{{.*}})"
# CHECK-SAME: <closure1: !Int2, T: !Int1, closure2: !Int, U: !Int1>
# CHECK-SAME: [imm *"[[L0:.*]]`", imm *"[[L1:.*]]`"]
# CHECK-SAME: (%closure1: !lit.ref<:!Int2 closure1, imm *"[[L0]]`"> read_mem
# CHECK-SAME:, %closure2: !lit.ref<:!Int closure2, imm *"[[L1]]`"> read_mem, |, %x: !Int1) -> !kgen.none


fn take_closures[
    closure1: fn (y: Int) unified -> Int,
    T: Int,
    closure2: fn (y: Int, z: Int) unified -> Int,
    U: Int,
](x: Int):
    pass


# // -----

# COM: Unified Closure Parameters compose


# CHECK: lit.fn @"nested[{{.*}})"
# CHECK-SAME: <x: !Int, +>[imm *"x`"]
# CHECK-SAME: (%x: !lit.ref<:!Int x, imm *"x`"> read_mem, |) -> !kgen.none
fn nested[x: fn[y: fn (z: Int) unified -> Int] (u: Int) unified -> Int, //]():
    pass
