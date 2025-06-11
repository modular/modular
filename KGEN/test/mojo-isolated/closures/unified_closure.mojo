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

# CHECK: lit.struct.decl @"fn(y: Int) -> Int_wrapper"<impl: [[TRAIT]], |> attributes {isSynthetic} {
# CHECK:  lit.struct.field field0 : !kgen.param<:[[TRAIT]] impl>
# CHECK: }


# CHECK: lit.trait.decl @"fn(y: Int) -> Int"<?, *"_Self`": [[TRAIT]]>([[PARENT]])  unspecified attributes {dtorSig = !kgen.generator<!lit.generator<<[[TRAIT]], |>[1]("self": !lit.ref<:[[TRAIT]] *(0,0), mut *[0,0]> owned_in_mem, |) -> !kgen.none>>
# CHECK-NEXT:  lit.fn @"__call__({{.*}})"
# CHECK-SAME: [mut *"self`"](%{{.*}}: !lit.ref<:!Int *"_Self`", mut *"self`"> mut, |, %y: [[INT]]) -> [[INT]]
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
# CHECK-SAME: [mut *"self`", imm *"[[L1:.*]]`"](%0[*""]: !lit.ref<:[[TRAIT]] *"_Self`", mut *"self`"> mut, |, %a: !lit.ref<:!MyInterface T, imm *"[[L1]]`"> read_mem) -> !kgen.none


fn make_closure(x: Int) -> Int:
    fn parametric[T: MyInterface, b: T, c: Foo[T, b]](a: T) unified:
        pass

    return x
