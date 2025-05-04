# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Argument Packs.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait SomeTrait:
    pass


struct SomeMem(SomeTrait):
    fn __copyinit__(out self, existing: Self):
        pass


@register_passable
struct SomeReg(SomeTrait):
    fn __init__(out self):
        pass


# ===----------------------------------------------------------------------=== #
# Trait packs
# ===----------------------------------------------------------------------=== #


# This function takes a pack of owned values by Trait.
fn takeOwnedAnyTypePack[*Ts: AnyType](owned *rest: *Ts):
    pass


# Test mangling:
# CHECK-LABEL: lit.fn @"takeOwnedAnyTypePack[*::AnyType](*$0)"

# Test implicit lifetimes / param list.
# CHECK-SAME: <Ts: variadic<!AnyType> pos_vararg>[mut *"rest`"

# Check the argument pack.
# CHECK-SAME: (%rest: !lit.ref<{{.*}}@VariadicPack<:!Bool {:i1 1}, {{.*}}origin<1> = *"rest`"},
# CHECK-SAME: :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> Ts>, mut *"rest`1"> owned_in_mem|pack_vararg)


# Check the argument pack.
# CHECK-LABEL: lit.fn @"takeOwnedSomeTraitPack
# CHECK-SAME: (%rest: !lit.ref<{{.*}}@VariadicPack<:!Bool {:i1 1}, {{.*}}origin<1> = *"rest`"},
# CHECK-SAME: :!lit.anytrait<!AnyType> !SomeTrait, :variadic<!SomeTrait> Ts>, mut *"rest`1"> owned_in_mem|pack_vararg)
fn takeOwnedSomeTraitPack[*Ts: SomeTrait](owned *rest: *Ts):
    pass


# CHECK-LABEL: lit.fn @"test_owned_trait
fn test_owned_trait():
    # CHECK-NEXT: %value1 = lit.var.decl
    var value1: SomeMem
    # CHECK-NEXT: %value2 = lit.var.decl
    var value2: SomeMem

    # Argument expressions emitted first
    # CHECK-NEXT: lit.ownership.use %value1
    # CHECK-NEXT: [[ANONSLOT:%.*]] = lit.var.decl "anonymous
    # CHECK-NEXT: [[V2I:%.*]] = lit.ref.immut %value2
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[V2I]], [[ANONSLOT]])

    # Coerce to common origin
    # CHECK-NEXT: [[V1C:%.*]] = kgen.rebind %value1 : !lit.ref<!SomeMem, mut *"value1`"> to !lit.ref<!SomeMem, mut {*"anonymous*`2", *"value1`"}>
    # CHECK-NEXT: [[V2C:%.*]] = kgen.rebind [[ANONSLOT]] : !lit.ref<!SomeMem, mut *"anonymous*`2"> to !lit.ref<!SomeMem, mut {*"anonymous*`2", *"value1`"}

    # Form pack and call
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create([[V1C]], [[V2C]])

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKVAL:%.*]] = lit.call @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACK]])
    # CHECK-NEXT: [[PACKTMP:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[PACKVAL]], [[PACKTMP]]

    # CHECK-NEXT: lit.call {{.*}}takeOwnedAnyTypePack{{.*}}([[PACKTMP]])
    takeOwnedAnyTypePack(value1^, value2)

    # Test register types.
    # CHECK-NEXT: %value3 = lit.var.decl
    var value3: SomeReg

    # Argument expressions emitted first
    # CHECK-NEXT: lit.ownership.use %value3
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}SomeReg::@"__init__{{.*}}()
    # CHECK-NEXT: [[ANONSLOT:%.*]] = lit.var.decl "anonymous
    # CHECK-NEXT: lit.ref.store [[RES]], [[ANONSLOT]]

    # Coerce to common origin
    # CHECK-NEXT: [[V3C:%.*]] = kgen.rebind %value3
    # CHECK-NEXT: [[V4C:%.*]] = kgen.rebind [[ANONSLOT]]
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create([[V3C]], [[V4C]])

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKVAL:%.*]] = lit.call @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACK]])
    # CHECK-NEXT: [[PACKTMP:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[PACKVAL]], [[PACKTMP]]
    # CHECK-NEXT: lit.call {{.*}}takeOwnedAnyTypePack{{.*}}([[PACKTMP]])
    takeOwnedAnyTypePack(value3^, SomeReg())


# Check the argument pack.
# CHECK-LABEL: lit.fn @"takeInoutSomeTraitPack
# CHECK-SAME: (%rest: !lit.ref<{{.*}}@VariadicPack<:!Bool {:i1 1}, {{.*}}origin<1> = *"rest`"}
# CHECK-SAME: :!lit.anytrait<!AnyType> !SomeTrait, :variadic<!SomeTrait> Ts>, imm *"rest`1"> mut|pack_vararg)
fn takeInoutSomeTraitPack[*Ts: SomeTrait](mut *rest: *Ts):
    pass


# CHECK-LABEL: lit.fn @"test_inout
fn test_inout():
    # CHECK-NEXT: %value1 = lit.var.decl
    var value1: SomeMem
    # CHECK-NEXT: %value2 = lit.var.decl
    var value2: SomeMem

    # Coerce to common origin
    # CHECK-NEXT: [[V1C:%.*]] = kgen.rebind %value1 : !lit.ref<!SomeMem, mut *"value1`"> to !lit.ref<!SomeMem, mut {*"value1`", *"value2`1"}>
    # CHECK-NEXT: [[V2C:%.*]] = kgen.rebind %value2 : !lit.ref<!SomeMem, mut *"value2`1"> to !lit.ref<!SomeMem, mut {*"value1`", *"value2`1"}>

    # Form pack and call
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create([[V1C]], [[V2C]])

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKVAL:%.*]] = lit.call @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACK]])
    # CHECK-NEXT: [[VARIADICPACK:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[PACKVAL]], [[VARIADICPACK]]

    # CHECK-NEXT: [[PACKIMM:%.*]] = lit.ref.immut [[VARIADICPACK]]
    # CHECK-NEXT: lit.call {{.*}}takeInoutSomeTraitPack{{.*}}([[PACKIMM]])
    takeInoutSomeTraitPack(value1, value2)

    # Test register types.
    # CHECK-NEXT: %value3 = lit.var.decl
    var value3: SomeReg

    # Coerce to common origin
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create(%value3)

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKVAL:%.*]] = lit.call @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACK]])
    # CHECK-NEXT: [[VARIADICPACK:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[PACKVAL]], [[VARIADICPACK]]
    # CHECK-NEXT: [[PACKIMM:%.*]] = lit.ref.immut [[VARIADICPACK]]

    # CHECK-NEXT: lit.call {{.*}}takeInoutSomeTraitPack{{.*}}([[PACKIMM]])
    takeInoutSomeTraitPack(value3)


struct not_nested_struct[*Ts: AnyType]:
    @implicit
    fn __init__(out self, mut *args: *Ts):
        pass


# CHECK-LABEL: lit.fn @"test_empty_pack
fn test_empty_pack():
    # Make sure we pass an immortal origin for the pack.
    # CHECK: lit.call {{.*}}VariadicPack::@"__init__{{.*}}origin<1> = {}},
    var s1 = not_nested_struct()


# ===----------------------------------------------------------------------=== #
# Other tests
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @MyTuple
# CHECK-SAME: <Ts: variadic<!AnyType> pos_vararg>
struct MyTuple[*Ts: AnyType]:
    @implicit
    fn __init__(out self, *args: *Ts):
        pass


# CHECK-LABEL: lit.fn @"pack
# CHECK-SAME: Ts: variadic<!AnyType> pos_vararg>
# CHECK-SAME: (%args: !lit.ref<{{.*}}@VariadicPack<:!Bool {:i1 0}, {{.*}}origin<0> = *"args`"}, :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> Ts>, imm *"args`1"> read_mem|pack_vararg)
fn pack[*Ts: AnyType](*args: *Ts):
    pass


# CHECK-LABEL: lit.fn @"packBorrowed[
# CHECK-SAME: Ts: variadic<!AnyType> pos_vararg>
# CHECK-SAME: (%args: !lit.ref<{{.*}}@VariadicPack<:!Bool {:i1 0}, {{.*}}origin<0> = *"args`"}, :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> Ts>, imm *"args`1"> read_mem|pack_vararg)
fn packBorrowed[*Ts: AnyType](*args: *Ts):
    pass


# Ensure that parameters can be bound correctly.
fn variadicParameter[*Ts: AnyTrivialRegType](x: Int):
    pass


# CHECK-LABEL: lit.fn @"usePacks
# CHECK-SAME: [[ARGX:%.*]]: !FloatDyn
# CHECK-SAME: [[ARGY:%.*]]: !Int
fn usePacks(x: FloatDyn, y: Int):
    # CHECK: lit.var.decl {{.*}} : !lit.ref<@packs::@MyTuple<:variadic<!AnyType> [#Int1]>
    var a: MyTuple[Int]
    # CHECK: lit.var.decl {{.*}} : !lit.ref<@packs::@MyTuple<:variadic<!AnyType> [#Int1, #FloatDyn1, #Int1]>
    var b: MyTuple[Int, FloatDyn, Int]
    # CHECK: lit.var.decl {{.*}} : !lit.ref<@packs::@MyTuple<:variadic<!AnyType> [#Int1]>
    var c = MyTuple[Int](1)
    # CHECK: lit.var.decl {{.*}} : !lit.ref<@packs::@MyTuple<:variadic<!AnyType> [#FloatDyn1, #type_value]>
    var d = MyTuple(3.14, Int(6).value)
    # CHECK: lit.var.decl {{.*}} : !lit.ref<@packs::@MyTuple<:variadic<!AnyType> []>
    var e = MyTuple()

    pack(Int(1).value)
    pack(Int(1).value, 3.14)
    pack()

    pack(Int(1).value, x, y)
    pack[Int, FloatDyn, Int](Int(1), x, y)

    packBorrowed(Int(1).value, x, y)

    # CHECK: lit.call {{.*}}variadicParameter{{.*}}<:variadic<type>  [!Int, !FloatDyn]>
    variadicParameter[Int, FloatDyn](1)
    # CHECK: lit.call {{.*}}variadicParameter{{.*}}<:variadic<type> []>
    variadicParameter(Int(2))


# CHECK-LABEL: test_comptime_call
fn test_comptime_call[a: Int]():
    # CHECK: lit.alias.decl *"foo`": none =
    # CHECK-SAME: <apply(:!lit.generator<[2](
    # CHECK-SAME: "args": !lit.ref<{{.*}}@VariadicPack<:!Bool {:i1 0}, :!Bool {:i1 0},{{.*}}origin<0> = {}}, :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1]>, imm {}> read_mem|pack_vararg)
    # CHECK-SAME: <store_to_mem(a)>))
    alias foo = pack(a)
