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

trait SomeTrait: pass

struct SomeMem(SomeTrait):
  fn __copyinit__(inout self, existing: Self):
    pass

@register_passable
struct SomeReg(SomeTrait):
  fn __init__(inout self): pass

# ===----------------------------------------------------------------------=== #
# Trait packs
# ===----------------------------------------------------------------------=== #

# This function takes a pack of owned values by Trait.
fn takeOwnedAnyTypePack[*Ts: AnyType](owned *rest: *Ts):
  pass

# Test mangling:
# CHECK-LABEL: lit.func @"takeOwnedAnyTypePack[*stdlib::builtin::stubs::AnyType](*$0)"

# Test implicit lifetimes / param list.
# CHECK-SAME: [mut *"rest`"]<Ts: variadic<!AnyType> var>

# Check the argument pack.
# CHECK-SAME: (%rest: !lit.declref<#VariadicPack <:i1 1, :lifetime<1> *"rest`",
# CHECK-SAME: :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> Ts>> owned_in_mem|pack)

# Check the argument pack.
# CHECK-LABEL: lit.func @"takeOwnedSomeTraitPack
# CHECK-SAME: (%rest: !lit.declref<#VariadicPack <:i1 1, :lifetime<1> *"rest`",
# CHECK-SAME: :!lit.anytrait<!AnyType> !SomeTrait, :variadic<!SomeTrait> Ts>> owned_in_mem|pack)
fn takeOwnedSomeTraitPack[*Ts: SomeTrait](owned *rest: *Ts):
  pass


# CHECK-LABEL: lit.func @"test_owned_trait
fn test_owned_trait():
    # CHECK-NEXT: %value1 = lit.var.decl
    var value1: SomeMem
    # CHECK-NEXT: %value2 = lit.var.decl
    var value2: SomeMem

    # Argument expressions emitted first
    # CHECK-NEXT: [[V1T:%.*]] = lit.transfer_mem_ownership %value
    # CHECK-NEXT: [[ANONSLOT:%.*]] = lit.var.decl "anonymous
    # CHECK-NEXT: [[V2I:%.*]] = lit.ref.immut %value2
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[ANONSLOT]], [[V2I]])

    # Coerce to common lifetime
    # CHECK-NEXT: [[V1C:%.*]] = kgen.rebind [[V1T]] : !lit.ref<!SomeMem, mut *"value1(transfer)`2"> to !lit.ref<!SomeMem, mut {*"anonymous*`3", *"value1(transfer)`2"}>
    # CHECK-NEXT: [[V2C:%.*]] = kgen.rebind [[ANONSLOT]] : !lit.ref<!SomeMem, mut *"anonymous*`3"> to !lit.ref<!SomeMem, mut {*"anonymous*`3", *"value1(transfer)`2"}>

    # Form pack and call
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create([[V1C]], [[V2C]])

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKTMP:%.*]] = lit.var.decl
    # CHECK-NEXT: [[ISOWNED:%.*]] = kgen.param.constant: !Bool = <{:i1 1}>
    # CHECK-NEXT: lit.call @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACKTMP]], [[PACK]], [[ISOWNED]])
    # CHECK-NEXT: [[VARIADICPACK:%.*]] = lit.ref.load [[PACKTMP]]

    # CHECK-NEXT: lit.call {{.*}}takeOwnedAnyTypePack{{.*}}([[VARIADICPACK]])
    takeOwnedAnyTypePack(value1^, value2)

    # Test register types.
    # CHECK-NEXT: %value3 = lit.var.decl
    var value3: SomeReg

    # Argument expressions emitted first
    # CHECK-NEXT: [[V3T:%.*]] = lit.transfer_mem_ownership %value3
    # CHECK-NEXT: [[ANONSLOT:%.*]] = lit.var.decl "anonymous
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[ANONSLOT]])
    # Coerce to common lifetime
    # CHECK-NEXT: [[V3C:%.*]] = kgen.rebind [[V3T]]
    # CHECK-NEXT: [[V4C:%.*]] = kgen.rebind [[ANONSLOT]]
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create([[V3C]], [[V4C]])

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKTMP:%.*]] = lit.var.decl
    # CHECK-NEXT: [[ISOWNED:%.*]] = kgen.param.constant: !Bool = <{:i1 1}>
    # CHECK-NEXT: lit.call @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACKTMP]], [[PACK]], [[ISOWNED]])
    # CHECK-NEXT: [[VARIADICPACK:%.*]] = lit.ref.load [[PACKTMP]]

    # CHECK-NEXT: lit.call {{.*}}takeOwnedAnyTypePack{{.*}}([[VARIADICPACK]])
    takeOwnedAnyTypePack(value3^, SomeReg())



# Check the argument pack.
# CHECK-LABEL: lit.func @"takeInoutSomeTraitPack
# CHECK-SAME: (%rest: !lit.declref<#VariadicPack <:i1 1, :lifetime<1> *"rest`",
# CHECK-SAME: :!lit.anytrait<!AnyType> !SomeTrait, :variadic<!SomeTrait> Ts>> byref|pack)
fn takeInoutSomeTraitPack[*Ts: SomeTrait](inout *rest: *Ts):
  pass


# CHECK-LABEL: lit.func @"test_inout
fn test_inout():
    # CHECK-NEXT: %value1 = lit.var.decl
    var value1: SomeMem
    # CHECK-NEXT: %value2 = lit.var.decl
    var value2: SomeMem

    # Coerce to common lifetime
    # CHECK-NEXT: [[V1C:%.*]] = kgen.rebind %value1 : !lit.ref<!SomeMem, mut *"value1`"> to !lit.ref<!SomeMem, mut {*"value1`", *"value2`1"}>
    # CHECK-NEXT: [[V2C:%.*]] = kgen.rebind %value2 : !lit.ref<!SomeMem, mut *"value2`1"> to !lit.ref<!SomeMem, mut {*"value1`", *"value2`1"}>

    # Form pack and call
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create([[V1C]], [[V2C]])

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKTMP:%.*]] = lit.var.decl
    # CHECK-NEXT: [[ISOWNED:%.*]] = kgen.param.constant: !Bool = <{:i1 0}>
    # CHECK-NEXT: lit.call @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACKTMP]], [[PACK]], [[ISOWNED]])
    # CHECK-NEXT: [[VARIADICPACK:%.*]] = lit.ref.load [[PACKTMP]]

    # CHECK-NEXT: lit.call {{.*}}takeInoutSomeTraitPack{{.*}}([[VARIADICPACK]])
    takeInoutSomeTraitPack(value1, value2)

    # Test register types.
    # CHECK-NEXT: %value3 = lit.var.decl
    var value3: SomeReg

    # Coerce to common lifetime
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create(%value3)

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKTMP:%.*]] = lit.var.decl
    # CHECK-NEXT: [[ISOWNED:%.*]] = kgen.param.constant: !Bool = <{:i1 0}>
    # CHECK-NEXT: lit.call @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACKTMP]], [[PACK]], [[ISOWNED]])
    # CHECK-NEXT: [[VARIADICPACK:%.*]] = lit.ref.load [[PACKTMP]]

    # CHECK-NEXT: lit.call {{.*}}takeInoutSomeTraitPack{{.*}}([[VARIADICPACK]])
    takeInoutSomeTraitPack(value3)


struct not_nested_struct[*Ts: AnyType]:
    fn __init__(inout self, inout *args: *Ts):
        pass

# CHECK-LABEL: lit.func @"test_empty_pack
fn test_empty_pack():
  # Make sure we pass an immortal lifetime for the pack.
  # CHECK: lit.call {{.*}}VariadicPack::@"__init__{{.*}}:lifetime<1> #lit.lifetime,
  var s1 = not_nested_struct()
