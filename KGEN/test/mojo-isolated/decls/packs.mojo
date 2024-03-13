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

# Test mangling:
# CHECK-LABEL: lit.func @"takeOwnedTraitPack[*packs::SomeTrait](*$0)"

# Test implicit lifetimes / param list.
# CHECK-SAME: [mut *"rest`"]<Ts: variadic<!SomeTrait> var>

# Check the argument pack.
# CHECK-SAME: (%rest: !lit.ref.pack<:variadic<!SomeTrait> Ts, mut *"rest`"> owned_in_mem|pack)
fn takeOwnedTraitPack[*Ts: SomeTrait](owned *rest: *Ts):
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
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create(%2, %3)
    # CHECK-NEXT: lit.call {{.*}}takeOwnedTraitPack{{.*}}([[PACK]])
    takeOwnedTraitPack(value1^, value2)

    # Test register types.
    # CHECK-NEXT: %value3 = lit.var.decl
    var value3: SomeReg

    # Argument expressions emitted first
    # CHECK-NEXT: [[V3T:%.*]] = lit.transfer_mem_ownership %value3
    # CHECK-NEXT: [[ANONSLOT:%.*]] = lit.var.decl "anonymous
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%anonymous2A_0)
    # Coerce to common lifetime
    # CHECK-NEXT: [[V3C:%.*]] = kgen.rebind [[V3T]]
    # CHECK-NEXT: [[V4C:%.*]] = kgen.rebind [[ANONSLOT]]
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create([[V3C]], [[V4C]])
    # CHECK-NEXT: lit.call {{.*}}takeOwnedTraitPack{{.*}}([[PACK]])
    takeOwnedTraitPack(value3^, SomeReg())
