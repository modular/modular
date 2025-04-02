# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# LIT dialect asm aliases for trait composition.
# CHECK-DAG: !Trait1 = !lit.trait<@trait_composition::@Trait1>
# CHECK-DAG: !Trait2 = !lit.trait<@trait_composition::@Trait2>
# CHECK-DAG: !Trait3 = !lit.trait<@trait_composition::@Trait3>
# CHECK-DAG: !Trait1_Trait2 = !lit.trait<@trait_composition::@Trait1, @trait_composition::@Trait2>
# CHECK-DAG: !Trait1_Trait2_Trait3 = !lit.trait<@trait_composition::@Trait1, @trait_composition::@Trait2, @trait_composition::@Trait3>

trait Trait1:
    fn f1(self):
        ...

trait Trait2:
    fn f2(self):
        ...

trait Trait3:
    fn f3(self):
        ...

alias Traits12 = Trait1 & Trait2
alias Traits123 = Trait1 & Trait2 & Trait3

@value
struct Struct123(Trait1, Trait2):
    fn f1(self):
        pass

    fn f2(self):
        pass

    fn f3(self):
        pass

fn useAny[T: AnyType](x: T):
    pass

fn use1[T: Trait1](x: T):
    pass

# Use aliased trait composition.
fn use12[T: Traits12](x: T):
    pass

# Use direct trait composition.
fn use23[T: Trait2 & Trait3](x: T):
    pass

fn use123[T: Traits123](x: T):
    pass

# CHECK: lit.fn @"main_use()"
fn main_use():
    s123 = Struct123()
    # CHECK: lit.call @trait_composition::@"useAny
    # CHECK-SAME: <:!AnyType {{.*}}"__del__"
    useAny(s123)
    # CHECK: lit.call @trait_composition::@"use1
    # CHECK-SAME: <:!Trait1 {{.*}}"f1"{{.*}}"__del__"
    use1(s123)
    # CHECK: lit.call @trait_composition::@"use12
    # CHECK-SAME: <:!Trait1_Trait2 {{.*}}"f1"{{.*}}"__del__"{{.*}}"f2"
    use12(s123)
    # CHECK: lit.call @trait_composition::@"use23
    # CHECK-SAME: <:!Trait2_Trait3 {{.*}}"f2"{{.*}}"__del__"{{.*}}"f3"
    use23(s123)
    # CHECK: lit.call @trait_composition::@"use123
    # CHECK-SAME: <:!Trait1_Trait2_Trait3 {{.*}}"f1"{{.*}}"__del__"{{.*}}"f2"{{.*}}"f3"
    use123(s123)
