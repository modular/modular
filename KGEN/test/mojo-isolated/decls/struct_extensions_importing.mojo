# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --mojo-disable-builtins -I %S/inputs %s -split-input-file | FileCheck %s


from struct_package import PlainStruct

# CHECK-LABEL: lit.extension.decl @"extension:PlainStruct"
# CHECK-SAME: targetStruct = @struct_package::@plain_struct::@PlainStruct
__extension PlainStruct:
    # CHECK-LABEL: lit.fn @"sparklebark
    # CHECK-SAME: %self: !lit.ref<!PlainStruct, imm *"{{.*}}">
   fn sparklebark(self: PlainStruct):
       pass


# // -----

# Make Sure We Gracefully Handle Redundant Imports (MSWGHRI)

from struct_package import PlainStruct
from struct_package import PlainStruct
from struct_package import PlainStruct

# CHECK-LABEL: lit.extension.decl @"extension:PlainStruct"
# CHECK-SAME: targetStruct = @struct_package::@plain_struct::@PlainStruct
__extension PlainStruct:
    # CHECK-LABEL: lit.fn @"sparklebark
    # CHECK-SAME: %self: !lit.ref<!PlainStruct, imm *"{{.*}}">
   fn sparklebark(self: PlainStruct):
       pass

# // -----

from struct_package import PlainStruct


alias int = __mlir_type.index
alias `2` = __mlir_attr.`2 : index`


trait Flying:
    fn fly_to(mut self, new_location: int):
        ...

# CHECK-LABEL: lit.extension.decl @"extension:PlainStruct"
# CHECK-SAME: immediateParents = #M<symbols[@struct_extensions_importing::@Flying]>
# CHECK-SAME: targetStruct = @struct_package::@plain_struct::@PlainStruct
__extension PlainStruct(Flying):
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!PlainStruct, mut *"{{.*}}">
    # CHECK-SAME: %new_location: index
    fn fly_to(mut self: PlainStruct, new_location: int):
        self.set_location(new_location)
    # CHECK: kgen.conformance @"struct_extensions_importing::Flying" {

fn launch_flying[F: Flying](mut flying: F):
    flying.fly_to(`2`)

# CHECK-LABEL: lit.fn @"launch_ship
fn launch_ship(mut ship: PlainStruct):
    # CHECK: lit.call @struct_extensions_importing::@"launch_flying[struct_extensions_importing::Flying]
    # CHECK-SAME: <:!Flying !PlainStruct>
    launch_flying(ship)

# // -----


from trait_package import Flying as ImportedFlying


alias int = __mlir_type.index
alias `2` = __mlir_attr.`2 : index`


struct Spaceship:
    var location: int

    fn __init__(out self):
        self.location = __mlir_attr.`0 : index`

    fn set_location(mut self, new_location: int):
        self.location = new_location

# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: immediateParents = #M<symbols[@trait_package::@plain_trait::@Flying]>
# CHECK-SAME: targetStruct = @struct_extensions_importing::@Spaceship
__extension Spaceship(ImportedFlying):
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: index
    fn fly_to(mut self: Spaceship, new_location: int):
        self.set_location(new_location)
    # CHECK: kgen.conformance @"trait_package::plain_trait::Flying" {

fn launch_flying2[F: ImportedFlying](mut flying: F):
    flying.fly_to(`2`)

# CHECK-LABEL: lit.fn @"launch_ship2
fn launch_ship2(mut ship: Spaceship):
    # CHECK: lit.call @struct_extensions_importing::@"launch_flying2[trait_package::plain_trait::Flying]
    # CHECK-SAME: <:!Flying !Spaceship>
    launch_flying2(ship)
