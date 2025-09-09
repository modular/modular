# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --mojo-disable-builtins -split-input-file | FileCheck %s

alias int = __mlir_type.index

struct Spaceship:
    var location: int
    fn set_location(mut self, new_location: int):
        self.location = new_location

# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: index
    fn fly_to(mut self: Spaceship, new_location: int):
        self.set_location(new_location)


# // -----


alias int = __mlir_type.index
alias `2` = __mlir_attr.`2 : index`

struct Spaceship:
    var location: int
    fn set_location(mut self, new_location: int):
        self.location = new_location

# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: index
    fn fly_to(mut self: Spaceship, new_location: int):
        self.set_location(new_location)

# CHECK-LABEL: lit.fn @"do_things
fn do_things(mut ship: Spaceship):
    # CHECK: lit.call {{.*}}@"fly_to
    # CHECK-SAME: "self": !lit.ref<!Spaceship, mut *[0,0]>
    ship.fly_to(`2`)

# // -----

# Tests that we handle multiple extensions, and they're uniquely named.

alias int = __mlir_type.index

struct Spaceship:
    var location: int
    fn set_location(mut self, new_location: int):
        self.location = new_location

# CHECK-LABEL: lit.struct.decl @Spaceship
# CHECK-NOT: lit.struct.decl @"extension:Spaceship"
# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: index
    fn fly_to(mut self: Spaceship, new_location: int):
        self.set_location(new_location)


# CHECK-LABEL: lit.extension.decl @"extension:Spaceship
# CHECK-NOT: lit.extension.decl @"extension:Spaceship"
# Note the quote at the end of this last line -------^
# This checks that we're naming this struct extension something different.
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    fn something_else(self: Spaceship):
        pass
