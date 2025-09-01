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
    fn fly_to(mut self, new_location: int):
        self.set_location(new_location)


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
