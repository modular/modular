# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo -split-input-file | FileCheck %s

struct Spaceship:
    var location: Index
    fn set_location(mut self, new_location: Index):
        self.location = new_location

# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    fn fly_to(mut self, new_location: Index):
        self.set_location(new_location)
