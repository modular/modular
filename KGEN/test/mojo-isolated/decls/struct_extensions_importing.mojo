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
