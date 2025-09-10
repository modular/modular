# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --mojo-disable-builtins -I %S/inputs %s -mlir-print-debuginfo -split-input-file -verify-diagnostics

# @expected-note @below {{conflicts with this previous declaration}}
from struct_package_for_conflict import PlainStruct

# @expected-error @below {{cannot define a struct here with name 'PlainStruct'}}
struct PlainStruct:
    pass

__extension PlainStruct:
   fn sparklebark(self: PlainStruct):
       pass
