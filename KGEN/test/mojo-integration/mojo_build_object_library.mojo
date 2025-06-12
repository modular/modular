# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir %t
# RUN: %mojo-build %s -o %t/output.o --emit object

# RUN: llvm-nm -U %t/output.o | FileCheck %s

# CHECK: T {{foo|_foo}}


@export
fn foo():
    pass
