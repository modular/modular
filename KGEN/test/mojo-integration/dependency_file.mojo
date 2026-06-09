# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen --emit=object %s -o %t.o -d=%t.d
# RUN: cat %t.d | FileCheck %s

# CHECK: {{.*}}.o: {{.*}}std.mojoc


@export
def use_int(a: Int) abi("Mojo"):
    return
