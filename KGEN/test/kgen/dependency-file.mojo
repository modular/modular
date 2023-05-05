# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -emit %s -o %t.o -d=%t.d
# RUN: cat %t.d | FileCheck %s

# CHECK: {{.*}}.o: {{.*}}Int.mojo


fn use_int(a: Int):
    return
