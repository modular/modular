# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not kgen -elaborate -elaboration-max-depth=128 %s 2>&1 | FileCheck %s --check-prefix=CHECK-MAX-DEPTH
# RUN: not mojo build -elaboration-max-depth=128 %s 2>&1 | FileCheck %s --check-prefix=CHECK-MAX-DEPTH

from collections.string.string_slice import _get_kgen_string
from sys import env_get_bool


# CHECK-MAX-DEPTH: elaborator expansion is {{[0-9]+}} levels deep - infinite recursion?
fn self_recursion[i: Int]() -> Int:
    var x = self_recursion[i + 1]()
    return x


fn main():
    _ = self_recursion[1]()
