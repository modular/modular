# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -elaborate -S -o - | FileCheck %s

from std.utils import Variant


# CHECK-LABEL: func export @param_refitem
@export
def param_refitem() abi("Mojo") -> Int:
    comptime vec = Variant[Int](42)
    comptime value = vec[Int]
    # CHECK-NEXT: constant: scalar<index> = <42>
    return value
