# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -elaborate -S -o - | FileCheck %s

from utils import Variant


# CHECK-LABEL: func export @param_refitem
@export
fn param_refitem() -> Int:
    comptime vec = Variant[Int](42)
    comptime value = vec[Int]
    # CHECK-NEXT: constant = <42>
    return value
