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
    alias vec = Variant[Int](42)
    alias value = vec[Int]
    # CHECK-NEXT: constant = <42>
    return value
