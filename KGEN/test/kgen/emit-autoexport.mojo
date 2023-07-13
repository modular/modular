# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -emit-header | FileCheck %s

from SIMD import Float32
from IO import print


@export("bar", ABI="C")
# CHECK: extern float bar();
fn foo() -> Float32:
    # OK to alias, not proper main
    return 0.0
