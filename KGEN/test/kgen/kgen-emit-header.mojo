# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -emit-header | FileCheck %s

from SIMD import Float32
from IO import print


@export
# CHECK: extern float call_me();
fn call_me() -> Float32:
    return 1.0


# CHECK: extern int32_t main(int32_t, void *);
fn main():
    _ = call_me()
