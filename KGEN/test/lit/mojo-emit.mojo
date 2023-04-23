# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s -emit-header | FileCheck %s

from SIMD import F32
from IO import print


@export
# CHECK: extern float call_me();
fn call_me() -> F32:
    return 1.0


# CHECK: extern void main();
fn main():
    _ = call_me()
