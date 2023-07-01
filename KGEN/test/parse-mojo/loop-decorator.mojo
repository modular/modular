# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s

from IO import print
from Range import range

# CHECK-LABEL: lit.func @"unroll_for()"
fn unroll_for():
    @unroll
    for i in range(1, 9, 2):
        print(i)
        @unroll
        for j in range (1, 4):
            print (i + j)
    # CHECK: } {unrollFactor = #hlcf<loop_unroll_full full>}
    # CHECK: } {unrollFactor = #hlcf<loop_unroll_full full>}
