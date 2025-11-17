# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


fn main():
    comptime lst = [1, 2, 3]
    var dyn_lst = materialize[lst]()
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    for v in dyn_lst:
        print(v)
