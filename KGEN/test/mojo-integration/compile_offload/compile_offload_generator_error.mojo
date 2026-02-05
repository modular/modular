# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not %mojo %s 2>&1 | FileCheck %s

# CHECK: 'get_linkage_name' expected a valid generator reference, but got
# CHECK-SAME: param_fn{{.*}}

from compile import compile_info


fn param_fn[x: Int, y: Int]() -> Int:
    return x + y


fn my_wrapper[f: fn() -> Int]() -> fn() -> Int:
    return f


def main():
    # intentionally passing a function ptr instead of a generator
    print(compile_info[my_wrapper[param_fn[1, 2]]()]())
