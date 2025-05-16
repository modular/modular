# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not %mojo %s 2>&1 | FileCheck %s

# CHECK: 'get_linkage_name' function is not fully bound
# CHECK-SAME: param_fn{{.*}} missing 1 parameter binding(s)

from compile import compile_info


fn param_fn[x: Int, y: Int]() -> Int:
    return x + y


def main():
    # intentionally missing one parameter
    alias myInstantiatedFn = param_fn[2]
    var asm = compile_info[myInstantiatedFn]()
    print(asm)
