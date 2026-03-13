# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not %mojo %s 2>&1 | FileCheck %s

# CHECK: failed to infer parameter 'y', specify the parameter or use '_' or '...' to unbind the parameter explicitly

from std.compile import compile_info


def param_fn[x: Int, y: Int]() -> Int:
    return x + y


def main() raises:
    # intentionally missing one parameter
    comptime myInstantiatedFn = param_fn[2]
    print(compile_info[myInstantiatedFn]())
