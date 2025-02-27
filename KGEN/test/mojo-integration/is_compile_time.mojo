# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


fn f(i: Int) -> Int:
    var t = 0
    # Intentionally have then and else branches mismatch logic for testing.
    if __mlir_op.`kgen.is_compile_time`():
        t = i + 40
    else:
        t = i + 1

    return t + 1


fn main():
    # CHECK: interpret value: 42
    alias a = f(1)
    print("interpret value:", a)

    # CHECK: runtime value: 3
    var b = f(1)
    print("runtime value:", b)
