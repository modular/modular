# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: system-linux
# RUN: %mojo %s -o %t
# RUN: llvm-objdump -t %t | FileCheck %s

from compile import compile_info
from sys import argv, sizeof


fn get_type(dtype: DType) -> DType:
    return dtype


fn compiled_fn[dtype: DType](M: SIMD[get_type(dtype), 4]) -> Int:
    alias b = sizeof[get_type(dtype)]()
    return b + Int(M[0])


def main():
    alias myCompiledFn = compiled_fn[DType.uint32]
    # compile myCompileFn into a shared object binary
    var myShared = compile_info[myCompiledFn, emission_kind="object"]()

    idx = 0
    args = argv()
    for arg in argv():
        idx = idx + 1
        if arg == "-o":
            break

    # write the shared object binary to a file for checking
    with open(args[idx], "w") as f:
        f.write(myShared)


# CHECK: dynamic
# CHECK: compile_shared::compiled_fn[DType]
