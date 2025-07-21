# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from compile import compile_info
from sys import sizeof


fn get_type(dtype: DType) -> DType:
    return dtype


fn compiled_fn[dtype: DType](M: SIMD[get_type(dtype), 4]) -> Int:
    alias b = sizeof[get_type(dtype)]()
    return b + Int(M[0])


fn main():
    alias myCompiledFn = compiled_fn[DType.uint32]
    print(compile_info[myCompiledFn, emission_kind="llvm"]())


# CHECK: define {{.*}} @"compile_assembly_debuginfo::compiled_fn{{.*}} !dbg ![[SP:[0-9]+]]
# CHECK-NOT @"compile_assembly_debuginfo::compiled_fn
# CHECK: ![[SP]] = distinct !DISubprogram({{.*}}type: ![[SUBROUTINE:[0-9]+]]
# CHECK: ![[SUBROUTINE]] = !DISubroutineType({{.*}}types: ![[FUNCTION_TYPE:[0-9]+]]
# CHECK: ![[FUNCTION_TYPE]] = !{!{{[0-9]+}}, ![[ARG_TYPE:[0-9]+]]}

# The function arg type should have been concretized into the actual dtype.
# CHECK: ![[ARG_TYPE]] = !DICompositeType(tag: DW_TAG_array_type, name: "!pop.simd<4, ui32>"
