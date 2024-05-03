// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' -mlir-print-debuginfo %s | FileCheck %s

// CHECK-DAG: ![[INDEX:.*]] = !debuginfo.basic<index
// CHECK-DAG: ![[SUBROUTINE:.*]] = !debuginfo.subroutine<() -> (![[INDEX]]): DW_CC_normal>
// CHECK-DAG: ![[PTR:.*]] = !debuginfo.ptr<![[SUBROUTINE]] {sizeInBits = 64, alignInBits = 64}>

// CHECK-DAG: !debuginfo.subroutine<(![[PTR]]) -> (): DW_CC_normal>

!test = !debuginfo.subroutine<(
  !debuginfo.unresolved<!co.routine<() -> index>>
) -> (): DW_CC_normal>

#subprogram = #debuginfo.subprogram<name = <"foo">> : !test

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="i64:64:64", simd_bit_width=128>} {
  kgen.func @foo() {
    kgen.return loc(fused<#subprogram>["foo.mlir":10:10])
  } loc(fused<#subprogram>["foo.mlir":10:10])
}
