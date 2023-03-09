// RUN: kgen-opt -lower-kgen-to-llvm -mlir-print-debuginfo %s | FileCheck %s

!subroutine = !debuginfo.subroutine<() -> (): DW_CC_normal>

#file = #debuginfo.file<"test.mlir" in "">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR", isOptimized = true, emissionKind = Full>

#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "kernel", linkageName = "kernel()", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized"> : !subroutine

#loc = loc(fused<#subprogram>["test.mlir":1:1])

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @kernel() {
    kgen.return loc(#loc)
  } loc(#loc)
  kgen.export @kernel as @kernel_special_name

  // CHECK-LABEL: llvm.func @kernel_special_name
  // CHECK-NEXT: llvm.return{{.*}}loc(#[[LOC:.*]])
  // CHECK-NEXT: loc(#[[LOC]])

  // CHECK: #[[SP1:.*]] = #debuginfo.subprogram<{{.*}} "kernel", linkageName = "kernel_special_name"
  // CHECK-DAG: #[[LOC]] = loc(fused<#[[SP1]]>[#{{.*}}])
}
