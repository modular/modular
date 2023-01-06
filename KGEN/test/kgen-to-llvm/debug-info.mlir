// RUN: kgen-opt -lower-kgen-to-llvm -mlir-print-debuginfo %s | FileCheck %s

!subroutine = !debuginfo.subroutine<() -> (): DW_CC_normal>

#file = #debuginfo.file<"test.mlir" in "">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR", isOptimized = true, emissionKind = Full>

#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "kernel", linkageName = "kernel()", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized"> : !subroutine

#loc = loc(fused<#subprogram>["test.mlir":1:1])

// CHECK-LABEL: llvm.func @"kernel()"
kgen.func @"kernel()"() {
  kgen.return
} loc(#loc)


// CHECK-LABEL: llvm.func @kernel___c
// CHECK-NEXT: llvm.call {{.*}} loc(#[[LOC:.*]])
// CHECK-NEXT: llvm.return{{.*}}loc(#[[LOC]])
// CHECK-NEXT: loc(#[[LOC]])

// CHECK: #[[SP1:.*]] = #debuginfo.subprogram<{{.*}} "kernel", linkageName = "kernel___c"
// CHECK: #[[LOC]] = loc(fused<#[[SP1]]>[#{{.*}}])

kgen.export [@"kernel()"]
