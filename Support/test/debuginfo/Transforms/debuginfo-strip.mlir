// RUN: support-dialect-opt %s -debuginfo-strip -mlir-print-debuginfo -allow-unregistered-dialect | FileCheck %s

!subroutine = !debuginfo.subroutine<() -> (): DW_CC_normal>
#file = #debuginfo.file<"foo" in "foo">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR", isOptimized = true, emissionKind = Full>
#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = <"fn">, linkageName = "fn", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized"> : !subroutine

#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "buf", file = #file, line = 159, arg = 1> : !debuginfo.unresolved<index>

// CHECK-NOT: #debuginfo.
// CHECK: "unknown_dialect.op"
// CHECK: "test.mlir":10:10

func.func @foo(%arg: index) {
  "unknown_dialect.op"() : () -> ()
  debuginfo.value #local_variable = %arg : index
  return
} loc(fused<#subprogram>["test.mlir":10:10])
