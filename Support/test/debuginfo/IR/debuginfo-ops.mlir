// RUN: support-dialect-opt -mlir-print-debuginfo %s | support-dialect-opt -mlir-print-debuginfo | FileCheck %s

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_C,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1,
  alignInBits = 32
> : !debuginfo.unresolved<i32>

#loc1 = loc("foo.mlir":7:8)
#loc2 = loc("bar.mlir":5:6)
#fusedLoc = loc(fused<#subprogram>[#loc2])
#loc3 = loc(callsite(#fusedLoc at #loc1))

// CHECK-LABEL: func @foo
// CHECK-SAME: (%[[ARG:.*]]: i32
func.func @foo(%arg: i32) {
  // CHECK: debuginfo.value #[[VAR:.*]] = %[[ARG]] : i32
  debuginfo.value #local_variable = %arg : i32 loc(callsite(#loc3 at #loc1))
  return
}
