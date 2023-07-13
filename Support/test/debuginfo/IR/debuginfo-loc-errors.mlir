// COM: Since errors involving incorrect locations cannot be handled by
// COM: -verify-diagnostics, we check manually.
// RUN: not support-dialect-opt %s 2>&1 | FileCheck %s

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

#loc = loc("foo.mlir":7:8)

func.func @foo(%arg: i32) {
  // CHECK: foo.mlir:7:8: error: 'debuginfo.value' op location scope must match variable scope: #debuginfo.file<"foo.c" in "/mlir/"> vs. #debuginfo.subprogram
  debuginfo.value #local_variable = %arg : i32 loc(fused<#file>[#loc])
  return
}
