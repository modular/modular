// RUN: support-dialect-opt %s | support-dialect-opt | FileCheck %s

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

// CHECK-LABEL: func @foo
// CHECK-SAME: (%[[ARG:.*]]: i32)
func.func @foo(%arg: i32) {
  // CHECK: debuginfo.value #[[VAR:.*]] = %[[ARG]] : i32
  debuginfo.value #local_variable = %arg : i32
  return
}
