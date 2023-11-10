// RUN: support-dialect-opt %s | support-dialect-opt | FileCheck %s
// RUN: support-dialect-opt -emit-bytecode %s | support-dialect-opt | FileCheck %s

// CHECK: ![[SUBROUTINE:.*]] = !debuginfo.subroutine<() -> (): DW_CC_normal>
// CHECK: ![[UNRESOLVED:.*]] = !debuginfo.unresolved<index>
// CHECK: #[[FILE:.*]] = #debuginfo.file<"foo.c" in "/mlir/">
#file = #debuginfo.file<"foo.c" in "/mlir/">

// CHECK: #[[CU:.*]] = #debuginfo.compile_unit<
// CHECK-SAME:   sourceLanguage = DW_LANG_Mojo,
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   producer = "MLIR",
// CHECK-SAME:   isOptimized = true,
// CHECK-SAME:   emissionKind = Full
// CHECK-SAME: >
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>

// CHECK: #[[SP:.*]] = #debuginfo.subprogram<
// CHECK-SAME:   compileUnit = #[[CU]],
// CHECK-SAME:   scope = #[[FILE]],
// CHECK-SAME:   name = <"foo">,
// CHECK-SAME:   linkageName = "foo",
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   line = 10,
// CHECK-SAME:   scopeLine = 10,
// CHECK-SAME:   subprogramFlags = Definition
// CHECK-SAME: > : ![[SUBROUTINE]]
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = <"foo">,
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<() -> (): DW_CC_normal>

// CHECK: #[[LEX_BLOCK:.*]] = #debuginfo.lexical_block<
// CHECK-SAME:   scope = #[[SP]],
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   line = 10,
// CHECK-SAME:   column = 1
// CHECK-SAME: >
#lex_block = #debuginfo.lexical_block<
  scope = #subprogram,
  file = #file,
  line = 10,
  column = 1
>

// CHECK: #[[VAR:.*]] = #debuginfo.local_variable<
// CHECK-SAME:   scope = #[[LEX_BLOCK]],
// CHECK-SAME:   name = "foo",
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   line = 10,
// CHECK-SAME:   arg = 1,
// CHECK-SAME:   alignInBits = 32
// CHECK-SAME: > : ![[UNRESOLVED]]
#local_variable = #debuginfo.local_variable<
  scope = #lex_block,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1,
  alignInBits = 32
> : !debuginfo.unresolved<index>

// CHECK: module attributes {test.loc = #[[VAR]]}
module attributes { test.loc = #local_variable } {}
