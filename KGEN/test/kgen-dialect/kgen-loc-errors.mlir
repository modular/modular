// COM: Since errors involving incorrect locations cannot be handled by
// COM: -verify-diagnostics, we check manually.
// RUN: not kgen-opt -split-input-file %s 2>&1 | FileCheck %s

#file = #debuginfo.file<"test.mlir" in "">
#loc = loc("foo.mlir":7:8)

kgen.func @foo() {
  kgen.return
// CHECK: foo.mlir:7:8: error: 'kgen.func' op must have subprogram scope in location, but got #debuginfo.file<"test.mlir" in "">
} loc(fused<#file>[#loc])

// -----

#file = #debuginfo.file<"test.mlir" in "">
#loc = loc("foo.mlir":7:8)

kgen.generator @foo() {
  kgen.return
// CHECK: foo.mlir:7:8: error: 'kgen.generator' op must have subprogram scope in location, but got #debuginfo.file<"test.mlir" in "">
} loc(fused<#file>[#loc])

// -----

#file = #debuginfo.file<"foo.mlir" in "/mlir/">
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

kgen.func @foo() {
  // CHECK: error: 'kgen.return' op location scope does not match scope of parent func location: #debuginfo.subprogram
  kgen.return
} loc(fused<#subprogram>[#loc])

// -----

#file = #debuginfo.file<"foo.mlir" in "/mlir/">
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

kgen.func @foo() {
  // CHECK: foo.mlir:7:8: error: 'kgen.return' op location scope does not match scope of parent func location: #debuginfo.subprogram
  kgen.return loc(fused<#file>[#loc])
} loc(fused<#subprogram>[#loc])

// -----

#file = #debuginfo.file<"foo.mlir" in "/mlir/">
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
#loc1 = loc("bar.mlir":5:6)
#funcLoc = loc(fused<#subprogram>[#loc])
#callsite = loc(callsite(#loc1 at #loc))

kgen.func @foo() {
  // CHECK: bar.mlir:5:6: error: 'kgen.param.constant' op location scope does not match scope of parent func location: #debuginfo.subprogram
  %index1 = kgen.param.constant = <1> loc(callsite(#loc1 at #callsite))
  kgen.return loc(#funcLoc)
} loc(#funcLoc)

// -----

#file = #debuginfo.file<"foo.mlir" in "/mlir/">
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
#subprogram1 = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "foo1",
  linkageName = "foo1",
  file = #file,
  line = 23,
  scopeLine = 23,
  subprogramFlags = Definition
> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#lexical_block = #debuginfo.lexical_block<scope = #subprogram1, file = #file, line = 104, column = 17>
#lexical_block1 = #debuginfo.lexical_block<scope = #lexical_block, file = #file, line = 120, column = 22>

#loc = loc("foo.mlir":7:8)
#loc1 = loc("foo.mlir":10:13)
#funcLoc = loc(fused<#subprogram>[#loc])

kgen.func @foo() {
  // CHECK: foo.mlir:10:13: error: 'kgen.param.constant' op location scope does not match scope of parent func location: #debuginfo.subprogram
  %index1 = kgen.param.constant = <2> loc(fused<#lexical_block1>[#loc1])
  kgen.return loc(#funcLoc)
} loc(#funcLoc)

// -----

#file = #debuginfo.file<"foo.mlir" in "/mlir/">
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
#loc1 = loc("bar.mlir":5:6)
#funcLoc = loc(fused<#subprogram>[#loc])

kgen.func @foo() {
  // CHECK: foo.mlir:7:8: error: 'kgen.param.constant' op contains inconsistent scopes in fused location
  %index1 = kgen.param.constant = <1> loc(callsite(#loc at fused[#loc1, #funcLoc]))
  kgen.return loc(#funcLoc)
} loc(#funcLoc)
