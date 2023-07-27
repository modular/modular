// COM: Since errors involving incorrect locations cannot be handled by
// COM: -verify-diagnostics, we check manually.
// RUN: not kgen-opt -split-input-file %s 2>&1 | FileCheck %s

#file = #debuginfo.file<"test.mlir" in "">
#loc = loc("foo.mlir":7:8)

lit.func @foo() {
  lit.return
// CHECK: foo.mlir:7:8: error: 'lit.func' op must have subprogram scope in location, but got #debuginfo.file<"test.mlir" in "">
} loc(fused<#file>[#loc])

// -----

#file = #debuginfo.file<"bar.mlir" in "">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_C,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#loc = loc("bar.mlir":10:5)

lit.struct.decl @Foo {
  lit.struct.field value : index
// CHECK: bar.mlir:10:5: error: 'lit.struct.decl' op must have file scope in location, but got #debuginfo.compile_unit
} loc(fused<#compile_unit>[#loc])

// -----

#file = #debuginfo.file<"foo.mlir" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 44,
  scopeLine = 44,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "SomeClosure",
  linkageName = "SomeClosure",
  file = #file,
  line = 325,
  scopeLine = 325,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)
#loc4 = loc(fused<#subprogram>[#loc1])
#loc5 = loc(fused<#subprogram1>[#loc2])

lit.func @foo() {
  %0 = lit.async.execute <() -> ()> {
    lit.async.return loc(#loc5)
  // CHECK: foo.mlir:325:11: error: 'lit.async.execute' op must have callsite location
  } loc(#loc5)
  lit.return loc(#loc4)
} loc(#loc4)
