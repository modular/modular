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
