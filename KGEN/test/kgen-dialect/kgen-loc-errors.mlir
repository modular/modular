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
