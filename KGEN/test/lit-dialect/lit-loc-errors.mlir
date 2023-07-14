// COM: Since errors involving incorrect locations cannot be handled by
// COM: -verify-diagnostics, we check manually.
// RUN: not kgen-opt %s 2>&1 | FileCheck %s

#file = #debuginfo.file<"test.mlir" in "">
#loc = loc("foo.mlir":7:8)

lit.func @foo() {
  lit.return
// CHECK: foo.mlir:7:8: error: 'lit.func' op must have subprogram scope in location, but got #debuginfo.file<"test.mlir" in "">
} loc(fused<#file>[#loc])
