// COM: Since errors involving incorrect locations cannot be handled by
// COM: -verify-diagnostics, we check manually.
// RUN: not support-dialect-opt -split-input-file %s 2>&1 | FileCheck %s

#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<f32>
#loc = loc("foo.mlir":7:8)

func.func @foo(%arg: i32) {
  // CHECK: foo.mlir:7:8: error: 'debuginfo.value' op conversion expression input expr.irvalue type 'f32' does not match actual IR Value type 'i32'
  debuginfo.value #local_variable = %arg : i32 loc(fused<#subprogram>[#loc])
  return
}

// -----

#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<f32>
#loc = loc("foo.mlir":7:8)

func.func @foo(%arg: i32) {
  // CHECK: foo.mlir:7:8: error: 'debuginfo.value' op conversion expression output type '!debuginfo.unresolved<i32>' does not match variable declared type '!debuginfo.unresolved<f32>'
  debuginfo.value #local_variable #debuginfo.expr.irvalue: !debuginfo.unresolved<i32> = %arg : i32 loc(fused<#subprogram>[#loc])
  return
}

// -----

#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<i32>
#loc = loc("foo.mlir":7:8)

#diExpr = #debuginfo.expr.refof<#debuginfo.expr.irvalue: !debuginfo.unresolved<i32>> : !debuginfo.ti.ptr<i32>

func.func @foo(%arg: i32) {
  // CHECK: foo.mlir:7:8: error: 'debuginfo.value' op conversion expression output type '!debuginfo.ti.ptr<i32>' does not match variable declared type '!debuginfo.unresolved<i32>'
  debuginfo.value #local_variable #diExpr = %arg : i32 loc(fused<#subprogram>[#loc])
  return
}
