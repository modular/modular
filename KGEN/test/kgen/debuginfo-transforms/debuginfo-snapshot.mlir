// RUN: kgen-opt -debuginfo-snapshot -mlir-print-debuginfo -split-input-file %s | FileCheck %s

// CHECK: ![[SP_INNER_TYPE:.+]] = !debuginfo.subroutine<() -> (): DW_CC_normal>
// CHECK: #[[SP_INNER:.+]] = #debuginfo.subprogram<{{.*}}name = <"kgen.param.declare.region">{{.*}}> : ![[SP_INNER_TYPE]]

// CHECK: kgen.generator @subprogram_scoped_op
kgen.generator @subprogram_scoped_op() always_inline_no_debug {
  // CHECK-NEXT: kgen.param.declare.region
  kgen.param.declare.region A = () -> () {
    // CHECK-NEXT: kgen.param.constant {{.*}} loc(#[[LOC_CONST:.+]])
    // CHECK-NEXT: debuginfo.value {{.*}} loc(#[[LOC_CONST]])
    %0 = kgen.param.constant = <1>
    // CHECK-NEXT: kgen.return loc(#[[LOC_RET:.+]])
    kgen.return
  // CHECK-NEXT: } loc(#[[LOC_SP_OP:.+]])
  }
  // CHECK-NEXT: kgen.return loc(#[[LOC_OUTER_RET:.+]])
  kgen.return
}

// CHECK-DAG: #[[LOC_CONST]] = loc(fused<#[[SP_INNER]]>
// CHECK-DAG: #[[LOC_RET]] = loc(fused<#[[SP_INNER]]>
// CHECK-DAG: #[[LOC_SP_OP]] = loc(fused<#[[SP_INNER]]>

// CHECK-DAG: #[[SP_OUTER:.+]] = #debuginfo.subprogram<{{.*}}name = <"subprogram_scoped_op">{{.*}}>
// CHECK-DAG: #[[LOC_OUTER_RET]] = loc(fused<#[[SP_OUTER]]>

// -----

// CHECK: kgen.func @inlined_subprogram_scoped_op
kgen.func @inlined_subprogram_scoped_op() always_inline {
  // CHECK-NEXT: kgen.stage_closure
  kgen.stage_closure = () {
    // CHECK-NEXT: kgen.return loc(#[[LOC_RET:.+]])
    kgen.return
  // CHECK-NEXT: } loc(#[[LOC_SP_OP:.+]])
  }
  // CHECK-NEXT: kgen.return loc(#[[LOC_OUTER_RET:.+]])
  kgen.return
}

// CHECK-DAG: #[[SP_OUTER:.+]] = #debuginfo.subprogram<{{.*}}name = <"inlined_subprogram_scoped_op">{{.*}}>
// CHECK-DAG: #[[SP_INNER:.+]] = #debuginfo.subprogram<{{.*}}name = <"kgen.stage_closure">{{.*}}>

// CHECK-DAG: #[[LOC_RET]] = loc(fused<#[[SP_INNER]]>

// CHECK-DAG: #[[LOC_SP_OP]] = loc(fused<#[[SP_INNER]]>[#[[LOC_SP_OP_RAW:.+]]])
// CHECK-DAG: #[[LOC_SP_OP_RAW]] = loc(fused<#[[CALL_LOC:.+]]>
// CHECK-DAG: #[[CALL_LOC]] = #debuginfo.call_loc<#[[LOC_SP_OP_CALL_LOC:.+]]>
// CHECK-DAG: #[[LOC_SP_OP_CALL_LOC]] = loc(fused<#[[SP_OUTER]]>

// CHECK-DAG: #[[LOC_OUTER_RET]] = loc(fused<#[[SP_OUTER]]>
