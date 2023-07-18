// RUN: kgen-opt %s -pass-pipeline='builtin.module(kgen.func(lower-loops, canonicalize))' | FileCheck %s

// CHECK-LABEL: @induction_var_no_retvals_no_iterargs
kgen.func @induction_var_no_retvals_no_iterargs() {
  // CHECK:      [[IDX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[IDX1:%.*]] = index.constant 1
  // CHECK-NEXT: hlcf.loop (%arg0 = [[IDX2]] : index) {
  // CHECK-NEXT:   [[V0:%.*]] = index.cmp slt([[IDX0]], %arg0)
  // CHECK-NEXT:   hlcf.if [[V0]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[V1:%.*]] = index.sub %arg0, [[IDX1]]
  // CHECK-NEXT:   kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.continue [[V1]] : index
  // CHECK-NEXT: }

  %index2 = index.constant 2
  %idx0 = index.constant 0
  %index1 = index.constant 1
  hlcf.for [%idx0 to %index2 step %index1] (%arg0 = %index2 : index) {
    %0 = index.sub %arg0, %index1
    kgen.call @foo(%0) : (index) -> ()
    hlcf.for.yield [induction_var (%0 : index)] [retvals ()] [iterargs ()]
  } {unrollFactor = #hlcf<loop_unroll_full none>}
  kgen.return
}

// CHECK-LABEL: @nested_unroll_loops
kgen.func @nested_unroll_loops() {
  // CHECK:      [[IDX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[IDX4:%.*]] = index.constant 4
  // CHECK-NEXT: [[IDX8:%.*]] = index.constant 8
  // CHECK-NEXT: [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[IDX1:%.*]] = index.constant 1
  // CHECK-NEXT: hlcf.loop (%arg0 = [[IDX2]] : index) {
  // CHECK-NEXT:   [[V0:%.*]] = index.cmp slt([[IDX0]], %arg0)
  // CHECK-NEXT:   hlcf.if [[V0]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[V1:%.*]] = index.sub %arg0, [[IDX1]]
  // CHECK-NEXT:   kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.loop (%arg1 = [[IDX4]] : index) {
  // CHECK-NEXT:     [[V3:%.*]] = index.cmp sgt([[IDX8]], %arg1)
  // CHECK-NEXT:     hlcf.if [[V3]] {
  // CHECK-NEXT:       hlcf.yield
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       hlcf.break
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[V4:%.*]] = index.add %arg1, [[IDX2]]
  // CHECK-NEXT:     kgen.call @foo([[V4]]) : (index) -> ()
  // CHECK-NEXT:     hlcf.continue [[V4]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   hlcf.continue [[V1]] : index
  // CHECK-NEXT: }

  %index2 = index.constant 2
  %index4 = index.constant 4
  %index8 = index.constant 8
  %idx0 = index.constant 0
  %index1 = index.constant 1
  hlcf.for [%idx0 to %index2 step %index1] (%arg0 = %index2 : index) {
    %0 = index.sub %arg0, %index1
    kgen.call @foo(%0) : (index) -> ()
    hlcf.for [%index4 to %index8 step %index2] (%arg1 = %index4 : index) {
      %3 = index.add %arg1, %index2
      kgen.call @foo(%3) : (index) -> ()
      hlcf.for.yield [induction_var (%3 : index)] [retvals ()] [iterargs ()]
    } {unrollFactor = #hlcf<loop_unroll_full none>}
    hlcf.for.yield [induction_var (%0 : index)] [retvals ()] [iterargs ()]
  } {unrollFactor = #hlcf<loop_unroll_full none>}
  kgen.return
}

// CHECK-LABEL: @loop_carried_dependency
kgen.func @loop_carried_dependency() {
  // CHECK:      [[IDX1:%.*]] = index.constant 1
  // CHECK-NEXT: [[IDX9:%.*]] = index.constant 9
  // CHECK-NEXT: [[IDX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[IDX4:%.*]] = index.constant 4
  // CHECK-NEXT: [[IDX8:%.*]] = index.constant 8
  // CHECK-NEXT: [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: %0:2 = hlcf.loop (%arg0 = [[IDX1]] : index, %arg1 = [[IDX0]] : index, %arg2 = [[IDX0]] : index) -> (index, index) {
  // CHECK-NEXT:   [[V1:%.*]] = index.cmp sgt([[IDX9]], %arg0)
  // CHECK-NEXT:   hlcf.if [[V1]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break %arg1, %arg2 : index, index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[V2:%.*]] = index.add %arg0, [[IDX2]]
  // CHECK-NEXT:   kgen.call @foo([[V2]], %arg1) : (index, index) -> ()
  // CHECK-NEXT:   [[V3:%.*]] = hlcf.loop (%arg3 = [[IDX4]] : index, %arg4 = %arg2 : index) -> index {
  // CHECK-NEXT:     [[V4:%.*]] = index.cmp sgt([[IDX8]], %arg3)
  // CHECK-NEXT:     hlcf.if [[V4]] {
  // CHECK-NEXT:       hlcf.yield
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       hlcf.break %arg4 : index
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[V5:%.*]] = index.add %arg3, [[IDX2]]
  // CHECK-NEXT:     kgen.call @foo([[V5]], %arg4) : (index, index) -> ()
  // CHECK-NEXT:     hlcf.continue [[V5]], [[V5]] : index, index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   hlcf.continue [[V2]], [[V3]], [[V2]] : index, index, index
  // CHECK-NEXT: }

  %index1 = index.constant 1
  %index9 = index.constant 9
  %index2 = index.constant 2
  %index4 = index.constant 4
  %index8 = index.constant 8
  %index0 = index.constant 0
  %0:2 = hlcf.for [%index1 to %index9 step %index2] (%arg2 = %index1 : index, %arg0 = %index0 : index, %arg1 = %index0 : index) -> (index, index) {
    %3 = index.add %arg2, %index2
    kgen.call @foo(%3, %arg0) : (index, index) -> ()
    %6 = hlcf.for [%index4 to %index8 step %index2] (%arg4 = %index4 : index, %arg3 = %arg1 : index) -> index {
      %7 = index.add %arg4, %index2
      kgen.call @foo(%7, %arg3) : (index, index) -> ()
      hlcf.for.yield [induction_var (%7 : index)] [retvals (%7: index)] [iterargs ()]
    } {unrollFactor = #hlcf<loop_unroll_full none>}
    hlcf.for.yield [induction_var (%3 : index)] [retvals (%6: index)] [iterargs (%3: index)]
  } {unrollFactor = #hlcf<loop_unroll_full none>}
  kgen.call @foo(%0#0) : (index) -> ()
  kgen.call @foo(%0#1) : (index) -> ()
  kgen.return
}
