// RUN: kgen-opt %s -pass-pipeline='builtin.module(kgen.func(lower-loops, canonicalize))' | FileCheck %s

// CHECK-LABEL: @induction_var_no_retvals_no_iterargs
kgen.func @induction_var_no_retvals_no_iterargs() {
  // CHECK:      [[IDX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[IDX1:%.*]] = index.constant 1
  // CHECK-NEXT: hlcf.loop (%arg0 = [[IDX2]] : index) {
  // CHECK-NEXT:   [[V0:%.*]] = index.cmp sgt(%arg0, [[IDX0]])
  // CHECK-NEXT:   hlcf.if [[V0]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[V1:%.*]] = index.sub %arg0, [[IDX1]]
  // CHECK-NEXT:   kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.continue [[V1]] : index
  // CHECK-NEXT: }

  %idx2 = index.constant 2
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  hlcf.for [%idx2 to %idx0 step %idx1 sgtlhs sub] (%arg0 = %idx2 : index) {
    %0 = index.sub %arg0, %idx1
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
  // CHECK-NEXT:   [[V0:%.*]] = index.cmp sgt(%arg0, [[IDX0]])
  // CHECK-NEXT:   hlcf.if [[V0]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[V1:%.*]] = index.sub %arg0, [[IDX1]]
  // CHECK-NEXT:   kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.loop (%arg1 = [[IDX4]] : index) {
  // CHECK-NEXT:     [[V3:%.*]] = index.cmp slt(%arg1, [[IDX8]])
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

  %idx2 = index.constant 2
  %idx4 = index.constant 4
  %idx8 = index.constant 8
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  hlcf.for [%idx2 to %idx0 step %idx1 sgtlhs sub] (%arg0 = %idx2 : index) {
    %0 = index.sub %arg0, %idx1
    kgen.call @foo(%0) : (index) -> ()
    hlcf.for [%idx4 to %idx8 step %idx2 sltlhs add] (%arg1 = %idx4 : index) {
      %3 = index.add %arg1, %idx2
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
  // CHECK-NEXT:   [[V1:%.*]] = index.cmp slt(%arg0, [[IDX9]])
  // CHECK-NEXT:   hlcf.if [[V1]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break %arg1, %arg2 : index, index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[V2:%.*]] = index.add %arg0, [[IDX2]]
  // CHECK-NEXT:   kgen.call @foo([[V2]], %arg1) : (index, index) -> ()
  // CHECK-NEXT:   [[V3:%.*]] = hlcf.loop (%arg3 = [[IDX4]] : index, %arg4 = %arg2 : index) -> index {
  // CHECK-NEXT:     [[V4:%.*]] = index.cmp slt(%arg3, [[IDX8]])
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

  %idx1 = index.constant 1
  %idx9 = index.constant 9
  %idx2 = index.constant 2
  %idx4 = index.constant 4
  %idx8 = index.constant 8
  %idx0 = index.constant 0
  %0:2 = hlcf.for [%idx1 to %idx9 step %idx2 sltlhs add] (%arg2 = %idx1 : index, %arg0 = %idx0 : index, %arg1 = %idx0 : index) -> (index, index) {
    %3 = index.add %arg2, %idx2
    kgen.call @foo(%3, %arg0) : (index, index) -> ()
    %6 = hlcf.for [%idx4 to %idx8 step %idx2 sltlhs add] (%arg4 = %idx4 : index, %arg3 = %arg1 : index) -> index {
      %7 = index.add %arg4, %idx2
      kgen.call @foo(%7, %arg3) : (index, index) -> ()
      hlcf.for.yield [induction_var (%7 : index)] [retvals (%7: index)] [iterargs ()]
    } {unrollFactor = #hlcf<loop_unroll_full none>}
    hlcf.for.yield [induction_var (%3 : index)] [retvals (%6, %3: index, index)] [iterargs ()]
  } {unrollFactor = #hlcf<loop_unroll_full none>}
  kgen.call @foo(%0#0) : (index) -> ()
  kgen.call @foo(%0#1) : (index) -> ()
  kgen.return
}
