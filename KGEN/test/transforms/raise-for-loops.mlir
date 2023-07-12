// RUN: kgen-opt %s -pass-pipeline='builtin.module(kgen.func(raise-for-loops, canonicalize))' | FileCheck %s

// CHECK-LABEL: @zero_starting_range
kgen.func @zero_starting_range() {
  %index2 = kgen.param.constant = <2>
  %idx0 = index.constant 0
  %index1 = kgen.param.constant = <1>
  %array = kgen.param.constant: array<0, i1> = <[]>

  // CHECK:      [[INDEX2:%.*]] = kgen.param.constant = <2>
  // CHECK-NEXT: [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[INDEX1:%.*]] = kgen.param.constant = <1>
  // CHECK-NEXT: hlcf.for [[[IDX0]] to [[INDEX2]] step [[INDEX1]]] (%arg0 = [[INDEX2]] : index) {
  // CHECK-NEXT:   [[IDX:%.*]] = index.sub %arg0, [[INDEX1]]
  // CHECK-NEXT:   [[V:%.*]] = index.sub [[INDEX2]], %arg0
  // CHECK-NEXT:   kgen.call @foo([[V]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.continue [[IDX]] : index
  // CHECK-NEXT: } {unrollFactor = #hlcf<loop_unroll_full full>}

  hlcf.loop (%arg0 = %index2 : index) {
    %0 = index.cmp sgt(%arg0, %idx0)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.sub %arg0, %index1
    %2 = index.sub %index2, %arg0
    kgen.call @foo(%2) : (index) -> ()
    hlcf.continue %1 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return
}

// CHECK-LABEL: @sequential_range
kgen.func @sequential_range() {
  %index1 = kgen.param.constant = <1>
  %index4 = kgen.param.constant = <4>
  %array = kgen.param.constant: array<0, i1> = <[]>

  // CHECK:      [[INDEX1:%.*]] = kgen.param.constant = <1>
  // CHECK-NEXT: [[INDEX4:%.*]] = kgen.param.constant = <4>
  // CHECK-NEXT: hlcf.for [[[INDEX1]] to [[INDEX4]] step [[INDEX1]]] (%arg0 = [[INDEX1]] : index) {
  // CHECK-NEXT:   [[IDX:%.*]] = index.add %arg0, [[INDEX1]]
  // CHECK-NEXT:   kgen.call @foo(%arg0) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.continue [[IDX]] : index
  // CHECK-NEXT: } {unrollFactor = #hlcf<loop_unroll_full full>}

  hlcf.loop (%arg0 = %index1 : index) {
    %0 = index.cmp slt(%arg0, %index4)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.add %arg0, %index1
    kgen.call @foo(%arg0) : (index) -> ()
    hlcf.continue %1 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return
}

// CHECK-LABEL: @strided_range
kgen.func @strided_range() {
  %index1 = kgen.param.constant = <1>
  %index6 = kgen.param.constant = <6>
  %index2 = kgen.param.constant = <2>

  // CHECK:      [[INDEX1:%.*]] = kgen.param.constant = <1>
  // CHECK-NEXT: [[INDEX6:%.*]] = kgen.param.constant = <6>
  // CHECK-NEXT: [[INDEX2:%.*]] = kgen.param.constant = <2>
  // CHECK-NEXT: hlcf.for [[[INDEX1]] to [[INDEX6]] step [[INDEX2]]] (%arg0 = [[INDEX1]] : index) {
  // CHECK-NEXT:   [[IDX:%.*]] = index.add %arg0, [[INDEX2]]
  // CHECK-NEXT:   kgen.call @foo(%arg0) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.continue [[IDX]]  : index
  // CHECK-NEXT: } {unrollFactor = #hlcf<loop_unroll_full full>}

  %array = kgen.param.constant: array<0, i1> = <[]>
  hlcf.loop (%arg0 = %index1 : index) {
    %0 = index.cmp slt(%arg0, %index6)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.add %arg0, %index2
    kgen.call @foo(%arg0) : (index) -> ()
    hlcf.continue %1 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return
}

// CHECK-LABEL: @nested_unroll_loops
kgen.func @nested_unroll_loops() {
  %index2 = kgen.param.constant = <2>
  %index4 = kgen.param.constant = <4>
  %index8 = kgen.param.constant = <8>
  %idx0 = index.constant 0
  %index1 = kgen.param.constant = <1>
  %array = kgen.param.constant: array<0, i1> = <[]>

  // CHECK:      [[INDEX2:%.*]] = kgen.param.constant = <2>
  // CHECK-NEXT: [[INDEX4:%.*]] = kgen.param.constant = <4>
  // CHECK-NEXT: [[INDEX8:%.*]] = kgen.param.constant = <8>
  // CHECK-NEXT: [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[INDEX1:%.*]] = kgen.param.constant = <1>
  // CHECK-NEXT: hlcf.for [[[IDX0]] to [[INDEX2]] step [[INDEX1]]] (%arg0 = [[INDEX2]] : index) {
  // CHECK-NEXT:   [[IDX0:%.*]] = index.sub %arg0, [[INDEX1]]
  // CHECK-NEXT:   [[V0:%.*]]  = index.sub [[INDEX2]], %arg0
  // CHECK-NEXT:   kgen.call @foo([[V0]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.for [[[INDEX4]] to [[INDEX8]] step [[INDEX2]]] (%arg1 = [[INDEX4]] : index) {
  // CHECK-NEXT:     [[IDX1:%.*]]  = index.add %arg1, [[INDEX2]]
  // CHECK-NEXT:     [[V1:%.*]] = index.add %1, %arg1
  // CHECK-NEXT:     kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT:     hlcf.for.continue [[IDX1]] : index
  // CHECK-NEXT:  } {unrollFactor = #hlcf<loop_unroll_full full>}
  // CHECK-NEXT:   hlcf.for.continue [[IDX0]] : index
  // CHECK-NEXT: } {unrollFactor = #hlcf<loop_unroll_full full>}

  hlcf.loop (%arg0 = %index2 : index) {
    %0 = index.cmp sgt(%arg0, %idx0)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.sub %arg0, %index1
    %2 = index.sub %index2, %arg0
    kgen.call @foo(%2) : (index) -> ()
    hlcf.loop (%arg1 = %index4 : index) {
      %4 = index.cmp slt(%arg1, %index8)
      hlcf.if %4 {
        hlcf.yield
      } else {
        hlcf.break
      }
      %5 = index.add %arg1, %index2
      %6 = index.add %2, %arg1
      kgen.call @foo(%6) : (index) -> ()
      hlcf.continue %5 : index
    } {unrollFactor = #hlcf<loop_unroll_full full>}
    hlcf.continue %1 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return
}

// For-loop not raised because loop has no unrollFactor
// CHECK-LABEL: @zero_starting_range_no_raise
kgen.func @zero_starting_range_no_raise() {
  %index2 = kgen.param.constant = <2>
  %idx0 = index.constant 0
  %index1 = kgen.param.constant = <1>
  %array = kgen.param.constant: array<0, i1> = <[]>
  // CHECK-NOT: hlcf.for
  hlcf.loop (%arg0 = %index2 : index) {
    %0 = index.cmp sgt(%arg0, %idx0)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.sub %arg0, %index1
    %2 = index.sub %index2, %arg0
    kgen.call @foo(%2) : (index) -> ()
    hlcf.continue %1 : index
  } {unrollFactor = #hlcf<loop_unroll_full none>}
  kgen.return
}

// CHECK-LABEL: @loop_carried_dependency
kgen.func @loop_carried_dependency() {
  %index1 = kgen.param.constant = <1>
  %index9 = kgen.param.constant = <9>
  %index2 = kgen.param.constant = <2>
  %index4 = kgen.param.constant = <4>
  %index8 = kgen.param.constant = <8>
  %array = kgen.param.constant: array<0, i1> = <[]>
  %index0 = kgen.param.constant = <0>

  // CHECK:      [[INDEX1:%.*]] = kgen.param.constant = <1>
  // CHECK-NEXT: [[INDEX9:%.*]] = kgen.param.constant = <9>
  // CHECK-NEXT: [[INDEX2:%.*]] = kgen.param.constant = <2>
  // CHECK-NEXT: [[INDEX4:%.*]] = kgen.param.constant = <4>
  // CHECK-NEXT: [[INDEX8:%.*]] = kgen.param.constant = <8>
  // CHECK-NEXT: [[INDEX0:%.*]] = kgen.param.constant = <0>
  // CHECK-NEXT: %0:2 = hlcf.for [[[INDEX1]] to [[INDEX9]] step [[INDEX2]]] (%arg0 = [[INDEX0]]  : index, %arg1 = [[INDEX0]] : index, %arg2 = [[INDEX1]]  : index) -> (index, index) {
  // CHECK-NEXT:   [[IDX0:%.*]] = index.add %arg2, [[INDEX2]]
  // CHECK-NEXT:   kgen.call @foo(%arg2) : (index) -> ()
  // CHECK-NEXT:   [[V0:%.*]] = index.add %arg0, %arg2
  // CHECK-NEXT:   [[V1:%.*]] = hlcf.for [[[INDEX4]] to [[INDEX8]] step [[INDEX2]]] (%arg3 = %arg1 : index, %arg4 = [[INDEX4]] : index) -> index {
  // CHECK-NEXT:     [[IDX1:%.*]] = index.add %arg4, [[INDEX2]]
  // CHECK-NEXT:     [[V2:%.*]] = index.add %arg2, %arg4
  // CHECK-NEXT:     kgen.call @foo([[V2]]) : (index) -> ()
  // CHECK-NEXT:     [[V3:%.*]] = index.add %arg3, %arg4
  // CHECK-NEXT:     hlcf.for.continue [[V3]], [[IDX1]] : index, index
  // CHECK-NEXT:   } {unrollFactor = #hlcf<loop_unroll_full full>}
  // CHECK-NEXT:   hlcf.for.continue [[V0]], [[V1]], [[IDX0]] : index, index, index
  // CHECK-NEXT: } {unrollFactor = #hlcf<loop_unroll_full full>}

  %0:2 = hlcf.loop (%arg0 = %index0 : index, %arg1 = %index0 : index, %arg2 = %index1 : index) -> (index, index) {
    %3 = index.cmp slt(%arg2, %index9)
    hlcf.if %3 {
      hlcf.yield
    } else {
      hlcf.break %arg0, %arg1 : index, index
    }
    %4 = index.add %arg2, %index2
    kgen.call @foo(%arg2) : (index) -> ()
    %6 = index.add %arg0, %arg2
    %7 = hlcf.loop (%arg3 = %arg1 : index, %arg4 = %index4 : index) -> index {
      %8 = index.cmp slt(%arg4, %index8)
      hlcf.if %8 {
        hlcf.yield
      } else {
        hlcf.break %arg3 : index
      }
      %9 = index.add %arg4, %index2
      %10 = index.add %arg2, %arg4
      kgen.call @foo(%10) : (index) -> ()
      %12 = index.add %arg3, %arg4
      hlcf.continue %12, %9 : index, index
    } {unrollFactor = #hlcf<loop_unroll_full full>}
    hlcf.continue %6, %7, %4 : index, index, index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.call @foo(%0#0) : (index) -> ()
  kgen.call @foo(%0#1) : (index) -> ()
  kgen.return
}
