// RUN: kgen-opt %s -pass-pipeline='builtin.module(kgen.func(raise-for-loops, canonicalize))' | FileCheck %s

// CHECK-LABEL: @zero_starting_range
kgen.func @zero_starting_range() -> !pop.array<0, i1> {
  %index2 = kgen.param.constant = <2>
  %idx0 = index.constant 0
  %index1 = kgen.param.constant = <1>
  %array = kgen.param.constant: array<0, i1> = <[]>

  // CHECK: hlcf.for [%idx0 to %index2 step %index1] (%arg0 = %index2 : index) {
  // CHECK-NEXT:   %0 = index.sub %arg0, %index1
  // CHECK-NEXT:   %1 = index.sub %index2, %arg0
  // CHECK-NEXT:   %2 = kgen.call @"$IO::print($Int::Int)"(%1) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:   hlcf.for.yield %0 : index
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
    %3 = kgen.call @"$IO::print($Int::Int)"(%2) : (index) -> !pop.array<0, i1>
    hlcf.continue %1 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return %array : !pop.array<0, i1>
}

// CHECK-LABEL: @sequential_range
kgen.func @sequential_range() -> !pop.array<0, i1> {
  %index1 = kgen.param.constant = <1>
  %index4 = kgen.param.constant = <4>
  %array = kgen.param.constant: array<0, i1> = <[]>

  // CHECK: hlcf.for [%index1 to %index4 step %index1] (%arg0 = %index1 : index) {
  // CHECK-NEXT:   %0 = index.add %arg0, %index1
  // CHECK-NEXT:   %1 = kgen.call @"$IO::print($Int::Int)"(%arg0) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:   hlcf.for.yield %0 : index
  // CHECK-NEXT: } {unrollFactor = #hlcf<loop_unroll_full full>}

  hlcf.loop (%arg0 = %index1 : index) {
    %0 = index.cmp slt(%arg0, %index4)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.add %arg0, %index1
    %2 = kgen.call @"$IO::print($Int::Int)"(%arg0) : (index) -> !pop.array<0, i1>
    hlcf.continue %1 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return %array : !pop.array<0, i1>
}

// CHECK-LABEL: @strided_range
kgen.func @strided_range() -> !pop.array<0, i1> {
  %index1 = kgen.param.constant = <1>
  %index6 = kgen.param.constant = <6>
  %index2 = kgen.param.constant = <2>

  // CHECK: hlcf.for [%index1 to %index6 step %index2] (%arg0 = %index1 : index) {
  // CHECK-NEXT:   %0 = index.add %arg0, %index2
  // CHECK-NEXT:   %1 = kgen.call @"$IO::print($Int::Int)"(%arg0) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:   hlcf.for.yield %0 : index
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
    %2 = kgen.call @"$IO::print($Int::Int)"(%arg0) : (index) -> !pop.array<0, i1>
    hlcf.continue %1 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return %array : !pop.array<0, i1>
}

// CHECK-LABEL: @nested_unroll_loops
kgen.func @nested_unroll_loops() -> !pop.array<0, i1> {
  %index2 = kgen.param.constant = <2>
  %index4 = kgen.param.constant = <4>
  %index8 = kgen.param.constant = <8>
  %idx0 = index.constant 0
  %index1 = kgen.param.constant = <1>
  %array = kgen.param.constant: array<0, i1> = <[]>

  // CHECK: hlcf.for [%idx0 to %index2 step %index1] (%arg0 = %index2 : index) {
  // CHECK-NEXT:   %0 = index.sub %arg0, %index1
  // CHECK-NEXT:   %1 = index.sub %index2, %arg0
  // CHECK-NEXT:   %2 = kgen.call @"$IO::print($Int::Int)"(%1) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:   hlcf.for [%index4 to %index8 step %index2] (%arg1 = %index4 : index) {
  // CHECK-NEXT:     %3 = index.add %arg1, %index2
  // CHECK-NEXT:     %4 = index.add %1, %arg1
  // CHECK-NEXT:     %5 = kgen.call @"$IO::print($Int::Int)"(%4) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:     hlcf.for.yield %3 : index
  // CHECK-NEXT:  } {unrollFactor = #hlcf<loop_unroll_full full>}
  // CHECK-NEXT:   hlcf.for.yield %0 : index
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
    %3 = kgen.call @"$IO::print($Int::Int)"(%2) : (index) -> !pop.array<0, i1>
    hlcf.loop (%arg1 = %index4 : index) {
      %4 = index.cmp slt(%arg1, %index8)
      hlcf.if %4 {
        hlcf.yield
      } else {
        hlcf.break
      }
      %5 = index.add %arg1, %index2
      %6 = index.add %2, %arg1
      %7 = kgen.call @"$IO::print($Int::Int)"(%6) : (index) -> !pop.array<0, i1>
      hlcf.continue %5 : index
    } {unrollFactor = #hlcf<loop_unroll_full full>}
    hlcf.continue %1 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return %array : !pop.array<0, i1>
}

// For-loop not raised because loop has no unrollFactor
// CHECK-LABEL: @zero_starting_range_no_raise
kgen.func @zero_starting_range_no_raise() -> !pop.array<0, i1> {
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
    %3 = kgen.call @"$IO::print($Int::Int)"(%2) : (index) -> !pop.array<0, i1>
    hlcf.continue %1 : index
  }
  kgen.return %array : !pop.array<0, i1>
}
