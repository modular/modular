// RUN: kgen-opt %s -pass-pipeline='builtin.module(kgen.func(loop-unrolling, canonicalize))' | FileCheck %s

// CHECK-LABEL: @zero_starting_range
kgen.func @zero_starting_range() -> !pop.array<0, i1> {
  // CHECK: %idx1 = index.constant 1
  // CHECK-NEXT: %idx0 = index.constant 0
  // CHECK-NEXT: %array = kgen.param.constant: array<0, i1> = <[]>
  // CHECK-NOT: hlcf.for
  // CHECK-NEXT: %0 = kgen.call @"$IO::print($Int::Int)"(%idx0) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %1 = kgen.call @"$IO::print($Int::Int)"(%idx1) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: kgen.return %array : !pop.array<0, i1>
  %index2 = kgen.param.constant = <2>
  %idx0 = index.constant 0
  %index1 = kgen.param.constant = <1>
  %array = kgen.param.constant: array<0, i1> = <[]>
  hlcf.for [%idx0 to %index2 step %index1] (%arg0 = %index2 : index) {
    %0 = index.sub %arg0, %index1
    %1 = index.sub %index2, %arg0
    %2 = kgen.call @"$IO::print($Int::Int)"(%1) : (index) -> !pop.array<0, i1>
    hlcf.for.yield %0 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return %array : !pop.array<0, i1>
}

// CHECK-LABEL: @sequential_range
kgen.func @sequential_range() -> !pop.array<0, i1> {
  // CHECK: %idx3 = index.constant 3
  // CHECK-NEXT: %idx2 = index.constant 2
  // CHECK-NEXT: %index1 = kgen.param.constant = <1>
  // CHECK-NEXT: %array = kgen.param.constant: array<0, i1> = <[]>
  // CHECK-NOT: hlcf.for
  // CHECK-NEXT: %0 = kgen.call @"$IO::print($Int::Int)"(%index1) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %1 = kgen.call @"$IO::print($Int::Int)"(%idx2) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %2 = kgen.call @"$IO::print($Int::Int)"(%idx3) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: kgen.return %array : !pop.array<0, i1>

  %index1 = kgen.param.constant = <1>
  %index4 = kgen.param.constant = <4>
  %array = kgen.param.constant: array<0, i1> = <[]>
  hlcf.for [%index1 to %index4 step %index1] (%arg0 = %index1 : index) {
    %0 = index.add %arg0, %index1
    %1 = kgen.call @"$IO::print($Int::Int)"(%arg0) : (index) -> !pop.array<0, i1>
    hlcf.for.yield %0 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return %array : !pop.array<0, i1>
}

// CHECK-LABEL: @strided_range
kgen.func @strided_range() -> !pop.array<0, i1> {
  // CHECK: %idx5 = index.constant 5
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: %index1 = kgen.param.constant = <1>
  // CHECK-NEXT: %array = kgen.param.constant: array<0, i1> = <[]>
  // CHECK-NOT: hlcf.for
  // CHECK-NEXT: %0 = kgen.call @"$IO::print($Int::Int)"(%index1) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %1 = kgen.call @"$IO::print($Int::Int)"(%idx3) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %2 = kgen.call @"$IO::print($Int::Int)"(%idx5) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: kgen.return %array : !pop.array<0, i1>

  %index1 = kgen.param.constant = <1>
  %index6 = kgen.param.constant = <6>
  %index2 = kgen.param.constant = <2>
  %array = kgen.param.constant: array<0, i1> = <[]>
  hlcf.for [%index1 to %index6 step %index2] (%arg0 = %index1 : index) {
    %0 = index.add %arg0, %index2
    %1 = kgen.call @"$IO::print($Int::Int)"(%arg0) : (index) -> !pop.array<0, i1>
    hlcf.for.yield %0 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return %array : !pop.array<0, i1>
}

// CHECK-LABEL: @nested_unroll_loops
kgen.func @nested_unroll_loops() -> !pop.array<0, i1> {
  // CHECK:  %idx7 = index.constant 7
  // CHECK-NEXT:  %idx5 = index.constant 5
  // CHECK-NEXT:  %idx6 = index.constant 6
  // CHECK-NEXT:  %idx1 = index.constant 1
  // CHECK-NEXT:  %idx4 = index.constant 4
  // CHECK-NEXT:  %idx0 = index.constant 0
  // CHECK-NEXT:  %array = kgen.param.constant: array<0, i1> = <[]>
  // CHECK-NOT: hlcf.for
  // CHECK-NEXT:  %0 = kgen.call @"$IO::print($Int::Int)"(%idx0) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:  %1 = kgen.call @"$IO::print($Int::Int)"(%idx4) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:  %2 = kgen.call @"$IO::print($Int::Int)"(%idx6) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:  %3 = kgen.call @"$IO::print($Int::Int)"(%idx1) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:  %4 = kgen.call @"$IO::print($Int::Int)"(%idx5) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:  %5 = kgen.call @"$IO::print($Int::Int)"(%idx7) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT:  kgen.return %array : !pop.array<0, i1>

  %index2 = kgen.param.constant = <2>
  %index4 = kgen.param.constant = <4>
  %index8 = kgen.param.constant = <8>
  %idx0 = index.constant 0
  %index1 = kgen.param.constant = <1>
  %array = kgen.param.constant: array<0, i1> = <[]>
  hlcf.for [%idx0 to %index2 step %index1] (%arg0 = %index2 : index) {
    %0 = index.sub %arg0, %index1
    %1 = index.sub %index2, %arg0
    %2 = kgen.call @"$IO::print($Int::Int)"(%1) : (index) -> !pop.array<0, i1>
    hlcf.for [%index4 to %index8 step %index2] (%arg1 = %index4 : index) {
      %3 = index.add %arg1, %index2
      %4 = index.add %1, %arg1
      %5 = kgen.call @"$IO::print($Int::Int)"(%4) : (index) -> !pop.array<0, i1>
      hlcf.for.yield %3 : index
    } {unrollFactor = #hlcf<loop_unroll_full full>}
    hlcf.for.yield %0 : index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return %array : !pop.array<0, i1>
}

// CHECK-LABEL: @loop_carried_dependency
kgen.func @loop_carried_dependency() -> !pop.array<0, i1> {
  // CHECK: %idx40 = index.constant 40
  // CHECK-NEXT: %idx13 = index.constant 13
  // CHECK-NEXT: %idx11 = index.constant 11
  // CHECK-NEXT: %idx16 = index.constant 16
  // CHECK-NEXT: %idx9 = index.constant 9
  // CHECK-NEXT: %idx7 = index.constant 7
  // CHECK-NEXT: %idx5 = index.constant 5
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: %index1 = kgen.param.constant = <1>
  // CHECK-NEXT: %array = kgen.param.constant: array<0, i1> = <[]>
  // CHECK-NOT: hlcf.for
  // CHECK-NEXT: %0 = kgen.call @"$IO::print($Int::Int)"(%index1) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %1 = kgen.call @"$IO::print($Int::Int)"(%idx5) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %2 = kgen.call @"$IO::print($Int::Int)"(%idx7) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %3 = kgen.call @"$IO::print($Int::Int)"(%idx3) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %4 = kgen.call @"$IO::print($Int::Int)"(%idx7) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %5 = kgen.call @"$IO::print($Int::Int)"(%idx9) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %6 = kgen.call @"$IO::print($Int::Int)"(%idx5) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %7 = kgen.call @"$IO::print($Int::Int)"(%idx9) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %8 = kgen.call @"$IO::print($Int::Int)"(%idx11) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %9 = kgen.call @"$IO::print($Int::Int)"(%idx7) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %10 = kgen.call @"$IO::print($Int::Int)"(%idx11) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %11 = kgen.call @"$IO::print($Int::Int)"(%idx13) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %12 = kgen.call @"$IO::print($Int::Int)"(%idx16) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: %13 = kgen.call @"$IO::print($Int::Int)"(%idx40) : (index) -> !pop.array<0, i1>
  // CHECK-NEXT: kgen.return %array : !pop.array<0, i1>

  %index1 = kgen.param.constant = <1>
  %index9 = kgen.param.constant = <9>
  %index2 = kgen.param.constant = <2>
  %index4 = kgen.param.constant = <4>
  %index8 = kgen.param.constant = <8>
  %array = kgen.param.constant: array<0, i1> = <[]>
  %index0 = kgen.param.constant = <0>
  %0:2 = hlcf.for [%index1 to %index9 step %index2] (%arg0 = %index0 : index, %arg1 = %index0 : index, %arg2 = %index1 : index) -> (index, index) {
    %3 = index.add %arg2, %index2
    %4 = kgen.call @"$IO::print($Int::Int)"(%arg2) : (index) -> !pop.array<0, i1>
    %5 = index.add %arg0, %arg2
    %6 = hlcf.for [%index4 to %index8 step %index2] (%arg3 = %arg1 : index, %arg4 = %index4 : index) -> index {
      %7 = index.add %arg4, %index2
      %8 = index.add %arg2, %arg4
      %9 = kgen.call @"$IO::print($Int::Int)"(%8) : (index) -> !pop.array<0, i1>
      %10 = index.add %arg3, %arg4
      hlcf.for.yield %10, %7 : index, index
    } {unrollFactor = #hlcf<loop_unroll_full full>}
    hlcf.for.yield %5, %6, %3 : index, index, index
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  %1 = kgen.call @"$IO::print($Int::Int)"(%0#0) : (index) -> !pop.array<0, i1>
  %2 = kgen.call @"$IO::print($Int::Int)"(%0#1) : (index) -> !pop.array<0, i1>
  kgen.return %array : !pop.array<0, i1>
}
