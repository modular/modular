// RUN: kgen-opt %s -pass-pipeline='builtin.module(kgen.func(loop-unrolling, canonicalize))' | FileCheck %s

// CHECK-LABEL: @zero_starting_range
kgen.func @zero_starting_range() {
  // CHECK: [[V1:%.*]] = index.constant 1
  // CHECK-NEXT: [[V0:%.*]] = index.constant 0
  // CHECK-NOT: hlcf.for
  // CHECK-NEXT: kgen.call @foo([[V0]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V1]]) : (index) -> ()

  %index2 = index.constant 2
  %idx0 = index.constant 0
  %index1 = index.constant 1
  hlcf.for [%idx0 to %index2 step %index1] (%arg0 = %index2 : index) {
    %0 = index.sub %arg0, %index1
    %1 = index.sub %index2, %arg0
    kgen.call @foo(%1) : (index) -> ()
    hlcf.for.yield [induction_var (%0 : index)] [retvals ()] [iterargs ()]
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return
}

// CHECK-LABEL: @sequential_range
kgen.func @sequential_range() {
  // CHECK: [[V2:%.*]] = index.constant 3
  // CHECK-NEXT: [[V1:%.*]] = index.constant 2
  // CHECK-NEXT: [[V0:%.*]] = index.constant 1
  // CHECK-NOT: hlcf.for
  // CHECK-NEXT: kgen.call @foo([[V0]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V2]]) : (index) -> ()

  %index1 = index.constant 1
  %index4 = index.constant 4
  hlcf.for [%index1 to %index4 step %index1] (%arg0 = %index1 : index) {
    %0 = index.add %arg0, %index1
    kgen.call @foo(%arg0) : (index) -> ()
    hlcf.for.yield [induction_var (%0 : index)] [retvals ()] [iterargs ()]
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return
}

// CHECK-LABEL: @strided_range
kgen.func @strided_range() {
  // CHECK: [[V0:%.*]] = index.constant 5
  // CHECK-NEXT: [[V1:%.*]] = index.constant 3
  // CHECK-NEXT: [[V2:%.*]] = index.constant 1
  // CHECK-NOT: hlcf.for
  // CHECK-NEXT: kgen.call @foo([[V2]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V0]]) : (index) -> ()

  %index1 = index.constant 1
  %index6 = index.constant 6
  %index2 = index.constant 2
  hlcf.for [%index1 to %index6 step %index2] (%arg0 = %index1 : index) {
    %0 = index.add %arg0, %index2
    kgen.call @foo(%arg0) : (index) -> ()
    hlcf.for.yield [induction_var (%0 : index)] [retvals ()] [iterargs ()]
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return
}

// CHECK-LABEL: @nested_unroll_loops
kgen.func @nested_unroll_loops() {
  // CHECK:  [[V0:%.*]] = index.constant 7
  // CHECK-NEXT:  [[V1:%.*]] = index.constant 5
  // CHECK-NEXT:  [[V2:%.*]] = index.constant 6
  // CHECK-NEXT:  [[V3:%.*]] = index.constant 1
  // CHECK-NEXT:  [[V4:%.*]] = index.constant 4
  // CHECK-NEXT:  [[V5:%.*]] = index.constant 0
  // CHECK-NOT: hlcf.for
  // CHECK-NEXT:  kgen.call @foo([[V5]]) : (index) -> ()
  // CHECK-NEXT:  kgen.call @foo([[V4]]) : (index) -> ()
  // CHECK-NEXT:  kgen.call @foo([[V2]]) : (index) -> ()
  // CHECK-NEXT:  kgen.call @foo([[V3]]) : (index) -> ()
  // CHECK-NEXT:  kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT:  kgen.call @foo([[V0]]) : (index) -> ()

  %index2 = index.constant 2
  %index4 = index.constant 4
  %index8 = index.constant 8
  %idx0 = index.constant 0
  %index1 = index.constant 1
  hlcf.for [%idx0 to %index2 step %index1] (%arg0 = %index2 : index) {
    %0 = index.sub %arg0, %index1
    %1 = index.sub %index2, %arg0
    kgen.call @foo(%1) : (index) -> ()
    hlcf.for [%index4 to %index8 step %index2] (%arg1 = %index4 : index) {
      %3 = index.add %arg1, %index2
      %4 = index.add %1, %arg1
      kgen.call @foo(%4) : (index) -> ()
      hlcf.for.yield [induction_var (%3 : index)] [retvals ()] [iterargs ()]
    } {unrollFactor = #hlcf<loop_unroll_full full>}
    hlcf.for.yield [induction_var (%0 : index)] [retvals ()] [iterargs ()]
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return
}

// CHECK-LABEL: @loop_carried_dependency
kgen.func @loop_carried_dependency() {
  // CHECK: [[V0:%.*]] = index.constant 40
  // CHECK-NEXT: [[V1:%.*]] = index.constant 13
  // CHECK-NEXT: [[V2:%.*]] = index.constant 11
  // CHECK-NEXT: [[V3:%.*]] = index.constant 16
  // CHECK-NEXT: [[V4:%.*]] = index.constant 9
  // CHECK-NEXT: [[V5:%.*]] = index.constant 7
  // CHECK-NEXT: [[V6:%.*]] = index.constant 5
  // CHECK-NEXT: [[V8:%.*]] = index.constant 1
  // CHECK-NEXT: [[V7:%.*]] = index.constant 3
  // CHECK-NOT: hlcf.for
  // CHECK-NEXT: kgen.call @foo([[V8]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V6]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V5]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V7]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V5]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V4]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V6]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V4]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V2]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V5]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V2]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V3]]) : (index) -> ()
  // CHECK-NEXT: kgen.call @foo([[V0]]) : (index) -> ()

  %index1 = index.constant 1
  %index9 = index.constant 9
  %index2 = index.constant 2
  %index4 = index.constant 4
  %index8 = index.constant 8
  %index0 = index.constant 0
  %0:2 = hlcf.for [%index1 to %index9 step %index2] (%arg2 = %index1 : index, %arg0 = %index0 : index, %arg1 = %index0 : index) -> (index, index) {
    %3 = index.add %arg2, %index2
    kgen.call @foo(%arg2) : (index) -> ()
    %5 = index.add %arg0, %arg2
    %6 = hlcf.for [%index4 to %index8 step %index2] (%arg4 = %index4 : index, %arg3 = %arg1 : index) -> index {
      %7 = index.add %arg4, %index2
      %8 = index.add %arg2, %arg4
      kgen.call @foo(%8) : (index) -> ()
      %10 = index.add %arg3, %arg4
      hlcf.for.yield [induction_var (%7 : index)] [retvals (%10: index)] [iterargs ()]
    } {unrollFactor = #hlcf<loop_unroll_full full>}
    hlcf.for.yield [induction_var (%3 : index)] [retvals (%5: index)] [iterargs (%6: index)]
  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.call @foo(%0#0) : (index) -> ()
  kgen.call @foo(%0#1) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: @loop_has_side_effect
kgen.func @loop_has_side_effect(%arg0: !pop.struct<pointer<scalar<f32>>, index, dtype>) -> index {
  // CHECK: [[IDX:%.*]] = index.constant 1
  // CHECK-NEXT: [[V0:%.*]] = pop.struct.extract %arg0[0] : !pop.struct<pointer<scalar<f32>>, index, dtype>
  // CHECK-NEXT: [[V1:%.*]] = pop.load [[V0]] align 1  : !pop.pointer<scalar<f32>>
  // CHECK-NEXT: [[V2:%.*]] = pop.cast [[V1]] : !pop.scalar<f32> to !pop.scalar<index>
  // CHECK-NEXT: [[V3:%.*]] = pop.cast_to_builtin [[V2]] : !pop.scalar<index> to index
  // CHECK-NEXT: [[V4:%.*]] = pop.offset %0[[[IDX]]] : !pop.pointer<scalar<f32>>
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] align 1  : !pop.pointer<scalar<f32>>
  // CHECK-NEXT: [[V6:%.*]] = pop.cast [[V5]] : !pop.scalar<f32> to !pop.scalar<index>
  // CHECK-NEXT: [[V7:%.*]] = pop.cast_to_builtin [[V6]] : !pop.scalar<index> to index
  // CHECK-NEXT: [[V8:%.*]] = index.add [[V3]], [[V7]]
  // CHECK-NEXT: kgen.return [[V8]] : index

  %index10 = index.constant 2
  %idx0 = index.constant 0
  %index1 = index.constant 1
  %index0 = index.constant 0
  %0 = pop.struct.extract %arg0[0] : !pop.struct<pointer<scalar<f32>>, index, dtype>
  %1 = hlcf.for [%idx0 to %index10 step %index1] (%arg3 = %index10 : index, %arg1 = %index0 : index, %arg2 = %0 : !pop.pointer<scalar<f32>>) -> index {
    %2 = index.cmp sgt(%arg3, %idx0)
    %3 = index.sub %arg3, %index1
    %4 = pop.load %arg2 align 1  : !pop.pointer<scalar<f32>>
    %5 = pop.cast %4 : !pop.scalar<f32> to !pop.scalar<index>
    %6 = pop.cast_to_builtin %5 : !pop.scalar<index> to index
    %7 = index.add %arg1, %6
    %8 = pop.offset %arg2[%index1] : !pop.pointer<scalar<f32>>
    hlcf.for.yield [induction_var (%3 : index)] [retvals (%7: index)] [iterargs (%8: !pop.pointer<scalar<f32>>)]

  } {unrollFactor = #hlcf<loop_unroll_full full>}
  kgen.return %1 : index
}

// CHECK-LABEL: @single_iteration_no_decorator
kgen.func @single_iteration_no_decorator(%arg0: !pop.struct<pointer<scalar<f32>>, index, dtype>) -> index {
  // CHECK:      [[V0:%.*]] = pop.struct.extract %arg0[0] : !pop.struct<pointer<scalar<f32>>, index, dtype>
  // CHECK-NEXT: [[V1:%.*]] = pop.load [[V0]] align 1  : !pop.pointer<scalar<f32>>
  // CHECK-NEXT: [[V2:%.*]] = pop.cast [[V1]] : !pop.scalar<f32> to !pop.scalar<index>
  // CHECK-NEXT: [[V3:%.*]] = pop.cast_to_builtin [[V2]] : !pop.scalar<index> to index
  // CHECK-NEXT: kgen.return [[V3]] : index

  %idx0 = index.constant 0
  %index1 = index.constant 1
  %index0 = index.constant 0
  %0 = pop.struct.extract %arg0[0] : !pop.struct<pointer<scalar<f32>>, index, dtype>
  %1 = hlcf.for [%idx0 to %index1 step %index1] (%arg3 = %index1 : index, %arg1 = %index0 : index, %arg2 = %0 : !pop.pointer<scalar<f32>>) -> index {
    %2 = index.cmp sgt(%arg3, %idx0)
    %3 = index.sub %arg3, %index1
    %4 = pop.load %arg2 align 1  : !pop.pointer<scalar<f32>>
    %5 = pop.cast %4 : !pop.scalar<f32> to !pop.scalar<index>
    %6 = pop.cast_to_builtin %5 : !pop.scalar<index> to index
    %7 = index.add %arg1, %6
    %8 = pop.offset %arg2[%index1] : !pop.pointer<scalar<f32>>
    hlcf.for.yield [induction_var (%3 : index)] [retvals (%7: index)] [iterargs (%8: !pop.pointer<scalar<f32>>)]

  } {unrollFactor = #hlcf<loop_unroll_full none>}
  kgen.return %1 : index
}
