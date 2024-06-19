// RUN: kgen-opt %s -pass-pipeline='builtin.module(kgen.func(raise-for-loops, canonicalize))' -split-input-file | FileCheck %s

// CHECK-LABEL: @zero_starting_range
kgen.func @zero_starting_range() {
  %index2 = index.constant 2
  %idx0 = index.constant 0
  %index1 = index.constant 1

  // CHECK:      [[INDEX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[INDEX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT: hlcf.for [[[INDEX2]] to [[INDEX0]] step [[INDEX1]] sgt sub] (%arg0 = [[INDEX2]] : index) {
  // CHECK-NEXT:   [[IDX:%.*]] = index.sub %arg0, [[INDEX1]]
  // CHECK-NEXT:   [[V:%.*]] = index.sub [[INDEX2]], %arg0
  // CHECK-NEXT:   kgen.call @foo([[V]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

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
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return
}

// CHECK-LABEL: @sequential_range
kgen.func @sequential_range() {
  %index1 = index.constant 1
  %index4 = index.constant 4

  // CHECK:      [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT: [[INDEX4:%.*]] = index.constant 4
  // CHECK-NEXT: hlcf.for [[[INDEX1]] to [[INDEX4]] step [[INDEX1]] slt add] (%arg0 = [[INDEX1]] : index) {
  // CHECK-NEXT:   [[IDX:%.*]] = index.add %arg0, [[INDEX1]]
  // CHECK-NEXT:   kgen.call @foo(%arg0) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

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
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return
}

// CHECK-LABEL: @strided_range
kgen.func @strided_range() {
  %index1 = index.constant 1
  %index6 = index.constant 6
  %index2 = index.constant 2

  // CHECK:      [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT: [[INDEX6:%.*]] = index.constant 6
  // CHECK-NEXT: [[INDEX2:%.*]] = index.constant 2
  // CHECK-NEXT: hlcf.for [[[INDEX1]] to [[INDEX6]] step [[INDEX2]] slt add] (%arg0 = [[INDEX1]] : index) {
  // CHECK-NEXT:   [[IDX:%.*]] = index.add %arg0, [[INDEX2]]
  // CHECK-NEXT:   kgen.call @foo(%arg0) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

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
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return
}

// CHECK-LABEL: @nested_unroll_loops
kgen.func @nested_unroll_loops() {
  %idx0 = index.constant 0
  %index1 = index.constant 1
  %index2 = index.constant 2
  %index4 = index.constant 4
  %index8 = index.constant 8

  // CHECK:      [[INDEX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT: [[INDEX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[INDEX4:%.*]] = index.constant 4
  // CHECK-NEXT: [[INDEX8:%.*]] = index.constant 8
  // CHECK-NEXT: hlcf.for [[[INDEX2]] to [[INDEX0]] step [[INDEX1]] sgt sub] (%arg0 = [[INDEX2]] : index) {
  // CHECK-NEXT:   [[IDX0:%.*]] = index.sub %arg0, [[INDEX1]]
  // CHECK-NEXT:   [[V0:%.*]]  = index.sub [[INDEX2]], %arg0
  // CHECK-NEXT:   kgen.call @foo([[V0]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.for [[[INDEX4]] to [[INDEX8]] step [[INDEX2]] slt add] (%arg1 = [[INDEX4]] : index) {
  // CHECK-NEXT:     [[IDX1:%.*]]  = index.add %arg1, [[INDEX2]]
  // CHECK-NEXT:     [[V1:%.*]] = index.add %1, %arg1
  // CHECK-NEXT:     kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT:     hlcf.for.yield [induction_var ([[IDX1]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT:   } {unrollLevel = #hlcf<unroll_level full>}
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX0]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

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
    } {unrollLevel = #hlcf<unroll_level full>}
    hlcf.continue %1 : index
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return
}

// CHECK-LABEL: @zero_starting_range_not_decorated
kgen.func @zero_starting_range_not_decorated() {
  %index2 = index.constant 2
  %idx0 = index.constant 0
  %index1 = index.constant 1
  // CHECK:      [[INDEX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[INDEX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT: hlcf.for [[[INDEX2]] to [[INDEX0]] step [[INDEX1]] sgt sub] (%arg0 = [[INDEX2]] : index) {
  // CHECK-NEXT:   [[IDX:%.*]] = index.sub %arg0, [[INDEX1]]
  // CHECK-NEXT:   [[V:%.*]] = index.sub [[INDEX2]], %arg0
  // CHECK-NEXT:   kgen.call @foo([[V]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: }
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
  }
  kgen.return
}

// CHECK-LABEL: @loop_carried_dependency
kgen.func @loop_carried_dependency() {
  %index1 = index.constant 1
  %index9 = index.constant 9
  %index2 = index.constant 2
  %index4 = index.constant 4
  %index8 = index.constant 8
  %index0 = index.constant 0

  // CHECK:      [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT: [[INDEX9:%.*]] = index.constant 9
  // CHECK-NEXT: [[INDEX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[INDEX4:%.*]] = index.constant 4
  // CHECK-NEXT: [[INDEX8:%.*]] = index.constant 8
  // CHECK-NEXT: [[INDEX0:%.*]] = index.constant 0
  // CHECK-NEXT: %0:2 = hlcf.for [[[INDEX1]] to [[INDEX9]] step [[INDEX2]] slt add] (%arg0 = [[INDEX1]]  : index, %arg1 = [[INDEX0]] : index, %arg2 = [[INDEX0]]  : index) -> (index, index) {
  // CHECK-NEXT:   [[IDX0:%.*]] = index.add %arg0, [[INDEX2]]
  // CHECK-NEXT:   kgen.call @foo(%arg0) : (index) -> ()
  // CHECK-NEXT:   [[V0:%.*]] = index.add %arg1, %arg0
  // CHECK-NEXT:   [[V1:%.*]] = hlcf.for [[[INDEX4]] to [[INDEX8]] step [[INDEX2]] slt add] (%arg3 =  [[INDEX4]] : index, %arg4 = %arg2 : index) -> index {
  // CHECK-NEXT:     [[V4:%.*]] = index.add %arg3, [[INDEX2]]
  // CHECK-NEXT:     [[V2:%.*]] = index.add %arg0, %arg3
  // CHECK-NEXT:     kgen.call @foo([[V2]]) : (index) -> ()
  // CHECK-NEXT:     [[V3:%.*]] = index.add %arg4, %arg3
  // CHECK-NEXT:     hlcf.for.yield [induction_var ([[V4]] : index)] [retvals ([[V3]] : index)] [iterargs ()]
  // CHECK-NEXT:   } {unrollLevel = #hlcf<unroll_level full>}
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX0]] : index)] [retvals ([[V0]], [[V1]] : index, index)] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

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
    } {unrollLevel = #hlcf<unroll_level full>}
    hlcf.continue %6, %7, %4 : index, index, index
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.call @foo(%0#0) : (index) -> ()
  kgen.call @foo(%0#1) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: @reorder_args
kgen.func @reorder_args(%arg0: !kgen.struct<(pointer<scalar<f32>>, index, dtype)>) -> index {
  %index10 = index.constant 10
  %index1 = index.constant 1
  %index0 = index.constant 0
  %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(pointer<scalar<f32>>, index, dtype)>

  // CHECK:       [[INDEX10:%.*]] = index.constant 10
  // CHECK-NEXT:  [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT:  [[INDEX0:%.*]] = index.constant 0
  // CHECK-NEXT:  [[V0:%.*]] = kgen.struct.extract %arg0[0] : !kgen.struct<(pointer<scalar<f32>>, index, dtype)>
  // CHECK-NEXT:  [[V1:%.*]] = hlcf.for [[[INDEX10]] to [[INDEX0]] step [[INDEX1]] sgt sub] (%arg1 = [[INDEX10]] : index, %arg2 = [[INDEX0]] : index, %arg3 = [[V0]] : !kgen.pointer<scalar<f32>>) -> index {
  // CHECK-NEXT:   [[V2:%.*]] = index.sub %arg1, [[INDEX1]]
  // CHECK-NEXT:   [[V3:%.*]] = pop.load %arg3 align<1> : !kgen.pointer<scalar<f32>>
  // CHECK-NEXT:   [[V4:%.*]] = pop.cast [[V3]] : !pop.scalar<f32> to !pop.scalar<index>
  // CHECK-NEXT:   [[V5:%.*]] = pop.cast_to_builtin [[V4]] : !pop.scalar<index> to index
  // CHECK-NEXT:   [[V6:%.*]] = index.add %arg2, [[V5]]
  // CHECK-NEXT:   [[V7:%.*]] = pop.offset %arg3[[[INDEX1]]] : !kgen.pointer<scalar<f32>>
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[V2]] : index)] [retvals ([[V6]] : index)] [iterargs ([[V7]] : !kgen.pointer<scalar<f32>>)]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

  %1 = hlcf.loop (%arg3 = %index10 : index, %arg1 = %0 : !kgen.pointer<scalar<f32>>, %arg2 = %index0 : index) -> index {
    %2 = index.cmp sgt(%arg3, %index0)
    hlcf.if %2 {
      hlcf.yield
    } else {
      hlcf.break %arg2 : index
    }
    %3 = index.sub %arg3, %index1
    %4 = pop.load %arg1 align<1> : !kgen.pointer<scalar<f32>>
    %5 = pop.cast %4 : !pop.scalar<f32> to !pop.scalar<index>
    %6 = pop.cast_to_builtin %5 : !pop.scalar<index> to index
    %7 = index.add %arg2, %6
    %8 = pop.offset %arg1[%index1] : !kgen.pointer<scalar<f32>>
    hlcf.continue %3, %8, %7 : index, !kgen.pointer<scalar<f32>>, index
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return %1 : index
}

// CHECK-LABEL: @complex_exit_logic_no_raise
kgen.func @complex_exit_logic_no_raise() {
  %idx2 = index.constant 2
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  // CHECK-NOT: hlcf.for
  hlcf.loop (%arg0 = %idx2 : index) {
    %0 = index.cmp sgt(%arg0, %idx0)
    hlcf.if %0 {
      hlcf.yield
    } else {
      kgen.call @bar(%arg0) : (index) -> ()
      hlcf.break
    }
    %1 = index.sub %arg0, %idx1
    kgen.call @foo(%1) : (index) -> ()
    hlcf.continue %1 : index
  }
  kgen.return
}

// CHECK-LABEL: @negative_step
kgen.func @negative_step() {
  %index5 = index.constant 5
  %index1 = index.constant 1
  %index-1 = index.constant -1

  // CHECK:       [[INDEX5:%.*]] = index.constant 5
  // CHECK-NEXT:  [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT:  [[INDEXN1:%.*]] = index.constant -1
  // CHECK-NEXT: hlcf.for [[[INDEX5]] to [[INDEX1]] step [[INDEXN1]]  sgt add] (%arg0 = [[INDEX5]] : index) {
  // CHECK-NEXT:   %0 = index.add %arg0, [[INDEXN1]]
  // CHECK-NEXT:   kgen.call @foo(%arg0) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.yield [induction_var (%0 : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

  hlcf.loop (%arg0 = %index5 : index) {
    %3 = index.cmp sgt(%arg0, %index1)
    hlcf.if %3 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %4 = index.add %arg0, %index-1
    kgen.call @foo(%arg0) : (index) -> ()
    hlcf.continue %4 : index
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return
}

// CHECK-LABEL: @nested_loops_no_unroll_inner
kgen.func @nested_loops_no_unroll_inner() {
  %index0 = index.constant 0
  %index1 = index.constant 1
  %index2 = index.constant 2

  // CHECK:      [[INDEX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT: [[INDEX2:%.*]] = index.constant 2
  // CHECK-NEXT: hlcf.for [[[INDEX2]] to [[INDEX0]] step [[INDEX1]] sgt sub] (%arg0 = [[INDEX2]] : index) {
  // CHECK-NEXT:   [[IDX0:%.*]] = index.sub %arg0, [[INDEX1]]
  // CHECK-NEXT:   [[V0:%.*]]  = index.sub [[INDEX2]], %arg0
  // CHECK-NEXT:   kgen.call @foo([[V0]]) : (index) -> ()
  // CHECK-NOT:    hlcf.for
  // CHECK-DAG:   hlcf.for.yield [induction_var ([[IDX0]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: }

  hlcf.loop (%arg0 = %index2 : index) {
    %0 = index.cmp sgt(%arg0, %index0)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.sub %arg0, %index1
    %2 = index.sub %index2, %arg0
    kgen.call @foo(%2) : (index) -> ()
    // Cannot raise this loop to a for-loop because the upper bound is %arg0 which changes
    // over parent loop's iterations.
    hlcf.loop (%arg1 = %index1 : index) {
      %4 = index.cmp slt(%arg1, %arg0)
      hlcf.if %4 {
        hlcf.yield
      } else {
        hlcf.break
      }
      %5 = index.add %arg1, %index2
      %6 = index.add %2, %arg1
      kgen.call @foo(%6) : (index) -> ()
      hlcf.continue %5 : index
    }
    hlcf.continue %1 : index
  }
  kgen.return
}

// CHECK-LABEL: @break_in_then
 kgen.func @break_in_then() {
   %index0 = index.constant 0
   %index1 = index.constant 1
   %index10 = index.constant 10

   // CHECK:      [[INDEX0:%.*]] = index.constant 0
   // CHECK-NEXT: [[INDEX1:%.*]] = index.constant 1
   // CHECK-NEXT: [[INDEX10:%.*]] = index.constant 10
   // CHECK-NEXT: hlcf.for [[[INDEX0]] to [[INDEX10]] step [[INDEX1]] sle add] (%arg0 = [[INDEX0]] : index)
   hlcf.loop (%arg0 = %index0 : index) {
     // when for-loop is raise, this condition will be inverted to sle
     %1 = index.cmp sgt(%arg0, %index10)
     hlcf.if %1 {
       hlcf.break
     } else {
       hlcf.yield
     }
     %2 = index.add %arg0, %index1
     hlcf.continue %2 : index
   }
   kgen.return
 }

// CHECK-LABEL: @return_value_same_as_iter_var
kgen.func @return_value_same_as_iter_var()  {
  %index0 = index.constant 0
  %index1 = index.constant 1
  %index2 = index.constant 2

  // CHECK:      [[INDEX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT: [[INDEX2:%.*]] = index.constant 2
  // CHECK-NEXT: hlcf.for [[[INDEX0]] to [[INDEX2]] step [[INDEX1]] slt add] (%arg0 = [[INDEX0]] : index, %arg1 = [[INDEX0]] : index) -> index
  // CHECK-NEXT:  [[V1:%.*]] = index.add %arg0, [[INDEX1]]
  // CHECK-NEXT:  hlcf.for.yield [induction_var ([[V1]] : index)] [retvals ([[V1]] : index)] [iterargs ()]

  %0 = hlcf.loop (%arg0 = %index0 : index) -> index {
    // loop return value is the same as iterVar: %arg0
    %2 = index.cmp slt(%arg0, %index2)
    hlcf.if %2 {
      hlcf.yield
    } else {
      hlcf.break %arg0 : index
    }
    %3 = index.add %arg0, %index1
    hlcf.continue %3 : index
  }
  kgen.return
}

// CHECK-LABEL: @stride_same_as_iter_var
kgen.func @stride_same_as_iter_var()  {
  %index0 = index.constant 0
  %index1 = index.constant 1
  // CHECK-NOT: hlcf.for
  %0 = hlcf.loop (%arg0 = %index0 : index) -> index {
    // loop stride value is the same as iterVar: %arg0
    %2 = index.cmp slt(%arg0, %index1)
    hlcf.if %2 {
      hlcf.yield
    } else {
      hlcf.break %arg0 : index
    }
    %3 = index.add %arg0, %arg0
    hlcf.continue %3 : index
  }
  kgen.return
}

// CHECK-LABEL: @non_const_loop_end
kgen.func @non_const_loop_end()  {
  %index0 = index.constant 0
  %index1 = index.constant 1
  // CHECK-NOT: hlcf.for
  %0 = hlcf.loop (%arg0 = %index0 : index, %arg1 = %index1 : index) -> index {
    // loop end is not always constant
    %2 = index.cmp slt(%arg0, %arg1)
    hlcf.if %2 {
      hlcf.yield
    } else {
      hlcf.break %arg0 : index
    }
    %3 = index.add %arg0, %index1
    hlcf.continue %3, %3 : index, index
  }
  kgen.return
}

// -----

// Tests for handling ops inside the conditional break branch of the loop.

// CHECK-LABEL: @simple_call_in_break_branch
kgen.func @simple_call_in_break_branch() {
  %index0 = index.constant 0
  %index1 = index.constant 1
  %index10 = index.constant 10

  // CHECK:      [[INDEX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT: [[INDEX10:%.*]] = index.constant 10
  // CHECK-NEXT: hlcf.for [[[INDEX0]] to [[INDEX10]] step [[INDEX1]] sgt add] (%arg0 = [[INDEX0]] : index)
  // CHECK:        hlcf.for.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: kgen.call @foo([[INDEX10]])
  hlcf.loop (%arg0 = %index0 : index) {
    %1 = index.cmp sgt(%arg0, %index10)
    hlcf.if %1 {
      hlcf.yield
    } else {
      kgen.call @foo(%index10) : (index) -> ()
      hlcf.break
    }
    %2 = index.add %arg0, %index1
    hlcf.continue %2 : index
  }
  kgen.return
}

// CHECK-LABEL: @intermediate_values_in_break_branch
kgen.func @intermediate_values_in_break_branch() {
  %index0 = index.constant 0
  %index1 = index.constant 1
  %index10 = index.constant 10

  // CHECK:      [[INDEX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[INDEX1:%.*]] = index.constant 1
  // CHECK-NEXT: [[INDEX10:%.*]] = index.constant 10
  // CHECK-NEXT: hlcf.for [[[INDEX0]] to [[INDEX10]] step [[INDEX1]] sgt add] (%arg0 = [[INDEX0]] : index)
  // CHECK:        hlcf.for.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V0:%.*]] = kgen.call @foo([[INDEX10]])
  // CHECK-NEXT: kgen.call @bar([[V0]])
  hlcf.loop (%arg0 = %index0 : index) {
    %1 = index.cmp sgt(%arg0, %index10)
    hlcf.if %1 {
      hlcf.yield
    } else {
      %v0 = kgen.call @foo(%index10) : (index) -> (index)
      kgen.call @bar(%v0) : (index) -> ()
      hlcf.break
    }
    %2 = index.add %arg0, %index1
    hlcf.continue %2 : index
  }
  kgen.return
}

// CHECK-LABEL: @break_dependent_on_break_branch
kgen.func @break_dependent_on_break_branch() {
  %index0 = index.constant 0
  %index1 = index.constant 1
  %index10 = index.constant 10

  // CHECK-NOT: hlcf.for
  %loop = hlcf.loop (%arg0 = %index0 : index) -> index {
    %1 = index.cmp sgt(%arg0, %index10)
    hlcf.if %1 {
      hlcf.yield
    } else {
      // Break is dependent on intermediate results from this branch. Abort.
      %v0 = kgen.call @foo(%index10) : (index) -> (index)
      hlcf.break %v0 : index
    }
    %2 = index.add %arg0, %index1
    hlcf.continue %2 : index
  }
  kgen.return
}

// CHECK-LABEL: @dependent_ops_in_break_branch
kgen.func @dependent_ops_in_break_branch() {
  %index0 = index.constant 0
  %index1 = index.constant 1
  %index10 = index.constant 10

  // CHECK-NOT: hlcf.for
  hlcf.loop (%arg0 = %index0 : index) {
    %1 = index.cmp sgt(%arg0, %index10)
    hlcf.if %1 {
      hlcf.yield
    } else {
      // Op depends on internal value. Can no longer convert.
      kgen.call @foo(%arg0) : (index) -> ()
      hlcf.break
    }
    %2 = index.add %arg0, %index1
    hlcf.continue %2 : index
  }
  kgen.return
}

// -----

// COM: MOCO-718 fix test.
// CHECK-LABEL: @donnot_crash_with_block_argument_cond()
kgen.func @donnot_crash_with_block_argument_cond() {
  %0 = kgen.param.constant: i1 = <0>
  %1 = kgen.param.constant: i1 = <1>
  hlcf.loop "_loop" (%arg2 = %1 : i1) {
    hlcf.if %arg2 {
      hlcf.yield
    } else {
      hlcf.break "_loop"
    }
    hlcf.continue %0 : i1
  }
  kgen.return
}
