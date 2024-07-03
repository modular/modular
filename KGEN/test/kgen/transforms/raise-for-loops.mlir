// RUN: kgen-opt %s -raise-for-loops -split-input-file | FileCheck %s

// CHECK-LABEL: @zero_starting_range
kgen.func @zero_starting_range() {
  %idx2 = index.constant 2
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  // CHECK: hlcf.for [%idx2 to %idx0 step %idx1 sgt sub] (%arg0 = %idx2 : index) {
  // CHECK:        [[IDX:%.*]] = index.sub %arg0, %idx1
  // CHECK-NEXT:   [[V:%.*]] = index.sub %idx2, %arg0
  // CHECK-NEXT:   kgen.call @foo([[V]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

  hlcf.loop (%arg0 = %idx2 : index) {
    %0 = index.cmp sgt(%arg0, %idx0)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.sub %arg0, %idx1
    %2 = index.sub %idx2, %arg0
    kgen.call @foo(%2) : (index) -> ()
    hlcf.continue %1 : index
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return
}

// CHECK-LABEL: @sequential_range
kgen.func @sequential_range() {
  %idx1 = index.constant 1
  %idx4 = index.constant 4

  // CHECK: hlcf.for [%idx1 to %idx4 step %idx1 slt add] (%arg0 = %idx1 : index) {
  // CHECK:        [[IDX:%.*]] = index.add %arg0, %idx1
  // CHECK-NEXT:   kgen.call @foo(%arg0) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

  hlcf.loop (%arg0 = %idx1 : index) {
    %0 = index.cmp slt(%arg0, %idx4)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.add %arg0, %idx1
    kgen.call @foo(%arg0) : (index) -> ()
    hlcf.continue %1 : index
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return
}

// CHECK-LABEL: @strided_range
kgen.func @strided_range() {
  %idx1 = index.constant 1
  %idx6 = index.constant 6
  %idx2 = index.constant 2

  // CHECK: hlcf.for [%idx1 to %idx6 step %idx2 slt add] (%arg0 = %idx1 : index) {
  // CHECK:        [[IDX:%.*]] = index.add %arg0, %idx2
  // CHECK-NEXT:   kgen.call @foo(%arg0) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

  hlcf.loop (%arg0 = %idx1 : index) {
    %0 = index.cmp slt(%arg0, %idx6)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.add %arg0, %idx2
    kgen.call @foo(%arg0) : (index) -> ()
    hlcf.continue %1 : index
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return
}

// CHECK-LABEL: @nested_unroll_loops
kgen.func @nested_unroll_loops() {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %idx4 = index.constant 4
  %idx8 = index.constant 8

  // CHECK: hlcf.for [%idx2 to %idx0 step %idx1 sgt sub] (%arg0 = %idx2 : index) {
  // CHECK:        [[IDX0:%.*]] = index.sub %arg0, %idx1
  // CHECK-NEXT:   [[V0:%.*]]  = index.sub %idx2, %arg0
  // CHECK-NEXT:   kgen.call @foo([[V0]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.for [%idx4 to %idx8 step %idx2 slt add] (%arg1 = %idx4 : index) {
  // CHECK:          [[IDX1:%.*]]  = index.add %arg1, %idx2
  // CHECK-NEXT:     [[V1:%.*]] = index.add [[V0]], %arg1
  // CHECK-NEXT:     kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT:     hlcf.for.yield [induction_var ([[IDX1]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT:   } {unrollLevel = #hlcf<unroll_level full>}
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX0]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

  hlcf.loop (%arg0 = %idx2 : index) {
    %0 = index.cmp sgt(%arg0, %idx0)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.sub %arg0, %idx1
    %2 = index.sub %idx2, %arg0
    kgen.call @foo(%2) : (index) -> ()
    hlcf.loop (%arg1 = %idx4 : index) {
      %4 = index.cmp slt(%arg1, %idx8)
      hlcf.if %4 {
        hlcf.yield
      } else {
        hlcf.break
      }
      %5 = index.add %arg1, %idx2
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
  %idx2 = index.constant 2
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  // CHECK: hlcf.for [%idx2 to %idx0 step %idx1 sgt sub] (%arg0 = %idx2 : index) {
  // CHECK:        [[IDX:%.*]] = index.sub %arg0, %idx1
  // CHECK-NEXT:   [[V:%.*]] = index.sub %idx2, %arg0
  // CHECK-NEXT:   kgen.call @foo([[V]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: }
  hlcf.loop (%arg0 = %idx2 : index) {
    %0 = index.cmp sgt(%arg0, %idx0)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.sub %arg0, %idx1
    %2 = index.sub %idx2, %arg0
    kgen.call @foo(%2) : (index) -> ()
    hlcf.continue %1 : index
  }
  kgen.return
}

// CHECK-LABEL: @loop_carried_dependency
kgen.func @loop_carried_dependency() {
  %idx1 = index.constant 1
  %idx9 = index.constant 9
  %idx2 = index.constant 2
  %idx4 = index.constant 4
  %idx8 = index.constant 8
  %idx0 = index.constant 0

  // CHECK: %0:2 = hlcf.for [%idx1 to %idx9 step %idx2 slt add] (%arg0 = %idx1  : index, %arg1 = %idx0 : index, %arg2 = %idx0  : index) -> (index, index) {
  // CHECK:        [[IDX0:%.*]] = index.add %arg0, %idx2
  // CHECK-NEXT:   kgen.call @foo(%arg0) : (index) -> ()
  // CHECK-NEXT:   [[V0:%.*]] = index.add %arg1, %arg0
  // CHECK-NEXT:   [[V1:%.*]] = hlcf.for [%idx4 to %idx8 step %idx2 slt add] (%arg3 =  %idx4 : index, %arg4 = %arg2 : index) -> index {
  // CHECK:          [[V4:%.*]] = index.add %arg3, %idx2
  // CHECK-NEXT:     [[V2:%.*]] = index.add %arg0, %arg3
  // CHECK-NEXT:     kgen.call @foo([[V2]]) : (index) -> ()
  // CHECK-NEXT:     [[V3:%.*]] = index.add %arg4, %arg3
  // CHECK-NEXT:     hlcf.for.yield [induction_var ([[V4]] : index)] [retvals ([[V3]] : index)] [iterargs ()]
  // CHECK-NEXT:   } {unrollLevel = #hlcf<unroll_level full>}
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[IDX0]] : index)] [retvals ([[V0]], [[V1]] : index, index)] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

  %0:2 = hlcf.loop (%arg0 = %idx0 : index, %arg1 = %idx0 : index, %arg2 = %idx1 : index) -> (index, index) {
    %3 = index.cmp slt(%arg2, %idx9)
    hlcf.if %3 {
      hlcf.yield
    } else {
      hlcf.break %arg0, %arg1 : index, index
    }
    %4 = index.add %arg2, %idx2
    kgen.call @foo(%arg2) : (index) -> ()
    %6 = index.add %arg0, %arg2
    %7 = hlcf.loop (%arg3 = %arg1 : index, %arg4 = %idx4 : index) -> index {
      %8 = index.cmp slt(%arg4, %idx8)
      hlcf.if %8 {
        hlcf.yield
      } else {
        hlcf.break %arg3 : index
      }
      %9 = index.add %arg4, %idx2
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
  %idx10 = index.constant 10
  %idx1 = index.constant 1
  %idx0 = index.constant 0
  %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(pointer<scalar<f32>>, index, dtype)>

  // CHECK:  [[V0:%.*]] = kgen.struct.extract %arg0[0] : !kgen.struct<(pointer<scalar<f32>>, index, dtype)>
  // CHECK-NEXT:  [[V1:%.*]] = hlcf.for [%idx10 to %idx0 step %idx1 sgt sub] (%arg1 = %idx10 : index, %arg2 = %idx0 : index, %arg3 = [[V0]] : !kgen.pointer<scalar<f32>>) -> index {
  // CHECK:        [[V2:%.*]] = index.sub %arg1, %idx1
  // CHECK-NEXT:   [[V3:%.*]] = pop.load %arg3 align<1> : !kgen.pointer<scalar<f32>>
  // CHECK-NEXT:   [[V4:%.*]] = pop.cast [[V3]] : !pop.scalar<f32> to !pop.scalar<index>
  // CHECK-NEXT:   [[V5:%.*]] = pop.cast_to_builtin [[V4]] : !pop.scalar<index> to index
  // CHECK-NEXT:   [[V6:%.*]] = index.add %arg2, [[V5]]
  // CHECK-NEXT:   [[V7:%.*]] = pop.offset %arg3[%idx1] : !kgen.pointer<scalar<f32>>
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[V2]] : index)] [retvals ([[V6]] : index)] [iterargs ([[V7]] : !kgen.pointer<scalar<f32>>)]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

  %1 = hlcf.loop (%arg3 = %idx10 : index, %arg1 = %0 : !kgen.pointer<scalar<f32>>, %arg2 = %idx0 : index) -> index {
    %2 = index.cmp sgt(%arg3, %idx0)
    hlcf.if %2 {
      hlcf.yield
    } else {
      hlcf.break %arg2 : index
    }
    %3 = index.sub %arg3, %idx1
    %4 = pop.load %arg1 align<1> : !kgen.pointer<scalar<f32>>
    %5 = pop.cast %4 : !pop.scalar<f32> to !pop.scalar<index>
    %6 = pop.cast_to_builtin %5 : !pop.scalar<index> to index
    %7 = index.add %arg2, %6
    %8 = pop.offset %arg1[%idx1] : !kgen.pointer<scalar<f32>>
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
  %idx5 = index.constant 5
  %idx1 = index.constant 1
  %index-1 = index.constant -1

  // CHECK: hlcf.for [%idx5 to %idx1 step %idx-1 sgt add] (%arg0 = %idx5 : index) {
  // CHECK:        [[V:%.*]] = index.add %arg0, %idx-1
  // CHECK-NEXT:   kgen.call @foo(%arg0) : (index) -> ()
  // CHECK-NEXT:   hlcf.for.yield [induction_var ([[V]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: } {unrollLevel = #hlcf<unroll_level full>}

  hlcf.loop (%arg0 = %idx5 : index) {
    %3 = index.cmp sgt(%arg0, %idx1)
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
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  // CHECK: hlcf.for [%idx2 to %idx0 step %idx1 sgt sub] (%arg0 = %idx2 : index) {
  // CHECK:        [[IDX0:%.*]] = index.sub %arg0, %idx1
  // CHECK-NEXT:   [[V0:%.*]]  = index.sub %idx2, %arg0
  // CHECK-NEXT:   kgen.call @foo([[V0]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.for [%idx1 to %arg0 step %idx2
  // CHECK-DAG:   hlcf.for.yield [induction_var ([[IDX0]] : index)] [retvals ()] [iterargs ()]
  // CHECK-NEXT: }

  hlcf.loop (%arg0 = %idx2 : index) {
    %0 = index.cmp sgt(%arg0, %idx0)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.sub %arg0, %idx1
    %2 = index.sub %idx2, %arg0
    kgen.call @foo(%2) : (index) -> ()
    // Cannot raise this loop to a for-loop because the upper bound is %arg0 which changes
    // over parent loop's iterations.
    hlcf.loop (%arg1 = %idx1 : index) {
      %4 = index.cmp slt(%arg1, %arg0)
      hlcf.if %4 {
        hlcf.yield
      } else {
        hlcf.break
      }
      %5 = index.add %arg1, %idx2
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
   %idx0 = index.constant 0
   %idx1 = index.constant 1
   %idx10 = index.constant 10

   // CHECK: hlcf.for [%idx0 to %idx10 step %idx1 sle add] (%arg0 = %idx0 : index)
   hlcf.loop (%arg0 = %idx0 : index) {
     // when for-loop is raise, this condition will be inverted to sle
     %1 = index.cmp sgt(%arg0, %idx10)
     hlcf.if %1 {
       hlcf.break
     } else {
       hlcf.yield
     }
     %2 = index.add %arg0, %idx1
     hlcf.continue %2 : index
   }
   kgen.return
 }

// CHECK-LABEL: @return_value_same_as_iter_var
kgen.func @return_value_same_as_iter_var()  {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  // CHECK: hlcf.for [%idx0 to %idx2 step %idx1 slt add] (%arg0 = %idx0 : index, %arg1 = %idx0 : index) -> index
  // CHECK:  [[V1:%.*]] = index.add %arg0, %idx1
  // CHECK-NEXT:  hlcf.for.yield [induction_var ([[V1]] : index)] [retvals ([[V1]] : index)] [iterargs ()]

  %0 = hlcf.loop (%arg0 = %idx0 : index) -> index {
    // loop return value is the same as iterVar: %arg0
    %2 = index.cmp slt(%arg0, %idx2)
    hlcf.if %2 {
      hlcf.yield
    } else {
      hlcf.break %arg0 : index
    }
    %3 = index.add %arg0, %idx1
    hlcf.continue %3 : index
  }
  kgen.return
}

// CHECK-LABEL: @stride_same_as_iter_var
kgen.func @stride_same_as_iter_var()  {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  // CHECK-NOT: hlcf.for
  %0 = hlcf.loop (%arg0 = %idx0 : index) -> index {
    // loop stride value is the same as iterVar: %arg0
    %2 = index.cmp slt(%arg0, %idx1)
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
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  // CHECK-NOT: hlcf.for
  %0 = hlcf.loop (%arg0 = %idx0 : index, %arg1 = %idx1 : index) -> index {
    // loop end is not always constant
    %2 = index.cmp slt(%arg0, %arg1)
    hlcf.if %2 {
      hlcf.yield
    } else {
      hlcf.break %arg0 : index
    }
    %3 = index.add %arg0, %idx1
    hlcf.continue %3, %3 : index, index
  }
  kgen.return
}

// -----

// Tests for handling ops inside the conditional break branch of the loop.

// CHECK-LABEL: @simple_call_in_break_branch
kgen.func @simple_call_in_break_branch() {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx10 = index.constant 10

  // CHECK: hlcf.for [%idx0 to %idx10 step %idx1 sgt add] (%arg0 = %idx0 : index)
  // CHECK:        hlcf.for.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: kgen.call @foo(%idx10)
  hlcf.loop (%arg0 = %idx0 : index) {
    %1 = index.cmp sgt(%arg0, %idx10)
    hlcf.if %1 {
      hlcf.yield
    } else {
      kgen.call @foo(%idx10) : (index) -> ()
      hlcf.break
    }
    %2 = index.add %arg0, %idx1
    hlcf.continue %2 : index
  }
  kgen.return
}

// CHECK-LABEL: @intermediate_values_in_break_branch
kgen.func @intermediate_values_in_break_branch() {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx10 = index.constant 10

  // CHECK: hlcf.for [%idx0 to %idx10 step %idx1 sgt add] (%arg0 = %idx0 : index)
  // CHECK:        hlcf.for.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V0:%.*]] = kgen.call @foo(%idx10)
  // CHECK-NEXT: kgen.call @bar([[V0]])
  hlcf.loop (%arg0 = %idx0 : index) {
    %1 = index.cmp sgt(%arg0, %idx10)
    hlcf.if %1 {
      hlcf.yield
    } else {
      %v0 = kgen.call @foo(%idx10) : (index) -> (index)
      kgen.call @bar(%v0) : (index) -> ()
      hlcf.break
    }
    %2 = index.add %arg0, %idx1
    hlcf.continue %2 : index
  }
  kgen.return
}

// CHECK-LABEL: @break_dependent_on_break_branch
kgen.func @break_dependent_on_break_branch() {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx10 = index.constant 10

  // CHECK-NOT: hlcf.for
  %loop = hlcf.loop (%arg0 = %idx0 : index) -> index {
    %1 = index.cmp sgt(%arg0, %idx10)
    hlcf.if %1 {
      hlcf.yield
    } else {
      // Break is dependent on intermediate results from this branch. Abort.
      %v0 = kgen.call @foo(%idx10) : (index) -> (index)
      hlcf.break %v0 : index
    }
    %2 = index.add %arg0, %idx1
    hlcf.continue %2 : index
  }
  kgen.return
}

// CHECK-LABEL: @dependent_ops_in_break_branch
kgen.func @dependent_ops_in_break_branch() {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx10 = index.constant 10

  // CHECK-NOT: hlcf.for
  hlcf.loop (%arg0 = %idx0 : index) {
    %1 = index.cmp sgt(%arg0, %idx10)
    hlcf.if %1 {
      hlcf.yield
    } else {
      // Op depends on internal value. Can no longer convert.
      kgen.call @foo(%arg0) : (index) -> ()
      hlcf.break
    }
    %2 = index.add %arg0, %idx1
    hlcf.continue %2 : index
  }
  kgen.return
}

// CHECK-LABEL: @dynamic_bounds
kgen.func @dynamic_bounds(%arg0: index) {
  %0 = kgen.call @bar() : () -> index
  %1 = kgen.call @bar() : () -> index
  // CHECK: hlcf.for [%0 to %1 step %arg0 slt add]
  hlcf.loop (%arg1 = %0 : index) {
    %2 = index.cmp slt(%arg1, %1)
    hlcf.if %2 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %3 = index.add %arg1, %arg0
    hlcf.continue %3 : index
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
