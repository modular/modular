// RUN: kgen-opt -sccp -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @loop_generates_constant
kgen.func @loop_generates_constant() -> (index, index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK: [[FALSE:%.*]] = index.bool.constant false
  // CHECK-DAG: [[IDX11:%.*]] = index.constant 11
  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  // The result of this loop will be 2
  %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
    // CHECK: index.cmp
    %3 = index.cmp slt(%arg0, %idx2)
    hlcf.if %3 {
      hlcf.yield
    } else {
      %4 = index.add %arg0, %1
      hlcf.break %4: index
    }
    %5 = index.add %arg0, %idx1
    hlcf.continue %5 : index
  }

  // COM: %2 will be a constant, so this cmp result will be a constant
  // CHECK-NOT: index.cmp
  %6 = index.cmp slt(%2, %idx2)

  // CHECK: [[V1:%.*]] = hlcf.if [[FALSE]]
  %7 = hlcf.if %6 -> index {
    hlcf.yield %idx0: index
  } else {
    // CHECK: hlcf.yield [[IDX9]]
    hlcf.yield %1: index
  }

  // CHECK: kgen.return [[IDX11]], [[IDX9]]
  kgen.return %2, %7 : index, index
}

// CHECK-LABEL: @not_much_can_be_known
kgen.func @not_much_can_be_known(%cond: i1) -> (index, index) {
  // COM: Not much can be folded except obvious one that has constant operands.
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0
  %2 = hlcf.loop(%arg0 = %idx0: index, %arg1 = %cond: i1) -> index {
    %3 = hlcf.if %arg1 -> index {
      hlcf.yield %idx0: index
    } else {
      hlcf.yield %idx1: index
    }

    %4 = index.cmp slt(%3, %arg0)
    hlcf.if %4 {
      hlcf.yield
    } else {
      %5 = index.add %3, %3
      hlcf.break %5: index
    }

    %6 = index.add %3, %idx1
    %7 = index.cmp slt(%3, %idx2)
    hlcf.continue %6, %7 : index, i1
  }

  // CHECK: kgen.return [[IDX9]], [[V0:%.*]]
  kgen.return %1, %2 : index, index
}

// CHECK-LABEL: @nested_if_constant_result
kgen.func @nested_if_constant_result(%cond: i1) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK: [[TRUE:%.*]] = index.bool.constant true
  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  %2 = hlcf.if %cond -> index {
    %3:2 = hlcf.if %cond -> index, index {
      // CHECK: hlcf.yield [[IDX9]], [[IDX1]]
      hlcf.yield %1, %idx1: index, index
    } else {
      // CHECK: hlcf.yield [[IDX2]], [[IDX1]]
      hlcf.yield %idx2, %idx1: index, index
    }
    kgen.call @foo(%3#0) : (index) -> ()
    // COM: This cmp has constant result.
    %4 = index.cmp slt (%3#1, %idx2)

    // CHECK: [[V2:%.*]] = hlcf.if [[TRUE]]
    %5 = hlcf.if %4 -> index {
      // CHECK: hlcf.yield [[IDX1]]
      hlcf.yield %3#1: index
    } else {
      hlcf.yield %3#0: index
    }

    // CHECK: hlcf.yield [[IDX1]]
    hlcf.yield %5: index
  } else {
    // CHECK: hlcf.yield [[IDX3]]
    hlcf.yield %0: index
  }

  kgen.return %2 : index
}

// CHECK-LABEL: @test_switches
kgen.func @test_switches(%arg0: index) -> (index, index, index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK-DAG: [[INDEX2:%.*]] = kgen.param.constant = <2>
  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  // COM: switch result is constant.
  %2 = hlcf.switch %arg0 -> index
  default {
    %3 = index.mul %0, %0

    // CHECK: hlcf.yield [[IDX9]]
    hlcf.yield %3: index
  }
  case 2 {
    // CHECK: hlcf.yield [[IDX9]]
    hlcf.yield %1: index
  }

  // COM: switch result is unknown.
  // CHECK: [[V4:%.*]] = hlcf.switch
  %4 = hlcf.switch %arg0 -> index
  default {
    hlcf.yield %arg0: index
  }
  case 1 {
    hlcf.yield %1: index
  }

  // COM: complex switch result is constant.
  %5 = hlcf.switch %arg0 -> index
  default {
    // COM: loop result is constant
    %6 = hlcf.loop(%arg1 = %idx0: index) -> index {
      %7 = index.cmp slt(%arg1, %idx2)
      hlcf.if %7 {
        hlcf.yield
      } else {
        hlcf.break %arg1: index
      }
      %8 = index.add %arg1, %idx1
      hlcf.continue %8 : index
    }
    // CHECK: hlcf.yield [[INDEX2]]
    hlcf.yield %6: index
  }
  case 2 {
    hlcf.yield %idx2: index
  }

  // CHECK: kgen.return [[IDX9]], [[V4]], [[INDEX2]]
  kgen.return %2, %4, %5: index, index, index
}

kgen.func @test_for_loop(%arg0: index) -> (index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  %0 = hlcf.for [%idx0 to %idx2 step %idx1 slt add] (%arg1 = %idx0 : index, %arg2 = %idx1: index) -> index {
    %1 = index.add %arg1, %idx1
    %2 = index.add %arg2, %idx1
    kgen.call @foo(%1, %arg0) : (index, index) -> ()
    hlcf.for.yield [induction_var (%1 : index)] [retvals (%2: index)] [iterargs ()]
  }

  // CHECK: kgen.return [[IDX3]]
  kgen.return %0: index
}

// CHECK-LABEL: @nested_loops
kgen.func @nested_loops() -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %idx4 = index.constant 4

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %idx1

  // CHECK-DAG: [[IDX10:%.*]] = index.constant 10
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX4:%.*]] = index.constant 4
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  // COM: The result of this loop will be 10
  %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
    %3 = hlcf.loop(%arg1 = %idx0: index) -> index {
      %4 = index.cmp slt(%arg1, %arg0)
      hlcf.if %4 {
        hlcf.yield
      } else {
        %5 = index.add %arg1, %1
        hlcf.break %5: index
      }
      %6 = index.add %arg1, %idx2
      hlcf.continue %6: index
    }

    %7 = index.cmp slt(%3, %idx4)
    hlcf.if %7 {
      hlcf.yield
    } else {
      %8 = index.add %3, %1
      hlcf.break %8: index
    }

    hlcf.continue %3 : index
  }

  // CHECK: kgen.return [[IDX10]]
  kgen.return %2: index
}

// CHECK-LABEL: @loop_generates_constant_but_hits_limit
kgen.func @loop_generates_constant_but_hits_limit() -> (index, index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %idx110 = index.constant 110

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX110:%.*]] = index.constant 110
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  // COM: The result of this loop will be 110, but hits analysis threshold before finishing,
  // COM: so result will be unknown.
  // CHECK: [[V2:%.*]] = hlcf.loop
  %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
    // CHECK: index.cmp
    %3 = index.cmp slt(%arg0, %idx110)
    hlcf.if %3 {
      hlcf.yield
    } else {
      %4 = index.add %arg0, %1
      hlcf.break %4: index
    }
    %5 = index.add %arg0, %idx1
    hlcf.continue %5 : index
  }

  // CHECK: [[V6:%.*]] = index.cmp slt([[V2]], [[IDX2]])
  %6 = index.cmp slt(%2, %idx2)

  // CHECK: [[V7:%.*]] = hlcf.if [[V6:%.*]]
  %7 = hlcf.if %6 -> index {
    hlcf.yield %idx0: index
  } else {
    // CHECK: hlcf.yield [[IDX9]]
    hlcf.yield %1: index
  }

  // CHECK: kgen.return [[V2]], [[V7]]
  kgen.return %2, %7 : index, index
}


 // CHECK-LABEL: @nested_if_breaks
 kgen.func @nested_if_breaks(%cond: i1) -> (index, index) {
   %idx0 = index.constant 0
   %idx1 = index.constant 1
   %idx2 = index.constant 2

   // CHECK-DAG: %idx6 = index.constant 6
   // CHECK-DAG: %idx3 = index.constant 3
   // CHECK-DAG: %idx2 = index.constant 2
   // CHECK-DAG: %idx1 = index.constant 1
   // CHECK-DAG: %idx0 = index.constant 0

   // %0 = 3
   %0 = index.add %idx1, %idx2
   // %1 = 6
   %1 = index.mul %0, %idx2

   // COM: break is in a nested hlcf.if
   // COM: loop generates constant 6.
   %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
     %3 = index.cmp slt(%arg0, %idx2)
     hlcf.if %3 {
       hlcf.yield
     } else {
       %4 = index.add %arg0, %idx2
       %5 = index.cmp slt(%4, %1)
       hlcf.if %5 {
         hlcf.yield
       } else {
         // break and return 6
         hlcf.break %4: index
       }
       hlcf.yield
     }
     %5 = index.add %arg0, %idx1
     hlcf.continue %5 : index
   }

   // COM: loop generates constant 1.
   %6 = hlcf.loop(%arg0 = %cond: i1) -> index {
     // COM: Both regions of hlcf.if will terminate the current iteration
     // COM: immediately so that %9 and operations after will never be evaluated.
     %8 = hlcf.if %arg0 -> index {
       hlcf.break %idx1: index
     } else {
       hlcf.break %idx1: index
     }
     %9 = index.add %8, %idx1
     %10 = index.cmp slt (%9, %idx2)
     hlcf.continue %10 : i1
   }
   // CHECK: kgen.return %idx6, %idx1
   kgen.return %2, %6 : index, index
 }

// CHECK-LABEL: @none_hlcf_controlflownode_donot_crash
kgen.generator @none_hlcf_controlflownode_donot_crash() -> index {
  // COM: Conservatively mark all results as Unknown, but process the subregions.
  kgen.param.declare condition: i1 = <0>
  %0 = kgen.param.if <condition> -> index {
    %i0 = index.constant 0
    kgen.param.yield %i0: index
  } else {
    %i1 = index.constant 1
    kgen.param.yield %i1: index
  }

  // CHECK: kgen.return [[V0:%.*]]
  kgen.return %0: index
}

// COM: This test should not fail with lattice value assertion due to early exits in the loop.
// CHECK-LABEL: @should_continue
kgen.func @should_continue() -> index {
  // CHECK: [[IDX0:%.*]] = kgen.param.constant = <0>
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = hlcf.loop (%arg0 = %idx0 : index, %arg1 = %idx1 : index) -> index {
    %2 = index.cmp sgt(%arg1, %idx0)
    hlcf.if %2 {
      hlcf.yield
    } else {
      hlcf.break %arg0 : index
    }
    %3 = index.sub %arg1, %idx1
    %4 = index.cmp eq(%idx1, %arg1)
    hlcf.if %4 {
      hlcf.continue %arg0, %3 : index, index
    } else {
      hlcf.yield
    }
    %5 = index.add %arg0, %idx1
    hlcf.continue %5, %3 : index, index
  }
  // CHECK: kgen.call @f([[IDX0]]) : (index) -> index
  %1 = kgen.call @f(%0) : (index) -> index
  kgen.return %1: index
}


// CHECK-LABEL: @indirect_loop_break
kgen.func @indirect_loop_break(%cond: index) -> index {
  // CHECK-DAG: %idx0 = index.constant 0
  // CHECK-DAG: %idx1 = index.constant 1
  // CHECK-DAG: %idx7 = index.constant 7
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx7 = index.constant 7

  // CHECK: [[V0:%.*]] = hlcf.loop
  %0 = hlcf.loop "inlined_cf_scope" () -> index {
    // This loop can't converge and it will lead to the outer loop
    // fail to converge too because one of the break inside breaks
    // the outer loop.
    %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
      // CHECK: index.cmp
      %3 = index.cmp slt(%arg0, %cond)
      hlcf.if %3 {
        // This breaks to the outer loop instead of the inner one.
        hlcf.break "inlined_cf_scope" %arg0 : index
      } else {
        %4 = index.cmp slt(%arg0, %idx7)
        hlcf.if %4 {
          hlcf.yield
        } else {
          %5 = index.add %arg0, %idx1
          hlcf.break %5: index
        }
        hlcf.yield
      }
      %5 = index.add %arg0, %idx1
      hlcf.continue %5 : index
    }
    hlcf.break "inlined_cf_scope" %idx0: index
  }
  // CHECK: kgen.call @f([[V0]])
  %1 = kgen.call @f(%0) : (index) -> index
  kgen.return %1: index
}


// CHECK-LABEL: @single_unreachable_indirect_loop_break
kgen.func @single_unreachable_indirect_loop_break(%cond: index) -> index {
  // CHECK-DAG: %idx0 = index.constant 0
  // CHECK-DAG: %idx1 = index.constant 1
  // CHECK-DAG: %idx100000 = index.constant 100000
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  // MOCO-1318: A large enough trip count such that `else` branch would never be
  // processed by sccp.
  %idx100000 = index.constant 100000

  // CHECK: hlcf.loop
  %0 = hlcf.loop "inlined_cf_scope" () -> index {
    hlcf.loop "_loop_0" (%arg2 = %idx100000 : index, %arg3 = %idx0 : index) {
      %18 = index.cmp sgt(%arg2, %idx0)
      hlcf.if %18 {
        hlcf.yield
      } else {
        hlcf.break "inlined_cf_scope" %idx1 : index
      }
      %19 = index.sub %arg2, %idx1
      hlcf.continue %19, %arg3 : index, index
    }
    kgen.unreachable
  }

  // CHECK: kgen.call @f([[V0]])
  %1 = kgen.call @f(%0) : (index) -> index
  kgen.return %1: index
}
