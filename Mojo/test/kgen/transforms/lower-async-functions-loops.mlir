// RUN: kgen-opt -lower-async-functions -split-input-file %s | FileCheck %s
// RUN: kgen-opt -lower-async-functions='use-liveness-frame-evaluation=true' -split-input-file %s | FileCheck %s

// COM: Verify Ramp + Resume + Async Calls are transformed correctly.
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_hot_ramp(
kgen.func @coroutine(%arg0: i1, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async -> index {
 // CHECK:      [[CORO:%.*]] = pop.aligned_alloc
 // CHECK:      [[V11:%.*]] = pop.pointer.bitcast [[CORO]]
 // CHECK-NEXT: hlcf.loop "_loop_0" ([[BLOCK_ARG:%.*]] =
 // Check that block arguments are stored in frame.
 // CHECK-NEXT: [[V9:%.*]] = kgen.struct.gep %0[[[#FRAME8:]]]
 // CHECK-NEXT: pop.store [[BLOCK_ARG]], [[V9]] : !kgen.pointer<index>

 // CHECK-NEXT: hlcf.loop "_loop_1" ([[BLOCK_ARG_INNER:%.*]] =
 // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %0[[[#FRAME8 - 1]]]
 // CHECK-NEXT: pop.store [[BLOCK_ARG_INNER]], [[V10]] : !kgen.pointer<index>

 // Verify suspension point in nested loops is properly replaced and
 // unreachable is inserted to terminate unreachable blocks.
 // CHECK-NEXT:   [[COND0:%.*]] = pop.cast_from_builtin %arg2 : i1 to !kgen.scalar<bool>
 // CHECK-NEXT:   hlcf.if [[COND0]] {
 // CHECK-NEXT:     hlcf.break "_loop_1"
 // CHECK-NEXT:   } else {
 // CHECK-NEXT:     hlcf.yield
 // CHECK-NEXT:   }
 // CHECK-NEXT:   [[V15:%.*]] = kgen.param.constant: i32 = <1>
 // CHECK-NEXT:   [[V16:%.*]] = kgen.struct.gep [[CORO]][0]
 // CHECK-NEXT:   pop.store [[V15]], [[V16]] : !kgen.pointer<i32>
 // CHECK-NEXT:   hlcf.return [[V11]]
 // CHECK-NEXT: }
 // CHECK-NEXT: kgen.call @print
 // CHECK-NEXT: hlcf.continue
 // CHECK-NEXT: }
 // CHECK-NEXT: hlcf.unreachable
 hlcf.loop "_loop_0" (%arg3 = %arg1 : index) {
   hlcf.loop "_loop_1" (%arg2 = %arg1 : index) {
     %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
     hlcf.if %arg0_sb {
       hlcf.break "_loop_1"
     } else {
       hlcf.yield
     }
     co.suspend (%hdl) {
       co.suspend.end
     }
     kgen.call @print1(%arg2) : (index) -> ()
     hlcf.continue %arg2 : index
   }
   kgen.call @print(%arg0) : (i1) -> ()
   hlcf.continue %arg3 : index
 }
 co.suspend (%hdl) {
   co.suspend.end
 }
 %final = index.add %arg1, %arg1
 hlcf.return %final : index
}

// Check that the operands of parents that are state 0 are replaced with constants. All other ops in state 0 will be erased.
// CHECK-LABEL:  kgen.func @coroutine_resume
// CHECK-NEXT:   [[UNDEF:%.*]] = kgen.param.constant = <#interp.uninitmem>
// CHECK-NEXT:   hlcf.loop "_loop_0" (%arg1 = [[UNDEF]] : index) {
kgen.func @trigger_creation(%arg0: i1, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async {
   %coro = co.hot_invoke[(i1, index, !kgen.pointer<index> byref_result) async -> index: @coroutine](%arg0, %arg1, %__result__)
   hlcf.return
}

}

// -----

// COM: Verify Loop With Await In Then Statement Is Correct

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine1_resume
kgen.func @coroutine1(%arg0: i1, %arg1: index, %arg2: index, %arg3: index) async -> index {
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V5:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[V6:%.*]] = pop.load [[V5]] : !kgen.pointer<index>

  // CHECK-NEXT: [[V7:%.*]] = kgen.call @foo(%idx3, [[V6]]) : (index, index) -> index
  // CHECK-NEXT: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: pop.store [[V7]], [[V8]] : !kgen.pointer<index>
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index

  // CHECK-NEXT: hlcf.loop
  hlcf.loop "_loop_0" {
    // CHECK-NEXT: [[V19:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
    // CHECK-NEXT: [[V20:%.*]] = pop.load [[V19]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V21:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 1]]]
    // CHECK-NEXT: [[V22:%.*]] = pop.load [[V21]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V23:%.*]] = kgen.call @bar([[V20]], [[V22]]) : (index, index) -> index
    %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index


    // CHECK-NEXT: [[V24:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 2]]]
    // CHECK-NEXT: [[V25:%.*]] = pop.load [[V24]] : !kgen.pointer<i1>
    // CHECK-NEXT: [[V25SB:%.*]] = pop.cast_from_builtin [[V25]] : i1 to !kgen.scalar<bool>
    // CHECK-NEXT: hlcf.if [[V25SB]] {
    // CHECK-NEXT:   kgen.param.constant
    // CHECK-NEXT:   kgen.struct.gep
    // CHECK-NEXT:   pop.store
    // CHECK-NEXT:   co.suspend {
    // CHECK-NEXT:     co.suspend.end
    // CHECK-NEXT:   }
    // CHECK-NEXT:   hlcf.yield
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   hlcf.break "_loop_0"
    // CHECK-NEXT: }
    %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
    hlcf.if %arg0_sb {
       co.suspend (%hdl) {
         co.suspend.end
       }
       hlcf.yield
    } else {
       hlcf.break "_loop_0"
    }
    // CHECK-NEXT: %idx3_0 = index.constant 3
    // CHECK-NEXT: [[V28:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 3]]]
    // CHECK-NEXT: [[V29:%.*]] = pop.load [[V28]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V30:%.*]] = kgen.call @foo(%idx3_0, [[V29]]) : (index, index) -> index
    %result6 = kgen.call @foo(%idx3, %arg2) : (index,index) -> index

    // CHECK-NEXT: hlcf.continue
    hlcf.continue
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V9:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[V10:%.*]] = pop.load [[V9]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
  // CHECK-NEXT: [[V12:%.*]] = pop.load [[V11]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @bar([[V10]], [[V12]]) : (index, index) -> index
  %result5 = kgen.call @bar(%result, %arg1): (index,index) -> index

  // CHECK-NEXT: [[V14:%.*]] = kgen.struct.gep %arg0[[[#PROMISE_IDX:]]]
  // CHECK-NEXT: [[PTR:%.*]] = kgen.struct.gep [[V14]][0]
  // CHECK-NEXT: pop.store [[V10]], [[PTR]] : !kgen.pointer<index>
  // CHECK-NEXT: hlcf.return
  hlcf.return %result : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg2: index, %arg3: index) {
  %coro = co.invoke[(i1, index, index, index) async -> index:@coroutine1](%arg0, %arg1, %arg2, %arg3)
  hlcf.return
}
}

// -----

// COM: Verify Loop With Await In Else Statement Is Correct

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine5_resume
kgen.func @coroutine5(%arg0: i1, %arg1: index, %arg3: index) async -> index {
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[NOT_IN_FRAME:%.*]] = kgen.call @foo(%idx3, [[V5]]) : (index, index) -> index
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  hlcf.loop "_loop_0" {
     %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
     hlcf.if %arg0_sb {
       hlcf.yield
     } else {
       // CHECK: } else {
       // CHECK-NEXT: [[V17:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 2]]]
       // CHECK-NEXT: [[V18:%.*]] = pop.load [[V17]] : !kgen.pointer<index>
       // CHECK-NEXT: [[V19:%.*]] = kgen.call @bar([[NOT_IN_FRAME]], [[V18]]) : (index, index) -> index
       // CHECK-NEXT: kgen.param.constant
       // CHECK-NEXT: kgen.struct.gep
       // CHECK-NEXT: pop.store
       // CHECK-NEXT: co.suspend
       %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index
       co.suspend (%hdl) {
         co.suspend.end
       }
       hlcf.break "_loop_0"
     }
     hlcf.continue
  }
  // CHECK:      [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[V9:%.*]] = pop.load [[V8]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %arg0[[[#PROMISE_IDX:]]]
  // CHECK-NEXT: [[PTR:%.*]] = kgen.struct.gep [[V10]][0]
  // CHECK-NEXT: pop.store [[V9]], [[PTR]]
  // CHECK-NEXT: hlcf.return
  hlcf.return %result : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index) {
  %coro = co.invoke[(i1, index, index) async -> index:@coroutine5](%arg0, %arg1, %arg3)
  hlcf.return
}

}

// -----

// COM: Verify Block With Multiple Awaits Is Correct

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine3_resume
kgen.func @coroutine3(%arg0: i1, %arg1: index, %arg3: index) async -> index {
  %idx3 = index.constant 3
  // CHECK: [[NIF:%.*]] = kgen.call @foo(%idx3, %{{.*}}) : (index, index) -> index
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  // CHECK: hlcf.loop "_loop_0"
  hlcf.loop "_loop_0" {
    // CHECK-NEXT: [[V13:%.*]] = kgen.struct.gep %arg0[[[#FRAME11:]]]
    // CHECK-NEXT: [[V14:%.*]] = pop.load [[V13]] : !kgen.pointer<i1>
    // CHECK-NEXT: [[V14SB:%.*]] = pop.cast_from_builtin [[V14]] : i1 to !kgen.scalar<bool>
    // CHECK-NEXT: hlcf.if [[V14SB]] {
    // CHECK-NEXT:   hlcf.yield
    // CHECK-NEXT: } else {
    // CHECK-NEXT: [[V15:%.*]] = kgen.struct.gep %arg0[[[#FRAME11 + 1]]]
    // CHECK-NEXT: [[V16:%.*]] = pop.load [[V15]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V17:%.*]] = kgen.call @bar([[NIF]], [[V16]]) : (index, index) -> index
    // CHECK-NEXT: [[V18:%.*]] = kgen.struct.gep %arg0[[[#FRAME11 - 3]]]
    // CHECK-NEXT: pop.store [[V17]], [[V18]] : !kgen.pointer<index>
    // CHECK-NEXT: kgen.param.constant
    // CHECK-NEXT: kgen.struct.gep
    // CHECK-NEXT: pop.store
    // CHECK-NEXT: co.suspend {
    // CHECK-NEXT:   co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V19:%.*]] = kgen.struct.gep %arg0[[[#FRAME11 - 3]]]
    // CHECK-NEXT: [[V20:%.*]] = pop.load [[V19]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V21:%.*]] = kgen.struct.gep %arg0[[[#FRAME11 + 1]]]
    // CHECK-NEXT: [[V22:%.*]] = pop.load [[V21]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V23:%.*]] = kgen.call @bar([[V20]], [[V22]]) : (index, index) -> index
    // CHECK-NEXT: kgen.param.constant
    // CHECK-NEXT: kgen.struct.gep
    // CHECK-NEXT: pop.store
    // CHECK-NEXT: co.suspend {
    // CHECK-NEXT:   co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: hlcf.break "_loop_0"
    // CHECK-NEXT: }
     %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
     hlcf.if %arg0_sb {
        hlcf.yield
     } else {
         %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index
         co.suspend (%hdl) {
           co.suspend.end
         }
         %result6 = kgen.call @bar(%result4, %arg3): (index,index) -> index
         co.suspend (%hdl) {
           co.suspend.end
         }
        hlcf.break "_loop_0"
     }
     hlcf.continue
  }
  hlcf.return %result : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index) {
  %coro = co.invoke[(i1, index, index) async -> index:@coroutine3](%arg0, %arg1, %arg3)
  hlcf.return
}
}

// -----

// COM: Verify Nested Control Flow Is Correct

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_nested_resume
kgen.func @coroutine_nested(%arg0: i1, %arg1: index, %arg3: index) async -> index {
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  hlcf.loop "_loop_0" {
     %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
     hlcf.if %arg0_sb {
        hlcf.yield
     } else {
         %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index
         // CHECK:      hlcf.loop "_loop_1" {
         // CHECK-NEXT: [[V23:%.*]] = kgen.struct.gep %arg0[[[#FRAME10:]]]
         // CHECK-NEXT: [[V24:%.*]] = pop.load [[V23]] : !kgen.pointer<i1>
         // CHECK-NEXT: [[V24SB:%.*]] = pop.cast_from_builtin [[V24]] : i1 to !kgen.scalar<bool>
         // CHECK-NEXT: hlcf.if [[V24SB]] {
         // CHECK-NEXT:   hlcf.yield
         // CHECK-NEXT: } else {
         // CHECK-NEXT:   kgen.param.constant
         // CHECK-NEXT:   kgen.struct.gep
         // CHECK-NEXT:   pop.store
         // CHECK-NEXT:   co.suspend {
         // CHECK-NEXT:     co.suspend.end
         // CHECK-NEXT:   }
         // CHECK-NEXT:   [[V25:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 - 2]]]
         // CHECK-NEXT:   [[V26:%.*]] = pop.load [[V25]] : !kgen.pointer<index>
         // CHECK-NEXT:   [[V27:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 + 1]]]
         // CHECK-NEXT:   [[V28:%.*]] = pop.load [[V27]] : !kgen.pointer<index>
         // CHECK-NEXT:   [[V29:%.*]] = kgen.call @bar([[V26]], [[V28]]) : (index, index) -> index
         // CHECK-NEXT:   hlcf.break "_loop_1"
         // CHECK-NEXT: }
         // CHECK-NEXT: hlcf.continue
         // CHECK-NEXT: }
         // CHECK-NEXT: [[V18:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 - 2]]]
         // CHECK-NEXT: [[V19:%.*]] = pop.load [[V18]] : !kgen.pointer<index>
         // CHECK-NEXT: [[V20:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 + 1]]]
         // CHECK-NEXT: [[V21:%.*]] = pop.load [[V20]] : !kgen.pointer<index>
         // CHECK-NEXT: [[V22:%.*]] = kgen.call @bar([[V19]], [[V21]]) : (index, index) -> index
         // CHECK-NEXT: hlcf.break "_loop_0"
         hlcf.loop "_loop_1" {
           %arg0_sb2 = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
           hlcf.if %arg0_sb2 {
             hlcf.yield
           } else {
             co.suspend (%hdl) {
               co.suspend.end
             }
             %result6 = kgen.call @bar(%result, %arg3): (index,index) -> index
             hlcf.break "_loop_1"
          }
          hlcf.continue
         }
         %result6 = kgen.call @bar(%result, %arg3): (index,index) -> index
         hlcf.break "_loop_0"
     }
     hlcf.continue
  }
  hlcf.return %result : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index) {
  %coro = co.invoke[(i1, index, index) async -> index:@coroutine_nested](%arg0, %arg1, %arg3)
  hlcf.return
}
}

// -----

// COM: Verify that Block Arguments Are Added To Frame.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_block_args3_resume
kgen.func @coroutine_block_args3(%arg0: index) async -> index {
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep %arg0[[[#FRAME6:]]]
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V6:%.*]] = hlcf.loop (%arg1 = [[V5]] : index) -> index {
  // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %arg0[[[#FRAME6 - 1]]]
  // CHECK-NEXT:  pop.store %arg1, [[V10]] : !kgen.pointer<index>
  // CHECK-NEXT:  %idx0 = index.constant 0
  // CHECK-NEXT:  [[V11:%.*]] = index.cmp slt(%arg1, %idx0)
  // CHECK-NEXT:  [[V11SB:%.*]] = pop.cast_from_builtin [[V11]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:  hlcf.if [[V11SB]] {
  // CHECK-NEXT:    kgen.param.constant
  // CHECK-NEXT:    kgen.struct.gep
  // CHECK-NEXT:    pop.store
  // CHECK-NEXT:    co.suspend {
  // CHECK-NEXT:      co.suspend.end
  // CHECK-NEXT:    }
  // CHECK-NEXT:    hlcf.yield
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    [[V14:%.*]] = kgen.struct.gep %arg0[[[#FRAME6 - 1]]]
  // CHECK-NEXT:    [[V15:%.*]] = pop.load [[V14]] : !kgen.pointer<index>
  // CHECK-NEXT:    hlcf.break [[V15]] : index
  // CHECK-NEXT:  }
  // CHECK-NEXT:  [[V12:%.*]] = kgen.struct.gep %arg0[[[#FRAME6 - 1]]]
  // CHECK-NEXT:  [[V13:%.*]] = pop.load [[V12]] : !kgen.pointer<index>
  // CHECK-NEXT:  hlcf.continue [[V13]] : index
  // CHECK-NEXT:  }
  %0 = hlcf.loop (%arg1 = %arg0 : index) -> index {
    %idx0 = index.constant 0
    %1 = index.cmp slt(%arg1, %idx0)
    %cb1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %cb1 {
      co.suspend (%hdl) {
        co.suspend.end
      }
      hlcf.yield
    } else {
      hlcf.break %arg1 : index
    }
    hlcf.continue %arg1 : index
  }
  %idx1 = index.constant 1
  hlcf.return %idx1 : index
}

kgen.func @triggerCold(%arg1: index) {
  %coro = co.invoke[(index) async -> index:@coroutine_block_args3](%arg1)
  hlcf.return
}

}

// -----

// COM: Verify that Block Arguments Are Not Referenced Directly Across Suspension.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_block_args1_resume
kgen.func @coroutine_block_args1(%arg0: index) async -> index {
  // CHECK: [[V6:%.*]] = hlcf.loop (%arg1 = %{{.*}} : index) -> index {
  // CHECK: [[V10:%.*]] = kgen.struct.gep %arg0[[[#FRAME5:]]]
  // CHECK: pop.store %arg1, [[V10]] : !kgen.pointer<index>
  // CHECK: co.suspend {
  // CHECK: co.suspend.end
  // CHECK: }
  // CHECK: %idx0 = index.constant 0
  // CHECK: [[V11:%.*]] = kgen.struct.gep %arg0[[[#FRAME5]]]
  // CHECK: [[V12:%.*]] = pop.load [[V11]] : !kgen.pointer<index>
  // CHECK: [[V13:%.*]] = index.cmp slt([[V12]], %idx0)
  %0 = hlcf.loop (%arg5 = %arg0 : index) -> index {
    co.suspend (%hdl) {
      co.suspend.end
    }
    %idx0 = index.constant 0
    %1 = index.cmp slt(%arg5, %idx0)
    %cb1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %cb1 {
      hlcf.yield
    } else {
      hlcf.break %arg5 : index
    }
    hlcf.continue %arg5 : index
  }
  %idx1 = index.constant 1
  hlcf.return %idx1 : index
}

kgen.func @triggerCold(%arg1: index) {
  %coro = co.invoke[(index) async -> index:@coroutine_block_args1](%arg1)
  hlcf.return
}

}

// -----

// COM: Verify that Block Arguments Are Not Put In Frame If Not Needed.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_block_args2_resume
kgen.func @coroutine_block_args2(%arg0: index) async -> index {
  // CHECK:      hlcf.loop (%arg1 = %{{.*}} : index) -> index {
  // CHECK-NEXT:   %idx0 = index.constant 0
  // CHECK-NEXT:   [[V10:%.*]] = index.cmp slt(%arg1, %idx0)
  // CHECK-NEXT:   [[V10SB:%.*]] = pop.cast_from_builtin [[V10]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:   hlcf.if [[V10SB]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break %arg1 : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:     hlcf.continue %arg1 : index
  // CHECK-NEXT:   }
  %0 = hlcf.loop (%arg5 = %arg0 : index) -> index {
    %idx0 = index.constant 0
    %1 = index.cmp slt(%arg5, %idx0)
    %cb1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %cb1 {
      hlcf.yield
    } else {
      hlcf.break %arg5 : index
    }
    hlcf.continue %arg5 : index
  }
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx1 = index.constant 1
  hlcf.return %idx1 : index
}

kgen.func @triggerCold(%arg1: index) {
  %coro = co.invoke[(index) async -> index:@coroutine_block_args2](%arg1)
  hlcf.return
}

}

// -----

// COM: Verify Dry Runs Terminate

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: kgen.func @f_resume
  kgen.func @f(%arg0: index) async {
    // CHECK: hlcf.loop
    hlcf.loop (%arg1 = %arg0 : index) {
      // CHECK-NEXT:      [[V2:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
      // CHECK-NEXT: pop.store %arg1, [[V2]]
      %0 = index.cmp slt(%arg1, %arg0)
      hlcf.loop (%arg2 = %arg1 : index) {
        %1 = index.cmp slt(%arg2, %arg0)
        %cb1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
        hlcf.if %cb1 {
          hlcf.yield
        } else {
          hlcf.break
        }
        %2 = index.add %arg2, %arg0
        hlcf.continue %2 : index
      }
      hlcf.loop (%arg2 = %arg1 : index) {
        %1 = index.cmp slt(%arg2, %arg0)
        %cb1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
        hlcf.if %cb1 {
          hlcf.yield
        } else {
          co.suspend(%hdl) {
            co.suspend.end
          }
          hlcf.break
        }
        %2 = index.add %arg2, %arg0
        hlcf.continue %2 : index
      }
      // CHECK: [[V6:%.*]] = kgen.struct.gep %arg0[[[#FRAME7]]]
      // CHECK-NEXT: [[V7:%.*]] = pop.load [[V6]] : !kgen.pointer<index>
      // CHECK-NEXT: hlcf.continue [[V7]] : index
      hlcf.continue %arg1 : index
    }
    hlcf.return
  }

  kgen.func @triggerCold(%arg0: index) {
   %coro = co.invoke[(index) async -> (): @f](%arg0)
   hlcf.return
  }
}

// -----

// COM: Verify that Nested Loops Terminate

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_nested_resume
kgen.func @coroutine_nested(%arg0: i1, %arg1: index, %arg3: index, %arg4: i1) async -> index {
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  hlcf.loop "_loop_0" {
     hlcf.loop "_loop_1" {
       // CHECK: hlcf.loop "_loop_1" {
       // CHECK-NEXT: [[V6:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
       // CHECK-NEXT: [[V7:%.*]] = pop.load [[V6]] : !kgen.pointer<index>
       // CHECK-NEXT: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME7+2]]]
       // CHECK-NEXT: [[V9:%.*]] = pop.load [[V8]] : !kgen.pointer<index>
       // CHECK-NEXT: [[V10:%.*]] = kgen.call @bar([[V7]], [[V9]]) : (index, index) -> index
       %isThisDetected = kgen.call @bar(%result, %arg3): (index,index) -> index
       %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
       hlcf.if %arg0_sb {
         hlcf.yield
       } else {
         hlcf.break "_loop_1"
       }
       %arg4_sb = pop.cast_from_builtin %arg4 : i1 to !kgen.scalar<bool>
       hlcf.if %arg4_sb {
         co.suspend (%hdl) {
          co.suspend.end
         }
         hlcf.continue
       } else {
         hlcf.yield
       }
       hlcf.continue
     }
     hlcf.continue
  }

  hlcf.return %arg1 : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index, %arg4: i1) {
 %coro = co.invoke[(i1, index, index, i1) async -> index:@coroutine_nested](%arg0, %arg1, %arg3, %arg4)
 hlcf.return
}
}

// -----

// COM: Verify DryRun Nodes With Multiple Predecessors

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_nested_resume
kgen.func @coroutine_nested(%arg0: i1, %arg1: index, %arg3: index, %arg4: i1) async -> index {
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  // CHECK: hlcf.loop
  hlcf.loop "_loop_1" {
    // CHECK-NEXT: [[V6:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
    // CHECK-NEXT: [[V7:%.*]] = pop.load [[V6]]
    // CHECK-NEXT: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 2]]]
    // CHECK-NEXT: [[V9:%.*]] = pop.load [[V8]]
    // CHECK-NEXT: kgen.call @bar([[V7]], [[V9]])
    %isThisDetected = kgen.call @bar(%result, %arg3): (index,index) -> index
    %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
    hlcf.if %arg0_sb {
      hlcf.continue
    } else {
      hlcf.yield
    }
    co.suspend (%hdl) {
      co.suspend.end
    }
    hlcf.continue
  }
  hlcf.return %arg1 : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index, %arg4: i1) {
 %coro = co.invoke[(i1, index, index, i1) async -> index:@coroutine_nested](%arg0, %arg1, %arg3, %arg4)
 hlcf.return
}
}

// -----

// COM: All Successors Must Be Updated After Dry Run

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
    // CHECK-LABEL: kgen.func @foo_resume
    kgen.func @foo(%arg0: !kgen.pointer<pointer<none>>, %arg1: i1, %arg2: index) async no_inline {
      %idx0 = index.constant 0
      %idx1 = index.constant 1
      hlcf.loop "_loop_0" {
        %0 = pop.stack_allocation 1 x struct<(pointer<none>, pointer<none>) memoryOnly> marked
        %1 = pop.load %arg0 : !kgen.pointer<pointer<none>>
        kgen.call @"CBatch::__init__"(%1, %0) : (!kgen.pointer<none>, !kgen.pointer<struct<(pointer<none>, pointer<none>) memoryOnly>> byref_result) -> ()
        co.suspend (%hdl) {
          co.suspend.end
        }
        hlcf.loop "_loop_1" (%arg3 = %arg2 : index) {
          %3 = index.cmp sgt(%arg3, %idx0)
          %cb3 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
          hlcf.if %cb3 {
            hlcf.yield
          } else {
            hlcf.break "_loop_1"
          }
          %4 = index.sub %arg2, %idx1
          hlcf.continue %4 : index
        }
        // CHECK:        hlcf.continue %
        // CHECK-NEXT: }
        // CHECK:      [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME9:]]]
        // CHECK-NEXT: [[V9:%.*]] = kgen.call @batch_size([[V8]])
        %2 = kgen.call @batch_size(%0) : (!kgen.pointer<struct<(pointer<none>, pointer<none>) memoryOnly>> imm_mem) -> index
        hlcf.continue
      }
      hlcf.return
    }

  kgen.func @triggerCold(%arg0: !kgen.pointer<pointer<none>>, %arg1: i1, %arg2: index) {
     %coro = co.invoke[(!kgen.pointer<pointer<none>>, i1, index) async -> ():@foo](%arg0, %arg1, %arg2)
     hlcf.return
  }
}

// -----

// COM: Nested Loops With Parent Suspend

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
    // CHECK-LABEL:  kgen.func @foo_resume
    kgen.func @foo(%arg0: index, %arg1: index, %arg2: index, %arg3: i1, %arg4: i1) async {
    // CHECK:      [[V2:%.*]] = kgen.call @foo3
    // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
    // CHECK-NEXT: pop.store [[V2]], [[V3]] : !kgen.pointer<index>
      %0 = kgen.call @foo3(%arg2) : (index) -> index
      %1:3 = hlcf.loop "_loop_2" (%arg5 = %arg0 : index, %arg6 = %arg1 : index, %arg7 = %arg1 : index, %arg8 = %arg1 : index) -> (index, index, index) {
        %2 = hlcf.loop "_loop_0" (%arg9 = %arg2 : index, %arg10 = %arg2 : index) -> index {
          %3 = kgen.call @bar1(%arg9) : (index) -> i1
          // CHECK: hlcf.loop "_loop_1"
          // CHECK-NEXT: [[V25:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
          // CHECK-NEXT: [[V26:%.*]] = pop.load [[V25]] : !kgen.pointer<index>
          // CHECK-NEXT: [[V27:%.*]] kgen.call @bar2([[V26]]) : (index) -> i1
          %4 = hlcf.loop "_loop_1" (%arg11 = %arg1 : index, %arg12 = %arg1 : index) -> index {
            %5 = kgen.call @bar2(%0) : (index) -> i1
            %cb5 = pop.cast_from_builtin %5 : i1 to !kgen.scalar<bool>
            hlcf.if %cb5 {
              hlcf.continue %arg1, %arg1 : index, index
            } else {
              hlcf.break "_loop_1" %arg11 : index
            }
            hlcf.unreachable
          }
          %arg3_sb1 = pop.cast_from_builtin %arg3 : i1 to !kgen.scalar<bool>
          hlcf.if %arg3_sb1 {
            hlcf.break "_loop_0" %arg9 : index
          } else {
            hlcf.yield
          }
          hlcf.continue %arg1, %arg1 : index, index
        }
        co.suspend (%hdl) {
          co.suspend.end
        }
        %arg3_sb2 = pop.cast_from_builtin %arg3 : i1 to !kgen.scalar<bool>
        hlcf.if %arg3_sb2 {
          hlcf.break "_loop_2" %arg1, %arg6, %arg7 : index, index, index
        } else {
          hlcf.continue %arg1, %arg6, %arg7, %arg8 : index, index, index, index
        }
        hlcf.unreachable
      }
      hlcf.return
    }

  kgen.func @triggerCold(%arg0: index, %arg1: index, %arg2: index, %arg3: i1, %arg4: i1) {
     %coro = co.invoke[(index, index, index, i1, i1) async -> ():@foo](%arg0, %arg1, %arg2, %arg3, %arg4)
     hlcf.return
  }
}

// -----

// COM: Frame Addresses Are Not Stored In Frame

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @gep(%arg0: i1 imm, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)> marked
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<struct<(index, index)>>
    %1 = kgen.call @fillMe(%0) : (!kgen.pointer<struct<(index, index)>> byref_result) -> index
    %2 = kgen.struct.gep %0[1] : <struct<(index, index)>>
    co.suspend (%hdl) {
      co.suspend.end
    }
    // CHECK-LABEL: kgen.func @gep_resume
    // CHECK: co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[7]
    // CHECK-SAME: <struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>, struct<()>, struct<(index, index)>)>>
    // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep [[V3]][1] : <struct<(index, index)>>
    // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
    // CHECK-NEXT:  kgen.call @doSomething([[V5]]) : (index) -> index
    %4 = pop.load %2 : !kgen.pointer<index>
    %3 = kgen.call @doSomething(%4) : (index) -> index
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<struct<(index, index)>>
    hlcf.return
  }
  kgen.func @offset(%arg0: i1 imm, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    // CHECK-LABEL: kgen.func @offset_resume
    // CHECK:      [[V1:%.*]] = index.constant 1
    // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
    // CHECK-NEXT: [[V4:%.*]] = pop.pointer.bitcast [[V3]]
    // CHECK-NEXT: [[V5:%.*]] = pop.offset [[V4]][[[V1]]] : !kgen.pointer<index>
    %0 = pop.stack_allocation 2 x index marked
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
    %idx1 = index.constant 1
    %1 = pop.offset %0[%idx1] : !kgen.pointer<index>
    hlcf.loop {
      %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
      hlcf.if %arg0_sb {
        hlcf.yield
      } else {
        hlcf.break
      }
      co.suspend (%hdl) {
        co.suspend.end
      }
      // CHECK:      [[V6:%.*]] = index.constant 1
      // CHECK-NEXT: [[V7:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
      // CHECK-NEXT: [[V8:%.*]] = pop.pointer.bitcast [[V7]]
      // CHECK-NEXT: [[V9:%.*]] = pop.offset [[V8]][[[V6]]] : !kgen.pointer<index>
      // CHECK-NEXT: [[V10:%.*]] = pop.load [[V9]]
      // CHECK-NEXT: [[V11:%.*]] = kgen.call @doSomething([[V10]])
      %4 = pop.load %1 : !kgen.pointer<index>
      %3 = kgen.call @doSomething(%4) : (index) -> index
      hlcf.continue
    }
    // CHECK:      [[V12:%.*]] = index.constant 1
    // CHECK-NEXT: [[V13:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
    // CHECK-NEXT: [[V14:%.*]] = pop.pointer.bitcast [[V13]]
    // CHECK-NEXT: [[V15:%.*]] = pop.offset [[V14]][[[V12]]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V16:%.*]] = pop.load [[V15]]
    // CHECK-NEXT: [[V17:%.*]] = kgen.call @doSomething([[V16]])
    %5 = pop.load %1 : !kgen.pointer<index>
    %6 = kgen.call @doSomething(%5) : (index) -> index
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
    hlcf.return
  }

  kgen.func @triggerCold(%arg0: i1) {
     %coro = co.invoke[(i1 imm, !kgen.pointer<none> byref_result) async -> (): @offset](%arg0)
     %coro2 = co.invoke[(i1 imm, !kgen.pointer<none> byref_result) async -> (): @gep](%arg0)
     hlcf.return
  }

}

// -----

// COM: Verify Block Storage and Unreachable inserts in Hot Ramp

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_hot_ramp(
kgen.func @coroutine(%arg0: i1, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async -> index {
 // CHECK:      [[CORO:%.*]] = pop.aligned_alloc
 // CHECK:      [[V11:%.*]] = pop.pointer.bitcast [[CORO]]
 // CHECK-NEXT: hlcf.loop "_loop_0" ([[BLOCK_ARG:%.*]] =
 // Check that block arguments are stored in frame.
 // CHECK-NEXT: [[V9:%.*]] = kgen.struct.gep %0[[[#FRAME8:]]]
 // CHECK-NEXT: pop.store [[BLOCK_ARG]], [[V9]] : !kgen.pointer<index>

 // CHECK-NEXT: hlcf.loop "_loop_1" ([[BLOCK_ARG_INNER:%.*]] =
 // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %0[[[#FRAME8 - 1]]]
 // CHECK-NEXT: pop.store [[BLOCK_ARG_INNER]], [[V10]] : !kgen.pointer<index>

 // Verify suspension point in nested loops is properly replaced and
 // unreachable is inserted to terminate unreachable blocks.
 // CHECK-NEXT:   [[COND0:%.*]] = pop.cast_from_builtin %arg2 : i1 to !kgen.scalar<bool>
 // CHECK-NEXT:   hlcf.if [[COND0]] {
 // CHECK-NEXT:     hlcf.break "_loop_1"
 // CHECK-NEXT:   } else {
 // CHECK-NEXT:     hlcf.yield
 // CHECK-NEXT:   }

 // CHECK-NEXT:   kgen.param.constant: i32 = <1>
 // CHECK-NEXT:   kgen.struct.gep [[CORO]][0]
 // CHECK-NEXT:   pop.store

 // CHECK-NEXT:   hlcf.return [[V11]]
 // CHECK-NEXT: }
 // CHECK-NEXT: kgen.call @print
 // CHECK-NEXT: hlcf.continue
 // CHECK-NEXT: }
 // CHECK-NEXT: hlcf.unreachable
 hlcf.loop "_loop_0" (%arg3 = %arg1 : index) {
   hlcf.loop "_loop_1" (%arg2 = %arg1 : index) {
     %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
     hlcf.if %arg0_sb {
       hlcf.break "_loop_1"
     } else {
       hlcf.yield
     }
     co.suspend (%hdl) {
       co.suspend.end
     }
     kgen.call @print1(%arg2) : (index) -> ()
     hlcf.continue %arg2 : index
   }
   kgen.call @print(%arg0) : (i1) -> ()
   hlcf.continue %arg3 : index
 }
 co.suspend (%hdl) {
   co.suspend.end
 }
 %final = index.add %arg1, %arg1
 hlcf.return %final : index
}

// Check that the operands of parents that are state 0 are replaced with constants. All other ops in state 0 will be erased.
// CHECK-LABEL:  kgen.func @coroutine_resume
// CHECK-NEXT:   [[UNDEF:%.*]] = kgen.param.constant = <#interp.uninitmem>
// CHECK-NEXT:   hlcf.loop "_loop_0" (%arg1 = [[UNDEF]] : index) {
kgen.func @trigger_creation(%arg0: i1, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async {
   %coro = co.hot_invoke[(i1, index, !kgen.pointer<index> byref_result) async -> index: @coroutine](%arg0, %arg1, %__result__)
   hlcf.return
}

}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
kgen.func @coroutine(%arg0: i1, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async -> index {
  // CHECK: [[V2:%.*]] = kgen.call @doSomething({{.*}}) : (index) -> index
  // CHECK-NOT: hlcf.loop "_loop_1" (%arg2 = [[V2]] : index) {
  %x = kgen.call @doSomething(%arg1) : (index) -> index
  hlcf.loop "_loop_0" (%arg3 = %arg1 : index) {
    hlcf.loop "_loop_1" (%arg2 = %x : index) {
      %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
      hlcf.if %arg0_sb {
        hlcf.break "_loop_1"
      } else {
        hlcf.yield
      }
      co.suspend (%hdl) {
        co.suspend.end
      }
      hlcf.continue %arg2 : index
    }
    hlcf.continue %arg3 : index
  }
  co.suspend (%hdl) {
    co.suspend.end
  }
  %final = index.add %arg1, %arg1
  hlcf.return %final : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index) {
   %coro = co.invoke[(i1, index, !kgen.pointer<index> byref_result) async -> index: @coroutine](%arg0, %arg1)
   hlcf.return
}

}
