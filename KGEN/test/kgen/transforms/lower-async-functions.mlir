// RUN: kgen-opt -lower-async-functions -split-input-file %s | FileCheck %s

// COM: Verify Ramp + Resume + Async Calls are transformed correctly.
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @coroutine1(%arg0: i1) async -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK:      kgen.func @coroutine_ramp(%arg0: i1) -> !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>> {
// CHECK-NEXT:   %idx64 = index.constant 64
// CHECK-NEXT:   %idx8 = index.constant 8
// CHECK-NEXT:   [[CONTINUATION:%.*]] = pop.aligned_alloc %idx8, %idx64 : <struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, struct<(index)>, i1)>>
// CHECK-NEXT:   [[ZERO:%.*]] = kgen.param.constant: i32 = <0>
// CHECK-NEXT:   [[STATESLOT:%.*]] = kgen.struct.gep [[CONTINUATION]][0]
// CHECK-NEXT:   pop.store [[ZERO]], [[STATESLOT]] : !kgen.pointer<i32>
// CHECK-NEXT:   [[RESUME_SLOT:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME1:]]]
// CHECK-NEXT:   [[RESUME_FNC_PTR:%.*]] = kgen.create_closure[{{.*}}: @coroutine_resume]()
// CHECK-NEXT:   [[RESUME_FNC_PTR_OPAQUE:%.*]] = pop.pointer.bitcast [[RESUME_FNC_PTR]]
// CHECK-NEXT:   pop.store [[RESUME_FNC_PTR_OPAQUE]], [[RESUME_SLOT]]
// CHECK-NEXT:   [[ARG0_SLOT:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME1:]]]
// CHECK-NEXT:   pop.store %arg0, [[ARG0_SLOT]] : !kgen.pointer<i1>
// CHECK-NEXT:   [[HEADER:%.*]] = pop.pointer.bitcast [[CONTINUATION]] : !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, struct<(index)>, i1)>> to !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
// CHECK-NEXT:   kgen.return [[HEADER]]
// CHECK-NEXT: }

// CHECK-LABEL: kgen.func @coroutine_resume
// CHECK-SAME:  coroutineType = !kgen.struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, struct<(index)>, i1)>
// CHECK-NEXT:    [[ARG0_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME1]]]
// CHECK-NEXT:    [[ARG0:%.*]] = pop.load [[ARG0_SLOT]] : !kgen.pointer<i1>
// CHECK-NEXT:    hlcf.if [[ARG0]] {
// CHECK-NEXT:      %idx1 = index.constant 1
// CHECK-NEXT:      [[PROMISE_SLOT:%.*]] = kgen.struct.gep %arg0[[[#PROMISE_IDX:]]]
// CHECK-NEXT:      [[RESULT_PTR:%.*]] = kgen.struct.gep [[PROMISE_SLOT]][0]
// CHECK-NEXT:      pop.store %idx1, [[RESULT_PTR]] : !kgen.pointer<index>
// CHECK-NEXT:      kgen.return
// CHECK-NEXT:    } else {
// CHECK-NEXT:      hlcf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    %true = index.bool.constant true
// CHECK-NEXT:    kgen.call @coroutine1_ramp(%true)
// CHECK:         co.suspend {
// CHECK-NEXT:      co.suspend.end
// CHECK-NEXT:    }
// CHECK-NEXT:    %idx0 = index.constant 0
// CHECK-NEXT:    [[PROMISE_SLOT:%.*]] = kgen.struct.gep %arg0[[[#PROMISE_IDX]]]
// CHECK-NEXT:    [[RESULT_PTR:%.*]] = kgen.struct.gep [[PROMISE_SLOT]][0]
// CHECK-NEXT:    pop.store %idx0, [[RESULT_PTR]] : !kgen.pointer<index>
// CHECK-NEXT:    kgen.return
// CHECK-NEXT:  }
kgen.func @coroutine(%arg0: i1) async -> index {
  hlcf.if %arg0 {
    %idx1 = index.constant 1
    kgen.return %idx1 : index
  } else {
    hlcf.yield
  }
  %true = index.bool.constant true
  %result = co.invoke[(i1) async -> index: @coroutine1](%true)
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @call_coroutine
kgen.func @call_coroutine() {
  %true = index.bool.constant true
  // CHECK: kgen.call @coroutine_ramp(%true) :
  // CHECK-SAME: (i1) -> !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
  %result = co.invoke[(i1) async -> index: @coroutine](%true)
  // CHECK-NEXT: kgen.return
  kgen.return
}
}

// -----

// COM: Verify Loop With Await In Then Statement Is Correct

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine1_resume
kgen.func @coroutine1(%arg0: i1, %arg1: index, %arg2: index, %arg3: index) async -> index {
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V5:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
  // CHECK-NEXT: [[V6:%.*]] = pop.load [[V5]] : !kgen.pointer<index>

  // CHECK-NEXT: [[V7:%.*]] = kgen.call @foo(%idx3, [[V6]]) : (index, index) -> index
  // CHECK-NEXT: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 1]]]
  // CHECK-NEXT: pop.store [[V7]], [[V8]] : !kgen.pointer<index>
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index

  // CHECK-NEXT: hlcf.loop
  hlcf.loop "_loop_0" {
    // CHECK-NEXT: [[V19:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 1]]]
    // CHECK-NEXT: [[V20:%.*]] = pop.load [[V19]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V21:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 2]]]
    // CHECK-NEXT: [[V22:%.*]] = pop.load [[V21]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V23:%.*]] = kgen.call @bar([[V20]], [[V22]]) : (index, index) -> index
    %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index


    // CHECK-NEXT: [[V24:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 3]]]
    // CHECK-NEXT: [[V25:%.*]] = pop.load [[V24]] : !kgen.pointer<i1>
    // CHECK-NEXT: hlcf.if [[V25]] {
    // CHECK-NEXT:   co.suspend {
    // CHECK-NEXT:     co.suspend.end
    // CHECK-NEXT:   }
    // CHECK-NEXT:   hlcf.yield
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   hlcf.break "_loop_0"
    // CHECK-NEXT: }
    hlcf.if %arg0 {
       co.suspend (%hdl) {
         co.suspend.end
       }
       hlcf.yield
    } else {
       hlcf.break "_loop_0"
    }
    // CHECK-NEXT: %idx3_0 = index.constant 3
    // CHECK-NEXT: [[V28:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 4]]]
    // CHECK-NEXT: [[V29:%.*]] = pop.load [[V28]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V30:%.*]] = kgen.call @foo(%idx3_0, [[V29]]) : (index, index) -> index
    %result6 = kgen.call @foo(%idx3, %arg2) : (index,index) -> index

    // CHECK-NEXT: hlcf.continue
    hlcf.continue
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V9:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 1]]]
  // CHECK-NEXT: [[V10:%.*]] = pop.load [[V9]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep %arg0[[[#FRAME7]]]
  // CHECK-NEXT: [[V12:%.*]] = pop.load [[V11]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @bar([[V10]], [[V12]]) : (index, index) -> index
  %result5 = kgen.call @bar(%result, %arg1): (index,index) -> index

  // CHECK-NEXT: [[V14:%.*]] = kgen.struct.gep %arg0[[[#PROMISE_IDX:]]]
  // CHECK-NEXT: [[PTR:%.*]] = kgen.struct.gep [[V14]][0]
  // CHECK-NEXT: pop.store [[V10]], [[PTR]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.return
  kgen.return %result : index
}

kgen.func @foo(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @bar(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
}

// -----

// COM: Verify Loop With Await In Else Statement Is Correct

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine2_resume
kgen.func @coroutine2(%arg0: i1, %arg1: index, %arg3: index) async -> index {
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep %arg0[[[#FRAME5:]]]
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[NOT_IN_FRAME:%.*]] = kgen.call @foo(%idx3, [[V5]]) : (index, index) -> index
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  hlcf.loop "_loop_0" {
     hlcf.if %arg0 {
       hlcf.yield
     } else {
       // CHECK: } else {
       // CHECK-NEXT: [[V17:%.*]] = kgen.struct.gep %arg0[[[#FRAME5 + 1]]]
       // CHECK-NEXT: [[V18:%.*]] = pop.load [[V17]] : !kgen.pointer<index>
       // CHECK-NEXT: [[V19:%.*]] = kgen.call @bar([[NOT_IN_FRAME]], [[V18]]) : (index, index) -> index
       // CHECK-NEXT: co.suspend
       %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index
       co.suspend (%hdl) {
         co.suspend.end
       }
       hlcf.break "_loop_0"
     }
     hlcf.continue
  }
  // CHECK:      [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME5 + 3]]]
  // CHECK-NEXT: [[V9:%.*]] = pop.load [[V8]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %arg0[[[#PROMISE_IDX:]]]
  // CHECK-NEXT: [[PTR:%.*]] = kgen.struct.gep [[V10]][0]
  // CHECK-NEXT: pop.store [[V9]], [[PTR]]
  // CHECK-NEXT: kgen.return
  kgen.return %result : index
}

kgen.func @foo(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @bar(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
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
    // CHECK-NEXT: [[V13:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
    // CHECK-NEXT: [[V14:%.*]] = pop.load [[V13]] : !kgen.pointer<i1>
    // CHECK-NEXT: hlcf.if [[V14]] {
    // CHECK-NEXT:   hlcf.yield
    // CHECK-NEXT: } else {
    // CHECK-NEXT: [[V15:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 2]]]
    // CHECK-NEXT: [[V16:%.*]] = pop.load [[V15]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V17:%.*]] = kgen.call @bar([[NIF]], [[V16]]) : (index, index) -> index
    // CHECK-NEXT: [[V18:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
    // CHECK-NEXT: pop.store [[V17]], [[V18]] : !kgen.pointer<index>
    // CHECK-NEXT: co.suspend {
    // CHECK-NEXT:   co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V19:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
    // CHECK-NEXT: [[V20:%.*]] = pop.load [[V19]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V21:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 2]]]
    // CHECK-NEXT: [[V22:%.*]] = pop.load [[V21]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V23:%.*]] = kgen.call @bar([[V20]], [[V22]]) : (index, index) -> index
    // CHECK-NEXT: co.suspend {
    // CHECK-NEXT:   co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: hlcf.break "_loop_0"
    // CHECK-NEXT: }
     hlcf.if %arg0 {
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
  kgen.return %result : index
}

kgen.func @foo(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @bar(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
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
     hlcf.if %arg0 {
        hlcf.yield
     } else {
         %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index
         // CHECK:      hlcf.loop "_loop_1" {
         // CHECK-NEXT: [[V23:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
         // CHECK-NEXT: [[V24:%.*]] = pop.load [[V23]] : !kgen.pointer<i1>
         // CHECK-NEXT: hlcf.if [[V24]] {
         // CHECK-NEXT:   hlcf.yield
         // CHECK-NEXT: } else {
         // CHECK-NEXT:   co.suspend {
         // CHECK-NEXT:     co.suspend.end
         // CHECK-NEXT:   }
         // CHECK-NEXT:   [[V25:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
         // CHECK-NEXT:   [[V26:%.*]] = pop.load [[V25]] : !kgen.pointer<index>
         // CHECK-NEXT:   [[V27:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 2]]]
         // CHECK-NEXT:   [[V28:%.*]] = pop.load [[V27]] : !kgen.pointer<index>
         // CHECK-NEXT:   [[V29:%.*]] = kgen.call @bar([[V26]], [[V28]]) : (index, index) -> index
         // CHECK-NEXT:   hlcf.break "_loop_1"
         // CHECK-NEXT: }
         // CHECK-NEXT: hlcf.continue
         // CHECK-NEXT: }
         // CHECK-NEXT: [[V18:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
         // CHECK-NEXT: [[V19:%.*]] = pop.load [[V18]] : !kgen.pointer<index>
         // CHECK-NEXT: [[V20:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 2]]]
         // CHECK-NEXT: [[V21:%.*]] = pop.load [[V20]] : !kgen.pointer<index>
         // CHECK-NEXT: [[V22:%.*]] = kgen.call @bar([[V19]], [[V21]]) : (index, index) -> index
         // CHECK-NEXT: hlcf.break "_loop_0"
         hlcf.loop "_loop_1" {
           hlcf.if %arg0 {
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
  kgen.return %result : index
}

kgen.func @foo(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @bar(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
}

// -----

// COM: Verify that Await Code Loads/Stores From Frame.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine2_resume
kgen.func @coroutine2(%arg0: i1, %arg1: index, %arg3: index) async -> index {
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V2:%.*]] = kgen.call @foo(%idx3, [[V5]]) : (index, index) -> index
  // CHECK-NEXT: [[V7:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 2]]]
  // CHECK-NEXT: pop.store [[V2]], [[V7]] : !kgen.pointer<index>
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 1]]]
  // CHECK-NEXT:   [[V9:%.*]] = pop.load [[V8]]
  // CHECK-NEXT:   kgen.call @bar([[V2]], [[V9]]) : (index, index) -> index
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT:  }
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  co.suspend (%hdl) {
    %result2 = kgen.call @bar(%result, %arg3): (index,index) -> index
    co.suspend.end
  }
  kgen.return %result : index
}

kgen.func @foo(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @bar(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
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
  // CHECK-NEXT:  hlcf.if [[V11]] {
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
    hlcf.if %1 {
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
  kgen.return %idx1 : index
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
    hlcf.if %1 {
      hlcf.yield
    } else {
      hlcf.break %arg5 : index
    }
    hlcf.continue %arg5 : index
  }
  %idx1 = index.constant 1
  kgen.return %idx1 : index
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
  // CHECK-NEXT:   hlcf.if [[V10]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break %arg1 : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:     hlcf.continue %arg1 : index
  // CHECK-NEXT:   }
  %0 = hlcf.loop (%arg5 = %arg0 : index) -> index {
    %idx0 = index.constant 0
    %1 = index.cmp slt(%arg5, %idx0)
    hlcf.if %1 {
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
  kgen.return %idx1 : index
}
}

// -----

// COM: Verify that Unused Arguments Are Not Put In Frame.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @unused_args_resume
// CHECK-SAME: coroutineType = !kgen.struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, struct<(index)>, index)>
kgen.func @unused_args(%arg0: index, %arg1: index) async -> index {
  co.suspend (%hdl) {
    co.suspend.end
  }
  %result = kgen.call @foo(%arg0) : (index) -> index
  kgen.return %result : index
}

kgen.func @bar(%arg0: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
}

// -----

// COM: Test Try Raise

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @tryraise_resume
kgen.func @tryraise(%arg1: index, %arg2 : index) async -> index {
  // CHECK: [[NIF:%.*]] = kgen.call @foo1
  %result3 = kgen.call @foo1(%arg1) : (index) -> index
  lit.try {
    hlcf.elif {
      %result = kgen.call @bar(%arg2) : (index) -> i1
      hlcf.elif.yield %result : i1
    } then {
      // CHECK: } then {
      // CHECK-NEXT: [[V7:%.*]] = kgen.struct.gep %arg0[[[#FRAME6:]]]
      // CHECK-NEXT: [[V8:%.*]] = pop.load [[V7]] : !kgen.pointer<index>
      // CHECK-NEXT: [[V9:%.*]] = kgen.call @foo([[V8]]) : (index) -> index
      // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %arg0[[[#FRAME6 + 1]]]
      // CHECK-NEXT: pop.store [[V9]], [[V10]] : !kgen.pointer<index>
      // CHECK-NEXT: co.suspend {
      // CHECK-NEXT:   co.suspend.end
      // CHECK-NEXT: }
      // CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep %arg0[[[#FRAME6 + 1]]]
      // CHECK-NEXT: [[V12:%.*]] = pop.load [[V11]] : !kgen.pointer<index>
      // CHECK-NEXT: lit.try.raise [[V12]] : index
      %result2 = kgen.call @foo(%arg2) : (index) -> index
      co.suspend (%hdl) {
        co.suspend.end
      }
      lit.try.raise %result2 : index
    } else {
      hlcf.yield
    }
    lit.try.yield
  } except (%e: index) {
    // CHECK:     } except (%arg1: index) {
    // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %arg0[[[#FRAME6 + 2]]]
    // CHECK-NEXT: pop.store %arg1, [[V10]] : !kgen.pointer<index>
    // CHECK-NEXT: co.suspend {
    // CHECK-NEXT: co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep %arg0[[[#FRAME6 + 2]]]
    // CHECK-NEXT: [[V12:%.*]] = pop.load [[V11]]
    // CHECK:      pop.store [[V12]], {{.*}} : !kgen.pointer<index>
    // CHECK-NEXT: kgen.return
    co.suspend (%hdl) {
      co.suspend.end
    }
    kgen.return %e : index
  } else {
    // CHECK: } else {
    // CHECK:      pop.store [[NIF]], {{.*}}
    // CHECK-NEXT: kgen.return
    kgen.return %result3 : index
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
kgen.func @bar(%arg0: index) -> i1 {
  %7 = kgen.param.constant: i1 = <0>
  kgen.return %7 : i1
}
kgen.func @foo(%arg0: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
kgen.func @foo1(%arg0: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
}

// -----

// COM: Verify Set Error/Results Op is Lowered Correctly

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @populate(%__result__: !kgen.pointer<index> byref_result) -> i1 {
  %true = index.bool.constant true
  kgen.return %true : i1
}

kgen.func @use(%a: !kgen.pointer<index> borrow_in_mem) -> i1 {
  %true = index.bool.constant true
  kgen.return %true : i1
}

// CHECK-LABEL: kgen.func @throwing_coroutine_resume
kgen.func @throwing_coroutine(%__error__: !kgen.pointer<index> byref_error,
                              %__result__: !kgen.pointer<index> byref_result) throws|async -> i1 {
  // CHECK-NEXT: [[RESSLOT:%.*]] = kgen.struct.gep %arg0[[[#RESULT:]]]
  // CHECK-NEXT: [[RES:%.*]] = pop.load [[RESSLOT]] : !kgen.pointer<pointer<none>>
  // CHECK-NEXT: [[RESTYPED:%.*]] = pop.pointer.bitcast [[RES]] : !kgen.pointer<none> to !kgen.pointer<index>
  // CHECK-NEXT: [[V4:%.*]] = kgen.call @populate([[RESTYPED]]) : (!kgen.pointer<index> byref_result) -> i1
  // CHECK-NEXT: kgen.call @use([[RESTYPED]]) : (!kgen.pointer<index> borrow_in_mem) -> i1
  %0 = kgen.call @populate(%__result__) : (!kgen.pointer<index> byref_result) -> i1
  %2 = kgen.call @use(%__result__) : (!kgen.pointer<index> borrow_in_mem) -> i1
  hlcf.if %0 {
    // CHECK: hlcf.if [[V4]] {
    // CHECK-NEXT: [[ERRSLOT:%.*]] = kgen.struct.gep %arg0[[[#ERROR:]]]
    // CHECK-NEXT: [[ERR:%.*]] = pop.load [[ERRSLOT]] : !kgen.pointer<pointer<none>>
    // CHECK-NEXT: [[TYPEDERR:%.*]] = pop.pointer.bitcast [[ERR]] : !kgen.pointer<none> to !kgen.pointer<index>
    // CHECK-NEXT: kgen.call @populate([[TYPEDERR]]) : (!kgen.pointer<index> byref_result) -> i1
    %1 = kgen.call @populate(%__error__) : (!kgen.pointer<index> byref_result) -> i1
    kgen.return %1 : i1
  } else {
    hlcf.yield
  }
  co.suspend (%hdl) {
    co.suspend.end
  }
  %1 = kgen.call @populate(%__result__) : (!kgen.pointer<index> byref_result) -> i1
  kgen.return %1 : i1
}

// CHECK-LABEL: kgen.func @call_throwing_coro
kgen.func @call_throwing_coro() {
  %size = index.constant 1
  %align = index.constant 8
  // CHECK:      [[CONT:%.*]] = kgen.call @throwing_coroutine_ramp()
  // CHECK-NEXT: [[ERR:%.*]] = pop.aligned_alloc %idx8, %idx1 : <index>
  // CHECK-NEXT: [[RES:%.*]] = pop.aligned_alloc %idx8, %idx1 : <index>
  // CHECK-NEXT: [[ERRORSLOT:%.*]] = kgen.struct.gep [[CONT]][[[#ERROR]]]
  // CHECK-NEXT: [[TYPED_ERRORSLOT:%.*]] = pop.pointer.bitcast [[ERRORSLOT]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store [[ERR]], [[TYPED_ERRORSLOT]] : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: [[RESSLOT:%.*]] = kgen.struct.gep [[CONT]][[[#RESULT]]]
  // CHECK-NEXT: [[TYPED_RESSLOT:%.*]] = pop.pointer.bitcast [[RESSLOT]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store [[RES]], [[TYPED_RESSLOT]] : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: kgen.return
  %coro = co.invoke[(!kgen.pointer<index> byref_error, !kgen.pointer<index> byref_result) throws|async -> i1: @throwing_coroutine]()
  %0 = pop.aligned_alloc %align, %size : <index>
  %1 = pop.aligned_alloc %align, %size : <index>
  co.set_byref_error_result %coro(%1, %0) : !co.routine, !kgen.pointer<index>, !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @use2
kgen.func @use2(%a: !co.routine) -> i1 {
  %true = index.bool.constant true
  kgen.return %true : i1
}

// CHECK-LABEL: kgen.func @opaque_coro
kgen.func @opaque_coro(%coro: !co.routine, %arg1: !kgen.pointer<index>, %arg2: !kgen.pointer<index>) {
  // CHECK-NEXT: kgen.call @use2(%arg0)
  %2 = kgen.call @use2(%coro) : (!co.routine) -> i1

  // CHECK-NEXT: [[v3:%.*]] = kgen.struct.gep %arg0[[[#ERROR]]]
  // CHECK-NEXT: [[v4:%.*]] = pop.pointer.bitcast [[v3]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store %arg1, [[v4]] : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: [[v5:%.*]] = kgen.struct.gep %arg0[[[#RESULT]]]
  // CHECK-NEXT: [[v6:%.*]] = pop.pointer.bitcast [[v5]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store %arg2, [[v6]] : !kgen.pointer<pointer<index>>
  co.set_byref_error_result %coro(%arg2, %arg1) : !co.routine, !kgen.pointer<index>, !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @no_error_slot
kgen.func @no_error_slot(%arg0: !co.routine, %arg1: !kgen.pointer<index>) {
  // CHECK-NEXT: [[v5:%.*]] = kgen.struct.gep %arg0[[[#RESULT]]]
  // CHECK-NEXT: [[v6:%.*]] = pop.pointer.bitcast [[v5]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store %arg1, [[v6]] : !kgen.pointer<pointer<index>>
  co.set_byref_error_result %arg0(%arg1) : !co.routine, !kgen.pointer<index>
  kgen.return
}

// CEHCK-LABEL: kgen.func @set_byref_none
kgen.func @set_byref_none(%arg0: !co.routine, %arg1: !kgen.pointer<none>) {
  // CHECK-NOT: kgen.struct.gep %arg0[[[#RESULT]]]
  co.set_byref_error_result %arg0(%arg1) : !co.routine, !kgen.pointer<none>
  co.set_byref_error_result %arg0(%arg1, %arg1) : !co.routine, !kgen.pointer<none>, !kgen.pointer<none>
  kgen.return
}

}

// -----

// COM: Stack Allocations Are Lowered To Frame Allocations

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @use(%arg0:!kgen.pointer<index>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg1: index, %arg2: index) async -> index {
  %0 = pop.stack_allocation 2 x index marked
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  %idx1 = index.constant 1
  // CHECK-NEXT: %idx1 = index.constant 1
  // CHECK-NEXT: [[V1:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
  // CHECK-NEXT: [[V2:%.*]] = pop.load [[V1]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[[[#FRAME7+2]]]
  // CHECK-NEXT: [[V4:%.*]] = pop.pointer.bitcast [[V3]] : !kgen.pointer<array<2, index>> to !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[V2]], [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V5:%.*]] = pop.offset [[V4]][%idx1] : !kgen.pointer<index>
  // CHECK-NEXT: [[V6:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 1]]]
  // CHECK-NEXT: [[V7:%.*]] = pop.load [[V6]]
  // CHECK-NEXT: pop.store [[V7]], [[V5]]
  // CHECK-NEXT: kgen.call @use([[V5]])
  // CHECK-NEXT: kgen.call @use([[V4]])
  pop.store %arg1, %0 : !kgen.pointer<index>
  %1 = pop.offset %0[%idx1] : !kgen.pointer<index>
  pop.store %arg2, %1 : !kgen.pointer<index>
  %3 = kgen.call @use(%1) : (!kgen.pointer<index>) -> index
  %22 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 2]]]
  // CHECK-NEXT: [[V9:%.*]] = pop.pointer.bitcast [[V8]] : !kgen.pointer<array<2, index>> to !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @use([[V9]]) : (!kgen.pointer<index>) -> index
  // CHECK-NOT:  pop.stack_alloc.lifetime
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
}

// -----

// COM: Stack Allocations Are Not Lowered To Frame Allocations If Lifetime Contained In State

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @use(%arg0:!kgen.pointer<index>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg1: index) async -> index {
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK:      co.suspend
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V1:%.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_alloc.lifetime.start
  // CHECK-NEXT: [[V2:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
  // CHECK-NEXT: [[V3:%.*]] = pop.load [[V2]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[V3]], [[V1]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @use([[V1]]) : (!kgen.pointer<index>) -> index
  // CHECK:  pop.stack_alloc.lifetime
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  kgen.return %2 : index
}
}

// -----

// COM: Stack Allocations Are Lowered To Frame Allocations When Stack Allocation Not Used Outside Lifetime End

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {


kgen.func @use(%arg0:!kgen.pointer<index>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg1: index, %arg2: index) async -> index {
  %0 = pop.stack_allocation 2 x index marked
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  // CHECK-NEXT: %idx1 = index.constant 1
  %idx1 = index.constant 1
  // Extract pointer to inline frame memory instead of stack allocation.
  // CHECK-NEXT: [[V0:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[V1:%.*]] = pop.pointer.bitcast [[V0]] : !kgen.pointer<array<2, index>> to !kgen.pointer<index>


  // CHECK-NEXT: [[V2:%.*]] = pop.offset [[V1]][%idx1] : !kgen.pointer<index>
  // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[V4:%.*]] = pop.load [[V3]]
  // CHECK-NEXT: pop.store [[V4]], [[V2]]
  %1 = pop.offset %0[%idx1] : !kgen.pointer<index>
  pop.store %arg2, %1 : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK:      %idx1_0 = index.constant 1
  // CHECK-NEXT: [[STACK_MEM:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
  // CHECK-NEXT: [[V6:%.*]] = pop.pointer.bitcast [[STACK_MEM]]
  // CHECK-NEXT: [[V7:%.*]] = pop.offset [[V6]][%idx1_0]
  // CHECK-NEXT: [[V8:%.*]] = kgen.call @use([[V7]]) : (!kgen.pointer<index>) -> index
  // CHECK-NOT:  pop.stack_alloc.lifetime
  %2 = kgen.call @use(%1) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  kgen.return %2 : index
}
}

// -----

// COM: Stack Allocations Of Size 1 Do Not Have Array Types

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @use(%arg0:!kgen.pointer<struct<(index, index)>>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg1: index) async -> index {
  // CHECK-NEXT: [[STACK_ALLOC:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[SLOT:%.*]] = kgen.struct.gep [[STACK_ALLOC]][1] : <struct<(index, index)>>
  // CHECK-NEXT: [[ARG_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[ARG:%.*]] = pop.load [[ARG_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[ARG]], [[SLOT]] : !kgen.pointer<index>
  %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)> marked
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<struct<(index, index)>>
  %1 = kgen.struct.gep %0[1] : !kgen.pointer<struct<(index,index)>>
  pop.store %arg1, %1 : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK:       [[STACK_ALLOC2:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
  // CHECK-NEXT:  kgen.call @use([[STACK_ALLOC2]])
  %2 = kgen.call @use(%0) : (!kgen.pointer<struct<(index, index)>>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<struct<(index, index)>>
  kgen.return %2 : index
}
}

// -----


// COM: Do not remove lifetime markers of stack allocations not added to frame.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @use(%arg0:!kgen.pointer<index>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg1: index) async -> index {
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
      co.suspend.end
  }
  // CHECK: pop.stack_alloc.lifetime.start
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  // CHECK: pop.stack_alloc.lifetime.end
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  kgen.return %2 : index
}
}

// -----

// COM: Stack Allocation With Multiple Lifetimes

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
kgen.func @use(%arg0:!kgen.pointer<index>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @in_frame_resume
kgen.func @in_frame(%arg1: index, %arg2: index) async -> index {
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
      co.suspend.end
  }
  // CHECK:      [[ARG_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
  // CHECK-NEXT: [[ARG:%.*]] = pop.load [[ARG_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: [[FRAME_SA:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 1]]]
  // CHECK-NEXT: pop.store [[ARG]], [[FRAME_SA]]
  // CHECK-NEXT: kgen.call @use([[FRAME_SA]])
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>

  co.suspend (%hdl) {
    co.suspend.end
  }

  // CHECK-NEXT: [[ARG2_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 2]]]
  // CHECK-NEXT: [[ARG2:%.*]] = pop.load [[ARG2_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: [[FRAME_SA:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 1]]]
  // CHECK-NEXT: pop.store [[ARG2]], [[FRAME_SA]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @use([[FRAME_SA]]) : (!kgen.pointer<index>) -> index
  // CHECK-NEXT: [[ARG1_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME7]]]
  // CHECK-NEXT: [[ARG1:%.*]] = pop.load [[ARG1_SLOT]]
  // CHECK-NEXT: pop.store [[ARG1]], [[FRAME_SA]] : !kgen.pointer<index>
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[FRAME_SA2:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 1]]]
  // CHECK-NEXT: kgen.call @use([[FRAME_SA2]]) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg2, %0 : !kgen.pointer<index>
  %3 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.store %arg1, %0 : !kgen.pointer<index>

  co.suspend (%hdl) {
    co.suspend.end
  }
  %4 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
}

// -----

// COM: Stack Allocations With Control Flow In Frame

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @use(%arg0:!kgen.pointer<index>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @in_frame_cf_resume
kgen.func @in_frame_cf(%arg1: index, %arg2: index, %arg3: i1) async -> index {
  // CHECK-NEXT: hlcf.elif {
  // CHECK-NEXT:  [[ARG3_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT:  [[ARG3:%.*]] = pop.load [[ARG3_SLOT]] : !kgen.pointer<i1>
  // CHECK-NEXT:  hlcf.elif.yield [[ARG3]] : i1
  // CHECK-NEXT: } then {
  // CHECK-NEXT: [[ARG2_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 1]]]
  // CHECK-NEXT: [[ARG2:%.*]] = pop.load [[ARG2_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: [[SA:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 2]]]
  // CHECK-NEXT: pop.store [[ARG2]], [[SA]] : !kgen.pointer<index>
  // CHECK-NEXT: hlcf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT: [[ARG1_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[ARG1:%.*]] = pop.load [[ARG1_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: [[SA2:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 2]]]
  // CHECK-NEXT: pop.store [[ARG1]], [[SA2]] : !kgen.pointer<index>
  // CHECK-NEXT: hlcf.yield
  // CHECK-NEXT: }
  %0 = pop.stack_allocation 1 x index marked
  hlcf.elif {
    hlcf.elif.yield %arg3 : i1
  } then {
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
    pop.store %arg2, %0 : !kgen.pointer<index>
    hlcf.yield
  } else {
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
    pop.store %arg1, %0 : !kgen.pointer<index>
    hlcf.yield
  }
  // CHECK-NEXT: co.suspend
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  co.suspend (%hdl) {
    co.suspend.end
  }
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  // CHECK-NEXT: [[SA3:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 2]]]
  // CHECK-NEXT: kgen.call @use([[SA3]]) : (!kgen.pointer<index>) -> index
  // CHECK-NEXT: hlcf.elif {
  // CHECK-NEXT:   [[ARG3_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT:   [[ARG3:%.*]] = pop.load [[ARG3_SLOT]] : !kgen.pointer<i1>
  // CHECK-NEXT:   hlcf.elif.yield [[ARG3]] : i1
  // CHECK-NEXT: } then {
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: }
  hlcf.elif {
    hlcf.elif.yield %arg3 : i1
  } then {
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
    hlcf.yield
  } else {
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
    hlcf.yield
  }
  kgen.return %2 : index
}
}

// -----

// COM: Stack Allocations With Control Flow Not In Frame

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @use(%arg0:!kgen.pointer<index>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @not_in_frame_cf_resume
kgen.func @not_in_frame_cf(%arg1: index, %arg2: index, %arg3: i1) async -> index {
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[ARG3_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[ARG3:%.*]] = pop.load [[ARG3_SLOT]] : !kgen.pointer<i1>
  // CHECK-NEXT: hlcf.if [[ARG3]] {
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:  co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[SA:%.*]] = pop.stack_allocation 1 x index

  // CHECK: pop.stack_alloc.lifetime.start([[SA]])
  // CHECK: pop.stack_alloc.lifetime.start([[SA]])
  // CHECK: pop.stack_alloc.lifetime.start([[SA]])
  // CHECK: pop.stack_alloc.lifetime.end([[SA]])
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
    co.suspend.end
  }
  hlcf.if %arg3 {
    co.suspend (%hdl) {
      co.suspend.end
    }
    hlcf.elif {
      hlcf.elif.yield %arg3 : i1
    } then {
      hlcf.if %arg3 {
        pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
        pop.store %arg1, %0 : !kgen.pointer<index>
        hlcf.yield
      } else {
        pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
        pop.store %arg2, %0 : !kgen.pointer<index>
        hlcf.yield
      }
      hlcf.yield
    } else {
      pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
      pop.store %arg1, %0 : !kgen.pointer<index>
      hlcf.yield
    }
    %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
    hlcf.yield
  } else {
    hlcf.yield
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
}

// -----

// COM: LifetimeMarkers With Multiple Operands.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @use(%arg0:!kgen.pointer<index>, %arg1:!kgen.pointer<index>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @in_frame_resume
kgen.func @in_frame(%arg1: index) async -> index {
  // CHECK-NOT: pop.stack_alloc.lifetime.start
  // CHECK-NOT: pop.stack_alloc.lifetime.end
  %0 = pop.stack_allocation 1 x index marked
  %1 = pop.stack_allocation 1 x index marked
  pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  pop.store %arg1, %1 : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  %2 = kgen.call @use(%0, %1) : (!kgen.pointer<index>, !kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  kgen.return %2 : index
}

// CHECK-LABEL: kgen.func @not_in_frame_resume
kgen.func @not_in_frame(%arg1: index) async -> index {
  // CHECK: [[V1:%.*]] = pop.stack_allocation 1 x index
  // CHECK: [[V2:%.*]] = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x index marked
  %1 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK: pop.stack_alloc.lifetime.start
  pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  pop.store %arg1, %1 : !kgen.pointer<index>
  %2 = kgen.call @use(%0, %1) : (!kgen.pointer<index>, !kgen.pointer<index>) -> index
  // CHECK: pop.stack_alloc.lifetime.end
  pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>

  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @multiple_lifetimes_frame_resume
kgen.func @multiple_lifetimes_frame(%arg1: index, %arg2: i1) async -> index {
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V1:%.*]] = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x index marked
  %1 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK: hlcf.if {{.*}} {
  // CHECK-NEXT: [[V10:%.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_alloc.lifetime.start([[V1]], [[V10]])
  hlcf.if %arg2 {
    pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
    pop.store %arg1, %0 : !kgen.pointer<index>
    pop.store %arg1, %1 : !kgen.pointer<index>
    %2 = kgen.call @use(%0, %1) : (!kgen.pointer<index>, !kgen.pointer<index>) -> index
    pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: pop.stack_alloc.lifetime.start([[V1]])
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  %4 = kgen.call @use(%0, %0) : (!kgen.pointer<index>, !kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

}

// -----


// COM: Lower GetCallbackPtrOp

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @coroutine1(%arg0: i1) async -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @callback(%arg0: !kgen.pointer<none>) -> !kgen.none {
  %none = kgen.param.constant: !kgen.none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg0: i1) async -> index {
  hlcf.if %arg0 {
    %idx1 = index.constant 1
    kgen.return %idx1 : index
  } else {
    hlcf.yield
  }
  %true = index.bool.constant true
  // CHECK:      [[CORO:%.*]] = kgen.call @coroutine1_ramp(%true)
  // CHECK-NEXT: [[CALLBACK:%.*]] = kgen.create_closure[(!kgen.pointer<none>) -> !kgen.none: @callback]()
  // CHECK-NEXT: [[SLOT:%.*]] = kgen.struct.gep [[CORO]][[[#FRAME2:]]]
  // CHECK-NEXT: [[CAST:%.*]] = pop.pointer.bitcast [[SLOT]]
  // CHECK-NEXT: [[SLOT2:%.*]] = kgen.struct.gep [[CAST]][0] : <struct<((!kgen.pointer<none>) -> !kgen.none, pointer<none>)>>
  // CHECK-NEXT: pop.store [[CALLBACK]], [[SLOT2]] : !kgen.pointer<(!kgen.pointer<none>) -> !kgen.none>
  // CHECK-NEXT: co.suspend {
  %coro = co.invoke[(i1) async -> index: @coroutine1](%true)
  %callback = kgen.create_closure[(!kgen.pointer<none>) -> !kgen.none: @callback]()
  %ptr = co.get_callback_ptr %coro : <struct<(!kgen.signature<(!kgen.pointer<none>) -> !kgen.none>, pointer<none>)>>
  %callbackSlot = kgen.struct.gep %ptr[0] : <struct<(!kgen.signature<(!kgen.pointer<none>) -> !kgen.none>, pointer<none>)>>
  pop.store %callback, %callbackSlot : !kgen.pointer<!kgen.signature<(!kgen.pointer<none>) -> !kgen.none>>
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
}

// -----

// COM: Lower DestroyOp

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @coroutine1(%arg0: i1) async -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg0: i1) async -> index {
  hlcf.if %arg0 {
    %idx1 = index.constant 1
    kgen.return %idx1 : index
  } else {
    hlcf.yield
  }
  %true = index.bool.constant true
  // CHECK: [[CORO:%.*]] = kgen.call @coroutine1_ramp(%true)
  // CHECK-NEXT: [[CORO_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: pop.store [[CORO]], [[CORO_SLOT]]
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  %coro = co.invoke[(i1) async -> index: @coroutine1](%true)
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK-NEXT: [[CORO2_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
  // CHECK-NEXT:  [[CORO2:%.*]] = pop.load [[CORO2_SLOT]]
  // CHECK-NEXT: pop.aligned_free [[CORO2]]
  co.destroy %coro
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
}

// -----

// COM: Ensure Non-Result Args Are Mapped to Resume.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_ramp(%arg0: index) -> !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
kgen.func @coroutine(%arg0: index, %__result__: !kgen.pointer<index> byref_result) async -> index {
  %idx1 = index.constant 1
  %result1 = index.add %arg0, %idx1
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

}

// -----

// COM: Default Behavior For Stack Allocations Without Markers.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @use(%arg0:!kgen.pointer<struct<(index, index)>>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
kgen.func @use2(%arg0:!kgen.pointer<index>) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @missing_markers_resume
kgen.func @missing_markers(%arg1: index, %arg2: i1) async -> index {
  // CHECK-NEXT: [[STACK_ALLOC:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
  // CHECK-NEXT: [[V2:%.*]] = kgen.struct.gep [[STACK_ALLOC]][1] : <struct<(index, index)>>
  %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)>
  %1 = kgen.struct.gep %0[1] : !kgen.pointer<struct<(index,index)>>
  // CHECK: hlcf.if
  hlcf.if %arg2 {
    // CHECK-NEXT: [[V12:%.*]] = pop.stack_allocation 1 x index
    // CHECK-NEXT: [[V13:%.*]] = kgen.call @use2([[V12]]) : (!kgen.pointer<index>) -> index
    %3 = pop.stack_allocation 1 x index
    %4 = kgen.call @use2(%3) : (!kgen.pointer<index>) -> index
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK:      [[INFRAME:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 2]]]
  // CHECK-NEXT: [[V6:%.*]] = pop.load [[INFRAME]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[V6]], [[V2]] : !kgen.pointer<index>
  // CHECK-NEXT: co.suspend {
  pop.store %arg1, %1 : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  %2 = kgen.call @use(%0) : (!kgen.pointer<struct<(index, index)>>) -> index
  kgen.return %2 : index
}

}

// -----

// COM: Ensure the Result of Coro Invoke Is Compatible With Other Coro Ops.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @coroutine(%arg0: index) async -> index {
  co.suspend (%hdl) {
    co.suspend.end
  }
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.func @call_coroutine
kgen.func @call_coroutine(%arg0: index) -> index {
  // CHECK:      [[CORO:%.*]] = kgen.call @coroutine_ramp
  // CHECK-NEXT: [[RESUMESLOT:%.*]] = kgen.struct.gep [[CORO]][[[#FRAME1:]]]
  // CHECK-NEXT: [[TYPED_RESUMESLOT:%.*]] = pop.pointer.bitcast [[RESUMESLOT]] : !kgen.pointer<pointer<none>> to !kgen.pointer<(!kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>) -> ()>
  // CHECK-NEXT: [[RESUME:%.*]] = pop.load [[TYPED_RESUMESLOT]]
  // CHECK-NEXT: kgen.call_indirect [[RESUME]]([[CORO]])
  %coro = co.invoke[(index) async -> index: @coroutine](%arg0)
  %fn = co.resume %coro : <(!co.routine) -> ()>
  kgen.call_indirect %fn(%coro) : (!co.routine) -> ()
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @get_results
kgen.func @get_results(%arg0: !co.routine) {
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0
  // CHECK-NEXT: [[PROMISE_PTR:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6:]]]
  // CHECK-NEXT: [[VALUE_PTR:%.*]] = kgen.struct.gep [[PROMISE_PTR]][0]
  // CHECK-NEXT: pop.load [[VALUE_PTR]]
  %0 = co.get_results %arg0 : index
  kgen.return
}

// CHECK-LABEL: kgen.func @multiple_results_resume
kgen.func @multiple_results(%arg0: !co.routine) async -> (i32, i64) {
  // CHECK:      [[PROMISE_PTR:%.*]] = kgen.struct.gep {{.*}}[6]
  // CHECK-NEXT: [[R0_PTR:%.*]] = kgen.struct.gep [[PROMISE_PTR]][0]
  // CHECK-NEXT: [[R0:%.*]] = pop.load [[R0_PTR]]
  // CHECK-NEXT: [[R1_PTR:%.*]] = kgen.struct.gep [[PROMISE_PTR]][1]
  // CHECK-NEXT: [[R1:%.*]] = pop.load [[R1_PTR]]
  %0, %1 = co.get_results %arg0 : i32, i64
  // CHECK-NEXT: [[PROMISE_PTR:%.*]] = kgen.struct.gep {{.*}}[6]
  // CHECK-NEXT: [[R0_PTR:%.*]] = kgen.struct.gep [[PROMISE_PTR]][0]
  // CHECK-NEXT: store [[R0]], [[R0_PTR]]
  // CHECK-NEXT: [[R1_PTR:%.*]] = kgen.struct.gep [[PROMISE_PTR]][1]
  // CHECK-NEXT: store [[R1]], [[R1_PTR]]
  kgen.return %0, %1 : i32, i64
}

// CHECK-LABEL: kgen.func @no_results_ramp
// CHECK: aligned_alloc %idx8, %idx48
kgen.func @no_results() async {
  kgen.return
}

// CHECK-LABEL: kgen.func @use_of_suspend_resume
kgen.func @use_of_suspend() async -> i32 {
  // CHECK: co.suspend {
  co.suspend (%hdl) {
    // CHECK: [[HDL:%.*]] = pop.pointer.bitcast %arg0 : {{.*}} to !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
    %0 = pop.stack_allocation 1 x !co.routine
    // CHECK: store [[HDL]], %{{.*}}
    pop.store %hdl, %0 : !kgen.pointer<!co.routine>
    co.suspend.end
  }
  kgen.unreachable
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
        hlcf.if %1 {
          hlcf.yield
        } else {
          hlcf.break
        }
        %2 = index.add %arg2, %arg0
        hlcf.continue %2 : index
      }
      hlcf.loop (%arg2 = %arg1 : index) {
        %1 = index.cmp slt(%arg2, %arg0)
        hlcf.if %1 {
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
    kgen.return
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
       // CHECK-NEXT: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME7+1]]]
       // CHECK-NEXT: [[V9:%.*]] = pop.load [[V8]] : !kgen.pointer<index>
       // CHECK-NEXT: [[V10:%.*]] = kgen.call @bar([[V7]], [[V9]]) : (index, index) -> index
       %isThisDetected = kgen.call @bar(%result, %arg3): (index,index) -> index
       hlcf.if %arg0 {
         hlcf.yield
       } else {
         hlcf.break "_loop_1"
       }
       hlcf.if %arg4 {
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

  kgen.return %arg1 : index
}

kgen.func @foo(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @bar(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
}

// -----

// COM: Co.Suspend Has Correct Predecessors Set

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @foo(%arg0: index borrow, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    co.suspend (%hdl) {
      co.suspend.end
    }
    %idx1 = index.constant 1
    %0 = kgen.call @foo(%idx1) : (index) -> index
    // CHECK: kgen.call @foo
    // CHECK-NEXT: [[V0:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
    co.suspend (%hdl) {
      %12 = index.add %0, %0
      co.suspend.end
    }
    // CHECK: [[V1:%.*]] = kgen.struct.gep %arg0[[[#FRAME7]]]
    %11 = index.add %0, %0
    kgen.return
  }
}

// -----

// COM: Lower ResumeOp

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @coroutine1(%arg0: i1) async -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg0: i1) async -> index {
  hlcf.if %arg0 {
    %idx1 = index.constant 1
    kgen.return %idx1 : index
  } else {
    hlcf.yield
  }
  %true = index.bool.constant true
  %coro = co.invoke[(i1) async -> index: @coroutine1](%true)
  // CHECK:      [[CORO:%.*]] = kgen.call @coroutine1_ramp(%true)
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   [[RESUME_SLOT:%.*]] = kgen.struct.gep [[CORO]][[[#FRAME1:]]]
  // CHECK-NEXT:   [[TYPED_RESUME:%.*]] = pop.pointer.bitcast [[RESUME_SLOT]] : !kgen.pointer<pointer<none>> to !kgen.pointer<(!kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>) -> ()>
  // CHECK-NEXT:   [[RESUME:%.*]] = pop.load [[TYPED_RESUME]]
  // CHECK-NEXT:   kgen.call_indirect [[RESUME]]([[CORO]])
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  co.suspend (%hdl) {
    %fn = co.resume %coro : <(!co.routine) -> ()>
    kgen.call_indirect %fn(%coro) : (!co.routine) -> ()
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
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
    // CHECK-NEXT: [[V6:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
    // CHECK-NEXT: [[V7:%.*]] = pop.load [[V6]]
    // CHECK-NEXT: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 1]]]
    // CHECK-NEXT: [[V9:%.*]] = pop.load [[V8]]
    // CHECK-NEXT: kgen.call @bar([[V7]], [[V9]])
    %isThisDetected = kgen.call @bar(%result, %arg3): (index,index) -> index
    hlcf.if %arg0 {
      hlcf.continue
    } else {
      hlcf.yield
    }
    co.suspend (%hdl) {
      co.suspend.end
    }
    hlcf.continue
  }
  kgen.return %arg1 : index
}

kgen.func @foo(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @bar(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
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
        kgen.call @"CBatch::__init__"(%0, %1) : (!kgen.pointer<struct<(pointer<none>, pointer<none>) memoryOnly>> init_self, !kgen.pointer<none>) -> ()
        co.suspend (%hdl) {
          co.suspend.end
        }
        hlcf.loop "_loop_1" (%arg3 = %arg2 : index) {
          %3 = index.cmp sgt(%arg3, %idx0)
          hlcf.if %3 {
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
        %2 = kgen.call @batch_size(%0) : (!kgen.pointer<struct<(pointer<none>, pointer<none>) memoryOnly>> borrow_in_mem) -> index
        hlcf.continue
      }
      kgen.return
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
            hlcf.if %5 {
              hlcf.continue %arg1, %arg1 : index, index
            } else {
              hlcf.break "_loop_1" %arg11 : index
            }
            kgen.unreachable
          }
          hlcf.if %arg3 {
            hlcf.break "_loop_0" %arg9 : index
          } else {
            hlcf.yield
          }
          hlcf.continue %arg1, %arg1 : index, index
        }
        co.suspend (%hdl) {
          co.suspend.end
        }
        hlcf.if %arg3 {
          hlcf.break "_loop_2" %arg1, %arg6, %arg7 : index, index, index
        } else {
          hlcf.continue %arg1, %arg6, %arg7, %arg8 : index, index, index, index
        }
        kgen.unreachable
      }
      kgen.return
    }
}

// -----

// COM: Verify that Constants Are Not Stored In Frame

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: kgen.func @foo_resume
  kgen.func @foo(%arg0: i1 borrow, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    %idx1 = index.constant 1
    co.suspend (%hdl) {
      co.suspend.end
    }
    // CHECK: %idx1_0 = index.constant 1
    // CHECK-NEXT: index.mul %idx1_0, %idx1_0
    %14 = index.mul %idx1, %idx1
    kgen.return
  }
  // CHECK-LABEL: kgen.func @needsLift_resume
  kgen.func @needsLift(%arg0: i1 borrow, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    %idx1 = index.constant 1
    co.suspend (%hdl) {
      co.suspend.end
    }
    // CHECK: %idx1_0 = index.constant 1
    // CHECK-NEXT: kgen.struct.gep %arg0
    // CHECK-NEXT: pop.load
    // CHECK-NEXT: hlcf.if
    // CHECK-NEXT: index.sub %idx1_0, %idx1_0
    hlcf.if %arg0 {
      %13 = index.sub %idx1, %idx1
      hlcf.yield
    } else {
      hlcf.yield
    }
    // CHECK: index.add %idx1_0, %idx1_0
    hlcf.if %arg0 {
      %14 = index.add %idx1, %idx1
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.return
  }
}

// -----

// COM: Frame Addresses Are Not Stored In Frame

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @gep(%arg0: i1 borrow, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)> marked
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<struct<(index, index)>>
    %1 = kgen.call @fillMe(%0) : (!kgen.pointer<struct<(index, index)>> byref_result) -> index
    %2 = kgen.struct.gep %0[1] : <struct<(index, index)>>
    co.suspend (%hdl) {
      co.suspend.end
    }
    // CHECK: co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[7]
    // CHECK-SAME: <struct<(i32, pointer<none>, (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, struct<()>, struct<(index, index)>)>>
    // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep [[V3]][1] : <struct<(index, index)>>
    // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
    // CHECK-NEXT:  kgen.call @doSomething([[V5]]) : (index) -> index
    %4 = pop.load %2 : !kgen.pointer<index>
    %3 = kgen.call @doSomething(%4) : (index) -> index
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<struct<(index, index)>>
    kgen.return
  }
  kgen.func @offset(%arg0: i1 borrow, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    // CHECK:      [[V1:%.*]] = index.constant 1
    // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
    // CHECK-NEXT: [[V4:%.*]] = pop.pointer.bitcast [[V3]]
    // CHECK-NEXT: [[V5:%.*]] = pop.offset [[V4]][[[V1]]] : !kgen.pointer<index>
    %0 = pop.stack_allocation 2 x index marked
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
    %idx1 = index.constant 1
    %1 = pop.offset %0[%idx1] : !kgen.pointer<index>
    hlcf.loop {
      hlcf.if %arg0 {
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
    kgen.return
  }
}
