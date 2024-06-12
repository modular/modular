// RUN: kgen-opt -lower-async-functions -split-input-file %s | FileCheck %s

// COM: Verify Ramp + Resume + Async Calls are transformed correctly.
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @coroutine1(%arg0: i1) async -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK:      kgen.func @coroutine_ramp(%arg0: i1) -> !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>> {
// CHECK-NEXT:   %idx64 = index.constant 64
// CHECK-NEXT:   %idx8 = index.constant 8
// CHECK-NEXT:   [[CONTINUATION:%.*]] = pop.aligned_alloc %idx8, %idx64 : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>>
// CHECK-NEXT:   [[RESUME_SLOT:%.*]] = kgen.struct.gep %0[1] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>>
// CHECK-NEXT:   [[RESUME_FNC_PTR:%.*]] = kgen.create_closure[(!kgen.pointer<none>) -> (): @coroutine_resume]()
// CHECK-NEXT:   pop.store [[RESUME_FNC_PTR]], [[RESUME_SLOT]] : !kgen.pointer<(!kgen.pointer<none>) -> ()>
// CHECK-NEXT:   [[ARG0_SLOT:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME1:]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>>
// CHECK-NEXT:   pop.store %arg0, [[ARG0_SLOT]] : !kgen.pointer<i1>
// CHECK-NEXT:   kgen.return [[CONTINUATION]] : !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>>
// CHECK-NEXT: }

// CHECK-LABEL: kgen.func @coroutine_resume(%arg0: !kgen.pointer<none>) attributes {coroutineType = !kgen.struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>} {
// CHECK-NEXT:    [[CONTINUATION:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>>
// CHECK-NEXT:    [[ARG0_SLOT:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME1]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>>
// CHECK-NEXT:    [[ARG0:%.*]] = pop.load [[ARG0_SLOT]] : !kgen.pointer<i1>
// CHECK-NEXT:    hlcf.if [[ARG0]] {
// CHECK-NEXT:      %idx1 = index.constant 1
// CHECK-NEXT:      [[PROMISE_SLOT:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#PROMISE_IDX:]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>
// CHECK-NEXT:      [[PROMISE_OPAQUE:%.*]] = pop.load [[PROMISE_SLOT]] : !kgen.pointer<pointer<none>>
// CHECK-NEXT:      [[PROMISE:%.*]] = pop.pointer.bitcast [[PROMISE_OPAQUE]] : !kgen.pointer<none> to !kgen.pointer<index>
// CHECK-NEXT:      pop.store %idx1, [[PROMISE]] : !kgen.pointer<index>
// CHECK-NEXT:      kgen.return
// CHECK-NEXT:    } else {
// CHECK-NEXT:      hlcf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    %true = index.bool.constant true
// CHECK-NEXT:    kgen.call @coroutine1_ramp(%true) : (i1) -> !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>)>>
// CHECK-NEXT:    co.suspend {
// CHECK-NEXT:      co.suspend.end
// CHECK-NEXT:    }
// CHECK-NEXT:    %idx0 = index.constant 0
// CHECK-NEXT:    [[PROMISE_SLOT:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#PROMISE_IDX]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>>
// CHECK-NEXT:    [[PROMISE_OPAQUE:%.*]] = pop.load [[PROMISE_SLOT]] : !kgen.pointer<pointer<none>>
// CHECK-NEXT:    [[PROMISE:%.*]] = pop.pointer.bitcast [[PROMISE_OPAQUE]] : !kgen.pointer<none> to !kgen.pointer<index>
// CHECK-NEXT:    pop.store %idx0, [[PROMISE]] : !kgen.pointer<index>
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
  // CHECK-SAME: (i1) -> !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, i1)>>
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
  // CHECK-NEXT: [[CONTINUATION:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index, index, i1, index, index)>>
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME9:]]]
  // CHECK-NEXT: pop.store %idx3, [[V4]] : !kgen.pointer<index>

  // CHECK-NEXT: [[V5:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME9 - 4]]]
  // CHECK-NEXT: [[V6:%.*]] = pop.load [[V5]] : !kgen.pointer<index>

  // CHECK-NEXT: [[V7:%.*]] = kgen.call @foo(%idx3, [[V6]]) : (index, index) -> index
  // CHECK-NEXT: [[V8:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME9 - 3]]]
  // CHECK-NEXT: pop.store [[V7]], [[V8]] : !kgen.pointer<index>
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index

  // CHECK-NEXT: hlcf.loop
  hlcf.loop "_loop_0" {
    // CHECK-NEXT: [[V19:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME9 - 3]]]
    // CHECK-NEXT: [[V20:%.*]] = pop.load [[V19]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V21:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME9 - 2]]]
    // CHECK-NEXT: [[V22:%.*]] = pop.load [[V21]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V23:%.*]] = kgen.call @bar([[V20]], [[V22]]) : (index, index) -> index
    %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index


    // CHECK-NEXT: [[V24:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME9 - 1]]]
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

    // CHECK-NEXT: [[V26:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME9]]]
    // CHECK-NEXT: [[V27:%.*]] = pop.load [[V26]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V28:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME9 + 1]]]
    // CHECK-NEXT: [[V29:%.*]] = pop.load [[V28]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V30:%.*]] = kgen.call @foo([[V27]], [[V29]]) : (index, index) -> index
    %result6 = kgen.call @foo(%idx3, %arg2) : (index,index) -> index

    // CHECK-NEXT: hlcf.continue
    hlcf.continue
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V9:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME9 - 3]]]
  // CHECK-NEXT: [[V10:%.*]] = pop.load [[V9]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#FRAME9 - 4]]]
  // CHECK-NEXT: [[V12:%.*]] = pop.load [[V11]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @bar([[V10]], [[V12]]) : (index, index) -> index
  %result5 = kgen.call @bar(%result, %arg1): (index,index) -> index

  // CHECK-NEXT: [[V14:%.*]] = kgen.struct.gep [[CONTINUATION]][[[#PROMISE_IDX:]]]
  // CHECK-NEXT: [[V15:%.*]] = pop.load [[V14]] : !kgen.pointer<pointer<none>>
  // CHECK-NEXT: [[PROMISE:%.*]] = pop.pointer.bitcast [[V15]] : !kgen.pointer<none> to !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[V10]], [[PROMISE]] : !kgen.pointer<index>
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
  // CHECK: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index, i1, index)>>
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME5:]]]
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[NOT_IN_FRAME:%.*]] = kgen.call @foo(%idx3, [[V5]]) : (index, index) -> index
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  hlcf.loop "_loop_0" {
     hlcf.if %arg0 {
       hlcf.yield
     } else {
       // CHECK: } else {
       // CHECK-NEXT: [[V17:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME5 + 1]]]
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
  // CHECK:      [[V8:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME5 + 3]]]
  // CHECK-NEXT: [[V9:%.*]] = pop.load [[V8]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep [[CONT]][[[#PROMISE_IDX:]]]
  // CHECK-NEXT: [[V11:%.*]] = pop.load [[V10]] : !kgen.pointer<pointer<none>>
  // CHECK-NEXT: [[PROMISE:%.*]] = pop.pointer.bitcast [[V11]] : !kgen.pointer<none> to !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[V9]], [[PROMISE]] : !kgen.pointer<index>
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index, index, i1, index)>>

  %idx3 = index.constant 3
  // CHECK: [[NIF:%.*]] = kgen.call @foo(%idx3, %{{.*}}) : (index, index) -> index
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  // CHECK: hlcf.loop "_loop_0"
  hlcf.loop "_loop_0" {
    // CHECK-NEXT: [[V13:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8:]]]
    // CHECK-NEXT: [[V14:%.*]] = pop.load [[V13]] : !kgen.pointer<i1>
    // CHECK-NEXT: hlcf.if [[V14]] {
    // CHECK-NEXT:   hlcf.yield
    // CHECK-NEXT: } else {
    // CHECK-NEXT: [[V15:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 - 2]]]
    // CHECK-NEXT: [[V16:%.*]] = pop.load [[V15]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V17:%.*]] = kgen.call @bar([[NIF]], [[V16]]) : (index, index) -> index
    // CHECK-NEXT: [[V18:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 - 1]]]
    // CHECK-NEXT: pop.store [[V17]], [[V18]] : !kgen.pointer<index>
    // CHECK-NEXT: co.suspend {
    // CHECK-NEXT:   co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V19:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 - 1]]]
    // CHECK-NEXT: [[V20:%.*]] = pop.load [[V19]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V21:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 - 2]]]
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index, index, i1)>>

  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  hlcf.loop "_loop_0" {
     hlcf.if %arg0 {
        hlcf.yield
     } else {
         %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index
         // CHECK:      hlcf.loop "_loop_1" {
         // CHECK-NEXT: [[V23:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8:]]]
         // CHECK-NEXT: [[V24:%.*]] = pop.load [[V23]] : !kgen.pointer<i1>
         // CHECK-NEXT: hlcf.if [[V24]] {
         // CHECK-NEXT:   hlcf.yield
         // CHECK-NEXT: } else {
         // CHECK-NEXT:   co.suspend {
         // CHECK-NEXT:     co.suspend.end
         // CHECK-NEXT:   }
         // CHECK-NEXT:   [[V25:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 - 1]]]
         // CHECK-NEXT:   [[V26:%.*]] = pop.load [[V25]] : !kgen.pointer<index>
         // CHECK-NEXT:   [[V27:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 - 2]]]
         // CHECK-NEXT:   [[V28:%.*]] = pop.load [[V27]] : !kgen.pointer<index>
         // CHECK-NEXT:   [[V29:%.*]] = kgen.call @bar([[V26]], [[V28]]) : (index, index) -> index
         // CHECK-NEXT:   hlcf.break "_loop_1"
         // CHECK-NEXT: }
         // CHECK-NEXT: hlcf.continue
         // CHECK-NEXT: }
         // CHECK-NEXT: [[V18:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 - 1]]]
         // CHECK-NEXT: [[V19:%.*]] = pop.load [[V18]] : !kgen.pointer<index>
         // CHECK-NEXT: [[V20:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 - 2]]]
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index, index)>>

  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME5:]]]
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[NotInFrame:%.*]] = kgen.call @foo(%idx3, [[V5]]) : (index, index) -> index
  // CHECK-NEXT: [[V7:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME5 + 2]]]
  // CHECK-NEXT: pop.store [[NotInFrame]], [[V7]] : !kgen.pointer<index>
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   [[V13:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME5 + 1]]]
  // CHECK-NEXT:   [[V14:%.*]] = pop.load [[V13]] : !kgen.pointer<index>
  // CHECK-NEXT:   [[V15:%.*]] = kgen.call @bar([[NotInFrame]], [[V14]]) : (index, index) -> index
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT:  }
  // CHECK-NEXT:  [[V8:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME5 + 2]]]
  // CHECK-NEXT:  [[V9:%.*]] = pop.load [[V8]] : !kgen.pointer<index>
  // CHECK-NEXT:  [[V10:%.*]] = kgen.struct.gep [[CONT]][[[#PROMISE_IDX:]]]
  // CHECK-NEXT:  [[V11:%.*]] = pop.load [[V10]] : !kgen.pointer<pointer<none>>
  // CHECK-NEXT:  [[V12:%.*]] =  pop.pointer.bitcast [[V11]] : !kgen.pointer<none> to !kgen.pointer<index>
  // CHECK-NEXT:  pop.store [[V9]], [[V12]] : !kgen.pointer<index>
  // CHECK-NEXT:  kgen.return
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index)>>

  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6:]]]
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V6:%.*]] = hlcf.loop (%arg1 = [[V5]] : index) -> index {
  // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6 - 1]]]
  // CHECK-NEXT:  pop.store %arg1, [[V10]] : !kgen.pointer<index>
  // CHECK-NEXT:  %idx0 = index.constant 0
  // CHECK-NEXT:  [[V11:%.*]] = index.cmp slt(%arg1, %idx0)
  // CHECK-NEXT:  hlcf.if [[V11]] {
  // CHECK-NEXT:    co.suspend {
  // CHECK-NEXT:      co.suspend.end
  // CHECK-NEXT:    }
  // CHECK-NEXT:    hlcf.yield
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    [[V14:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6 - 1]]]
  // CHECK-NEXT:    [[V15:%.*]] = pop.load [[V14]] : !kgen.pointer<index>
  // CHECK-NEXT:    hlcf.break [[V15]] : index
  // CHECK-NEXT:  }
  // CHECK-NEXT:  [[V12:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6 - 1]]]
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index)>>
  // CHECK: [[V6:%.*]] = hlcf.loop (%arg1 = %{{.*}} : index) -> index {
  // CHECK: [[V10:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME5:]]]
  // CHECK: pop.store %arg1, [[V10]] : !kgen.pointer<index>
  // CHECK: co.suspend {
  // CHECK: co.suspend.end
  // CHECK: }
  // CHECK: %idx0 = index.constant 0
  // CHECK: [[V11:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME5]]]
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index)>>
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
kgen.func @unused_args(%arg0: index, %arg1: index) async -> index {
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index)>>
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index, index, index)>>

  // CHECK: [[NIF:%.*]] = kgen.call @foo1
  %result3 = kgen.call @foo1(%arg1) : (index) -> index
  lit.try {
    hlcf.elif {
      %result = kgen.call @bar(%arg2) : (index) -> i1
      hlcf.elif.yield %result : i1
    } then {
      // CHECK: } then {
      // CHECK-NEXT: [[V7:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6:]]]
      // CHECK-NEXT: [[V8:%.*]] = pop.load [[V7]] : !kgen.pointer<index>
      // CHECK-NEXT: [[V9:%.*]] = kgen.call @foo([[V8]]) : (index) -> index
      // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6 + 1]]]
      // CHECK-NEXT: pop.store [[V9]], [[V10]] : !kgen.pointer<index>
      // CHECK-NEXT: co.suspend {
      // CHECK-NEXT:   co.suspend.end
      // CHECK-NEXT: }
      // CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6 + 1]]]
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
    // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6 + 2]]]
    // CHECK-NEXT: pop.store %arg1, [[V10]] : !kgen.pointer<index>
    // CHECK-NEXT: co.suspend {
    // CHECK-NEXT: co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6 + 2]]]
    // CHECK-NEXT: [[V12:%.*]] = pop.load [[V11]]
    // CHECK-NEXT: [[V13:%.*]] = kgen.struct.gep [[CONT]][[[#PROMISE_IDX:]]]
    // CHECK-NEXT: [[V14:%.*]] = pop.load
    // CHECK-NEXT: [[V15:%.*]] = pop.pointer.bitcast [[V14]]
    // CHECK-NEXT: pop.store [[V12]], [[V15]] : !kgen.pointer<index>
    // CHECK-NEXT: kgen.return
    co.suspend (%hdl) {
      co.suspend.end
    }
    kgen.return %e : index
  } else {
    // CHECK: } else {
    // CHECK-NEXT: [[V10:%.*]] =  kgen.struct.gep [[CONT]][[[#PROMISE_IDX]]]
    // CHECK-NEXT: [[V11:%.*]] = pop.load [[V10]] : !kgen.pointer<pointer<none>>
    // CHECK-NEXT: [[V12:%.*]] = pop.pointer.bitcast [[V11]] : !kgen.pointer<none> to !kgen.pointer<index>
    // CHECK-NEXT: pop.store [[NIF]], [[V12]] : !kgen.pointer<index>
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
  // CHECK: [[CONT:%.*]] = pop.pointer.bitcast %arg0
  // CHECK-NEXT: [[RESSLOT:%.*]] = kgen.struct.gep [[CONT]][[[#RESULT:]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>)>>
  // CHECK-NEXT: [[RES:%.*]] = pop.load [[RESSLOT]] : !kgen.pointer<pointer<none>>
  // CHECK-NEXT: [[RESTYPED:%.*]] = pop.pointer.bitcast [[RES]] : !kgen.pointer<none> to !kgen.pointer<index>
  // CHECK-NEXT: [[V4:%.*]] = kgen.call @populate([[RESTYPED]]) : (!kgen.pointer<index> byref_result) -> i1
  // CHECK-NEXT: kgen.call @use([[RESTYPED]]) : (!kgen.pointer<index> borrow_in_mem) -> i1
  %0 = kgen.call @populate(%__result__) : (!kgen.pointer<index> byref_result) -> i1
  %2 = kgen.call @use(%__result__) : (!kgen.pointer<index> borrow_in_mem) -> i1
  hlcf.if %0 {
    // CHECK: hlcf.if [[V4]] {
    // CHECK-NEXT: [[ERRSLOT:%.*]] = kgen.struct.gep [[CONT]][[[#ERROR:]]]
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
  // CHECK-NEXT: [[ERRORSLOT:%.*]] = kgen.struct.gep [[CONT]][[[#ERROR]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>)>>
  // CHECK-NEXT: [[TYPED_ERRORSLOT:%.*]] = pop.pointer.bitcast [[ERRORSLOT]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store [[ERR]], [[TYPED_ERRORSLOT]] : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: [[RESSLOT:%.*]] = kgen.struct.gep [[CONT]][[[#RESULT]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>)>>
  // CHECK-NEXT: [[TYPED_RESSLOT:%.*]] = pop.pointer.bitcast [[RESSLOT]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store [[RES]], [[TYPED_RESSLOT]] : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: kgen.return
  %coro = co.invoke[(!kgen.pointer<index> byref_error, !kgen.pointer<index> byref_result) throws|async -> i1: @throwing_coroutine]()
  %0 = pop.aligned_alloc %align, %size : <index>
  %1 = pop.aligned_alloc %align, %size : <index>
  co.set_byref_error_result %coro(%1, %0) : !co.routine, !kgen.pointer<index>, !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @use2(%arg0: !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>)>>) -> i1
kgen.func @use2(%a: !co.routine) -> i1 {
  %true = index.bool.constant true
  kgen.return %true : i1
}

// CHECK-LABEL: kgen.func @opaque_coro
kgen.func @opaque_coro(%coro: !co.routine, %arg1: !kgen.pointer<index>, %arg2: !kgen.pointer<index>) {
  // CHECK-NEXT: kgen.call @use2(%arg0)
  %2 = kgen.call @use2(%coro) : (!co.routine) -> i1

  // CHECK-NEXT: [[v3:%.*]] = kgen.struct.gep %arg0[[[#ERROR]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>)>>
  // CHECK-NEXT: [[v4:%.*]] = pop.pointer.bitcast [[v3]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store %arg1, [[v4]] : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: [[v5:%.*]] = kgen.struct.gep %arg0[[[#RESULT]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>)>>
  // CHECK-NEXT: [[v6:%.*]] = pop.pointer.bitcast [[v5]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store %arg2, [[v6]] : !kgen.pointer<pointer<index>>
  co.set_byref_error_result %coro(%arg2, %arg1) : !co.routine, !kgen.pointer<index>, !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @no_error_slot
kgen.func @no_error_slot(%arg0: !co.routine, %arg1: !kgen.pointer<index>) {
  // CHECK-NEXT: [[v5:%.*]] = kgen.struct.gep %arg0[[[#RESULT]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>)>>
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0
  %0 = pop.stack_allocation 2 x index marked
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  %idx1 = index.constant 1
  // CHECK-NEXT: %idx1 = index.constant 1
  // CHECK-NEXT: [[V1:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7:]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index, array<2, index>)>>
  // CHECK-NEXT: [[V2:%.*]] = pop.load [[V1]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7+2]]]
  // CHECK-NEXT: [[V4:%.*]] = pop.pointer.bitcast [[V3]] : !kgen.pointer<array<2, index>> to !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[V2]], [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V5:%.*]] = pop.offset [[V4]][%idx1] : !kgen.pointer<index>
  // CHECK-NEXT: [[V6:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7 + 1]]]
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
  // CHECK: [[V8:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7 + 2]]]
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
  // CHECK-NEXT: [[V0:%.*]] = pop.pointer.bitcast %arg0
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK:      co.suspend
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V1:%.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_alloc.lifetime.start
  // CHECK-NEXT: [[V2:%.*]] = kgen.struct.gep [[V0]][[[#FRAME7:]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index)>>
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0
  %0 = pop.stack_allocation 2 x index marked
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  // CHECK-NEXT: %idx1 = index.constant 1
  %idx1 = index.constant 1
  // Extract pointer to inline frame memory instead of stack allocation.
  // CHECK-NEXT: [[V1:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME9:]]] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, pointer<index>, array<2, index>)>>
  // CHECK-NEXT: [[V2:%.*]] = pop.pointer.bitcast [[V1]] : !kgen.pointer<array<2, index>> to !kgen.pointer<index>

  // Store pointer to frame memory in frame since it's used across states.
  // CHECK-NEXT: [[V3:%.*]] = pop.offset [[V2]][%idx1] : !kgen.pointer<index>
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME9 - 1]]]
  // CHECK-NEXT: pop.store [[V3]], [[V4]]

  // Store the argument into the frame variable
  // CHECK-NEXT: [[V5:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME9 - 2]]]
  // CHECK-NEXT: [[V6:%.*]] = pop.load [[V5]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[V6]], [[V3]] : !kgen.pointer<index>
  %1 = pop.offset %0[%idx1] : !kgen.pointer<index>
  pop.store %arg2, %1 : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  // Extract out the pointer to the frame memory
  // CHECK: [[V7:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME9 - 1]]]
  // CHECK-NEXT: [[V8:%.*]] = pop.load [[V7]]
  // CHECK-NEXT: kgen.call @use([[V8]]) : (!kgen.pointer<index>) -> index
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, struct<(index, index)>)>>
  // CHECK-NEXT: [[STACK_ALLOC:%.*]] = kgen.struct.gep %0[[[#FRAME8:]]]
  // CHECK-NEXT: [[SLOT:%.*]] = kgen.struct.gep [[STACK_ALLOC]][1] : <struct<(index, index)>>
  // CHECK-NEXT: [[ARG_SLOT:%.*]] = kgen.struct.gep %0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[ARG:%.*]] = pop.load [[ARG_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[ARG]], [[SLOT]] : !kgen.pointer<index>
  %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)> marked
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<struct<(index, index)>>
  %1 = kgen.struct.gep %0[1] : !kgen.pointer<struct<(index,index)>>
  pop.store %arg1, %1 : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK:       [[STACK_ALLOC2:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8]]]
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

// CHECK-LABEL: kgen.func @not_in_frame_resume
kgen.func @not_in_frame(%arg1: index, %arg2: index) async -> index {
  // CHECK: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index)>>
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
      co.suspend.end
  }
  // CHECK:      [[OG_SA:%.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_alloc.lifetime.start([[OG_SA]]) : !kgen.pointer<index>
  // CHECK-NEXT: [[ARG_SLOT:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7:]]]
  // CHECK-NEXT: [[ARG:%.*]] = pop.load [[ARG_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[ARG]], [[OG_SA]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @use([[OG_SA]])
  // CHECK-NEXT: pop.stack_alloc.lifetime.end([[OG_SA]])
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>

  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK-NEXT: [[CLONE:%.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_alloc.lifetime.start([[CLONE]]) : !kgen.pointer<index>
  // CHECK-NEXT: [[ARG1_SLOT:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7 + 1]]]
  // CHECK-NEXT: [[ARG:%.*]] = pop.load [[ARG1_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[ARG]], [[CLONE]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @use([[CLONE]]) : (!kgen.pointer<index>) -> index
  // CHECK-NEXT: pop.stack_alloc.lifetime.end([[CLONE]]) : !kgen.pointer<index>
  // CHECK-NEXT: co.suspend {
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg2, %0 : !kgen.pointer<index>
  %3 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>

  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @in_frame_resume
kgen.func @in_frame(%arg1: index, %arg2: index) async -> index {
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index, index)>>
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
      co.suspend.end
  }
  // CHECK-NEXT: [[OG_SA:%.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_alloc.lifetime.start([[OG_SA]]) : !kgen.pointer<index>
  // CHECK-NEXT: [[ARG_SLOT:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7:]]]
  // CHECK-NEXT: [[ARG:%.*]] = pop.load [[ARG_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[ARG]], [[OG_SA]]
  // CHECK-NEXT: kgen.call @use([[OG_SA]])
  // CHECK-NEXT: pop.stack_alloc.lifetime.end([[OG_SA]]) : !kgen.pointer<index>
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

  // CHECK-NEXT: [[ARG2_SLOT:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7 + 1]]]
  // CHECK-NEXT: [[ARG2:%.*]] = pop.load [[ARG2_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: [[FRAME_SA:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7 + 2]]]
  // CHECK-NEXT: pop.store [[ARG2]], [[FRAME_SA]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @use([[FRAME_SA]]) : (!kgen.pointer<index>) -> index
  // CHECK-NEXT: [[ARG1_SLOT:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7]]]
  // CHECK-NEXT: [[ARG1:%.*]] = pop.load [[ARG1_SLOT]]
  // CHECK-NEXT: pop.store [[ARG1]], [[FRAME_SA]] : !kgen.pointer<index>
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[FRAME_SA2:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7 + 2]]]
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
  // CHECK:      [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, i1, index, index)>>
  // CHECK-NEXT: hlcf.elif {
  // CHECK-NEXT:  [[ARG3_SLOT:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8:]]]
  // CHECK-NEXT:  [[ARG3:%.*]] = pop.load [[ARG3_SLOT]] : !kgen.pointer<i1>
  // CHECK-NEXT:  hlcf.elif.yield [[ARG3]] : i1
  // CHECK-NEXT: } then {
  // CHECK-NEXT: [[ARG2_SLOT:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 + 1]]]
  // CHECK-NEXT: [[ARG2:%.*]] = pop.load [[ARG2_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: [[SA:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 + 2]]]
  // CHECK-NEXT: pop.store [[ARG2]], [[SA]] : !kgen.pointer<index>
  // CHECK-NEXT: hlcf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT: [[ARG1_SLOT:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[ARG1:%.*]] = pop.load %6 : !kgen.pointer<index>
  // CHECK-NEXT: [[SA2:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 + 2]]]
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
  // CHECK-NEXT: [[SA3:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8 + 2]]]
  // CHECK-NEXT: kgen.call @use([[SA3]]) : (!kgen.pointer<index>) -> index
  // CHECK-NEXT: hlcf.elif {
  // CHECK-NEXT:   [[ARG3_SLOT:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8:]]]
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, i1, index)>>
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[ARG3_SLOT:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8:]]]
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
  // CHECK: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to
  // CHECK-SAME: !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, index, index)>>
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
  // CHECK: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to
  // CHECK-SAME: !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index)>>
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
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>, index, i1)>>
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
