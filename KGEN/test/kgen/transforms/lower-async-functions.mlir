// RUN: kgen-opt -lower-async-functions %s | FileCheck %s

kgen.func @coroutine1(%arg0: i1) async -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK:      kgen.func @coroutine_ramp(%arg0: i1) -> !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>> {
// CHECK-NEXT:   %index = kgen.param.constant = <get_sizeof(struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>, current_target())>
// CHECK-NEXT:   %index_0 = kgen.param.constant = <get_alignof(struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>, current_target())>
// CHECK-NEXT:   [[CONTINUATION:%.*]] = pop.aligned_alloc %index_0, %index : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
// CHECK-NEXT:   [[RESUME_SLOT:%.*]] = kgen.struct.gep %0[1] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
// CHECK-NEXT:   [[RESUME_FNC_PTR:%.*]] = kgen.create_closure[(!kgen.pointer<none>) -> (): @coroutine_resume]()
// CHECK-NEXT:   pop.store [[RESUME_FNC_PTR]], [[RESUME_SLOT]] : !kgen.pointer<(!kgen.pointer<none>) -> ()>
// CHECK-NEXT:   [[FRAME_SLOT:%.*]] = kgen.struct.gep %0[4] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
// CHECK-NEXT:   [[OPAQUE_FRAME:%.*]] = pop.load [[FRAME_SLOT]]
// CHECK-NEXT:   [[FRAME:%.*]] = pop.pointer.bitcast [[OPAQUE_FRAME]] : !kgen.pointer<none> to !kgen.pointer<struct<(i1)>>
// CHECK-NEXT:   [[ARG0_SLOT:%.*]] = kgen.struct.gep [[FRAME]][0] : <struct<(i1)>>
// CHECK-NEXT:   pop.store %arg0, [[ARG0_SLOT]] : !kgen.pointer<i1>
// CHECK-NEXT:   kgen.return [[CONTINUATION]] : !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
// CHECK-NEXT: }

// CHECK-LABEL: kgen.func @coroutine_resume(%arg0: !kgen.pointer<none>) attributes {coro} {
// CHECK-NEXT:    [[CONTINUATION:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
// CHECK-NEXT:    [[FRAME_SLOT:%.*]] = kgen.struct.gep [[CONTINUATION]][4] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
// CHECK-NEXT:    [[FRAME_OPAQUE:%.*]] = pop.load [[FRAME_SLOT]]
// CHECK-NEXT:    [[FRAME:%.*]] = pop.pointer.bitcast [[FRAME_OPAQUE]] : !kgen.pointer<none> to !kgen.pointer<struct<(i1)>>
// CHECK-NEXT:    [[ARG0_SLOT:%.*]] = kgen.struct.gep [[FRAME]][0] : <struct<(i1)>>
// CHECK-NEXT:    [[ARG0:%.*]] = pop.load [[ARG0_SLOT]] : !kgen.pointer<i1>
// CHECK-NEXT:    hlcf.if [[ARG0]] {
// CHECK-NEXT:      %idx1 = index.constant 1
// CHECK-NEXT:      [[PROMISE_SLOT:%.*]] = kgen.struct.gep [[CONTINUATION]][5] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
// CHECK-NEXT:      [[PROMISE_OPAQUE:%.*]] = pop.load [[PROMISE_SLOT]] : !kgen.pointer<pointer<none>>
// CHECK-NEXT:      [[PROMISE:%.*]] = pop.pointer.bitcast [[PROMISE_OPAQUE]] : !kgen.pointer<none> to !kgen.pointer<index>
// CHECK-NEXT:      pop.store %idx1, [[PROMISE]] : !kgen.pointer<index>
// CHECK-NEXT:      kgen.return
// CHECK-NEXT:    } else {
// CHECK-NEXT:      hlcf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    %true = index.bool.constant true
// CHECK-NEXT:    kgen.call @coroutine1_ramp(%true) : (i1) -> !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
// CHECK-NEXT:    co.await {
// CHECK-NEXT:      co.await.end
// CHECK-NEXT:    }
// CHECK-NEXT:    %idx0 = index.constant 0
// CHECK-NEXT:    [[PROMISE_SLOT:%.*]] = kgen.struct.gep [[CONTINUATION]][5] : <struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
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
  co.await {
    co.await.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @call_coroutine
kgen.func @call_coroutine() {
  %true = index.bool.constant true
  // CHECK: kgen.call @coroutine_ramp(%true) :
  // CHECK-SAME: (i1) -> !kgen.pointer<struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>)>>
  %result = co.invoke[(i1) async -> index: @coroutine](%true)
  // CHECK-NEXT: kgen.return
  kgen.return
}
