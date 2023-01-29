// RUN: kgen-opt -pass-pipeline='builtin.module(lower-kgen-to-llvm, llvm.func(lower-coroutines, canonicalize))' %s | FileCheck %s

// CHECK-LABEL: llvm.func @coroutine
// CHECK-SAME: -> !llvm.ptr<i8>
// CHECK-SAME: attributes {passthrough = ["presplitcoroutine"]}
kgen.func @coroutine(%arg0: i32) -> !pop.coroutine<() -> i32> {
  // CHECK-DAG: %[[TRUE:.*]] = llvm.mlir.constant(true)
  // CHECK-DAG: %[[FALSE:.*]] = llvm.mlir.constant(false)
  // CHECK-DAG: %[[C0_i64:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK-DAG: %[[C1_i32:.*]] = llvm.mlir.constant(1 : i32)
  // CHECK-NEXT: %[[PROMISE_MEM:.*]] = llvm.alloca %[[C1_i32]] x !llvm.struct<(i32, struct<(ptr, ptr)>)> {alignment = 8 : i64}
  // CHECK-NEXT: %[[PROMISE:.*]] = llvm.bitcast %[[PROMISE_MEM]] : !llvm.ptr<struct<(i32, struct<(ptr, ptr)>)>> to !llvm.ptr<i8>
  // CHECK-NEXT: %[[NULLPTR:.*]] = llvm.inttoptr %[[C0_i64]] : i64 to !llvm.ptr<i8>
  // CHECK-NEXT: %[[CORO_ALIGN:.*]] = llvm.intr.coro.align : i32
  // CHECK-NEXT: %[[ID:.*]] = llvm.intr.coro.id %[[CORO_ALIGN]], %[[PROMISE]], %[[NULLPTR]], %[[NULLPTR]]
  // CHECK-NEXT: %[[NEED_DYN:.*]] = llvm.call_intrinsic "llvm.coro.alloc"(%[[ID]])
  // CHECK-NEXT: llvm.cond_br %[[NEED_DYN]], ^[[DYN_ALLOC:.*]], ^[[BEGIN:.*]](%[[NULLPTR]] : !llvm.ptr<i8>)

// CHECK: ^[[DYN_ALLOC]]:
  // CHECK-NEXT: %[[CORO_SIZE:.*]] = llvm.intr.coro.size : i64
  // CHECK-NEXT: %[[MEM:.*]] = pop.external_call @malloc(%[[CORO_SIZE]])
  // CHECK-NEXT: %[[FRAME_MEM:.*]] = llvm.bitcast %[[MEM]] : !llvm.ptr to !llvm.ptr<i8>
  // CHECK-NEXT: llvm.br ^[[BEGIN]](%[[FRAME_MEM]] : !llvm.ptr<i8>)

// CHECK: ^[[BEGIN]](%[[FRAME:.*]]: !llvm.ptr<i8>):
  // CHECK-NEXT: %[[HDL:.*]] = llvm.intr.coro.begin %[[ID]], %[[FRAME]]
  %hdl = pop.coroutine.handle : <() -> i32>
  // CHECK-NEXT: %[[TOK:.*]] = llvm.intr.coro.save %[[HDL]]
  // CHECK-NEXT: %[[STATE:.*]] = llvm.intr.coro.suspend %[[TOK]], %[[FALSE]] : i8
  // CHECK-NEXT: llvm.switch %[[STATE]] : i8, ^[[SUSPEND_BLOCK:.*]] [
  // CHECK-NEXT:   0: ^[[CONTINUE:.*]],
  // CHECK-NEXT:   1: ^[[CLEANUP:.*]]

// CHECK: ^[[CONTINUE]]:
  // CHECK-NEXT: %[[TOK:.*]] = llvm.intr.coro.save %[[HDL]]
  // CHECK-NEXT: %[[STATE:.*]] = llvm.intr.coro.suspend %[[TOK]], %[[TRUE]] : i8
  // CHECK-NEXT: llvm.switch %[[STATE]] : i8, ^[[SUSPEND_BLOCK]] [
  // CHECK-NEXT:   0: ^[[TRAP:.*]],
  // CHECK-NEXT:   1: ^[[CLEANUP]]

// CHECK: ^[[TRAP]]:
  // CHECK-NEXT: llvm.call_intrinsic "llvm.trap"()
  // CHECK-NEXT: llvm.unreachable

// CHECK: ^[[CLEANUP]]:
  // CHECK-NEXT: %[[MEM_TO_FREE:.*]] = llvm.intr.coro.free %[[ID]], %[[HDL]]
  // CHECK-NEXT: %[[NEED_DYN:.*]] = llvm.icmp "ne" %[[MEM_TO_FREE]], %[[NULLPTR]] : !llvm.ptr<i8>
  // CHECK-NEXT: llvm.cond_br %[[NEED_DYN]], ^bb6, ^[[SUSPEND_BLOCK]]

// CHECK: ^bb6:
  // CHECK-NEXT: %[[FRAME_TO_FREE:.*]] = llvm.bitcast %19 : !llvm.ptr<i8> to !llvm.ptr
  // CHECK-NEXT: pop.external_call @free(%[[FRAME_TO_FREE]]) : (!llvm.ptr) -> ()
  // CHECK-NEXT: llvm.br ^[[SUSPEND_BLOCK]]

// CHECK: ^[[SUSPEND_BLOCK]]:
  // CHECK-NEXT: %{{.*}} = llvm.intr.coro.end %[[NULLPTR]], %[[FALSE]] : i1
  // CHECK-NEXT: llvm.return %[[HDL]] : !llvm.ptr<i8>
  kgen.return %hdl : !pop.coroutine<() -> i32>
}

// CHECK-LABEL: @coroutine_await
kgen.func @coroutine_await() -> !pop.coroutine<() -> i32> {
  // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false)
  %hdl = pop.coroutine.handle : <() -> i32>
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  // CHECK-NEXT: llvm.intr.coro.save
  // CHECK-NEXT: llvm.intr.coro.suspend
  // CHECK-NEXT: llvm.switch %{{.*}} : i8, ^[[SUSPEND:.*]] [
  // CHECK-NEXT: 0: ^[[BODY:.*]],
  // CHECK-NEXT: 1: ^[[CLEANUP:.*]]

// CHECK: ^[[BODY]]:
  // CHECK-NEXT: %[[TOK:.*]] = llvm.intr.coro.save %[[HDL]] : !llvm.token
  // CHECK-NEXT: llvm.intr.coro.resume %[[HDL]]
  // CHECK-NEXT: %[[STATE:.*]] = llvm.intr.coro.suspend %[[TOK]], %[[FALSE]] : i8
  // CHECK-NEXT: llvm.switch %[[STATE]] : i8, ^[[SUSPEND]] [
  // CHECK-NEXT:   0: ^{{.*}},
  // CHECK-NEXT:   1: ^[[CLEANUP]]
  // CHECK-NEXT: ]
  pop.coroutine.await {
    pop.coroutine.resume %hdl : <() -> i32>
  }
  kgen.return %hdl : !pop.coroutine<() -> i32>
}

// CHECK-LABEL: @other_coroutine_ops
kgen.func @other_coroutine_ops(%a: !pop.coroutine<() -> i32>) -> !pop.pointer<struct<i32>> {
  // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false)
  // CHECK: %[[ALIGN:.*]] = llvm.mlir.constant(8 : i32)

  // CHECK: %[[PROMISE_PTR:.*]] = llvm.call_intrinsic "llvm.coro.promise"(%arg0, %[[ALIGN]], %[[FALSE]])
  // CHECK: %[[PROMISE:.*]] = llvm.bitcast %[[PROMISE_PTR:.*]] : !llvm.ptr<i8> to !llvm.ptr<struct<(i32, struct<(ptr, ptr)>)>>
  // CHECK: %[[PROMISE_RESULT:.*]] = llvm.bitcast %[[PROMISE]] : !llvm.ptr<struct<(i32, struct<(ptr, ptr)>)>> to !llvm.ptr<i32>
  %promise = pop.coroutine.promise %a : <() -> i32>

  // CHECK: llvm.call_intrinsic "llvm.coro.destroy"(%arg0) : (!llvm.ptr<i8>) -> ()
  pop.coroutine.destroy %a : <() -> i32>

  // CHECK: llvm.return %[[PROMISE_RESULT]] : !llvm.ptr<i32>
  kgen.return %promise : !pop.pointer<struct<i32>>
}

kgen.export @coroutine
