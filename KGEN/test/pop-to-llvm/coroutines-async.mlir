// RUN: kgen-opt %s -lower-coroutines-async -allow-unregistered-dialect | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func @coro_promise
llvm.func @coro_promise() {
  %0 = "make_handle"() : () -> !pop.coroutine<() -> (i32, i64)>
  // CHECK: %2 = llvm.getelementptr inbounds %1[24] : (!llvm.ptr<i8>) -> !llvm.ptr<struct<(i32, i64)>>
  %1 = pop.coroutine.promise %0 : <() -> (i32, i64)>
  // CHECK: "use"(%2)
  "use"(%1) : (!pop.pointer<struct<i32, i64>>) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @coro_resume
llvm.func @coro_resume() {
  %0 = "make_handle"() : () -> !pop.coroutine<() -> (i32, i64)>
  // CHECK: %2 = llvm.bitcast %1
  // CHECK: %3 = llvm.load %2
  // CHECK-NEXT: llvm.call %3(%1)
  pop.coroutine.resume %0 : !pop.coroutine<() -> (i32, i64)>
  llvm.return
}

// CHECK-LABEL: llvm.func @coro_destroy
llvm.func @coro_destroy() {
  %0 = "make_handle"() : () -> !pop.coroutine<() -> (i32, i64)>
  // CHECK: %2 = llvm.bitcast %1
  // CHECK-NEXT: external_call @KGEN_CompilerRT_AlignedFree(%2)
  pop.coroutine.destroy %0 : !pop.coroutine<() -> (i32, i64)>
  llvm.return
}

// CHECK-LABEL: llvm.func internal @async_fn_af.suspend
// CHECK-SAME: (%arg0: i64)
// CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg0 : i64 to index
// CHECK-NEXT: "do_something"(%0)


// CHECK-LABEL: llvm.func @async_fn_af
// CHECK-SAME: (%arg0: !llvm.ptr<i8>)
llvm.func @async_fn(%arg0: i32) -> !llvm.ptr<i8> {
  // CHECK: %[[C32:.*]] = llvm.mlir.constant(40 : i32)
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(8 : i32)
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: %[[AFP:.*]] = llvm.mlir.addressof @async_fn_afp
  // CHECK: %[[AFP_CAST:.*]] = llvm.bitcast %[[AFP]]
  // CHECK: %[[TOK:.*]] = llvm.call_intrinsic "llvm.coro.id.async"(%[[C32]], %[[C1]], %[[C0]], %[[AFP_CAST]])
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin %[[TOK]]
  %hdl = pop.coroutine.handle : <() -> (i64)>
  // CHECK: %[[BASE_CTXT_HDL:.*]] = llvm.getelementptr inbounds %[[HDL]][-40]
  // CHECK: %[[CTXT_PTR_HDL:.*]] = llvm.bitcast %[[BASE_CTXT_HDL]]
  // CHECK-NEXT: unrealized_conversion_cast %[[CTXT_PTR_HDL]]
  %0 = builtin.unrealized_conversion_cast %hdl : !pop.coroutine<() -> (i64)> to !llvm.ptr<i8>
  // CHECK: %[[BASE_CTXT_ARG:.*]] = llvm.getelementptr inbounds %[[HDL]][-40]
  // CHECK: %[[ARG_PTR:.*]] = llvm.getelementptr inbounds %[[BASE_CTXT_ARG]][0, 3]
  // CHECK: %[[ARG:.*]] = llvm.load %[[ARG_PTR]]
  // CHECK: "use"(%[[ARG]])
  "use"(%arg0) : (i32) -> ()
  %idx1 = index.constant 1
  // CHECK: %[[CAPTURED:.*]] = builtin.unrealized_conversion_cast %idx1
  // CHECK: %[[RESUME_FN:.*]] = llvm.call_intrinsic "llvm.coro.async.resume"
  // CHECK: %[[BASE_CTXT_RESUME:.*]] = llvm.getelementptr inbounds %[[HDL]][-40]
  // CHECK: %[[RESUME_FN_PTR:.*]] = llvm.getelementptr inbounds %[[BASE_CTXT_RESUME]][0, 0]
  // CHECK: llvm.store %[[RESUME_FN]], %[[RESUME_FN_PTR]]
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: %[[PROJ_FN:.*]] = llvm.mlir.addressof @__kgen_coro_ctxt_proj_fn
  // CHECK: %[[PROJ_FN_OPAQUE:.*]] = llvm.bitcast %[[PROJ_FN]]
  // CHECK: %[[SUSPEND_FN:.*]] = llvm.mlir.addressof @async_fn_af.suspend
  // CHECK-NEXT: external_call @llvm.coro.suspend.async.sl_p0p0p0s
  // CHECK-SAME: (%[[C0]], %[[RESUME_FN]], %[[PROJ_FN_OPAQUE]], %[[SUSPEND_FN]], %[[CAPTURED]])
  pop.coroutine.await {
  ^bb0:
    "do_something"(%idx1) : (index) -> ()
  }
  // CHECK: %[[BASE_CTXT:.*]] = llvm.getelementptr inbounds %[[HDL]][-40]
  // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false)
  // CHECK: %[[END_FN:.*]] = llvm.mlir.addressof @__kgen_coro_end_fn
  // CHECK: llvm.call_intrinsic "llvm.coro.end.async"(%[[HDL]], %[[FALSE]], %[[END_FN]], %[[BASE_CTXT]])
  llvm.return %0 : !llvm.ptr<i8>
}

// CHECK-LABEL: llvm.mlir.global internal constant @async_fn_afp
// CHECK-SAME: !llvm.struct<(i32, i32)> {
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
// CHECK-NEXT: %[[AFP:.*]] = llvm.mlir.addressof @async_fn_afp : !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT: %[[AFP_VALUE:.*]] = llvm.getelementptr inbounds %[[AFP]][0, 1] : (!llvm.ptr<struct<(i32, i32)>>) -> !llvm.ptr<i32>
// CHECK-DAG: %[[AF:.*]] = llvm.mlir.addressof @async_fn_af : !llvm.ptr<func<void (ptr<i8>)>>
// CHECK-DAG: %[[AF_INT:.*]] = llvm.ptrtoint %[[AF]] : !llvm.ptr<func<void (ptr<i8>)>> to i64
// CHECK-DAG: %[[AFP_INT:.*]] = llvm.ptrtoint %[[AFP_VALUE]] : !llvm.ptr<i32> to i64
// CHECK-NEXT: %[[OFFSET_i32:.*]] = llvm.sub %[[AF_INT]], %[[AFP_INT]]  : i64
// CHECK-NEXT: %[[OFFSET:.*]] = llvm.trunc %[[OFFSET_i32]] : i64 to i32
// CHECK-NEXT: %[[V0:.*]] = llvm.insertvalue %[[OFFSET]], %[[UNDEF]][0] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: %[[CTXT_SZ:.*]] = llvm.mlir.constant(40 : i32) : i32
// CHECK-NEXT: %[[RESULT:.*]] = llvm.insertvalue %[[CTXT_SZ]], %[[V0]][1] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: llvm.return %[[RESULT]] : !llvm.struct<(i32, i32)>

// CHECK-LABEL: llvm.func @async_fn(%arg0: i32) -> !llvm.ptr<i8>
// CHECK-NEXT: %[[AFP:.*]] = llvm.mlir.addressof @async_fn_afp
// CHECK-NEXT: %[[AFP_i8ptr:.*]] = llvm.bitcast %[[AFP]]
// CHECK-NEXT: %[[AFP_PREPARE:.*]] = llvm.call_intrinsic "llvm.coro.prepare.async"(%[[AFP_i8ptr]])
// CHECK-NEXT: %[[AFP_CASTED:.*]] = llvm.bitcast %[[AFP_PREPARE]]
// CHECK-NEXT: %[[CTXT_SZ_PTR:.*]] = llvm.getelementptr inbounds %[[AFP_CASTED]][0, 1]
// CHECK-NEXT: %[[CTXT_SZ_i32:.*]] = llvm.load %[[CTXT_SZ_PTR]]
// CHECK-NEXT: %[[CTXT_ALIGN:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT: %[[CTXT_SZ:.*]] = llvm.zext %[[CTXT_SZ_i32]] : i32 to i64
// CHECK-NEXT: %[[MEM:.*]] = pop.external_call @KGEN_CompilerRT_AlignedAlloc(%[[CTXT_ALIGN]], %[[CTXT_SZ]])
// CHECK-NEXT: %[[FRAME:.*]] = llvm.bitcast %[[MEM]]
// CHECK-DAG: %[[AF:.*]] = llvm.mlir.addressof @async_fn_af
// CHECK-DAG: %[[RESUME_FN_PTR:.*]] = llvm.getelementptr inbounds %[[FRAME]][0, 0]
// CHECK-NEXT: llvm.store %[[AF]], %[[RESUME_FN_PTR]]
// CHECK-NEXT: %[[ARG_PTR:.*]] = llvm.getelementptr inbounds %[[FRAME]][0, 3]
// CHECK-NEXT: llvm.store %arg0, %[[ARG_PTR]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[FRAME]]
// CHECK-NEXT: llvm.return %[[RESULT]] : !llvm.ptr<i8>

// CHECK-LABEL: llvm.func internal @__kgen_coro_end_fn(%arg0: !llvm.ptr<i8>)
// CHECK-NEXT: %[[CLOSURE:.*]] = llvm.getelementptr inbounds %arg0[8] : (!llvm.ptr<i8>) -> !llvm.ptr<struct<(ptr<func<void (ptr<i8>)>>, ptr<i8>)>>
// CHECK-NEXT: %[[FN_PTR:.*]] = llvm.getelementptr inbounds %[[CLOSURE]][0, 0]
// CHECK-NEXT: %[[FN:.*]] = llvm.load %[[FN_PTR]]
// CHECK-NEXT: %[[ARG_PTR:.*]] = llvm.getelementptr inbounds %[[CLOSURE]][0, 1]
// CHECK-NEXT: %[[ARG:.*]] = llvm.load %[[ARG_PTR]]
// CHECK-NEXT: llvm.call %[[FN]](%[[ARG]])
// CHECK-NEXT: llvm.return

// CHECK-LABEL: llvm.func internal @__kgen_coro_ctxt_proj_fn(%arg0: !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK-NEXT: llvm.return %arg0 : !llvm.ptr<i8>

}
