// RUN: kgen-opt %s -split-input-file -lower-coroutines-async -allow-unregistered-dialect -mlir-print-debuginfo | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func @coro_resume
llvm.func @coro_resume() {
  %0 = "make_handle"() : () -> !co.routine
  // CHECK: [[FUNC:%.*]] = llvm.load [[HND:%.*]] : !llvm.ptr -> !llvm.ptr
  %1 = co.resume %0 : <(!co.routine) -> ()>
  llvm.return
}

// CHECK-LABEL: llvm.func @coro_destroy
llvm.func @coro_destroy() {
  %0 = "make_handle"() : () -> !co.routine
  // CHECK: pop.aligned_free {{.*}} : <i8>
  co.destroy %0
  llvm.return
}

// CHECK-LABEL: llvm.func @coro_get_results
llvm.func @coro_get_results() {
  %0 = builtin.unrealized_conversion_cast to !co.routine
  // CHECK: [[PTR:%.*]] = llvm.getelementptr inbounds {{%.*}}[40]
  // CHECK-NEXT: [[STRUCT:%.*]] = llvm.load [[PTR]] : !llvm.ptr -> !llvm.struct<(i32, i64)>
  // CHECK-NEXT: [[R0:%.*]] = llvm.extractvalue [[STRUCT]][0]
  // CHECK-NEXT: [[R1:%.*]] = llvm.extractvalue [[STRUCT]][1]
  // CHECK-NEXT: unrealized_conversion_cast [[R1]] : i64 to index
  %1:2 = co.get_results %0 : i32, index
  "use"(%1#0, %1#1) : (i32, index) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @coro_get_results_none
llvm.func @coro_get_results_none() {
  // CHECK: unrealized_conversion_cast
  %0 = builtin.unrealized_conversion_cast to !co.routine
  co.get_results %0
  // CHECK-NEXT: return
  llvm.return
}


// CHECK-LABEL: llvm.func @coro_get_results_one
llvm.func @coro_get_results_one() {
  %0 = builtin.unrealized_conversion_cast to !co.routine
  // CHECK: [[PTR:%.*]] = llvm.getelementptr inbounds {{%.*}}[40]
  // CHECK-NEXT: [[R:%.*]] = llvm.load [[PTR]] : !llvm.ptr -> i64
  // CHECK-NEXT: unrealized_conversion_cast [[R]] : i64 to index
  %1 = co.get_results %0 : index
  "use"(%1) : (index) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @coro_set_results
llvm.func @coro_set_results(%arg0: i32) {
  %0 = builtin.unrealized_conversion_cast to !co.routine
  %idx0 = index.constant 0
  // CHECK: [[PTR:%.*]] = llvm.getelementptr inbounds {{%.*}}[40]
  // CHECK-NEXT: [[V0:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, i32)>
  // CHECK-NEXT: [[ELE:%.*]] = builtin.unrealized_conversion_cast %idx0 : index to i64
  // CHECK-NEXT: [[V1:%.*]] = llvm.insertvalue [[ELE]], [[V0]][0]
  // CHECK-NEXT: [[V2:%.*]] = llvm.insertvalue %arg0, [[V1]][1]
  // CHECK-NEXT: store [[V2]], [[PTR]]
  co.set_results %0(%idx0, %arg0) : index, i32
  llvm.return
}

// CHECK-LABEL: llvm.func @coro_set_results_none
llvm.func @coro_set_results_none() {
  // CHECK-NEXT: unrealized_conversion_cast
  %0 = builtin.unrealized_conversion_cast to !co.routine
  co.set_results %0()
  // CHECK-NEXT: return
  llvm.return
}

// CHECK-LABEL: llvm.func @coro_set_results_one
llvm.func @coro_set_results_one() {
  %0 = builtin.unrealized_conversion_cast to !co.routine
  %idx0 = index.constant 0
  // CHECK: [[PTR:%.*]] = llvm.getelementptr inbounds {{%.*}}[40]
  // CHECK-NEXT: [[ELE:%.*]] = builtin.unrealized_conversion_cast %idx0 : index to i64
  // CHECK-NEXT: store [[ELE]], [[PTR]]
  co.set_results %0(%idx0) : index
  llvm.return
}

// CHECK-LABEL: llvm.func @coro_byref_results
llvm.func @coro_byref_results() {
  %0 = builtin.unrealized_conversion_cast to !co.routine
  %1 = kgen.param.constant: pointer<index> = <0>
  // CHECK: [[RES:%.*]] = llvm.getelementptr inbounds [[CORO:%.*]][32]
  // CHECK-NEXT: store {{.*}}, [[RES]]
  // CHECK: [[ERR:%.*]] = llvm.getelementptr inbounds [[CORO:%.*]][24]
  // CHECK-NEXT: store {{.*}}, [[ERR]]
  // CHECK-NOT: set_byref_error_result
  co.set_byref_error_result %0(%1, %1) : !co.routine, !kgen.pointer<index>, !kgen.pointer<index>

  // CHECK-NOT: [[ERR:%.*]] = llvm.getelementptr inbounds [[CORO:%.*]][24]
  // CHECK: [[RES:%.*]] = llvm.getelementptr inbounds [[CORO:%.*]][32]
  // CHECK-NEXT: store {{.*}}, [[RES]]
  // CHECK-NOT: set_byref_error_result
  co.set_byref_error_result %0(%1) : !co.routine, !kgen.pointer<index>

  // CHECK: [[RES_PTR:%.*]] = llvm.getelementptr inbounds [[CORO:%.*]][32]
  // CHECK-NEXT: [[RES:%.*]] = llvm.load [[RES_PTR]]
  // CHECK: [[ERR_PTR:%.*]] = llvm.getelementptr inbounds [[CORO:%.*]][24]
  // CHECK-NEXT: [[ERR:%.*]] = llvm.load [[ERR_PTR]]
  %result, %error = co.get_byref_error_result %0 : !kgen.pointer<index>, !kgen.pointer<index>

  // CHECK: [[RES_PTR:%.*]] = llvm.getelementptr inbounds [[CORO:%.*]][32]
  // CHECK-NEXT: [[RES:%.*]] = llvm.load [[RES_PTR]]
  %result0 = co.get_byref_error_result %0 : !kgen.pointer<index>
  llvm.return
}

// CHECK-LABEL: llvm.func @set_byref_none
llvm.func @set_byref_none() {
  %0 = builtin.unrealized_conversion_cast to !co.routine
  %1 = kgen.param.constant: pointer<none> = <0>
  // CHECK-NOT: llvm.getelementptr inbounds [[CORO:%.*]][32]
  co.set_byref_error_result %0(%1, %1) : !co.routine, !kgen.pointer<none>, !kgen.pointer<none>
  co.set_byref_error_result %0(%1) : !co.routine, !kgen.pointer<none>
  llvm.return
}

}

// -----

// CHECK-DAG: #[[ASYNC_FN_NAME:.*]] = #debuginfo.source_name<"async_function" from <"async_fn">>
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<name = #[[ASYNC_FN_NAME]], linkageName = "async_fn_af">

!subroutine = !debuginfo.subroutine<(i32) -> (!llvm.ptr): DW_CC_normal>
#subprogram = #debuginfo.subprogram<name = <"async_fn">, linkageName = "async_fn"> : !subroutine


// CHECK-DAG: #[[SUSPEND_LOC:.*]] = loc("foo.mlir":10:5)

#loc = loc(fused<#subprogram>["foo.mlir":17:8])

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func internal @async_fn_af_suspend
// CHECK-SAME: (%arg0: !llvm.ptr loc({{.*}}), %arg1: i64 loc({{.*}}))
// CHECK-NEXT:   %0 = builtin.unrealized_conversion_cast %arg1 : i64 to index
// CHECK-NEXT:   "do_something"(%0)
// CHECK-NEXT:   llvm.return loc(#[[SUSPEND_LOC]])
// CHECK-NEXT: } loc(#[[SUSPEND_LOC]])


// CHECK-LABEL: llvm.func @async_fn_af
// CHECK-SAME: (%arg0: !llvm.ptr loc({{.*}}))
llvm.func @async_fn(%arg0: i32) -> !llvm.ptr {
  // CHECK: %[[AFP:.*]] = llvm.mlir.addressof @async_fn_afp
  // CHECK: %[[AFP_CAST:.*]] = llvm.bitcast %[[AFP]]
  // CHECK: %[[C32:.*]] = llvm.mlir.constant(56 : i32)
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(8 : i32)
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: %[[TOK:.*]] = llvm.call_intrinsic "llvm.coro.id.async"(%[[C32]], %[[C1]], %[[C0]], %[[AFP_CAST]])
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin %[[TOK]]
  %hdl = co.handle : i64
  // CHECK: %[[BASE_CTXT_HDL:.*]] = llvm.getelementptr inbounds %[[HDL]][-56]
  // CHECK-NEXT: unrealized_conversion_cast %[[BASE_CTXT_HDL]]
  %0 = builtin.unrealized_conversion_cast %hdl : !co.routine to !llvm.ptr
  // CHECK: %[[BASE_CTXT_ARG:.*]] = llvm.getelementptr inbounds %[[HDL]][-56] {{.*}} loc(#[[LOC_ARG:.*]])
  // CHECK: %[[ARG_PTR:.*]] = llvm.getelementptr inbounds %[[BASE_CTXT_ARG]][0, 5] {{.*}} loc(#[[LOC_ARG]])
  // CHECK: %[[ARG:.*]] = llvm.load %[[ARG_PTR]] {{.*}} loc(#[[LOC_ARG]])
  // CHECK: "use"(%[[ARG]])
  "use"(%arg0) : (i32) -> ()
  %idx1 = index.constant 1
  // CHECK: %[[CAPTURED:.*]] = builtin.unrealized_conversion_cast %idx1
  // CHECK: %[[RESUME_FN:.*]] = llvm.call_intrinsic "llvm.coro.async.resume"
  // CHECK: %[[BASE_CTXT_RESUME:.*]] = llvm.getelementptr inbounds %[[HDL]][-56]
  // CHECK: %[[RESUME_FN_PTR:.*]] = llvm.getelementptr inbounds %[[BASE_CTXT_RESUME]][0, 0]
  // CHECK: llvm.store %[[RESUME_FN]], %[[RESUME_FN_PTR]]
  // CHECK: %[[PROJ_FN:.*]] = llvm.mlir.addressof @__kgen_coro_ctxt_proj_fn
  // CHECK: %[[PROJ_FN_OPAQUE:.*]] = llvm.bitcast %[[PROJ_FN]]
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: %[[SUSPEND_FN:.*]] = llvm.mlir.addressof @async_fn_af_suspend
  // CHECK-NEXT: external_call @llvm.coro.suspend.async.sl_p0p0p0s
  // CHECK-SAME: (%[[C0]], %[[RESUME_FN]], %[[PROJ_FN_OPAQUE]], %[[SUSPEND_FN]], %[[BASE_CTXT_RESUME]], %[[CAPTURED]])
  co.suspend (%hdl0) {
    "do_something"(%idx1) : (index) -> ()
    co.suspend.end
  } loc(callsite("foo.mlir":10:5 at "bar.mlir":12:7))
  // CHECK: %[[BASE_CTXT:.*]] = llvm.getelementptr inbounds %[[HDL]][-56]
  // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false)
  // CHECK: %[[END_FN:.*]] = llvm.mlir.addressof @__kgen_coro_end_fn
  // CHECK: llvm.call_intrinsic "llvm.coro.end.async"(%[[HDL]], %[[FALSE]], %[[END_FN]], %[[BASE_CTXT]])
  llvm.return %0 : !llvm.ptr
} loc(#loc)

// CHECK-LABEL: llvm.mlir.global internal constant @async_fn_afp
// CHECK-SAME: !llvm.struct<(i32, i32)> {
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
// CHECK-NEXT: %[[AFP:.*]] = llvm.mlir.addressof @async_fn_afp : !llvm.ptr
// CHECK-NEXT: %[[AFP_VALUE:.*]] = llvm.getelementptr inbounds %[[AFP]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
// CHECK-DAG: %[[AF:.*]] = llvm.mlir.addressof @async_fn_af : !llvm.ptr
// CHECK-DAG: %[[AF_INT:.*]] = llvm.ptrtoint %[[AF]] : !llvm.ptr to i64
// CHECK-DAG: %[[AFP_INT:.*]] = llvm.ptrtoint %[[AFP_VALUE]] : !llvm.ptr to i64
// CHECK-NEXT: %[[OFFSET_i32:.*]] = llvm.sub %[[AF_INT]], %[[AFP_INT]]  : i64
// CHECK-NEXT: %[[OFFSET:.*]] = llvm.trunc %[[OFFSET_i32]] : i64 to i32
// CHECK-NEXT: %[[V0:.*]] = llvm.insertvalue %[[OFFSET]], %[[UNDEF]][0] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: %[[CTXT_SZ:.*]] = llvm.mlir.constant(56 : i32) : i32
// CHECK-NEXT: %[[RESULT:.*]] = llvm.insertvalue %[[CTXT_SZ]], %[[V0]][1] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: llvm.return %[[RESULT]] : !llvm.struct<(i32, i32)>

// CHECK-LABEL: llvm.func @async_fn(%arg0: i32 loc({{.*}})) -> !llvm.ptr
// CHECK-NEXT: %[[AFP:.*]] = llvm.mlir.addressof @async_fn_afp
// CHECK-NEXT: %[[AFP_i8ptr:.*]] = llvm.bitcast %[[AFP]]
// CHECK-NEXT: %[[AFP_PREPARE:.*]] = llvm.call_intrinsic "llvm.coro.prepare.async"(%[[AFP_i8ptr]])
// CHECK-NEXT: %[[AFP_CASTED:.*]] = llvm.bitcast %[[AFP_PREPARE]]
// CHECK-NEXT: %[[CTXT_SZ_PTR:.*]] = llvm.getelementptr inbounds %[[AFP_CASTED]][0, 1]
// CHECK-NEXT: %[[CTXT_SZ_i32:.*]] = llvm.load %[[CTXT_SZ_PTR]]
// CHECK-NEXT: %[[CTXT_ALIGN:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT: %[[CTXT_ALIGN_INDEX:.*]] = builtin.unrealized_conversion_cast %[[CTXT_ALIGN]] : i64 to index
// CHECK-NEXT: %[[CTXT_SZ:.*]] = llvm.zext %[[CTXT_SZ_i32]] : i32 to i64
// CHECK-NEXT: %[[CTXT_SZ_INDEX:.*]] = builtin.unrealized_conversion_cast %[[CTXT_SZ]] : i64 to index
// CHECK-NEXT: %[[MEM_I8:.*]] = pop.aligned_alloc %[[CTXT_ALIGN_INDEX]], %[[CTXT_SZ_INDEX]] : <i8>
// CHECK-NEXT: %[[MEM:.*]] = builtin.unrealized_conversion_cast %[[MEM_I8]]
// CHECK-DAG: %[[AF:.*]] = llvm.mlir.addressof @async_fn_af
// CHECK-DAG: %[[RESUME_FN_PTR:.*]] = llvm.getelementptr inbounds %[[MEM]][0, 0]
// CHECK-NEXT: llvm.store %[[AF]], %[[RESUME_FN_PTR]]
// CHECK-NEXT: %[[ARG_PTR:.*]] = llvm.getelementptr inbounds %[[MEM]][0, 5]
// CHECK-NEXT: llvm.store %arg0, %[[ARG_PTR]]
// CHECK-NEXT: llvm.return %[[MEM]] : !llvm.ptr

// CHECK-LABEL: llvm.func internal @__kgen_coro_end_fn(%arg0: !llvm.ptr loc({{.*}}))
// CHECK-NEXT: %[[CLOSURE:.*]] = llvm.getelementptr inbounds %arg0[8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT: %[[FN_PTR:.*]] = llvm.getelementptr inbounds %[[CLOSURE]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK-NEXT: %[[FN:.*]] = llvm.load %[[FN_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT: %[[ARG_PTR:.*]] = llvm.getelementptr inbounds %[[CLOSURE]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK-NEXT: %[[ARG:.*]] = llvm.load %[[ARG_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT: llvm.call %[[FN]](%[[ARG]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT: llvm.return

// CHECK-LABEL: llvm.func internal @__kgen_coro_ctxt_proj_fn(%arg0: !llvm.ptr loc({{.*}})) -> !llvm.ptr
// CHECK-NEXT: llvm.return %arg0 : !llvm.ptr

}

// CHECK: #[[LOC_ARG]] = loc(fused<#[[SP]]>

// -----

// CHECK-DAG: #[[FOO_NAME:.*]] = #debuginfo.source_name<"async_function" from <"foo">>
// CHECK-DAG: #[[SP_AF:.*]] = #debuginfo.subprogram<name = #[[FOO_NAME]], linkageName = "foo_af"
// CHECK-DAG: #[[LOC0:.*]] = loc("foo.mlir":41:11)
// CHECK-DAG: #[[LOC_AF:.*]] = loc(fused<#[[SP_AF]]>[#[[LOC0]]])
#subprogram = #debuginfo.subprogram<name = <"foo">, linkageName = "foo"> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#loc8 = loc(fused<#subprogram>["foo.mlir":41:11])

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  llvm.func internal @foo() -> !llvm.ptr {
    %0 = co.handle : index loc(#loc8)
    %1 = builtin.unrealized_conversion_cast %0 : !co.routine to !llvm.ptr loc(#loc8)

    // CHECK-LABEL: llvm.func internal @foo_af_suspend_0(%arg0: !llvm.ptr loc({{.*}}))
    // CHECK-NEXT:    %0 = llvm.mlir.constant(1 : i64) : i64 loc(#[[LOC_INL_CONST:.*]])
    // CHECK-NEXT:    llvm.return loc(#[[LOC_SUSP1:.*]])
    // CHECK-NEXT:  } loc(#[[LOC_SUSP1]])
    co.suspend (%hdl0) {
      %2 = llvm.mlir.constant(1 : i64) : i64 loc(#loc11)
      co.suspend.end
    } loc(#loc11)

    // CHECK-LABEL: llvm.func internal @foo_af_suspend_1(%arg0: !llvm.ptr loc({{.*}}))
    // CHECK-NEXT:    %0 = llvm.mlir.constant(2 : i64) : i64 loc(#[[LOC_INL_CONST:.*]])
    // CHECK-NEXT:    llvm.return loc(#[[LOC_SUSP2:.*]])
    // CHECK-NEXT:  } loc(#[[LOC_SUSP2]])
    co.suspend (%hdl1) {
      %2 = llvm.mlir.constant(2 : i64) : i64 loc(#loc13)
      co.suspend.end
    } loc(#loc13)

    llvm.return %1 : !llvm.ptr loc(#loc8)
  } loc(#loc8)
}

// CHECK-DAG: ![[SUSP_TYPE:.*]] = !debuginfo.subroutine<(!llvm.ptr) -> (): DW_CC_normal>
// CHECK-DAG: #[[SP1:.*]] = #debuginfo.subprogram<name = <"suspend.0" from #[[FOO_NAME]]>, linkageName = "foo_af_suspend_0"> : ![[SUSP_TYPE]]
// CHECK-DAG: #[[SP2:.*]] = #debuginfo.subprogram<name = <"suspend.1" from #[[FOO_NAME]]>, linkageName = "foo_af_suspend_1"> : ![[SUSP_TYPE]]

// CHECK-DAG: #[[LOC1:.*]] = loc("foo.mlir":44:38)
// CHECK-DAG: #[[LOC2:.*]] = loc("foo.mlir":42:41)
// CHECK-DAG: #[[LOC_CALLEE1:.*]] = loc(fused<#[[SP_AF]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_CALLEE2:.*]] = loc(fused<#[[SP_AF]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_SUSP1]] = loc(fused<#[[SP1]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_SUSP2]] = loc(fused<#[[SP2]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_INL_CONST]] = loc(unknown)

#loc10 = loc(fused<#subprogram>["foo.mlir":42:16])
#loc11 = loc(fused<#subprogram>["foo.mlir":44:38])
#loc13 = loc(fused<#subprogram>["foo.mlir":42:41])
