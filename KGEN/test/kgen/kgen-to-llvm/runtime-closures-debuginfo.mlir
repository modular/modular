// RUN: kgen-opt %s -lower-runtime-closures -allow-unregistered-dialect -mlir-print-debuginfo | FileCheck %s

#file = #debuginfo.file<"foo.mlir" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 44,
  scopeLine = 44,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":46:8)
#loc3 = loc("foo.mlir":48:8)
#loc4 = loc(fused<#subprogram>[#loc1])
#loc5 = loc(fused<#subprogram>[#loc2])
#loc6 = loc(fused<#subprogram>[#loc3])
#loc7 = loc(callsite(#loc5 at #loc4))
#loc8 = loc(callsite(#loc6 at #loc4))

// CHECK-DAG: #[[LOC0:.*]] = loc("foo.mlir":46:8)
// CHECK-DAG: #[[LOC1:.*]] = loc("foo.mlir":44:1)
// CHECK-DAG: #[[LOC2:.*]] = loc("foo.mlir":48:8)
// CHECK-DAG: #[[SUB_PROG:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "foo", linkageName = "foo"
// CHECK-DAG: #[[LOC3:.*]] = loc(fused<#[[SUB_PROG]]>[#[[LOC0]]])
// CHECK-DAG: #[[LOC4:.*]] = loc(fused<#[[SUB_PROG]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC5:.*]] = loc(fused<#[[SUB_PROG]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC6:.*]] = loc(callsite(#[[LOC3]] at #[[LOC4]]))
// CHECK-DAG: #[[LOC7:.*]] = loc(callsite(#[[LOC5]] at #[[LOC4]]))

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  llvm.func @h(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }

  llvm.func @g(%arg0: i64) -> i64 {
      llvm.return %arg0 : i64
  }

  // CHECK-LABEL: @main_closure
  llvm.func internal @main_closure() {
    %idx98 = index.constant 98
    %0 = kgen.create_closure [(index) -> index: @h](%idx98) loc(#loc7)
    %1 = kgen.create_closure [(index) -> index: @g](%idx98) loc(#loc8)
    "use.closure"(%0) : (!kgen.signature<() capturing -> index>) -> ()
    "use.closure"(%1) : (!kgen.signature<() capturing -> index>) -> ()
    llvm.return
  }

  // CHECK-LABEL: llvm.func internal @closure_wrapper_fn(%arg0: !llvm.ptr
  // CHECK: %0 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr<struct<(i64)>> loc(#[[CLOSURE_WRAPPER_FN_LOC:.*]])
  // CHECK: %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<struct<(i64)>>) -> !llvm.ptr<i64> loc(#[[CLOSURE_WRAPPER_FN_LOC]])
  // CHECK: %2 = llvm.load %1 : !llvm.ptr<i64> loc(#[[CLOSURE_WRAPPER_FN_LOC]])
  // CHECK: %3 = llvm.call @h(%2) {fastmathFlags = #llvm.fastmath<contract>} : (i64) -> i64  loc(#[[CLOSURE_WRAPPER_FN_LOC]])
  // CHECK: llvm.return %3 : i64 loc(#[[CLOSURE_WRAPPER_FN_LOC]])
  // CHECK:} loc(#[[CLOSURE_WRAPPER_FN_LOC]])

  // CHECK-LABEL: llvm.func internal @closure_wrapper_fn_0(%arg0: !llvm.ptr
  // CHECK: %0 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr<struct<(i64)>> loc(#[[CLOSURE_WRAPPER_FN_LOC0:.*]])
  // CHECK: %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<struct<(i64)>>) -> !llvm.ptr<i64> loc(#[[CLOSURE_WRAPPER_FN_LOC0]])
  // CHECK: %2 = llvm.load %1 : !llvm.ptr<i64> loc(#[[CLOSURE_WRAPPER_FN_LOC0]])
  // CHECK: %3 = llvm.call @g(%2) {fastmathFlags = #llvm.fastmath<contract>} : (i64) -> i64 loc(#[[CLOSURE_WRAPPER_FN_LOC0]])
  // CHECK: llvm.return %3 : i64 loc(#[[CLOSURE_WRAPPER_FN_LOC0]])
  // CHECK:} loc(#[[CLOSURE_WRAPPER_FN_LOC0]])
}

// CHECK: #[[SUB_PROG1:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit1, scope = #file, name = "closure_wrapper_fn", linkageName = "closure_wrapper_fn", file = #file, line = 46, scopeLine = 46, subprogramFlags = Definition> : !subroutine1
// CHECK: #[[SUB_PROG2:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit1, scope = #file, name = "closure_wrapper_fn_0", linkageName = "closure_wrapper_fn_0", file = #file, line = 48, scopeLine = 48, subprogramFlags = Definition> : !subroutine1
// CHECK: #[[CLOSURE_WRAPPER_FN_LOC]] = loc(fused<#[[SUB_PROG1]]>[#[[LOC0]]])
// CHECK: #[[CLOSURE_WRAPPER_FN_LOC0]] = loc(fused<#[[SUB_PROG2]]>[#[[LOC2]]])
