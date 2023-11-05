// RUN: kgen-opt %s --lower-runtime-closures -allow-unregistered-dialect | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @take_closure_no_args
  llvm.func @take_closure_no_args(%arg0: !llvm.struct<(ptr, ptr)>) {
    // CHECK: %1 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, ptr)>
    // CHECK: %2 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, ptr)>
    // CHECK: %3 = llvm.bitcast %1 : !llvm.ptr to !llvm.ptr<func<i64 (ptr)>>
    // CHECK: llvm.call %3(%2) {fastmathFlags = #llvm.fastmath<contract>}
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(ptr, ptr)> to !kgen.signature<() capturing -> index>
    %1 = kgen.call_signature %0() : () capturing -> index
    llvm.return
  }
  llvm.func @h(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }
  // CHECK-LABEL: @main_closure_arg
  llvm.func internal @main_closure_arg() {
    // CHECK: [[ARG:%.*]] = builtin.unrealized_conversion_cast %idx98 : index to i64
    // CHECK: [[UNDEF:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    // CHECK: [[ADDR:%.*]] = llvm.mlir.addressof @closure_wrapper_fn_0 : !llvm.ptr
    // CHECK: [[OPAQUE:%.*]] = llvm.bitcast [[ADDR]] : !llvm.ptr to !llvm.ptr
    // CHECK: [[S0:%.*]] = llvm.insertvalue [[OPAQUE]], [[UNDEF]][0] : !llvm.struct<(ptr, ptr)>
    // CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : i8) : i8
    // CHECK: [[STATE:%.*]] = llvm.alloca [[C1]] x !llvm.struct<(i64)> : (i8) -> !llvm.ptr<struct<(i64)>>
    // CHECK: llvm.intr.lifetime.start 8, [[STATE]] : !llvm.ptr<struct<(i64)>>
    // CHECK: [[ARGPTR:%.*]] = llvm.getelementptr [[STATE]][0, 0] : (!llvm.ptr<struct<(i64)>>) -> !llvm.ptr<i64>
    // CHECK: llvm.store [[ARG]], [[ARGPTR]] : !llvm.ptr<i64>
    // CHECK: [[OPAQUE_STATE:%.*]] = llvm.bitcast [[STATE]] : !llvm.ptr<struct<(i64)>> to !llvm.ptr
    // CHECK: [[CLOSURE:%.*]] = llvm.insertvalue [[OPAQUE_STATE]], [[S0]][1] : !llvm.struct<(ptr, ptr)>
    // CHECK-NEXT: unrealized_conversion_cast [[CLOSURE]]
    %idx98 = index.constant 98
    %0 = kgen.create_closure [(index) -> index: @h](%idx98)
    "use.closure"(%0) : (!kgen.signature<() capturing -> index>) -> ()
    llvm.return
  }
  // CHECK-LABEL: llvm.func internal @closure_wrapper_fn_0(%arg0: !llvm.ptr) -> i64
  // CHECK: %0 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr<struct<(i64)>>
  // CHECK: %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<struct<(i64)>>) -> !llvm.ptr<i64>
  // CHECK: %2 = llvm.load %1 : !llvm.ptr<i64>
  // CHECK: %3 = llvm.call @h(%2)
  // CHECK: llvm.return %3 : i64
}
