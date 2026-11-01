// RUN: kgen-opt %s --lower-runtime-closures -allow-unregistered-dialect -split-input-file | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @take_closure_no_args
  llvm.func @take_closure_no_args(%arg0: !llvm.ptr) {
    // CHECK: llvm.call %arg0() : !llvm.ptr, () -> i64
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !kgen.generator<() -> index>
    %1 = kgen.call_indirect %0() : () -> index
    llvm.return
  }
  llvm.func @h(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }
  // CHECK-LABEL: @main_closure_arg
  llvm.func internal @main_closure_arg() {
    // CHECK: [[ADDR:%.*]] = llvm.mlir.addressof @h : !llvm.ptr
    // CHECK-NEXT: unrealized_conversion_cast [[ADDR]]
    %0 = kgen.create_func_ptr[(index) -> index: @h]()
    "use.closure"(%0) : (!kgen.generator<(index) -> index>) -> ()
    llvm.return
  }
}

// -----

// COM: Check TailCallKind is passed to LLVM.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  llvm.func @h(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }
  // CHECK-LABEL: llvm.func @tailcall
  llvm.func @tailcall(%arg0: i64) -> i64 {
    %0 = kgen.create_func_ptr[(i64) -> i64: @h]()
    // CHECK: llvm.call musttail
    %1 = kgen.call_indirect musttail %0(%arg0) : (i64) -> i64
    llvm.return %1 : i64
  }
}
