// RUN: kgen-opt %s --lower-runtime-closures | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK:  %1 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, ptr)>
  // CHECK:  %2 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, ptr)>
  // CHECK:  %3 = llvm.bitcast %1 : !llvm.ptr to !llvm.ptr<func<i64 (ptr)>>
  // CHECK:  %4 = llvm.call %3(%2) : !llvm.ptr<func<i64 (ptr)>>, (!llvm.ptr) -> i64
  llvm.func internal @take_closure_no_args(%arg0: !llvm.struct<(ptr, ptr)>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(ptr, ptr)> to !kgen.signature<() capturing -> index>
    %1 = kgen.call_signature  %0() : () capturing -> index
    llvm.return
  }
  llvm.func internal @h(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }
  // CHECK:  %0 = builtin.unrealized_conversion_cast %idx98 : index to i64
  // CHECK:  %1 = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
  // CHECK:  %2 = llvm.mlir.addressof @closure_wrapper_fn : !llvm.ptr<func<i64 (ptr)>>
  // CHECK:  %3 = llvm.bitcast %2 : !llvm.ptr<func<i64 (ptr)>> to !llvm.ptr
  // CHECK:  %4 = llvm.insertvalue %3, %1[0] : !llvm.struct<(ptr, ptr)>
  // CHECK:  %5 = llvm.mlir.constant(1 : i8) : i8
  // CHECK:  %6 = llvm.alloca %5 x !llvm.struct<(i64)> : (i8) -> !llvm.ptr<struct<(i64)>>
  // CHECK:  llvm.intr.lifetime.start 8, %6 : !llvm.ptr<struct<(i64)>>
  // CHECK:  %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr<struct<(i64)>>) -> !llvm.ptr<i64>
  // CHECK:  llvm.store %0, %7 : !llvm.ptr<i64>
  // CHECK:  %8 = llvm.bitcast %6 : !llvm.ptr<struct<(i64)>> to !llvm.ptr
  // CHECK:  %9 = llvm.insertvalue %8, %4[1] : !llvm.struct<(ptr, ptr)>
  // CHECK:  %10 = builtin.unrealized_conversion_cast %9 : !llvm.struct<(ptr, ptr)> to !kgen.signature<() capturing -> index>
  // CHECK:  %11 = builtin.unrealized_conversion_cast %10 : !kgen.signature<() capturing -> index> to !llvm.struct<(ptr, ptr)>
  llvm.func internal @main_closure_arg() {
    %idx98 = index.constant 98
    %0 = kgen.create_closure [(index) -> index: @h](%idx98)
    %1 = builtin.unrealized_conversion_cast %0 : !kgen.signature<() capturing -> index> to !llvm.struct<(ptr, ptr)>
    llvm.call @take_closure_no_args(%1) {fastmathFlags = #llvm.fastmath<contract>} : (!llvm.struct<(ptr, ptr)>) -> ()
    llvm.return
  }
  // CHECK:  llvm.func internal @closure_wrapper_fn(%arg0: !llvm.ptr) -> i64 {
  // CHECK:  %0 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr<struct<(i64)>>
  // CHECK:  %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<struct<(i64)>>) -> !llvm.ptr<i64>
  // CHECK:  %2 = llvm.load %1 : !llvm.ptr<i64>
  // CHECK:  %3 = llvm.call @h(%2) : (i64) -> i64
  // CHECK:  llvm.return %3 : i64
}
