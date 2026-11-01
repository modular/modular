// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @addrspace(ptr addrspace(7) noundef %0)
kgen.func @addrspace(%arg0: !kgen.pointer<i32, 7>) {
  hlcf.return
}

// CHECK-LABEL: @borrow(ptr noundef %0)
kgen.func @borrow(%arg0: !kgen.pointer<i32> imm) {
  hlcf.return
}

// CHECK-LABEL: @imm_mem(ptr noundef nonnull %0)
kgen.func @imm_mem(%arg0: !kgen.pointer<i32> imm_mem) {
  hlcf.return
}

// CHECK-LABEL: @owned(ptr noundef %0)
kgen.func @owned(%arg0: !kgen.pointer<i32> owned) {
  hlcf.return
}

// CHECK-LABEL: @owned_in_mem(ptr noalias noundef nonnull %0)
kgen.func @owned_in_mem(%arg0: !kgen.pointer<i32> owned_in_mem) {
  hlcf.return
}

// CHECK-LABEL: @mut(ptr noalias noundef nonnull %0)
kgen.func @mut(%arg0: !kgen.pointer<i32> mut) {
  hlcf.return
}

// CHECK-LABEL: @ref(ptr noundef nonnull %0)
kgen.func @ref(%arg0: !kgen.pointer<i32> ref) {
  hlcf.return
}

// CHECK-LABEL: @immref(ptr noalias noundef nonnull %0)
kgen.func @immref(%arg0: !kgen.pointer<i32> mutref) {
  hlcf.return
}

// CHECK-LABEL: @byref_result(ptr noalias noundef nonnull %0)
kgen.func @byref_result(%arg0: !kgen.pointer<i32> byref_result) {
  hlcf.return
}

// CHECK-LABEL: @byref_error(ptr noalias noundef nonnull %0, ptr noalias noundef nonnull %1)
kgen.func @byref_error(%arg0: !kgen.pointer<i32> byref_error, %arg1: !kgen.pointer<i32> byref_result) throws {
  hlcf.return
}
}
