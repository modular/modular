// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @exclusive_ptr(ptr noalias noundef %0)
kgen.func @exclusive_ptr(%arg0: !kgen.pointer<i32 exclusive(1)>) {
  kgen.return
}

// CHECK-LABEL: @exclusive_ptr_addrspace(ptr addrspace(7) noalias noundef %0)
kgen.func @exclusive_ptr_addrspace(%arg0: !kgen.pointer<i32, 7 exclusive(1)>) {
  kgen.return
}

// CHECK-LABEL: @borrow
kgen.func @borrow(%arg0: !kgen.pointer<i32> borrow) {
  kgen.return
}

// CHECK-LABEL: @borrow_in_mem
kgen.func @borrow_in_mem(%arg0: !kgen.pointer<i32> borrow_in_mem) {
  kgen.return
}

// CHECK-LABEL: @owned
kgen.func @owned(%arg0: !kgen.pointer<i32> owned) {
  kgen.return
}

// CHECK-LABEL: @owned_in_mem
kgen.func @owned_in_mem(%arg0: !kgen.pointer<i32> owned_in_mem) {
  kgen.return
}

// CHECK-LABEL: @inout
kgen.func @inout(%arg0: !kgen.pointer<i32> inout) {
  kgen.return
}

// CHECK-LABEL: @ref
kgen.func @ref(%arg0: !kgen.pointer<i32> ref) {
  kgen.return
}

// CHECK-LABEL: @byref_result
kgen.func @byref_result(%arg0: !kgen.pointer<i32> byref_result) {
  kgen.return
}

// CHECK-LABEL: @byref_error
kgen.func @byref_error(%arg0: !kgen.pointer<i32> byref_error, %arg1: !kgen.pointer<i32> byref_result) throws {
  kgen.return
}

// CHECK-LABEL: @init_self
kgen.func @init_self(%arg0: !kgen.pointer<i32> init_self) {
  kgen.return
}

}
