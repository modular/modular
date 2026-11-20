// 1. Accept unions as bare values, array elements and nested union members.
// 2. Lower exact-fit global constants and external union symbols.
// RUN: kgen-opt %s -kgen-verifier -lower-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

// CHECK-DAG: @global_constant = internal constant <{ i8 }> <{ i8 1 }>, align 1
// CHECK-DAG: @raw_union_external_storage = external dso_local global { i8 }, align 1

module attributes {M.target_info = #M.target<triple="arm64-apple-darwin25.6.0", arch="", features="", data_layout="e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32", simd_bit_width=128>} {

// CHECK-LABEL: define internal void @bare_storage({ i8 } noundef
kgen.func @bare_storage(%arg: !kgen.untagged_union<i8>) {
  hlcf.return
}

// CHECK-LABEL: define internal void @array_storage([2 x { i8 }] noundef
kgen.func @array_storage(%arg: !pop.array<2, untagged_union<i8>>) {
  hlcf.return
}

// CHECK-LABEL: define internal void @nested_union_storage({ i8 } noundef
kgen.func @nested_union_storage(%arg: !pop.union<untagged_union<i8>>) {
  hlcf.return
}

// CHECK-LABEL: define internal void @constant_storage(
kgen.func @constant_storage() {
  %storage = pop.global_constant: !kgen.untagged_union<i8> = <#kgen.untagged_union<[1 : i8]> : !kgen.untagged_union<i8>>
  hlcf.return
}

// CHECK-LABEL: define internal void @external_storage(
kgen.func @external_storage() {
  %storage = pop.extern_ptr_symbol "raw_union_external_storage" alignment <1> : !kgen.pointer<untagged_union<i8>>
  hlcf.return
}

}
