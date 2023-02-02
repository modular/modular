// RUN: kgen-opt -allow-unregistered-dialect -lower-kgen-to-llvm %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_bit_width=64, simd_bit_width=128>} {

// CHECK-LABEL: llvm.func @array_arg
kgen.func @array_arg(%arr: !pop.array<4, i32>) {
  "use"(%arr) : (!pop.array<4, i32>) -> ()
  kgen.return
}

// CHECK-LABEL: llvm.func @array_arg_c
// CHECK-SAME: %[[ARR:.*]]: !llvm.ptr<array<4 x i32>>
// CHECK-NEXT: %[[V:.*]] = llvm.load %[[ARR]]
// CHECK-NEXT: llvm.call @array_arg(%[[V]])

// CHECK-LABEL: llvm.func @array_in_struct
kgen.func @array_in_struct(%s: !pop.struct<array<4, i32>>) {
  "use"(%s) : (!pop.struct<array<4, i32>>) -> ()
  kgen.return
}

// CHECK-LABEL: llvm.func @array_in_struct_c
// CHECK-SAME: %[[ARR_PTR:.*]]: !llvm.ptr<array<4 x i32>>
// CHECK-NEXT: %[[ARR:.*]] = llvm.load %[[ARR_PTR]]
// CHECK-NEXT: llvm.call @array_in_struct(%[[ARR]])

kgen.export @array_arg
kgen.export @array_in_struct

}
