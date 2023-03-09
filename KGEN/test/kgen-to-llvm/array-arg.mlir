// RUN: kgen-opt -allow-unregistered-dialect -lower-kgen-to-llvm %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @array_arg(%arr: !pop.array<4, i32>) {
  "use"(%arr) : (!pop.array<4, i32>) -> ()
  kgen.return
}

// CHECK-LABEL: llvm.func @array_arg
// CHECK-SAME: %[[ARR:.*]]: !llvm.ptr<array<4 x i32>>
// CHECK-NEXT: %[[V:.*]] = llvm.load %[[ARR]]
// CHECK-NEXT: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[V]]
// CHECK-NEXT: "use"(%[[CAST]])

kgen.func @array_in_struct(%s: !pop.struct<array<4, i32>>) {
  "use"(%s) : (!pop.struct<array<4, i32>>) -> ()
  kgen.return
}

// CHECK-LABEL: llvm.func @array_in_struct
// CHECK-SAME: %[[ARR_PTR:.*]]: !llvm.ptr<array<4 x i32>>
// CHECK-NEXT: %[[ARR:.*]] = llvm.load %[[ARR_PTR]]
// CHECK-NEXT: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARR]]
// CHECK-NEXT: "use"(%[[CAST]])

kgen.export @array_arg to C
kgen.export @array_in_struct to C

}
