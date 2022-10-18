// RUN: kgen-opt -allow-unregistered-dialect -lower-kgen-to-llvm="c-call=array_arg,array_in_struct emit-opaque-wrappers" %s | FileCheck %s

// CHECK-LABEL: llvm.func @array_arg
// CHECK-SAME: %[[ARR:.*]]: !llvm.ptr<array<4 x i32>>
kgen.func public @array_arg(%arr: !pop.array<4, i32>) {
  // CHECK: %[[V:.*]] = llvm.load %[[ARR]]
  // CHECK: unrealized_conversion_cast %[[V]]
  "use"(%arr) : (!pop.array<4, i32>) -> ()
  kgen.return
}

// CHECK-LABEL: llvm.func @array_arg_opaque_wrapper
// CHECK-SAME: %[[ARG:.*]]: !llvm.ptr<struct<packed (array<4 x i32>)>>
// CHECK-NEXT: %[[PTR:.*]] = llvm.getelementptr %[[ARG]][0, 0]
// CHECK-NEXT: llvm.call @array_arg(%[[PTR]]) : (!llvm.ptr<array<4 x i32>>)

// CHECK-LABEL: llvm.func @array_in_struct
// CHECK-SAME: %[[ARR_PTR:.*]]: !llvm.ptr<array<4 x i32>>
kgen.func public @array_in_struct(%s: !pop.struct<array<4, i32>>) {
  // CHECK: %[[S:.*]] = llvm.mlir.undef
  // CHECK: %[[ARR:.*]] = llvm.load %[[ARR_PTR]]
  // CHECK: llvm.insertvalue %[[ARR]], %[[S]][0]
  "use"(%s) : (!pop.struct<array<4, i32>>) -> ()
  kgen.return
}

// CHECK-LABEL: llvm.func @array_in_struct_opaque_wrapper
// CHECK-SAME: %[[ARG:.*]]: !llvm.ptr<struct<packed (struct<packed (array<4 x i32>)>)>>
// CHECK-NEXT: %[[PTR:.*]] = llvm.getelementptr %[[ARG]][0, 0, 0]
// CHECK-NEXT: llvm.call @array_in_struct(%[[PTR]])
