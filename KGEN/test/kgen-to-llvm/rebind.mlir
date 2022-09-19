
// RUN: kgen-opt -split-input-file -lower-kgen-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func private @pointer_rebind
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<f32>, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: !llvm.ptr<i64>
kgen.func @pointer_rebind(
  %arg0: !meta.pointer<!meta.scalar<f32>>,
  %arg1: !meta.pointer<?>,
  %arg2: !meta.pointer<!meta.scalar<ui64>>
) -> (!meta.pointer<?>, !meta.pointer<!meta.scalar<si32>>, !meta.pointer<!meta.scalar<ui64>>) {
  // CHECK: %[[V0:.*]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<f32> to !llvm.ptr
  %0 = meta.pointer.rebind %arg0 : !meta.pointer<!meta.scalar<f32>> to !meta.pointer<?>
  // CHECK: %[[V1:.*]] = llvm.bitcast %[[ARG1]] : !llvm.ptr to !llvm.ptr<i32>
  %1 = meta.pointer.rebind %arg1 : !meta.pointer<?> to !meta.pointer<!meta.scalar<si32>>
  %2 = meta.pointer.rebind %arg2 : !meta.pointer<!meta.scalar<ui64>> to !meta.pointer<!meta.scalar<ui64>>
  // CHECK: llvm.insertvalue %[[V0]], {{.*}}[0]
  // CHECK: llvm.insertvalue %[[V1]], {{.*}}[1]
  // CHECK: llvm.insertvalue %[[ARG2]], {{.*}}[2]
  kgen.return %0, %1, %2 : !meta.pointer<?>, !meta.pointer<!meta.scalar<si32>>, !meta.pointer<!meta.scalar<ui64>>
}
