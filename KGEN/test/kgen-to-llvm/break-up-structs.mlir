// RUN: kgen-opt %s -split-input-file -convert-kgen-to-llvm=break-up-structs="kernel" | FileCheck %s

// CHECK-LABEL: @kernel
// CHECK-SAME: %[[SIZE:.*]]: [[INDEXTY:.*]], %[[PTR:.*]]: !llvm.ptr, %[[DTYPE:.*]]: i8
// CHECK-SAME: %[[SIZE_OUT:.*]]: !llvm.ptr<[[INDEXTY]]>, %[[PTR_OUT:.*]]: !llvm.ptr<ptr>, %[[DTYPE_OUT:.*]]: !llvm.ptr<i8>
// CHECK: %[[BUFFER:.*]] = llvm.mlir.undef
// CHECK: %[[B0:.*]] = llvm.insertvalue %[[SIZE]], %[[BUFFER]][0]
// CHECK: %[[B1:.*]] = llvm.insertvalue %[[PTR]], %[[B0]][1]
// CHECK: %[[RESULT_BUFFER:.*]] = llvm.insertvalue %[[DTYPE]], %[[B1]][2]
// CHECK: %[[SIZE_RESULT:.*]] = llvm.extractvalue %[[RESULT_BUFFER]][0]
// CHECK: llvm.store %[[SIZE_RESULT]], %[[SIZE_OUT]]
// CHECK: %[[PTR_RESULT:.*]] = llvm.extractvalue %[[RESULT_BUFFER]][1]
// CHECK: llvm.store %[[PTR_RESULT]], %[[PTR_OUT]]
// CHECK: %[[DTYPE_RESULT:.*]] = llvm.extractvalue %[[RESULT_BUFFER]][2]
// CHECK: llvm.store %[[DTYPE_RESULT]], %[[DTYPE_OUT]]
// CHECK: llvm.return

kgen.func @kernel(%a: !meta.buffer<?, ?>) -> !meta.buffer<?, ?> {
  kgen.return %a : !meta.buffer<?, ?>
}

// -----

// CHECK-LABEL: @kernel
// CHECK-SAME: %[[V:.*]]: f32, %[[V_OUT:.*]]: !llvm.ptr<f32>
// CHECK: %[[S0:.*]] = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(f32)>)>
// CHECK: %[[EMPTY:.*]] = llvm.mlir.undef : !llvm.struct<()>
// CHECK: %[[S1:.*]] = llvm.insertvalue %[[EMPTY]], %[[S0]][0]
// CHECK: %[[FP:.*]] = llvm.mlir.undef : !llvm.struct<(f32)>
// CHECK: %[[FP0:.*]] = llvm.insertvalue %[[V]], %[[FP]][0]
// CHECK: %[[RES:.*]] = llvm.insertvalue %[[FP0]], %[[S1]][1]
// CHECK: %[[R0:.*]] = llvm.extractvalue %[[RES]][1]
// CHECK: %[[V_RESULT:.*]] = llvm.extractvalue %[[R0]][0]
// CHECK: llvm.store %[[V_RESULT]], %[[V_OUT]]

!nestedStruct = !llvm.struct<(struct<()>, struct<(f32)>)>
llvm.func @kernel(%a: !nestedStruct) -> !nestedStruct {
  llvm.return %a : !nestedStruct
}

// -----

// CHECK-LABEL: @kernel
// CHECK-SAME: %[[I:.*]]: f32
// CHECK: llvm.return

kgen.func @kernel(%i: f32, %a: !meta.buffer<?, f32>) {
  kgen.return
}

// -----

// CHECK-LABEL: @kernel
// CHECK-SAME: -> i64
// CHECK: %[[RESULT:.*]] = llvm.mlir.constant
// CHECK: llvm.return %[[RESULT]]

kgen.func @kernel(%a: !meta.buffer<?, f32>) -> i64 {
  %0 = llvm.mlir.constant(1 : i64) : i64
  kgen.return %0 : i64
}

// -----

// CHECK-LABEL: @kernel
// CHECK-SAME: (%{{.*}}: f32) -> f32
// CHECK-NEXT: return %{{.*}}: f32
kgen.func @kernel(%a: f32) -> f32 {
  kgen.return %a : f32
}

// -----

!structTy = !llvm.struct<(i32, struct<(struct<(i32)>, f32)>)>

// CHECK-LABEL: @kernel
// CHECK-SAME: (%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: f32, %{{.*}}: i1,
// CHECK-SAME: %[[A:.*]]: !llvm.ptr<i32>, %[[B:.*]]: !llvm.ptr<i32>, %[[C:.*]]: !llvm.ptr<f32>)
llvm.func @kernel(%a: !structTy, %c: i1) -> !structTy {
  llvm.cond_br %c, ^bb1, ^bb2

// CHECK: ^bb1
^bb1:
  // CHECK: llvm.store %{{.*}}, %[[A]]
  // CHECK: llvm.store %{{.*}}, %[[B]]
  // CHECK: llvm.store %{{.*}}, %[[C]]
  llvm.return %a : !structTy

// CHECK: ^bb2
^bb2:
  // CHECK: llvm.store %{{.*}}, %[[A]]
  // CHECK: llvm.store %{{.*}}, %[[B]]
  // CHECK: llvm.store %{{.*}}, %[[C]]
  llvm.return %a : !structTy
}
