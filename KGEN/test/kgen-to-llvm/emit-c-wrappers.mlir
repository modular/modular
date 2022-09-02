// RUN: kgen-opt %s -split-input-file -convert-kgen-to-llvm=emit-c-wrappers="kernel" | FileCheck %s

// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME: attributes {c_wrapper = @kernel_c_wrapper}
kgen.kernel @kernel(%a: !meta.buffer<?, ?>) -> !meta.buffer<?, ?> {
  kgen.return %a : !meta.buffer<?, ?>
}

// CHECK: @kernel_c_wrapper
// CHECK-SAME: %[[SIZE:.*]]: [[INDEXTY:.*]], %[[PTR:.*]]: !llvm.ptr, %[[DTYPE:.*]]: i8
// CHECK-SAME: %[[SIZE_OUT:.*]]: !llvm.ptr<[[INDEXTY]]>, %[[PTR_OUT:.*]]: !llvm.ptr<ptr>, %[[DTYPE_OUT:.*]]: !llvm.ptr<i8>
// CHECK: %[[BUFFER:.*]] = llvm.mlir.undef
// CHECK: %[[B0:.*]] = llvm.insertvalue %[[SIZE]], %[[BUFFER]][0]
// CHECK: %[[B1:.*]] = llvm.insertvalue %[[PTR]], %[[B0]][1]
// CHECK: %[[B2:.*]] = llvm.insertvalue %[[DTYPE]], %[[B1]][2]
// CHECK: %[[RESULT_BUFFER:.*]] = llvm.call @kernel(%[[B2]])
// CHECK: %[[SIZE_RESULT:.*]] = llvm.extractvalue %[[RESULT_BUFFER]][0]
// CHECK: llvm.store %[[SIZE_RESULT]], %[[SIZE_OUT]]
// CHECK: %[[PTR_RESULT:.*]] = llvm.extractvalue %[[RESULT_BUFFER]][1]
// CHECK: llvm.store %[[PTR_RESULT]], %[[PTR_OUT]]
// CHECK: %[[DTYPE_RESULT:.*]] = llvm.extractvalue %[[RESULT_BUFFER]][2]
// CHECK: llvm.store %[[DTYPE_RESULT]], %[[DTYPE_OUT]]
// CHECK: llvm.return

// -----

// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME: attributes {c_wrapper = @kernel_c_wrapper}
kgen.kernel @kernel(%i: f32, %a: !meta.buffer<?, f32>) {
  kgen.return
}

// CHECK: @kernel_c_wrapper
// CHECK: llvm.call @kernel({{.*}}) : ({{.*}}) -> ()
// CHECK: llvm.return

// -----

// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME: attributes {c_wrapper = @kernel_c_wrapper}
kgen.kernel @kernel(%a: !meta.buffer<?, f32>) -> i64 {
  %0 = llvm.mlir.constant(1 : i64) : i64
  kgen.return %0 : i64
}

// CHECK: @kernel_c_wrapper
// CHECK-SAME: -> i64
// CHECK: %[[RES:.*]] = llvm.call @kernel({{.*}}) : ({{.*}}) -> i64
// CHECK: llvm.return %[[RES]]
