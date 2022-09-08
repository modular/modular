// RUN: kgen-opt -convert-pop-to-llvm %s | FileCheck %s

// CHECK-LABEL: @pop_buffer_stack_allocation
kgen.kernel @pop_buffer_stack_allocation() -> !meta.buffer<4, f32> {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(4 : index)
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[SIZE]] x f32 : (i64) -> !llvm.ptr<f32>
  // CHECK: %[[BUFFER:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]]
  %buf = pop.buffer.stack_allocation : !meta.buffer<4, f32>
  // CHECK: kgen.return %[[BUFFER]]
  kgen.return %buf : !meta.buffer<4, f32>
}
