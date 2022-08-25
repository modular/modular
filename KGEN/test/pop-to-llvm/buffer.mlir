// RUN: kgen-opt -convert-pop-to-llvm %s | FileCheck %s

// CHECK-LABEL: @buffer_load
kgen.kernel @buffer_load(%buf: !meta.buffer<4, f32>, %idx: index) -> !meta.scalar<f32> {
  // CHECK: %[[BUF:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[IDX:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %[[BUF]][%[[IDX]]]
  // CHECK: %[[LOAD:.*]] = llvm.load %[[PTR]] : !llvm.ptr<f32>
  // CHECK: unrealized_conversion_cast %[[LOAD]]
  %0 = pop.buffer.load %buf[%idx] : !meta.buffer<4, f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: @buffer_load_dynamic
kgen.kernel @buffer_load_dynamic(%buf: !meta.buffer<?, f32>, %idx: index) -> !meta.scalar<f32> {
  // CHECK: %[[BUF:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[IDX:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[PTR:.*]] = llvm.extractvalue %[[BUF]][1]
  // CHECK: %[[OFFSET:.*]] = llvm.getelementptr %[[PTR]][%[[IDX]]]
  // CHECK: %[[LOAD:.*]] = llvm.load %[[OFFSET]] : !llvm.ptr<f32>
  // CHECK: unrealized_conversion_cast %[[LOAD]]
  %0 = pop.buffer.load %buf[%idx] : !meta.buffer<?, f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: @buffer_store
kgen.kernel @buffer_store(%val: !meta.scalar<f32>, %buf: !meta.buffer<4, f32>, %idx: index) -> () {
  // CHECK: %[[BUF:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[VAL:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[IDX:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %[[BUF]][%[[IDX]]]
  // CHECK: lvm.store %[[VAL]], %[[PTR]] : !llvm.ptr<f32>
  pop.buffer.store %val, %buf[%idx] : !meta.buffer<4, f32>
  kgen.return
}

// CHECK-LABEL: @buffer_store_dynamic
kgen.kernel @buffer_store_dynamic(%val: !meta.scalar<f32>, %buf: !meta.buffer<?, f32>, %idx: index) -> () {
  // CHECK: %[[BUF:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[VAL:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[IDX:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[PTR:.*]] = llvm.extractvalue %[[BUF]][1]
  // CHECK: %[[OFFSET:.*]] = llvm.getelementptr %[[PTR]][%[[IDX]]]
  // CHECK: lvm.store %[[VAL]], %[[OFFSET]] : !llvm.ptr<f32>
  pop.buffer.store %val, %buf[%idx] : !meta.buffer<?, f32>
  kgen.return
}
