// RUN: kgen-opt -lower-zap-to-pop %s | FileCheck %s

// CHECK-LABEL: @buffer_load
// CHECK-SAME: %[[BUF:.*]]: !meta.buffer
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @buffer_load(%buf: !meta.buffer<4, f32>, %idx: index) -> !meta.scalar<f32> {
  // CHECK: %[[BASE:.*]] = meta.buffer.address %[[BUF]]
  // CHECK: %[[PTR:.*]] = pop.offset %[[BASE]][%[[IDX]]]
  // CHECK: %[[VAL:.*]] = pop.load %[[PTR]]
  // CHECK: return %[[VAL]]
  %0 = zap.buffer.load %buf[%idx] : !meta.buffer<4, f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: @buffer_store
// CHECK-SAME: %[[VAL:.*]]: !meta.scalar
// CHECK-SAME: %[[BUF:.*]]: !meta.buffer
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @buffer_store(%val: !meta.scalar<f32>, %buf: !meta.buffer<4, f32>, %idx: index) -> () {
  // CHECK: %[[BASE:.*]] = meta.buffer.address %[[BUF]]
  // CHECK: %[[PTR:.*]] = pop.offset %[[BASE]][%[[IDX]]]
  // CHECK: pop.store %[[VAL]], %[[PTR]]
  zap.buffer.store %val, %buf[%idx] : !meta.buffer<4, f32>
  kgen.return
}
