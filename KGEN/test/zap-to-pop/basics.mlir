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

// CHECK-LABEL: @simd_load
// CHECK-SAME: %[[BUF:.*]]: !meta.buffer
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @simd_load(%buf: !meta.buffer<4, f32>, %idx: index) -> !meta.simd<4, f32> {
  // CHECK: %[[BASE:.*]] = meta.buffer.address %[[BUF]]
  // CHECK: %[[PTR:.*]] = pop.offset %[[BASE]][%[[IDX]]]
  // CHECK: %[[BPTR:.*]] = pop.bitcast %[[PTR]] : !meta.pointer<!meta.scalar<f32>> to !meta.pointer<!meta.simd<4, f32>>
  // CHECK: %[[VAL:.*]] = pop.load %[[BPTR]]
  %0 = zap.simd.load %buf[%idx] : !meta.buffer<4, f32>, !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: @simd_store
// CHECK-SAME: %[[VAL:.*]]: !meta.simd
// CHECK-SAME: %[[BUF:.*]]: !meta.buffer
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @simd_store(%val : !meta.simd<4, f32>, %buf: !meta.buffer<4, f32>, %idx: index) {
  // CHECK: %[[BASE:.*]] = meta.buffer.address %[[BUF]]
  // CHECK: %[[PTR:.*]] = pop.offset %[[BASE]][%[[IDX]]]
  // CHECK: %[[BPTR:.*]] = pop.bitcast %[[PTR]] : !meta.pointer<!meta.scalar<f32>> to !meta.pointer<!meta.simd<4, f32>>
  // CHECK: pop.store %[[VAL]], %[[BPTR]]
  zap.simd.store %val, %buf[%idx] : !meta.simd<4, f32>, !meta.buffer<4, f32>
  kgen.return
}
