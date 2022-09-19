// RUN: kgen-opt -lower-zap-to-pop -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @buffer_stack_allocation
kgen.generator @buffer_stack_allocation<size, type: dtype>() {
  // CHECK: %[[PTR0:.*]] = pop.stack_allocation 4 : !meta.scalar<f32>
  // CHECK: %[[BUF0:.*]] = meta.buffer.construct %[[PTR0]] : !meta.buffer<4, f32>
  %0 = zap.buffer.stack_allocation : !meta.buffer<4, f32>
  // CHECK: %[[PTR1:.*]] = pop.stack_allocation size : !meta.scalar<type>
  // CHECK: %[[BUF1:.*]] = meta.buffer.construct %[[PTR1]] : !meta.buffer<size, type>
  %1 = zap.buffer.stack_allocation : !meta.buffer<size, type>
  // CHECK: "use"(%[[BUF0]], %[[BUF1]])
  "use"(%0, %1) : (!meta.buffer<4, f32>, !meta.buffer<size, type>) -> ()
  kgen.return
}

// CHECK-LABEL: @zap_buffer_constant
kgen.generator @zap_buffer_constant<type: dtype>() {
  // CHECK: %[[ARR0:.*]] = pop.global_constant(dense<[1.{{0+}}e+01, 1.2{{0+}}e+01, -2.{{0+}}e+00]>
  // CHECK: %[[PTR0:.*]] = pop.bitcast %[[ARR0]] : !meta.pointer<!pop.array<3, !meta.scalar<f32>>> to !meta.pointer<!meta.scalar<f32>>
  // CHECK: %[[BUF0:.*]] = meta.buffer.construct %[[PTR0]] : !meta.buffer<3, f32>
  // CHECK: %[[ARR1:.*]] = pop.global_constant(dense<[2, 3]>
  // CHECK: %[[PTR1:.*]] = pop.bitcast %[[ARR1]] : !meta.pointer<!pop.array<2, !meta.scalar<type>>> to !meta.pointer<!meta.scalar<type>>
  // CHECK: %[[BUF1:.*]] = meta.buffer.construct %[[PTR1]] : !meta.buffer<2, type>
  %0 = zap.buffer.constant(dense<[10.0, 12.0, -2.0]> : tensor<3xf32>) : f32
  %1 = zap.buffer.constant(dense<[2, 3]> : tensor<2xui8>) : type
  // CHECK: "use"(%[[BUF0]], %[[BUF1]])
  "use"(%0, %1) : (!meta.buffer<3, f32>, !meta.buffer<2, type>) -> ()
  kgen.return
}

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
