// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: @zap_buffer_stack_allocation
kgen.generator @zap_buffer_stack_allocation<type: dtype, size>() {
  // CHECK: zap.buffer.stack_allocation : !meta.buffer<4, f32>
  %0 = zap.buffer.stack_allocation : !meta.buffer<4, f32>
  // CHECK: zap.buffer.stack_allocation : !meta.buffer<size, f32>
  %1 = zap.buffer.stack_allocation : !meta.buffer<size, f32>
  // CHECK: zap.buffer.stack_allocation : !meta.buffer<size, type>
  %2 = zap.buffer.stack_allocation : !meta.buffer<size, type>
  kgen.return
}

// CHECK-LABEL: @zap_buffer_load
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @zap_buffer_load<size, type: dtype>(
    %a: !meta.buffer<size, type>,
    %b: !meta.buffer<size, f32>,
    %c: !meta.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: %[[U:.*]] = zap.buffer.load %[[A]][%[[IDX]]] : !meta.buffer<size, type>
  %u = zap.buffer.load %a[%idx] : !meta.buffer<size, type>
  // CHECK: %[[V:.*]] = zap.buffer.load %[[B]][%[[IDX]]] : !meta.buffer<size, f32>
  %v = zap.buffer.load %b[%idx] : !meta.buffer<size, f32>
  // CHECK: %[[W:.*]] = zap.buffer.load %[[C]][%[[IDX]]] : !meta.buffer<4, si32>
  %w = zap.buffer.load %c[%idx] : !meta.buffer<4, si32>
  kgen.return
}

// CHECK-LABEL: @zap_buffer_store
// CHECK-SAME: %[[V0:[a-z0-9]+]]:
// CHECK-SAME: %[[V1:[a-z0-9]+]]:
// CHECK-SAME: %[[V2:[a-z0-9]+]]:
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @zap_buffer_store<size, type: dtype>(
    %v0: !meta.scalar<type>,
    %v1: !meta.scalar<f32>,
    %v2: !meta.scalar<si32>,
    %a: !meta.buffer<size, type>,
    %b: !meta.buffer<size, f32>,
    %c: !meta.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: zap.buffer.store %[[V0]], %[[A]][%[[IDX]]] : !meta.buffer<size, type>
  zap.buffer.store %v0, %a[%idx] : !meta.buffer<size, type>
  // CHECK: zap.buffer.store %[[V1]], %[[B]][%[[IDX]]] : !meta.buffer<size, f32>
  zap.buffer.store %v1, %b[%idx] : !meta.buffer<size, f32>
  // CHECK: zap.buffer.store %[[V2]], %[[C]][%[[IDX]]] : !meta.buffer<4, si32>
  zap.buffer.store %v2, %c[%idx] : !meta.buffer<4, si32>
  kgen.return
}

// CHECK-LABEL: @zap_buffer_constant
kgen.generator @zap_buffer_constant<size, type: dtype>() {
  // CHECK: zap.buffer.constant(dense<[1.{{0+}}e+01, 1.2{{0+}}e+01, -2.{{0+}}e+00]> : tensor<3xf32>) : f32
  %0 = zap.buffer.constant(dense<[10.0, 12.0, -2.0]> : tensor<3xf32>) : f32
  // CHECK: zap.buffer.constant(dense<[2, 3]> : tensor<2xui8>) : type
  %1 = zap.buffer.constant(dense<[2, 3]> : tensor<2xui8>) : type
  kgen.return
}

// CHECK-LABEL: @zap_simd_load
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @zap_simd_load<size, type: dtype>(
    %a: !meta.buffer<size, type>,
    %b: !meta.buffer<size, f32>,
    %c: !meta.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 0
  // CHECK: %[[U:.*]] = zap.simd.load %[[A]][%[[IDX]]] : !meta.buffer<size, type>, !meta.simd<4, type>
  %u = zap.simd.load %a[%idx] : !meta.buffer<size, type>, !meta.simd<4, type>
  // CHECK: %[[V:.*]] = zap.simd.load %[[B]][%[[IDX]]] : !meta.buffer<size, f32>, !meta.simd<4, f32>
  %v = zap.simd.load %b[%idx] : !meta.buffer<size, f32>, !meta.simd<4, f32>
  // CHECK: %[[W:.*]] = zap.simd.load %[[C]][%[[IDX]]] : !meta.buffer<4, si32>, !meta.simd<4, si32>
  %w = zap.simd.load %c[%idx] : !meta.buffer<4, si32>, !meta.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @zap_simd_store
// CHECK-SAME: %[[V0:[a-z0-9]+]]:
// CHECK-SAME: %[[V1:[a-z0-9]+]]:
// CHECK-SAME: %[[V2:[a-z0-9]+]]:
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @zap_simd_store<size, type: dtype>(
    %v0: !meta.simd<size, type>,
    %v1: !meta.simd<8, f32>,
    %v2: !meta.simd<2, si32>,
    %a: !meta.buffer<size, type>,
    %b: !meta.buffer<size, f32>,
    %c: !meta.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 0
  // CHECK: zap.simd.store %[[V0]], %[[A]][%[[IDX]]] : !meta.simd<size, type>, !meta.buffer<size, type>
  zap.simd.store %v0, %a[%idx] : !meta.simd<size, type>, !meta.buffer<size, type>
  // CHECK: zap.simd.store %[[V1]], %[[B]][%[[IDX]]] : !meta.simd<8, f32>, !meta.buffer<size, f32>
  zap.simd.store %v1, %b[%idx] : !meta.simd<8, f32>, !meta.buffer<size, f32>
  // CHECK: zap.simd.store %[[V2]], %[[C]][%[[IDX]]] : !meta.simd<2, si32>, !meta.buffer<4, si32>
  zap.simd.store %v2, %c[%idx] : !meta.simd<2, si32>, !meta.buffer<4, si32>
  kgen.return
}
