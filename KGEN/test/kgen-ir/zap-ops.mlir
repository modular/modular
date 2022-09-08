// RUN: kgen-opt %s | FileCheck %s

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
  // CHECK: %[[w:.*]] = zap.buffer.load %[[C]][%[[IDX]]] : !meta.buffer<4, si32>
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
