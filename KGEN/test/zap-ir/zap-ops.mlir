// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: @zap_buffer_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer<!pop.scalar<f32>>
// CHECK-SAME: %[[OPAQUE:.*]]: !pop.pointer<?>
// CHECK-SAME: %[[SIZE:.*]]: index
// CHECK-SAME: %[[DTYPE:.*]]: !kgen.dtype
kgen.func @zap_buffer_construct(
  %ptr: !pop.pointer<!pop.scalar<f32>>,
  %opaque: !pop.pointer<?>,
  %size: index,
  %dtype: !kgen.dtype) -> (
  !zap.buffer<4, f32>,
  !zap.buffer<?, f32>,
  !zap.buffer<4, ?>,
  !zap.buffer<?, ?>
) {
  // CHECK: zap.buffer.construct %[[PTR]] : !zap.buffer<4, f32>
  %0 = zap.buffer.construct %ptr : !zap.buffer<4, f32>
  // CHECK: zap.buffer.construct %[[PTR]][%[[SIZE]]] : !zap.buffer<?, f32>
  %1 = zap.buffer.construct %ptr[%size] : !zap.buffer<?, f32>
  // CHECK: zap.buffer.construct %[[OPAQUE]] of %[[DTYPE]] : !zap.buffer<4, ?>
  %2 = zap.buffer.construct %opaque of %dtype : !zap.buffer<4, ?>
  // CHECK: zap.buffer.construct %[[OPAQUE]][%[[SIZE]]] of %[[DTYPE]] : !zap.buffer<?, ?>
  %3 = zap.buffer.construct %opaque[%size] of %dtype : !zap.buffer<?, ?>
  kgen.return %0, %1, %2, %3 :
    !zap.buffer<4, f32>,
    !zap.buffer<?, f32>,
    !zap.buffer<4, ?>,
    !zap.buffer<?, ?>
}

// CHECK-LABEL: @zap_buffer_size
// CHECK-SAME: %[[ARG0:.*]]: !zap.buffer<{{.*}}>, %[[ARG1:.*]]:
kgen.func @zap_buffer_size(%arg0: !zap.buffer<42, f32>, %arg1: !zap.buffer<?, f32>) {
  // CHECK: zap.buffer.size %[[ARG0]]
  %0 = zap.buffer.size %arg0 : !zap.buffer<42, f32>
  // CHECK: zap.buffer.size %[[ARG1]]
  %1 = zap.buffer.size %arg1 : !zap.buffer<?, f32>
  kgen.return
}

// CHECK-LABEL: @zap_buffer_dtype
// CHECK-SAME: %[[ARG0:.*]]: !zap.buffer<{{.*}}>, %[[ARG1:.*]]:
kgen.func @zap_buffer_dtype(%arg0: !zap.buffer<42, f32>, %arg1: !zap.buffer<42, ?>) {
  // CHECK: zap.buffer.dtype %[[ARG0]]
  %0 = zap.buffer.dtype %arg0 : !zap.buffer<42, f32>
  // CHECK: zap.buffer.dtype %[[ARG1]]
  %1 = zap.buffer.dtype %arg1 : !zap.buffer<42, ?>
  kgen.return
}

// CHECK-LABEL: @zap_buffer_address
kgen.generator @zap_buffer_address<dt: dtype, size>(
  // CHECK-SAME: %[[ARG0:.*]]: !zap.buffer<size
  // CHECK-SAME: %[[ARG1:.*]]: !zap.buffer<?
  // CHECK-SAME: %[[ARG2:.*]]: !zap.buffer<3
   %arg0: !zap.buffer<size, dt>, %arg1: !zap.buffer<?, ?>, %arg2: !zap.buffer<3, f32>) {
  // CHECK: zap.buffer.address %[[ARG0]]
  %0 = zap.buffer.address %arg0 : !zap.buffer<size, dt>
  // CHECK: zap.buffer.address %[[ARG1]]
  %1 = zap.buffer.address %arg1 : !zap.buffer<?, ?>
  // CHECK: zap.buffer.address %[[ARG2]]
  %2 = zap.buffer.address %arg2 : !zap.buffer<3, f32>
  kgen.return
}

// CHECK-LABEL: @zap_buffer_bitcast
// CHECK-SAME: %[[ARG0:.*]]:
kgen.generator @zap_buffer_bitcast<size, size2, dt: dtype>(%arg0: !zap.buffer<?, ?>) {
  // CHECK: %[[V0:.*]] = zap.buffer.bitcast %[[ARG0]] : !zap.buffer<?, ?> to !zap.buffer<42, f32>
  %0 = zap.buffer.bitcast %arg0 : !zap.buffer<?, ?> to !zap.buffer<42, f32>
  // CHECK: %[[V1:.*]] = zap.buffer.bitcast %[[V0]] : !zap.buffer<42, f32> to !zap.buffer<?, ?>
  %1 = zap.buffer.bitcast %0 : !zap.buffer<42, f32> to !zap.buffer<?, ?>
  // CHECK: %[[V2:.*]] = zap.buffer.bitcast %[[ARG0]] : !zap.buffer<?, ?> to !zap.buffer<?, f32>
  %2 = zap.buffer.bitcast %arg0 : !zap.buffer<?, ?> to !zap.buffer<?, f32>
  // CHECK: %[[V3:.*]] = zap.buffer.bitcast %[[ARG0]] : !zap.buffer<?, ?> to !zap.buffer<42, ?>
  %3 = zap.buffer.bitcast %arg0 : !zap.buffer<?, ?> to !zap.buffer<42, ?>
  // CHECK: %[[V4:.*]] = zap.buffer.bitcast %[[V2]] : !zap.buffer<?, f32> to !zap.buffer<42, f32>
  %4 = zap.buffer.bitcast %2 : !zap.buffer<?, f32> to !zap.buffer<42, f32>
  // CHECK: %[[V5:.*]] = zap.buffer.bitcast %[[V3]] : !zap.buffer<42, ?> to !zap.buffer<42, f32>
  %5 = zap.buffer.bitcast %3 : !zap.buffer<42, ?> to !zap.buffer<42, f32>
  // CHECK: %[[V6:.*]] = zap.buffer.bitcast %[[V0]] : !zap.buffer<42, f32> to !zap.buffer<42, f32>
  %6 = zap.buffer.bitcast %0 : !zap.buffer<42, f32> to !zap.buffer<42, f32>


  // Conversions between different unknown parameters are ok.
  // CHECK: %[[V7:.*]] = zap.buffer.bitcast %[[V0]] : !zap.buffer<42, f32> to !zap.buffer<size, f32>
  %7 = zap.buffer.bitcast %0 : !zap.buffer<42, f32> to !zap.buffer<size, f32>

  // CHECK: %[[V8:.*]] = zap.buffer.bitcast %[[V0]] : !zap.buffer<42, f32> to !zap.buffer<size, dt>
  %8 = zap.buffer.bitcast %0 : !zap.buffer<42, f32> to !zap.buffer<size, dt>

  // CHECK: %[[V9:.*]] = zap.buffer.bitcast %[[V8]] : !zap.buffer<size, dt> to !zap.buffer<size2, dt>
  %9 = zap.buffer.bitcast %8 : !zap.buffer<size, dt> to !zap.buffer<size2, dt>

  // Reinterpretations of the contains of the buffer are ok.
  // CHECK: %[[V10:.*]] = zap.buffer.bitcast %[[V4]] : !zap.buffer<42, f32> to !zap.buffer<1, f64>
  %10 = zap.buffer.bitcast %4 : !zap.buffer<42, f32> to !zap.buffer<1, f64>

  kgen.return
}

// CHECK-LABEL: @zap_buffer_stack_allocation
kgen.generator @zap_buffer_stack_allocation<type: dtype, size>() {
  // CHECK: zap.buffer.stack_allocation : !zap.buffer<4, f32>
  %0 = zap.buffer.stack_allocation : !zap.buffer<4, f32>
  // CHECK: zap.buffer.stack_allocation : !zap.buffer<size, f32>
  %1 = zap.buffer.stack_allocation : !zap.buffer<size, f32>
  // CHECK: zap.buffer.stack_allocation : !zap.buffer<size, type>
  %2 = zap.buffer.stack_allocation : !zap.buffer<size, type>
  kgen.return
}

// CHECK-LABEL: @zap_buffer_load
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @zap_buffer_load<size, type: dtype>(
    %a: !zap.buffer<size, type>,
    %b: !zap.buffer<size, f32>,
    %c: !zap.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: %[[U:.*]] = zap.buffer.load %[[A]][%[[IDX]]] : !zap.buffer<size, type>
  %u = zap.buffer.load %a[%idx] : !zap.buffer<size, type>
  // CHECK: %[[V:.*]] = zap.buffer.load %[[B]][%[[IDX]]] : !zap.buffer<size, f32>
  %v = zap.buffer.load %b[%idx] : !zap.buffer<size, f32>
  // CHECK: %[[W:.*]] = zap.buffer.load %[[C]][%[[IDX]]] : !zap.buffer<4, si32>
  %w = zap.buffer.load %c[%idx] : !zap.buffer<4, si32>
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
    %v0: !pop.scalar<type>,
    %v1: !pop.scalar<f32>,
    %v2: !pop.scalar<si32>,
    %a: !zap.buffer<size, type>,
    %b: !zap.buffer<size, f32>,
    %c: !zap.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: zap.buffer.store %[[V0]], %[[A]][%[[IDX]]] : !zap.buffer<size, type>
  zap.buffer.store %v0, %a[%idx] : !zap.buffer<size, type>
  // CHECK: zap.buffer.store %[[V1]], %[[B]][%[[IDX]]] : !zap.buffer<size, f32>
  zap.buffer.store %v1, %b[%idx] : !zap.buffer<size, f32>
  // CHECK: zap.buffer.store %[[V2]], %[[C]][%[[IDX]]] : !zap.buffer<4, si32>
  zap.buffer.store %v2, %c[%idx] : !zap.buffer<4, si32>
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
    %a: !zap.buffer<size, type>,
    %b: !zap.buffer<size, f32>,
    %c: !zap.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 0
  // CHECK: %[[U:.*]] = zap.simd.load %[[A]][%[[IDX]]] : !zap.buffer<size, type>, !meta.simd<4, type>
  %u = zap.simd.load %a[%idx] : !zap.buffer<size, type>, !meta.simd<4, type>
  // CHECK: %[[V:.*]] = zap.simd.load %[[B]][%[[IDX]]] : !zap.buffer<size, f32>, !meta.simd<4, f32>
  %v = zap.simd.load %b[%idx] : !zap.buffer<size, f32>, !meta.simd<4, f32>
  // CHECK: %[[W:.*]] = zap.simd.load %[[C]][%[[IDX]]] : !zap.buffer<4, si32>, !meta.simd<4, si32>
  %w = zap.simd.load %c[%idx] : !zap.buffer<4, si32>, !meta.simd<4, si32>
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
    %a: !zap.buffer<size, type>,
    %b: !zap.buffer<size, f32>,
    %c: !zap.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 0
  // CHECK: zap.simd.store %[[V0]], %[[A]][%[[IDX]]] : !meta.simd<size, type>, !zap.buffer<size, type>
  zap.simd.store %v0, %a[%idx] : !meta.simd<size, type>, !zap.buffer<size, type>
  // CHECK: zap.simd.store %[[V1]], %[[B]][%[[IDX]]] : !meta.simd<8, f32>, !zap.buffer<size, f32>
  zap.simd.store %v1, %b[%idx] : !meta.simd<8, f32>, !zap.buffer<size, f32>
  // CHECK: zap.simd.store %[[V2]], %[[C]][%[[IDX]]] : !meta.simd<2, si32>, !zap.buffer<4, si32>
  zap.simd.store %v2, %c[%idx] : !meta.simd<2, si32>, !zap.buffer<4, si32>
  kgen.return
}

// CHECK-LABEL: @zap_print
kgen.generator @zap_print(%a: !pop.scalar<f32>) {
  // CHECK: zap.print "foo %f"(%{{.*}}) : !pop.scalar<f32>
  zap.print "foo %f"(%a) : !pop.scalar<f32>
  kgen.return
}
