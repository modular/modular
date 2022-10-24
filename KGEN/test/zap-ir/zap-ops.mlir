// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: @zap_buffer_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer<simd<1, f32>>
// CHECK-SAME: %[[OPAQUE:.*]]: !pop.pointer<simd<1, invalid>>
// CHECK-SAME: %[[SIZE:.*]]: index
// CHECK-SAME: %[[DTYPE:.*]]: !kgen.dtype
kgen.func @zap_buffer_construct(
  %ptr: !pop.pointer<simd<1, f32>>,
  %opaque: !pop.pointer<simd<1, invalid>>,
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

// CHECK-LABEL: @zap_buffer_constant
kgen.generator @zap_buffer_constant<size, type: dtype>() {
  // CHECK: zap.buffer.constant(#M.dense_array<1.{{0+}}e+01, 1.2{{0+}}e+01, -2.{{0+}}e+00> : !M.array<3xf32>) : f32
  %0 = zap.buffer.constant(#M.dense_array<10.0, 12.0, -2.0> : !M.array<3xf32>) : f32
  // CHECK: zap.buffer.constant(#M.dense_array<2, 3> : !M.array<2xui8>) : type
  %1 = zap.buffer.constant(#M.dense_array<2, 3> : !M.array<2xui8>) : type
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
  // CHECK: %[[U:.*]] = zap.buffer.load %[[A]][%[[IDX]]] : !zap.buffer<size, type>, !pop.simd<4, type>
  %u = zap.buffer.load %a[%idx] : !zap.buffer<size, type>, !pop.simd<4, type>
  // CHECK: %[[V:.*]] = zap.buffer.load %[[B]][%[[IDX]]] : !zap.buffer<size, f32>, !pop.simd<4, f32>
  %v = zap.buffer.load %b[%idx] : !zap.buffer<size, f32>, !pop.simd<4, f32>
  // CHECK: %[[W:.*]] = zap.buffer.load %[[C]][%[[IDX]]] : !zap.buffer<4, si32>, !pop.simd<4, si32>
  %w = zap.buffer.load %c[%idx] : !zap.buffer<4, si32>, !pop.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @zap_buffer_aligned_load
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @zap_buffer_aligned_load<size, type: dtype>(
    %a: !zap.buffer<size, type>,
    %b: !zap.buffer<size, f32>,
    %c: !zap.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 0
  // CHECK: %[[U:.*]] = zap.buffer.load %[[A]][%[[IDX]]] align 8 : !zap.buffer<size, type>, !pop.simd<4, type>
  %u = zap.buffer.load %a[%idx] align 8 : !zap.buffer<size, type>, !pop.simd<4, type>
  // CHECK: %[[V:.*]] = zap.buffer.load %[[B]][%[[IDX]]] align 4 : !zap.buffer<size, f32>, !pop.simd<4, f32>
  %v = zap.buffer.load %b[%idx] align get_alignof(f32) : !zap.buffer<size, f32>, !pop.simd<4, f32>
  // CHECK: %[[W:.*]] = zap.buffer.load %[[C]][%[[IDX]]] align 8 : !zap.buffer<4, si32>, !pop.simd<4, si32>
  %w = zap.buffer.load %c[%idx] align get_alignof(f64) : !zap.buffer<4, si32>, !pop.simd<4, si32>
  // CHECK: %[[W:.*]] = zap.buffer.load %[[C]][%[[IDX]]] align size : !zap.buffer<4, si32>, !pop.simd<4, si32>
  %x = zap.buffer.load %c[%idx] align size : !zap.buffer<4, si32>, !pop.simd<4, si32>
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
    %v0: !pop.simd<size, type>,
    %v1: !pop.simd<8, f32>,
    %v2: !pop.simd<2, si32>,
    %a: !zap.buffer<size, type>,
    %b: !zap.buffer<size, f32>,
    %c: !zap.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 0
  // CHECK: zap.buffer.store %[[V0]], %[[A]][%[[IDX]]] : !pop.simd<size, type>, !zap.buffer<size, type>
  zap.buffer.store %v0, %a[%idx] : !pop.simd<size, type>, !zap.buffer<size, type>
  // CHECK: zap.buffer.store %[[V1]], %[[B]][%[[IDX]]] : !pop.simd<8, f32>, !zap.buffer<size, f32>
  zap.buffer.store %v1, %b[%idx] : !pop.simd<8, f32>, !zap.buffer<size, f32>
  // CHECK: zap.buffer.store %[[V2]], %[[C]][%[[IDX]]] : !pop.simd<2, si32>, !zap.buffer<4, si32>
  zap.buffer.store %v2, %c[%idx] : !pop.simd<2, si32>, !zap.buffer<4, si32>
  kgen.return
}

// CHECK-LABEL: @zap_buffer_aligned_store
// CHECK-SAME: %[[V0:[a-z0-9]+]]:
// CHECK-SAME: %[[V1:[a-z0-9]+]]:
// CHECK-SAME: %[[V2:[a-z0-9]+]]:
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @zap_buffer_aligned_store<size, type: dtype>(
    %v0: !pop.simd<size, type>,
    %v1: !pop.simd<8, f32>,
    %v2: !pop.simd<2, si32>,
    %a: !zap.buffer<size, type>,
    %b: !zap.buffer<size, f32>,
    %c: !zap.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 0
  // CHECK: zap.buffer.store %[[V0]], %[[A]][%[[IDX]]] align size : !pop.simd<size, type>, !zap.buffer<size, type>
  zap.buffer.store %v0, %a[%idx] align size : !pop.simd<size, type>, !zap.buffer<size, type>
  // CHECK: zap.buffer.store %[[V1]], %[[B]][%[[IDX]]] align 1 : !pop.simd<8, f32>, !zap.buffer<size, f32>
  zap.buffer.store %v1, %b[%idx] align 1 : !pop.simd<8, f32>, !zap.buffer<size, f32>
  // CHECK: zap.buffer.store %[[V2]], %[[C]][%[[IDX]]] align 8 : !pop.simd<2, si32>, !zap.buffer<4, si32>
  zap.buffer.store %v2, %c[%idx] align 8 : !pop.simd<2, si32>, !zap.buffer<4, si32>
  kgen.return
}

// CHECK-LABEL: @zap_ndbuffer
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[?, 4], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[4, 5], ?>
// CHECK-SAME: %[[NDBUFFER3:.*]]: !zap.ndbuffer<[?, ?, ?], ?>
kgen.func @zap_ndbuffer(
  %arg0 : !zap.ndbuffer<[4, 5, 3], f32>,
  %arg1 : !zap.ndbuffer<[?, 4], f32>,
  %arg2: !zap.ndbuffer<[4, 5], ?>,
  %arg3 : !zap.ndbuffer<[?, ?, ?], ?>) -> (!zap.ndbuffer<[4, 5, 3], f32>,
                                         !zap.ndbuffer<[?, 4], f32>,
                                         !zap.ndbuffer<[4, 5], ?>,
                                         !zap.ndbuffer<[?, ?, ?], ?>) {
  kgen.return %arg0, %arg1, %arg2, %arg3 : !zap.ndbuffer<[4, 5, 3], f32>,
                                           !zap.ndbuffer<[?, 4], f32>,
                                           !zap.ndbuffer<[4, 5], ?>,
                                           !zap.ndbuffer<[?, ?, ?], ?>
}

// CHECK-LABEL: @zap_ndbuffer_with_params
// CHECK-SAME: !zap.ndbuffer<[size, 5, 3], type>
// CHECK-SAME: !zap.ndbuffer<[size, 5, 3], f32>
// CHECK-SAME: !zap.ndbuffer<[size, size], f32>
// CHECK-SAME: !zap.ndbuffer<[?, 4, size], f32>
kgen.generator @zap_ndbuffer_with_params<type:dtype, size>(
    %arg0 : !zap.ndbuffer<[size, 5, 3], type>,
    %arg1 : !zap.ndbuffer<[size, 5, 3], f32>,
    %arg2 : !zap.ndbuffer<[size, size], f32>,
    %arg3 : !zap.ndbuffer<[?, 4, size], f32>
) -> (!zap.ndbuffer<[size, 5, 3], type>,
      !zap.ndbuffer<[size, 5, 3], f32>,
      !zap.ndbuffer<[size, size], f32>,
      !zap.ndbuffer<[?, 4, size], f32>) {
  kgen.return %arg0, %arg1, %arg2, %arg3 :
    !zap.ndbuffer<[size, 5, 3], type>,
    !zap.ndbuffer<[size, 5, 3], f32>,
    !zap.ndbuffer<[size, size], f32>,
    !zap.ndbuffer<[?, 4, size], f32>
}

// CHECK-LABEL: @zap_ndbuffer_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer<simd<1, f32>>
// CHECK-SAME: %[[OPAQUE:.*]]: !pop.pointer<simd<1, invalid>>
// CHECK-SAME: %[[SIZE:.*]]: index
// CHECK-SAME: %[[DTYPE:.*]]: !kgen.dtype
kgen.func @zap_ndbuffer_construct(
  %ptr: !pop.pointer<simd<1, f32>>,
  %opaque: !pop.pointer<simd<1, invalid>>,
  %size: index,
  %dtype: !kgen.dtype) -> (!zap.ndbuffer<[4, 5, 3], f32>,
                           !zap.ndbuffer<[?, 4], f32>,
                           !zap.ndbuffer<[4, 5], ?>,
                           !zap.ndbuffer<[?, ?, ?], ?>) {
  // CHECK: zap.ndbuffer.construct %[[PTR]] : !zap.ndbuffer<[4, 5, 3], f32>
  %0 = zap.ndbuffer.construct %ptr : !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.construct %[[PTR]][%[[SIZE]]] : !zap.ndbuffer<[?, 4], f32>
  %1 = zap.ndbuffer.construct %ptr[%size] : !zap.ndbuffer<[?, 4], f32>
  // CHECK: zap.ndbuffer.construct %[[OPAQUE]] of %[[DTYPE]] : !zap.ndbuffer<[4, 5], ?>
  %2 = zap.ndbuffer.construct %opaque of %dtype : !zap.ndbuffer<[4, 5], ?>
  // CHECK: zap.ndbuffer.construct %[[OPAQUE]][%[[SIZE]], %[[SIZE]], %[[SIZE]]] of %[[DTYPE]] : !zap.ndbuffer<[?, ?, ?], ?>
  %3 = zap.ndbuffer.construct %opaque[%size, %size, %size] of %dtype : !zap.ndbuffer<[?, ?, ?], ?>
  kgen.return %0, %1, %2, %3 : !zap.ndbuffer<[4, 5, 3], f32>,
                               !zap.ndbuffer<[?, 4], f32>,
                               !zap.ndbuffer<[4, 5], ?>,
                               !zap.ndbuffer<[?, ?, ?], ?>
}


// CHECK-LABEL: @zap_ndbuffer_stack_allocation
kgen.generator @zap_ndbuffer_stack_allocation<type: dtype, size>() {
  // CHECK: zap.ndbuffer.stack_allocation : !zap.ndbuffer<[4], f32>
  %0 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[4], f32>
  // CHECK: zap.ndbuffer.stack_allocation : !zap.ndbuffer<[42, size], f32>
  %1 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[42, size], f32>
  // CHECK: zap.ndbuffer.stack_allocation : !zap.ndbuffer<[3, 1, size, 42], type>
  %2 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[3, 1, size, 42], type>
  kgen.return
}

// CHECK-LABEL: @zap_ndbuffer_dim
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[?, 4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?, ?, ?], f32>
kgen.func @zap_ndbuffer_dim(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[?, 4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?, ?, ?], f32>) {
  // CHECK: zap.ndbuffer.dim %[[NDBUFFER0]][0] : !zap.ndbuffer<[4, 5, 3], f32>
  %0 = zap.ndbuffer.dim %ndbuffer0[0] : !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.dim %[[NDBUFFER1]][0] : !zap.ndbuffer<[?, 4, ?], f32>
  %1 = zap.ndbuffer.dim %ndbuffer1[0] : !zap.ndbuffer<[?, 4, ?], f32>
  // CHECK: zap.ndbuffer.dim %[[NDBUFFER1]][1] : !zap.ndbuffer<[?, 4, ?], f32>
  %2 = zap.ndbuffer.dim %ndbuffer1[1] : !zap.ndbuffer<[?, 4, ?], f32>
  // CHECK: zap.ndbuffer.dim %[[NDBUFFER2]][0] : !zap.ndbuffer<[?, ?, ?], f32>
  %3 = zap.ndbuffer.dim %ndbuffer2[0] : !zap.ndbuffer<[?, ?, ?], f32>
  // CHECK: zap.ndbuffer.dim %[[NDBUFFER2]][0] : !zap.ndbuffer<[?, ?, ?], f32>
  %4 = zap.ndbuffer.dim %ndbuffer2[0] : !zap.ndbuffer<[?, ?, ?], f32>
  kgen.return
}

// CHECK-LABEL: @zap_ndbuffer_dtype
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?, ?, ?, ?], f32>
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @zap_ndbuffer_dtype(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?, ?, ?, ?], f32>,
  %idx: index) {
  // CHECK: zap.ndbuffer.dtype %[[NDBUFFER0]] : !zap.ndbuffer<[4, 5, 3], f32>
  %0 = zap.ndbuffer.dtype %ndbuffer0 : !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.dtype %[[NDBUFFER1]] : !zap.ndbuffer<[4, ?], f32>
  %1 = zap.ndbuffer.dtype %ndbuffer1 : !zap.ndbuffer<[4, ?], f32>
  // CHECK: zap.ndbuffer.dtype %[[NDBUFFER2]] : !zap.ndbuffer<[?, ?, ?, ?], f32>
  %2 = zap.ndbuffer.dtype %ndbuffer2 : !zap.ndbuffer<[?, ?, ?, ?], f32>
  kgen.return
}

// CHECK-LABEL: @zap_ndbuffer_rank
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?, ?, ?, ?], f32>
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @zap_ndbuffer_rank(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?, ?, ?, ?], f32>,
  %idx: index) {
  // CHECK: zap.ndbuffer.rank %[[NDBUFFER0]] : !zap.ndbuffer<[4, 5, 3], f32>
  zap.ndbuffer.rank %ndbuffer0 : !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.rank %[[NDBUFFER1]] : !zap.ndbuffer<[4, ?], f32>
  zap.ndbuffer.rank %ndbuffer1 : !zap.ndbuffer<[4, ?], f32>
  // CHECK: zap.ndbuffer.rank %[[NDBUFFER2]] : !zap.ndbuffer<[?, ?, ?, ?], f32>
  zap.ndbuffer.rank %ndbuffer2 : !zap.ndbuffer<[?, ?, ?, ?], f32>
  kgen.return
}

// CHECK-LABEL: @zap_ndbuffer_address
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[?, 4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?, ?, ?], f32>
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @zap_ndbuffer_address(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[?, 4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?, ?, ?], f32>,
  %idx: index) {
  // CHECK: zap.ndbuffer.address %[[NDBUFFER0]] : !zap.ndbuffer<[4, 5, 3], f32>
  zap.ndbuffer.address %ndbuffer0 : !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.address %[[NDBUFFER1]] : !zap.ndbuffer<[?, 4, ?], f32>
  zap.ndbuffer.address %ndbuffer1 : !zap.ndbuffer<[?, 4, ?], f32>
  // CHECK: zap.ndbuffer.address %[[NDBUFFER2]] : !zap.ndbuffer<[?, ?, ?], f32>
  zap.ndbuffer.address %ndbuffer2 : !zap.ndbuffer<[?, ?, ?], f32>
  kgen.return
}

// CHECK-LABEL: @zap_ndbuffer_load
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?], f32>
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @zap_ndbuffer_load(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?], f32>,
  %idx: index) {
  // CHECK: %[[IDXZERO:.*]] =  index.constant
  %idxZero = index.constant 0
  // CHECK: %[[IDXONE:.*]] =  index.constant
  %idxOne = index.constant 1
  // CHECK: zap.ndbuffer.load %[[NDBUFFER0]][%[[IDX]], %[[IDXZERO]], %[[IDXONE]]] : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<1, f32>
  %0 = zap.ndbuffer.load %ndbuffer0[%idx, %idxZero, %idxOne] : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<1, f32>
  // CHECK: zap.ndbuffer.load %[[NDBUFFER0]][%[[IDX]], %[[IDX]], %[[IDX]]] : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<1, f32>
  %1 = zap.ndbuffer.load %ndbuffer0[%idx, %idx, %idx] : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<1, f32>
  // CHECK: zap.ndbuffer.load %[[NDBUFFER1]][%[[IDX]], %[[IDX]]] : !zap.ndbuffer<[4, ?], f32>, !pop.simd<1, f32>
  %2 = zap.ndbuffer.load %ndbuffer1[%idx, %idx] : !zap.ndbuffer<[4, ?], f32>, !pop.simd<1, f32>
  // CHECK: zap.ndbuffer.load %[[NDBUFFER2]][%[[IDX]]] : !zap.ndbuffer<[?], f32>, !pop.simd<1, f32>
  %3 = zap.ndbuffer.load %ndbuffer2[%idx] : !zap.ndbuffer<[?], f32>, !pop.simd<1, f32>
  kgen.return
}

// CHECK-LABEL: @zap.ndbuffer.load
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?], f32>
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @zap.ndbuffer.load(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?], f32>,
  %idx: index) {
  // CHECK: %[[IDXZERO:.*]] =  index.constant
  %idxZero = index.constant 0
  // CHECK: %[[IDXONE:.*]] =  index.constant
  %idxOne = index.constant 1
  // CHECK: zap.ndbuffer.load %[[NDBUFFER0]][%[[IDX]], %[[IDXZERO]], %[[IDXONE]]] : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<4, f32>
  %0 = zap.ndbuffer.load %ndbuffer0[%idx, %idxZero, %idxOne] : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<4, f32>
  // CHECK: zap.ndbuffer.load %[[NDBUFFER0]][%[[IDX]], %[[IDX]], %[[IDX]]] : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<3, f32>
  %1 = zap.ndbuffer.load %ndbuffer0[%idx, %idx, %idx] : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<3, f32>
  // CHECK: zap.ndbuffer.load %[[NDBUFFER1]][%[[IDX]], %[[IDX]]] : !zap.ndbuffer<[4, ?], f32>, !pop.simd<3, f32>
  %2 = zap.ndbuffer.load %ndbuffer1[%idx, %idx] : !zap.ndbuffer<[4, ?], f32>, !pop.simd<3, f32>
  // CHECK: zap.ndbuffer.load %[[NDBUFFER2]][%[[IDX]]] : !zap.ndbuffer<[?], f32>, !pop.simd<3, f32>
  %3 = zap.ndbuffer.load %ndbuffer2[%idx] : !zap.ndbuffer<[?], f32>, !pop.simd<3, f32>
  kgen.return
}

// CHECK-LABEL: @zap_ndbuffer_load_aligned
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?], f32>
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @zap_ndbuffer_load_aligned(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?], f32>,
  %idx: index) {
  // CHECK: %[[IDXZERO:.*]] =  index.constant
  %idxZero = index.constant 0
  // CHECK: %[[IDXONE:.*]] =  index.constant
  %idxOne = index.constant 1
  // CHECK: zap.ndbuffer.load %[[NDBUFFER0]][%[[IDX]], %[[IDXZERO]], %[[IDXONE]]] align 1 : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<4, f32>
  %0 = zap.ndbuffer.load %ndbuffer0[%idx, %idxZero, %idxOne] align 1 : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<4, f32>
  // CHECK: zap.ndbuffer.load %[[NDBUFFER0]][%[[IDX]], %[[IDX]], %[[IDX]]] align 8 : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<3, f32>
  %1 = zap.ndbuffer.load %ndbuffer0[%idx, %idx, %idx] align 8 : !zap.ndbuffer<[4, 5, 3], f32>, !pop.simd<3, f32>
  // CHECK: zap.ndbuffer.load %[[NDBUFFER1]][%[[IDX]], %[[IDX]]] align 4 : !zap.ndbuffer<[4, ?], f32>, !pop.simd<3, f32>
  %2 = zap.ndbuffer.load %ndbuffer1[%idx, %idx] align get_alignof(f32) : !zap.ndbuffer<[4, ?], f32>, !pop.simd<3, f32>
  // CHECK: zap.ndbuffer.load %[[NDBUFFER2]][%[[IDX]]] align 16 : !zap.ndbuffer<[?], f32>, !pop.simd<3, f32>
  %3 = zap.ndbuffer.load %ndbuffer2[%idx] align 16 : !zap.ndbuffer<[?], f32>, !pop.simd<3, f32>
  kgen.return
}


// CHECK-LABEL: @zap_ndbuffer_store
// CHECK-SAME: %[[VAL:.*]]: !pop.simd<1, f32>
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?], f32>
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @zap_ndbuffer_store(
  %val : !pop.simd<1, f32>,
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?], f32>,
  %idx: index) {
  // CHECK: %[[IDXZERO:.*]] =  index.constant
  %idxZero = index.constant 0
  // CHECK: %[[IDXONE:.*]] =  index.constant
  %idxOne = index.constant 1
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER0]][%[[IDX]], %[[IDXZERO]], %[[IDXONE]]] : !pop.simd<1, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  zap.ndbuffer.store %val, %ndbuffer0[%idx, %idxZero, %idxOne] : !pop.simd<1, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER0]][%[[IDX]], %[[IDX]], %[[IDX]]] : !pop.simd<1, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  zap.ndbuffer.store %val, %ndbuffer0[%idx, %idx, %idx] : !pop.simd<1, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER1]][%[[IDX]], %[[IDX]]] : !pop.simd<1, f32>, !zap.ndbuffer<[4, ?], f32>
  zap.ndbuffer.store %val, %ndbuffer1[%idx, %idx] : !pop.simd<1, f32>, !zap.ndbuffer<[4, ?], f32>
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER2]][%[[IDX]]] : !pop.simd<1, f32>, !zap.ndbuffer<[?], f32>
  zap.ndbuffer.store %val, %ndbuffer2[%idx] : !pop.simd<1, f32>, !zap.ndbuffer<[?], f32>
  kgen.return
}

// CHECK-LABEL: @zap.ndbuffer.store
// CHECK-SAME: %[[VAL:.*]]: !pop.simd<4, f32>
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?], f32>
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @zap.ndbuffer.store(
  %val : !pop.simd<4, f32>,
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?], f32>,
  %idx: index) {
  // CHECK: %[[IDXZERO:.*]] =  index.constant
  %idxZero = index.constant 0
  // CHECK: %[[IDXONE:.*]] =  index.constant
  %idxOne = index.constant 1
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER0]][%[[IDX]], %[[IDXZERO]], %[[IDXONE]]] : !pop.simd<4, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  zap.ndbuffer.store %val, %ndbuffer0[%idx, %idxZero, %idxOne] : !pop.simd<4, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER0]][%[[IDX]], %[[IDX]], %[[IDX]]] : !pop.simd<4, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  zap.ndbuffer.store %val, %ndbuffer0[%idx, %idx, %idx] : !pop.simd<4, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER1]][%[[IDX]], %[[IDX]]] : !pop.simd<4, f32>, !zap.ndbuffer<[4, ?], f32>
  zap.ndbuffer.store %val, %ndbuffer1[%idx, %idx] : !pop.simd<4, f32>, !zap.ndbuffer<[4, ?], f32>
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER2]][%[[IDX]]] : !pop.simd<4, f32>, !zap.ndbuffer<[?], f32>
  zap.ndbuffer.store %val, %ndbuffer2[%idx] : !pop.simd<4, f32>, !zap.ndbuffer<[?], f32>
  kgen.return
}

// CHECK-LABEL: @zap_ndbuffer_store_aligned
// CHECK-SAME: %[[VAL:.*]]: !pop.simd<4, f32>
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?], f32>
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @zap_ndbuffer_store_aligned(
  %val : !pop.simd<4, f32>,
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?], f32>,
  %idx: index) {
  // CHECK: %[[IDXZERO:.*]] =  index.constant
  %idxZero = index.constant 0
  // CHECK: %[[IDXONE:.*]] =  index.constant
  %idxOne = index.constant 1
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER0]][%[[IDX]], %[[IDXZERO]], %[[IDXONE]]] align 1 : !pop.simd<4, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  zap.ndbuffer.store %val, %ndbuffer0[%idx, %idxZero, %idxOne] align 1 : !pop.simd<4, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER0]][%[[IDX]], %[[IDX]], %[[IDX]]] align 8 : !pop.simd<4, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  zap.ndbuffer.store %val, %ndbuffer0[%idx, %idx, %idx] align 8 : !pop.simd<4, f32>, !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER1]][%[[IDX]], %[[IDX]]] align 4 : !pop.simd<4, f32>, !zap.ndbuffer<[4, ?], f32>
  zap.ndbuffer.store %val, %ndbuffer1[%idx, %idx] align get_alignof(f32) : !pop.simd<4, f32>, !zap.ndbuffer<[4, ?], f32>
  // CHECK: zap.ndbuffer.store %[[VAL]], %[[NDBUFFER2]][%[[IDX]]] align 8 : !pop.simd<4, f32>, !zap.ndbuffer<[?], f32>
  zap.ndbuffer.store %val, %ndbuffer2[%idx] align get_alignof(f64) : !pop.simd<4, f32>, !zap.ndbuffer<[?], f32>
  kgen.return
}

// CHECK-LABEL: @zap_ndbuffer_size
// CHECK-SAME: %[[NDBUFFER0:.*]]: !zap.ndbuffer<[4, 5, 3], f32>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !zap.ndbuffer<[4, ?], f32>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !zap.ndbuffer<[?, ?, ?, ?], f32>
// CHECK-SAME: %[[IDX:.*]]: index
kgen.func @zap_ndbuffer_size(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[4, ?], f32>,
  %ndbuffer2: !zap.ndbuffer<[?, ?, ?, ?], f32>,
  %idx: index) {
  // CHECK: zap.ndbuffer.size %[[NDBUFFER0]] : !zap.ndbuffer<[4, 5, 3], f32>
  zap.ndbuffer.size %ndbuffer0 : !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: zap.ndbuffer.size %[[NDBUFFER1]] : !zap.ndbuffer<[4, ?], f32>
  zap.ndbuffer.size %ndbuffer1 : !zap.ndbuffer<[4, ?], f32>
  // CHECK: zap.ndbuffer.size %[[NDBUFFER2]] : !zap.ndbuffer<[?, ?, ?, ?], f32>
  zap.ndbuffer.size %ndbuffer2 : !zap.ndbuffer<[?, ?, ?, ?], f32>
  kgen.return
}


// CHECK-LABEL: @zap_ndbuffer_bitcast
// CHECK-SAME: %[[ARG0:.*]]:
kgen.generator @zap_ndbuffer_bitcast<size, size2, dt: dtype>(%arg0: !zap.ndbuffer<[?, ?], f32>) {
  // CHECK: %[[V0:.*]] = zap.ndbuffer.bitcast %[[ARG0]] : !zap.ndbuffer<[?, ?], f32> to !zap.ndbuffer<[?, 42], f32>
  %0 = zap.ndbuffer.bitcast %arg0 : !zap.ndbuffer<[?, ?], f32> to !zap.ndbuffer<[?, 42], f32>
  // CHECK: %[[V1:.*]] = zap.ndbuffer.bitcast %[[V0]] : !zap.ndbuffer<[?, 42], f32> to !zap.ndbuffer<[?, ?], si64>
  %1 = zap.ndbuffer.bitcast %0 : !zap.ndbuffer<[?, 42], f32> to !zap.ndbuffer<[?, ?], si64>
  // CHECK: %[[V2:.*]] = zap.ndbuffer.bitcast %[[ARG0]] : !zap.ndbuffer<[?, ?], f32> to !zap.ndbuffer<[42, ?], f32>
  %2 = zap.ndbuffer.bitcast %arg0 : !zap.ndbuffer<[?, ?], f32> to !zap.ndbuffer<[42, ?], f32>
  // CHECK: %[[V3:.*]] = zap.ndbuffer.bitcast %[[ARG0]] : !zap.ndbuffer<[?, ?], f32> to !zap.ndbuffer<[42, 42], si64>
  %3 = zap.ndbuffer.bitcast %arg0 : !zap.ndbuffer<[?, ?], f32> to !zap.ndbuffer<[42, 42], si64>
  // CHECK: %[[V4:.*]] = zap.ndbuffer.bitcast %[[V2]] : !zap.ndbuffer<[42, ?], f32> to !zap.ndbuffer<[10, 42], f32>
  %4 = zap.ndbuffer.bitcast %2 : !zap.ndbuffer<[42, ?], f32> to !zap.ndbuffer<[10, 42], f32>
  // CHECK: %[[V5:.*]] = zap.ndbuffer.bitcast %[[V3]] : !zap.ndbuffer<[42, 42], si64> to !zap.ndbuffer<[42, 42], f32>
  %5 = zap.ndbuffer.bitcast %3 : !zap.ndbuffer<[42, 42], si64> to !zap.ndbuffer<[42, 42], f32>
  // CHECK: %[[V6:.*]] = zap.ndbuffer.bitcast %[[V0]] : !zap.ndbuffer<[?, 42], f32> to !zap.ndbuffer<[42, 42], f32>
  %6 = zap.ndbuffer.bitcast %0 : !zap.ndbuffer<[?, 42], f32> to !zap.ndbuffer<[42, 42], f32>


  // Conversions between different unknown parameters are ok.
  // CHECK: %[[V7:.*]] = zap.ndbuffer.bitcast %[[V0]] : !zap.ndbuffer<[?, 42], f32> to !zap.ndbuffer<[size, ?], f32>
  %7 = zap.ndbuffer.bitcast %0 : !zap.ndbuffer<[?, 42], f32> to !zap.ndbuffer<[size, ?], f32>

  // CHECK: %[[V8:.*]] = zap.ndbuffer.bitcast %[[V0]] : !zap.ndbuffer<[?, 42], f32> to !zap.ndbuffer<[size, size], dt>
  %8 = zap.ndbuffer.bitcast %0 : !zap.ndbuffer<[?, 42], f32> to !zap.ndbuffer<[size, size], dt>

  // CHECK: %[[V9:.*]] = zap.ndbuffer.bitcast %[[V8]] : !zap.ndbuffer<[size, size], dt> to !zap.ndbuffer<[size2, size], dt>
  %9 = zap.ndbuffer.bitcast %8 : !zap.ndbuffer<[size, size], dt> to !zap.ndbuffer<[size2, size], dt>

  // Reinterpretations of the contains of the buffer are ok.
  // CHECK: %[[V10:.*]] = zap.ndbuffer.bitcast %[[V4]] : !zap.ndbuffer<[10, 42], f32> to !zap.ndbuffer<[1, 1], f64>
  %10 = zap.ndbuffer.bitcast %4 : !zap.ndbuffer<[10, 42], f32> to !zap.ndbuffer<[1, 1], f64>

  kgen.return
}

// CHECK-LABEL: @zap_print
kgen.generator @zap_print(%a: !pop.simd<1, f32>) {
  // CHECK: zap.print "foo %f"(%{{.*}}) : !pop.simd<1, f32>
  zap.print "foo %f"(%a) : !pop.simd<1, f32>
  kgen.return
}

// CHECK-LABEL: @zap_assert
// CHECK-SAME: %[[V0:[a-z0-9]+]]:
kgen.generator @zap_assert(%a: !pop.simd<1, bool>) {
  // CHECK: zap.debug_assert %[[V0]], "my message"
  zap.debug_assert %a, "my message" : !pop.simd<1, bool>
  kgen.return
}

// CHECK-LABEL: @global_string
kgen.generator @global_string() -> !pop.pointer<array<14, scalar<si8>>> {
  // CHECK: %{{.*}} = zap.global_string "hello world!!\00"[14]
  %0 = zap.global_string "hello world!!\00"[14]
  // CHECK: return %{{.*}} : !pop.pointer<array<14, scalar<si8>>>
  kgen.return %0 : !pop.pointer<array<14, scalar<si8>>>
}
