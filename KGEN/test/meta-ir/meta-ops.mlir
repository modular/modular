// RUN: kgen-opt %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @meta_buffer_size
// CHECK-SAME: %[[ARG0:.*]]: !meta.buffer<{{.*}}>, %[[ARG1:.*]]:
kgen.func @meta_buffer_size(%arg0: !meta.buffer<42, f32>, %arg1: !meta.buffer<?, f32>) {
  // CHECK: meta.buffer.size %[[ARG0]]
  %0 = meta.buffer.size %arg0 : !meta.buffer<42, f32>
  // CHECK: meta.buffer.size %[[ARG1]]
  %1 = meta.buffer.size %arg1 : !meta.buffer<?, f32>
  kgen.return
}

// CHECK-LABEL: @meta_buffer_dtype
// CHECK-SAME: %[[ARG0:.*]]: !meta.buffer<{{.*}}>, %[[ARG1:.*]]:
kgen.func @meta_buffer_dtype(%arg0: !meta.buffer<42, f32>, %arg1: !meta.buffer<42, ?>) {
  // CHECK: meta.buffer.dtype %[[ARG0]]
  %0 = meta.buffer.dtype %arg0 : !meta.buffer<42, f32>
  // CHECK: meta.buffer.dtype %[[ARG1]]
  %1 = meta.buffer.dtype %arg1 : !meta.buffer<42, ?>
  kgen.return
}

// CHECK-LABEL: @pointer_types
kgen.generator @pointer_types<dt: dtype>(
  // CHECK-SAME: %{{.*}}: !meta.pointer<!meta.scalar<dt>>, %{{.*}}: !meta.pointer<!meta.scalar<f32>>, %{{.*}}: !meta.pointer<?>
  %arg0: !meta.pointer<!meta.scalar<dt>>, %arg1: !meta.pointer<!meta.scalar<f32>>, %arg2: !meta.pointer<?>) {
  kgen.return
}

// CHECK-LABEL: @meta_buffer_address
kgen.generator @meta_buffer_address<dt: dtype, size>(
  // CHECK-SAME: %[[ARG0:.*]]: !meta.buffer<size
  // CHECK-SAME: %[[ARG1:.*]]: !meta.buffer<?
  // CHECK-SAME: %[[ARG2:.*]]: !meta.buffer<3
   %arg0: !meta.buffer<size, dt>, %arg1: !meta.buffer<?, ?>, %arg2: !meta.buffer<3, f32>) {
  // CHECK: meta.buffer.address %[[ARG0]]
  %0 = meta.buffer.address %arg0 : !meta.buffer<size, dt>
  // CHECK: meta.buffer.address %[[ARG1]]
  %1 = meta.buffer.address %arg1 : !meta.buffer<?, ?>
  // CHECK: meta.buffer.address %[[ARG2]]
  %2 = meta.buffer.address %arg2 : !meta.buffer<3, f32>
  kgen.return
}

// CHECK-LABEL: @meta_buffer_convert
// CHECK-SAME: %[[ARG0:.*]]:
kgen.generator @meta_buffer_convert<size, size2, dt: dtype>(%arg0: !meta.buffer<?, ?>) {
  // CHECK: %[[V0:.*]] = meta.buffer.convert %[[ARG0]] : !meta.buffer<?, ?> to !meta.buffer<42, f32>
  %0 = meta.buffer.convert %arg0 : !meta.buffer<?, ?> to !meta.buffer<42, f32>
  // CHECK: %[[V1:.*]] = meta.buffer.convert %[[V0]] : !meta.buffer<42, f32> to !meta.buffer<?, ?>
  %1 = meta.buffer.convert %0 : !meta.buffer<42, f32> to !meta.buffer<?, ?>
  // CHECK: %[[V2:.*]] = meta.buffer.convert %[[ARG0]] : !meta.buffer<?, ?> to !meta.buffer<?, f32>
  %2 = meta.buffer.convert %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, f32>
  // CHECK: %[[V3:.*]] = meta.buffer.convert %[[ARG0]] : !meta.buffer<?, ?> to !meta.buffer<42, ?>
  %3 = meta.buffer.convert %arg0 : !meta.buffer<?, ?> to !meta.buffer<42, ?>
  // CHECK: %[[V4:.*]] = meta.buffer.convert %[[V2]] : !meta.buffer<?, f32> to !meta.buffer<42, f32>
  %4 = meta.buffer.convert %2 : !meta.buffer<?, f32> to !meta.buffer<42, f32>
  // CHECK: %[[V5:.*]] = meta.buffer.convert %[[V3]] : !meta.buffer<42, ?> to !meta.buffer<42, f32>
  %5 = meta.buffer.convert %3 : !meta.buffer<42, ?> to !meta.buffer<42, f32>
  // CHECK: %[[V6:.*]] = meta.buffer.convert %[[V0]] : !meta.buffer<42, f32> to !meta.buffer<42, f32>
  %6 = meta.buffer.convert %0 : !meta.buffer<42, f32> to !meta.buffer<42, f32>


  // Conversions between different unknown parameters are ok.
  // CHECK: %[[V7:.*]] = meta.buffer.convert %[[V0]] : !meta.buffer<42, f32> to !meta.buffer<size, f32>
  %7 = meta.buffer.convert %0 : !meta.buffer<42, f32> to !meta.buffer<size, f32>

  // CHECK: %[[V8:.*]] = meta.buffer.convert %[[V0]] : !meta.buffer<42, f32> to !meta.buffer<size, dt>
  %8 = meta.buffer.convert %0 : !meta.buffer<42, f32> to !meta.buffer<size, dt>

  // CHECK: %[[V9:.*]] = meta.buffer.convert %[[V8]] : !meta.buffer<size, dt> to !meta.buffer<size2, dt>
  %9 = meta.buffer.convert %8 : !meta.buffer<size, dt> to !meta.buffer<size2, dt>

  kgen.return
}

// CHECK-LABEL: @cast_to_builtin
// CHECK-SAME: %[[ARG0:.*]]: !meta.scalar<f32>, %[[ARG1:.*]]: !meta.scalar<si32>
kgen.func @cast_to_builtin(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<si32>) {
  // CHECK: meta.cast_to_builtin %[[ARG0]] : !meta.scalar<f32> to f32
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<f32> to f32
  // CHECK: meta.cast_to_builtin %[[ARG1]] : !meta.scalar<si32> to i32
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<si32> to i32
  kgen.return
}

// CHECK-LABEL: @cast_from_builtin
// CHECK-SAME: %[[ARG0:.*]]: f32, %[[ARG1:.*]]: ui32
kgen.func @cast_from_builtin(%arg0: f32, %arg1: ui32) {
  // CHECK: meta.cast_from_builtin %[[ARG0]] : f32 to !meta.scalar<f32>
  %0 = meta.cast_from_builtin %arg0: f32 to !meta.scalar<f32>
  // CHECK: meta.cast_from_builtin %[[ARG1]] : ui32 to !meta.scalar<ui32>
  %1 = meta.cast_from_builtin %arg1: ui32 to !meta.scalar<ui32>
  kgen.return
}

// CHECK-LABEL: @cast_from_builtin_vector
// CHECK-SAME: %[[ARG:.*]]:
kgen.func @cast_from_builtin_vector(%arg0: vector<1xf32>) -> !meta.simd<1, f32> {
  // CHECK: %[[V0:.*]] = meta.cast_from_builtin %[[ARG]] : vector<1xf32> to !meta.simd<1, f32>
  %0 = meta.cast_from_builtin %arg0 : vector<1xf32> to !meta.simd<1, f32>
  // CHECK: kgen.return  %[[V0:.*]] : !meta.simd<1, f32>
  kgen.return %0 : !meta.simd<1, f32>
}

// CHECK-LABEL: @meta_buffer_construct
kgen.func @meta_buffer_construct(%ptr: !meta.pointer<!meta.scalar<f32>>, %opaque: !meta.pointer<?>,
                              %size: index, %dtype: !kgen.dtype) {
  // CHECK: meta.buffer.construct %{{.*}} : !meta.buffer<4, f32>
  %0 = meta.buffer.construct %ptr : !meta.buffer<4, f32>
  // CHECK: meta.buffer.construct %{{.*}}[%{{.*}}] : !meta.buffer<?, f32>
  %1 = meta.buffer.construct %ptr[%size] : !meta.buffer<?, f32>
  // CHECK: meta.buffer.construct %{{.*}} of %{{.*}} : !meta.buffer<4, ?>
  %2 = meta.buffer.construct %opaque of %dtype : !meta.buffer<4, ?>
  // CHECK: meta.buffer.construct %{{.*}}[%{{.*}}] of %{{.*}} : !meta.buffer<?, ?>
  %3 = meta.buffer.construct %opaque[%size] of %dtype : !meta.buffer<?, ?>
  // CHECK: meta.buffer.construct %{{.*}}[%{{.*}}] of %{{.*}} : !meta.buffer<4, f32>
  %4 = meta.buffer.construct %ptr[%size] of %dtype : !meta.buffer<4, f32>
  kgen.return
}
