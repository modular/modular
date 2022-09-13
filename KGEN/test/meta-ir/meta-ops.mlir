// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.func @meta_buffer_size(%arg0: !meta.buffer<42, f32>, %arg1: !meta.buffer<?, f32>) -> index {
kgen.func @meta_buffer_size(%arg0: !meta.buffer<42, f32>, %arg1: !meta.buffer<?, f32>) -> index {
  // CHECK: %0 = meta.buffer.size %arg0 : !meta.buffer<42, f32>
  %0 = meta.buffer.size %arg0 : !meta.buffer<42, f32>
  // CHECK: %1 = meta.buffer.size %arg1 : !meta.buffer<?, f32>
  %1 = meta.buffer.size %arg1 : !meta.buffer<?, f32>
  // CHECK: kgen.return %1 : index
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @meta_buffer_dtype(%arg0: !meta.buffer<42, f32>, %arg1: !meta.buffer<42, ?>) -> !kgen.dtype {
kgen.func @meta_buffer_dtype(%arg0: !meta.buffer<42, f32>, %arg1: !meta.buffer<42, ?>) -> !kgen.dtype {
  // CHECK: %0 = meta.buffer.dtype %arg0 : !meta.buffer<42, f32>
  %0 = meta.buffer.dtype %arg0 : !meta.buffer<42, f32>
  // CHECK: %1 = meta.buffer.dtype %arg1 : !meta.buffer<42, ?>
  %1 = meta.buffer.dtype %arg1 : !meta.buffer<42, ?>
  // CHECK: kgen.return %1 : !kgen.dtype
  kgen.return %1 : !kgen.dtype
}

// CHECK-LABEL: kgen.generator @pointer_types<dt: dtype>(
kgen.generator @pointer_types<dt: dtype>(
  // CHECK: %arg0: !meta.pointer<!meta.scalar<dt>>, %arg1: !meta.pointer<!meta.scalar<f32>>, %arg2: !meta.pointer<?>) {
  %arg0: !meta.pointer<!meta.scalar<dt>>, %arg1: !meta.pointer<!meta.scalar<f32>>, %arg2: !meta.pointer<?>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @meta_buffer_address<dt: dtype, size>(
kgen.generator @meta_buffer_address<dt: dtype, size>(
   %arg0: !meta.buffer<size, dt>, %arg1: !meta.buffer<?, ?>, %arg2: !meta.buffer<3, f32>) -> (!meta.pointer<!meta.scalar<dt>>, !meta.pointer<?>, !meta.pointer<!meta.scalar<f32>>) {
  // CHECK: %0 = meta.buffer.address %arg0 : !meta.buffer<size, dt>
  %0 = meta.buffer.address %arg0 : !meta.buffer<size, dt>
  // CHECK: %1 = meta.buffer.address %arg1 : !meta.buffer<?, ?>
  %1 = meta.buffer.address %arg1 : !meta.buffer<?, ?>
  // CHECK: %2 = meta.buffer.address %arg2 : !meta.buffer<3, f32>
  %2 = meta.buffer.address %arg2 : !meta.buffer<3, f32>
  kgen.return %0, %1, %2 : !meta.pointer<!meta.scalar<dt>>, !meta.pointer<?>, !meta.pointer<!meta.scalar<f32>>
}

// CHECK-LABEL: kgen.generator @meta_buffer_rebind<size, size2, dt: dtype>(%arg0: !meta.buffer<?, ?>) -> !meta.buffer<42, f32> {
kgen.generator @meta_buffer_rebind<size, size2, dt: dtype>(%arg0: !meta.buffer<?, ?>) -> !meta.buffer<42, f32> {
  // CHECK: %0 = meta.buffer.rebind %arg0 : !meta.buffer<?, ?> to !meta.buffer<42, f32>
  %0 = meta.buffer.rebind %arg0 : !meta.buffer<?, ?> to !meta.buffer<42, f32>
  // CHECK: = meta.buffer.rebind %0 : !meta.buffer<42, f32> to !meta.buffer<?, ?>
  %1 = meta.buffer.rebind %0 : !meta.buffer<42, f32> to !meta.buffer<?, ?>
  // CHECK: = meta.buffer.rebind %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, f32>
  %2 = meta.buffer.rebind %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, f32>
  // CHECK: = meta.buffer.rebind %arg0 : !meta.buffer<?, ?> to !meta.buffer<42, ?>
  %3 = meta.buffer.rebind %arg0 : !meta.buffer<?, ?> to !meta.buffer<42, ?>
  // CHECK: = meta.buffer.rebind %2 : !meta.buffer<?, f32> to !meta.buffer<42, f32>
  %4 = meta.buffer.rebind %2 : !meta.buffer<?, f32> to !meta.buffer<42, f32>
  // CHECK: = meta.buffer.rebind %3 : !meta.buffer<42, ?> to !meta.buffer<42, f32>
  %5 = meta.buffer.rebind %3 : !meta.buffer<42, ?> to !meta.buffer<42, f32>
  // CHECK: = meta.buffer.rebind %0 : !meta.buffer<42, f32> to !meta.buffer<42, f32>
  %6 = meta.buffer.rebind %0 : !meta.buffer<42, f32> to !meta.buffer<42, f32>


  // Casts between different unknown parameters are ok.
  // CHECK: = meta.buffer.rebind %0 : !meta.buffer<42, f32> to !meta.buffer<size, f32>
  %7 = meta.buffer.rebind %0 : !meta.buffer<42, f32> to !meta.buffer<size, f32>

  // CHECK: = meta.buffer.rebind %0 : !meta.buffer<42, f32> to !meta.buffer<size, dt>
  %8 = meta.buffer.rebind %0 : !meta.buffer<42, f32> to !meta.buffer<size, dt>

  // CHECK:  = meta.buffer.rebind %8 : !meta.buffer<size, dt> to !meta.buffer<size2, dt>
  %9 = meta.buffer.rebind %8 : !meta.buffer<size, dt> to !meta.buffer<size2, dt>

  // CHECK: kgen.return %0 : !meta.buffer<42, f32>
  kgen.return %0 : !meta.buffer<42, f32>
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

// CHECK-LABEL: @meta_scalar_rebind
// CHECK-SAME: %[[ARG0:.*]]:
kgen.generator @meta_scalar_rebind<type1: dtype, type2: dtype>(%arg0: !meta.scalar<f32>) -> !meta.scalar<si32> {
  // CHECK: %[[V0:.*]] = meta.scalar.rebind %[[ARG0]] : !meta.scalar<f32> to !meta.scalar<type1>
  %0 = meta.scalar.rebind %arg0 : !meta.scalar<f32> to !meta.scalar<type1>
  // CHECK: %[[V1:.*]] = meta.scalar.rebind %[[V0]] : !meta.scalar<type1> to !meta.scalar<type2>
  %1 = meta.scalar.rebind %0 : !meta.scalar<type1> to !meta.scalar<type2>
  // CHECK: %[[V2:.*]] = meta.scalar.rebind %[[V1]] : !meta.scalar<type2> to !meta.scalar<si32>
  %2 = meta.scalar.rebind %1 : !meta.scalar<type2> to !meta.scalar<si32>
  // CHECK: return %[[V2]] : !meta.scalar<si32>
  kgen.return %2 : !meta.scalar<si32>
}

// CHECK-LABEL: @meta_simd_rebind
// CHECK-SAME: %[[ARG0:.*]]:
kgen.generator @meta_simd_rebind<size1, size2, type1: dtype, type2: dtype>(%arg0: !meta.simd<2, f32>) -> !meta.simd<4, ui64> {
  // CHECK: %[[V0:.*]] = meta.simd.rebind %[[ARG0]] : !meta.simd<2, f32> to !meta.simd<size1, type1>
  %0 = meta.simd.rebind %arg0 : !meta.simd<2, f32> to !meta.simd<size1, type1>
  // CHECK: %[[V1:.*]] = meta.simd.rebind %[[V0]] : !meta.simd<size1, type1> to !meta.simd<size2, type2>
  %1 = meta.simd.rebind %0 : !meta.simd<size1, type1> to !meta.simd<size2, type2>
  // CHECK: %[[V2:.*]] = meta.simd.rebind %[[V1]] : !meta.simd<size2, type2> to !meta.simd<4, ui64>
  %2 = meta.simd.rebind %1 : !meta.simd<size2, type2> to !meta.simd<4, ui64>
  // CHECK: return %[[V2]] : !meta.simd<4, ui64>
  kgen.return %2 : !meta.simd<4, ui64>
}

// CHECK-LABEL: @meta_pointer_rebind
// CHECK-SAME: %[[ARG0:.*]]:
kgen.generator @meta_pointer_rebind<type1: dtype, type2: dtype>(%arg0: !meta.pointer<?>) -> !meta.pointer<!meta.scalar<f32>> {
  // CHECK: %[[V0]] = meta.pointer.rebind %[[ARG0]] : !meta.pointer<?> to !meta.pointer<!meta.scalar<type1>>
  %0 = meta.pointer.rebind %arg0 : !meta.pointer<?> to !meta.pointer<!meta.scalar<type1>>
  // CHECK: %[[V1]] = meta.pointer.rebind %[[V0]] : !meta.pointer<!meta.scalar<type1>> to !meta.pointer<!meta.scalar<type2>>
  %1 = meta.pointer.rebind %0 : !meta.pointer<!meta.scalar<type1>> to !meta.pointer<!meta.scalar<type2>>
  // CHECK: %[[V2]] = meta.pointer.rebind %[[V1]] : !meta.pointer<!meta.scalar<type2>> to !meta.pointer<!meta.scalar<f32>>
  %2 = meta.pointer.rebind %1 : !meta.pointer<!meta.scalar<type2>> to !meta.pointer<!meta.scalar<f32>>
  // CHECK: return %[[V2]] : !meta.pointer<!meta.scalar<f32>>
  kgen.return %2 : !meta.pointer<!meta.scalar<f32>>
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
