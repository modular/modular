// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.kernel @meta_buffer_size(%arg0: !meta.buffer<42, f32>, %arg1: !meta.buffer<?, f32>) -> index {
kgen.kernel @meta_buffer_size(%arg0: !meta.buffer<42, f32>, %arg1: !meta.buffer<?, f32>) -> index {
  // CHECK: %0 = meta.buffer.size %arg0 : !meta.buffer<42, f32>
  %0 = meta.buffer.size %arg0 : !meta.buffer<42, f32>
  // CHECK: %1 = meta.buffer.size %arg1 : !meta.buffer<?, f32>
  %1 = meta.buffer.size %arg1 : !meta.buffer<?, f32>
  // CHECK: kgen.return %1 : index
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.kernel @meta_buffer_dtype(%arg0: !meta.buffer<42, f32>, %arg1: !meta.buffer<42, ?>) -> !kgen.dtype {
kgen.kernel @meta_buffer_dtype(%arg0: !meta.buffer<42, f32>, %arg1: !meta.buffer<42, ?>) -> !kgen.dtype {
  // CHECK: %0 = meta.buffer.dtype %arg0 : !meta.buffer<42, f32>
  %0 = meta.buffer.dtype %arg0 : !meta.buffer<42, f32>
  // CHECK: %1 = meta.buffer.dtype %arg1 : !meta.buffer<42, ?>
  %1 = meta.buffer.dtype %arg1 : !meta.buffer<42, ?>
  // CHECK: kgen.return %1 : !kgen.dtype
  kgen.return %1 : !kgen.dtype
}

// CHECK-LABEL: kgen.kernel @meta_buffer_cast(%arg0: !meta.buffer<?, ?>) -> !meta.buffer<42, f32> {
kgen.kernel @meta_buffer_cast(%arg0: !meta.buffer<?, ?>) -> !meta.buffer<42, f32> {
  // CHECK: %0 = meta.buffer.cast %arg0 : !meta.buffer<?, ?> to !meta.buffer<42, f32>
  %0 = meta.buffer.cast %arg0 : !meta.buffer<?, ?> to !meta.buffer<42, f32>
  // CHECK: %1 = meta.buffer.cast %0 : !meta.buffer<42, f32> to !meta.buffer<?, ?>
  %1 = meta.buffer.cast %0 : !meta.buffer<42, f32> to !meta.buffer<?, ?>
  // CHECK: %2 = meta.buffer.cast %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, f32>
  %2 = meta.buffer.cast %arg0 : !meta.buffer<?, ?> to !meta.buffer<?, f32>
  // CHECK: %3 = meta.buffer.cast %arg0 : !meta.buffer<?, ?> to !meta.buffer<42, ?>
  %3 = meta.buffer.cast %arg0 : !meta.buffer<?, ?> to !meta.buffer<42, ?>
  // CHECK: %4 = meta.buffer.cast %2 : !meta.buffer<?, f32> to !meta.buffer<42, f32>
  %4 = meta.buffer.cast %2 : !meta.buffer<?, f32> to !meta.buffer<42, f32>
  // CHECK: %5 = meta.buffer.cast %3 : !meta.buffer<42, ?> to !meta.buffer<42, f32>
  %5 = meta.buffer.cast %3 : !meta.buffer<42, ?> to !meta.buffer<42, f32>
  // CHECK: %6 = meta.buffer.cast %0 : !meta.buffer<42, f32> to !meta.buffer<42, f32>
  %6 = meta.buffer.cast %0 : !meta.buffer<42, f32> to !meta.buffer<42, f32>
  // CHECK: kgen.return %0 : !meta.buffer<42, f32>
  kgen.return %0 : !meta.buffer<42, f32>
}

// CHECK-LABEL: kgen.kernel @cast_to_builtin(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<si32>) -> ui32 {
kgen.kernel @cast_to_builtin(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<si32>) -> ui32 {
  // CHECK: %0 = meta.cast_to_builtin %arg0 : !meta.scalar<f32> to f32
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<f32> to f32
  // CHECK: %1 = meta.cast_to_builtin %arg1 : !meta.scalar<si32> to ui32
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<si32> to ui32
  // CHECK: kgen.return %1 : ui32
  kgen.return %1 : ui32
}

// CHECK-LABEL: kgen.kernel @cast_from_builtin(%arg0: f32, %arg1: ui32) -> !meta.scalar<si32> {
kgen.kernel @cast_from_builtin(%arg0: f32, %arg1: ui32) -> !meta.scalar<si32> {
  // CHECK: %0 = meta.cast_from_builtin %arg0 : f32 to !meta.scalar<f32>
  %0 = meta.cast_from_builtin %arg0: f32 to !meta.scalar<f32>
  // CHECK: %1 = meta.cast_from_builtin %arg1 : ui32 to !meta.scalar<si32>
  %1 = meta.cast_from_builtin %arg1: ui32 to !meta.scalar<si32>
  // CHECK: kgen.return %1 : !meta.scalar<si32>
  kgen.return %1 : !meta.scalar<si32>
}
