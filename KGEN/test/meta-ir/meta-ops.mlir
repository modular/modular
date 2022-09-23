// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// CHECK-LABEL: @pointer_types
kgen.generator @pointer_types<dt: dtype>(
  // CHECK-SAME: %{{.*}}: !meta.pointer<!meta.scalar<dt>>, %{{.*}}: !meta.pointer<!meta.scalar<f32>>, %{{.*}}: !meta.pointer<?>
  %arg0: !meta.pointer<!meta.scalar<dt>>, %arg1: !meta.pointer<!meta.scalar<f32>>, %arg2: !meta.pointer<?>) {
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
