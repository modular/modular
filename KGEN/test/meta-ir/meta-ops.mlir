// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// CHECK-LABEL: @pointer_types
kgen.generator @pointer_types<dt: dtype>(
  // CHECK-SAME: %{{.*}}: !pop.pointer<!meta.scalar<dt>>, %{{.*}}: !pop.pointer<!meta.scalar<f32>>, %{{.*}}: !pop.pointer<?>
  %arg0: !pop.pointer<!meta.scalar<dt>>, %arg1: !pop.pointer<!meta.scalar<f32>>, %arg2: !pop.pointer<?>) {
  kgen.return
}

// CHECK-LABEL: @cast_to_builtin
// CHECK-SAME: %[[ARG0:.*]]: !meta.scalar<f32>, %[[ARG1:.*]]: !meta.scalar<si32>
kgen.func @cast_to_builtin(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<si32>) {
  // CHECK: pop.type_lower %[[ARG0]] : !meta.scalar<f32> to f32
  %0 = pop.type_lower %arg0: !meta.scalar<f32> to f32
  // CHECK: pop.type_lower %[[ARG1]] : !meta.scalar<si32> to i32
  %1 = pop.type_lower %arg1: !meta.scalar<si32> to i32
  kgen.return
}

// CHECK-LABEL: @cast_from_builtin
// CHECK-SAME: %[[ARG0:.*]]: f32, %[[ARG1:.*]]: ui32
kgen.func @cast_from_builtin(%arg0: f32, %arg1: ui32) {
  // CHECK: pop.type_raise %[[ARG0]] : f32 to !meta.scalar<f32>
  %0 = pop.type_raise %arg0: f32 to !meta.scalar<f32>
  // CHECK: pop.type_raise %[[ARG1]] : ui32 to !meta.scalar<ui32>
  %1 = pop.type_raise %arg1: ui32 to !meta.scalar<ui32>
  kgen.return
}

// CHECK-LABEL: @cast_from_builtin_vector
// CHECK-SAME: %[[ARG:.*]]:
kgen.func @cast_from_builtin_vector(%arg0: vector<1xf32>) -> !meta.simd<1, f32> {
  // CHECK: %[[V0:.*]] = pop.type_raise %[[ARG]] : vector<1xf32> to !meta.simd<1, f32>
  %0 = pop.type_raise %arg0 : vector<1xf32> to !meta.simd<1, f32>
  // CHECK: kgen.return  %[[V0:.*]] : !meta.simd<1, f32>
  kgen.return %0 : !meta.simd<1, f32>
}
