// RUN: kgen-opt %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @array
kgen.generator @array<size, type: type>(
  // CHECK-SAME: !pop.array<4, scalar<f32>>
  %arg0: !pop.array<4, scalar<f32>>,
  // CHECK-SAME: !pop.array<size, type>
  %arg1: !pop.array<size, type>
) {
  kgen.return
}

// CHECK-LABEL: @struct
kgen.generator @struct<size, dtype: dtype, type: type>(
  // CHECK-SAME: !pop.struct<scalar<f32>, simd<4, ui64>>
  %arg0: !pop.struct<scalar<f32>, simd<4, ui64>>,
  // CHECK-SAME: !pop.struct<pointer<simd<4, si8>>, array<24, scalar<si64>>, struct<scalar<f32>, scalar<f64>>>
  %arg1: !pop.struct<
    !pop.pointer<simd<4, si8>>,
    !pop.array<24, scalar<si64>>,
    !pop.struct<
      !pop.scalar<f32>,
      !pop.scalar<f64>
    >
  >,
  // CHECK: !pop.struct<type, array<size, scalar<dtype>>>
  %arg2: !pop.struct<type, array<size, scalar<dtype>>>
) {
  kgen.return
}

// CHECK-LABEL: @variadic
kgen.generator @variadic<type: type>(
  // CHECK-SAME: !kgen.variadic<!pop.scalar<f32>>
  %arg0: !kgen.variadic<!pop.scalar<f32>>,
  // CHECK-SAME: !kgen.variadic<type>
  %arg1: !kgen.variadic<type>
) {
  kgen.return
}
