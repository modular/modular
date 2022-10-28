// RUN: kgen-opt %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @array
kgen.generator @array<size, type: type>(
  // CHECK-SAME: !pop.array<4, simd<1, f32>>
  %arg0: !pop.array<4, simd<1, f32>>,
  // CHECK-SAME: !pop.array<size, type>
  %arg1: !pop.array<size, type>
) {
  kgen.return
}

// CHECK-LABEL: @struct
kgen.generator @struct<size, dtype: dtype, type: type>(
  // CHECK-SAME: !pop.struct<simd<1, f32>, simd<4, ui64>>
  %arg0: !pop.struct<simd<1, f32>, simd<4, ui64>>,
  // CHECK-SAME: !pop.struct<pointer<simd<4, si8>>, array<24, simd<1, si64>>, struct<simd<1, f32>, simd<1, f64>>>
  %arg1: !pop.struct<
    !pop.pointer<simd<4, si8>>,
    !pop.array<24, simd<1, si64>>,
    !pop.struct<
      !pop.simd<1, f32>,
      !pop.simd<1, f64>
    >
  >,
  // CHECK: !pop.struct<type, array<size, simd<1, dtype>>>
  %arg2: !pop.struct<type, array<size, simd<1, dtype>>>
) {
  kgen.return
}
