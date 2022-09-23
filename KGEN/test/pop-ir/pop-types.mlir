// RUN: kgen-opt %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @array
kgen.generator @array<size, type: type>(
  // CHECK-SAME: !pop.array<4, !pop.scalar<f32>>
  %arg0: !pop.array<4, !pop.scalar<f32>>,
  // CHECK-SAME: !pop.array<size, type>
  %arg1: !pop.array<size, type>
) {
  kgen.return
}

// CHECK-LABEL: @struct
kgen.generator @struct<size, dtype: dtype, type: type>(
  // CHECK-SAME: !pop.struct<!pop.scalar<f32>, !meta.simd<4, ui64>>
  %arg0: !pop.struct<!pop.scalar<f32>, !meta.simd<4, ui64>>,
  // CHECK-SAME: !pop.struct<!pop.pointer<!meta.simd<4, si8>>, !pop.array<24, !pop.scalar<si64>>, !pop.struct<!pop.scalar<f32>, !pop.scalar<f64>>>
  %arg1: !pop.struct<
    !pop.pointer<!meta.simd<4, si8>>,
    !pop.array<24, !pop.scalar<si64>>,
    !pop.struct<
      !pop.scalar<f32>,
      !pop.scalar<f64>
    >
  >,
  // CHECK: !pop.struct<type, !pop.array<size, !pop.scalar<dtype>>>
  %arg2: !pop.struct<type, !pop.array<size, !pop.scalar<dtype>>>
) {
  kgen.return
}
