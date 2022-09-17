// RUN: kgen-opt %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @array
kgen.generator @array<size, type: type>(
  // CHECK-SAME: !pop.array<4, !meta.scalar<f32>>
  %arg0: !pop.array<4, !meta.scalar<f32>>,
  // CHECK-SAME: !pop.array<size, type>
  %arg1: !pop.array<size, type>
) {
  kgen.return
}
