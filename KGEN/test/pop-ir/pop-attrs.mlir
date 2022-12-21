// RUN: kgen-opt %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @simd_constants
kgen.func @simd_constants() {
  // CHECK: !pop.simd<2, f32> = <#pop.simd<"12.375", "77">>
  %0 = kgen.param.constant: !pop.simd<2, f32> = <#pop.simd<"12.375", "77">>
  // CHECK: !pop.scalar<si64> = <#pop.simd<1234>>
  %1 = kgen.param.constant: !pop.scalar<si64> = <#pop.simd<1234>>
  // CHECK: !pop.simd<2, bool> = <#pop.simd<true, false>>
  %2 = kgen.param.constant: !pop.simd<2, bool> = <#pop.simd<true, false>>
  // CHECK: !pop.scalar<f64> = <#pop.simd<"0.01171875">>
  %3 = kgen.param.constant: !pop.scalar<f64> = <#pop.simd<"0.01171875">>
  kgen.return
}

// CHECK-LABEL: @array_struct_constants
kgen.func @array_struct_constants() {
  // CHECK: !pop.struct<index, f32> = <#pop.struct<1, 2.5{{0+}}e+00>>
  %0 = kgen.param.constant: !pop.struct<index, f32> = <#pop.struct<1, 2.5>>
  // CHECK: !pop.array<2, index> = <#pop.array<1, 2>>
  %1 = kgen.param.constant: !pop.array<2, index> = <#pop.array<1, 2>>
  kgen.return
}
