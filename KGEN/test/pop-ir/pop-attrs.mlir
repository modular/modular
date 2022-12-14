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
