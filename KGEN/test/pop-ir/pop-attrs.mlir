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
  // CHECK: !pop.simd<6, ui2> = <#pop.simd<0, 1, 2, 3, 3, 2>>
  %4 = kgen.param.constant: !pop.simd<6, ui2> = <#pop.simd<0, 1, 2, 3, 3, 2>>
  // CHECK: !pop.simd<2, index> = <#pop.simd<-54321, 12345>>
  %5 = kgen.param.constant: !pop.simd<2, index> = <#pop.simd<-54321, 12345>>
  // CHECK: !pop.scalar<f32> = <#pop.simd<"0.100000001">>
  %6 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"0.1">>
  kgen.return
}

// CHECK-LABEL: @array_struct_constants
kgen.generator @array_struct_constants<T: type, A: !kgen.paramref<T>, value: !pop.scalar<f32>>() {
  // CHECK: !pop.struct<index, f32> = <#pop.struct<1, 2.5{{0+}}e+00>>
  kgen.param.constant: !pop.struct<index, f32> = <#pop.struct<1, 2.5>>
  // CHECK: !pop.array<2, index> = <#pop.array<1, 2>>
  kgen.param.constant: !pop.array<2, index> = <#pop.array<1, 2>>
  // CHECK: !pop.struct<scalar<f32>> = <#pop.struct<value>>
  kgen.param.constant: !pop.struct<scalar<f32>> = <#pop.struct<value>>
  // CHECK: !pop.array<2, dtype> = <#pop.array<ui4, si4>>
  kgen.param.constant: !pop.array<2, dtype> = <#pop.array<ui4, si4>>
  // CHECK: !pop.struct<T> = <#pop.struct<A>>
  kgen.param.constant: !pop.struct<T> = <#pop.struct<A>>
  // CHECK: !pop.array<2, T> = <#pop.array<A, A>>
  kgen.param.constant: !pop.array<2, T> = <#pop.array<A, A>>
  kgen.return
}

// CHECK-LABEL: @variant_constants
kgen.generator @variant_constants<T: type, U: type, value: !kgen.paramref<T>>() {
  // CHECK: !pop.variant<f32, f64> = <#pop.variant<:f32 2.5{{0+}}e+00>>
  %0 = kgen.param.constant: !pop.variant<f32, f64> = <#pop.variant<:f32 2.5>>
  // CHECK: !pop.variant<T, U> = <#pop.variant<:!kgen.paramref<T> value>>
  %1 = kgen.param.constant: !pop.variant<T, U> = <#pop.variant<:!kgen.paramref<T> value>>
  kgen.return
}
