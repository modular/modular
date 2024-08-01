// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @simd_constants
kgen.generator @simd_constants<N, value: !pop.simd<N, si32>>() {
  // CHECK: simd<2, f32> = <<"12.375", "77">>
  %0 = kgen.param.constant: simd<2, f32> = <<"12.375", "77">>
  // CHECK: scalar<si64> = <1234>
  %1 = kgen.param.constant: scalar<si64> = <1234>
  // CHECK: simd<2, bool> = <<true, false>>
  %2 = kgen.param.constant: simd<2, bool> = <<true, false>>
  // CHECK: scalar<f64> = <"0.01171875">
  %3 = kgen.param.constant: scalar<f64> = <"0.01171875">
  // CHECK: simd<6, ui2> = <<0, 1, 2, 3, 3, 2>>
  %4 = kgen.param.constant: simd<6, ui2> = <<0, 1, 2, 3, 3, 2>>
  // CHECK: simd<2, index> = <<-54321, 12345>>
  %5 = kgen.param.constant: simd<2, index> = <<-54321, 12345>>
  // CHECK: scalar<f32> = <"0.100000001">
  %6 = kgen.param.constant: scalar<f32> = <"0.1">


  // CHECK: scalar<f16> = <"1.7285E-5">
  kgen.param.constant: scalar<f16> = <"1.7285E-5">

  // CHECK: #pop.simd<1, 2>
  "simd.const"() {a = #pop.simd<1, 2> : !pop.simd<2, si32>} : () -> ()
  // CHECK: #pop<simd 1>
  "simd.const"() {a = #pop<simd 1> : !pop.simd<2, si32>} : () -> ()
  // CHECK: simd<N, si32> = <value>
  kgen.param.constant: simd<N, si32> = <value>
  kgen.return
}

// CHECK-LABEL: @array_constants
kgen.generator @array_constants<T: type, A: !kgen.paramref<T>>() {
  // CHECK: array<2, index> = <[1, 2]>
  kgen.param.constant: array<2, index> = <[1, 2]>
  // CHECK: array<2, dtype> = <[ui4, si4]>
  kgen.param.constant: array<2, dtype> = <[ui4, si4]>
  // CHECK: array<2, T> = <[A, A]>
  kgen.param.constant: array<2, T> = <[A, A]>
  kgen.return
}

// CHECK-LABEL: @variadic_constants
kgen.generator @variadic_constants<T: type, value: si32>() {
  // CHECK: variadic<si32> = <[1, value]>
  kgen.param.constant: variadic<si32> = <[1, value]>
  // CHECK: variadic<T> = <[]>
  kgen.param.constant: variadic<T> = <[]>
  kgen.return
}

// CHECK-LABEL: @union_constants
kgen.func @union_constants() {
  // CHECK: constant: union<i32, i64> = <{:i32 42}>
  kgen.param.constant: union<i32, i64> = <{:i32 42}>
  kgen.return
}

// CHECK: f0 = #pop.union<:i32 42> : !pop.union<i32, i64>
"union.attr"() { f0 = #pop.union<:i32 42> : !pop.union<i32, i64> } : () -> ()

// CHECK: f0 = #pop<fmf none>
// CHECK: f1 = #pop<fmf reassoc>
// CHECK: f2 = #pop<fmf nnan|ninf|reassoc>
// CHECK: f3 = #pop<fmf fast>
"enums.op"() {
  f0 = #pop<fmf none>,
  f1 = #pop<fmf reassoc>,
  f2 = #pop<fmf reassoc|ninf|nnan>,
  f3 = #pop<fmf fast>
} : () -> ()
