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

// CHECK-LABEL: @array_struct_constants
kgen.generator @array_struct_constants<T: type, A: !kgen.paramref<T>, value: !pop.scalar<f32>>() {
  // CHECK: struct<index, f32> = <{ 1, 2.5{{0+}}e+00 }>
  kgen.param.constant: struct<index, f32> = <{ 1, 2.5 }>
  // CHECK: array<2, index> = <[1, 2]>
  kgen.param.constant: array<2, index> = <[1, 2]>
  // CHECK: struct<scalar<f32>> = <{ value }>
  kgen.param.constant: struct<scalar<f32>> = <{ value }>
  // CHECK: array<2, dtype> = <[ui4, si4]>
  kgen.param.constant: array<2, dtype> = <[ui4, si4]>
  // CHECK: struct<T> = <{ A }>
  kgen.param.constant: struct<T> = <{ A }>
  // CHECK: array<2, T> = <[A, A]>
  kgen.param.constant: array<2, T> = <[A, A]>
  kgen.return
}

// CHECK-LABEL: @pack_constants
kgen.generator @pack_constants<Ts: variadic<i32>>() {
  // CHECK: !kgen.pack<[i8, ui4, i32]> = <<3, 1, 4>>
  %0 = kgen.param.constant: !kgen.pack<[i8, ui4, i32]> = <<3, 1, 4>>
  // CHECK: !kgen.pack<[]> = <<>>
  %1 = kgen.param.constant: !kgen.pack<[]> = <<>>
  kgen.return
}

// CHECK-LABEL: @variant_constants
kgen.generator @variant_constants<T: type, U: type, value: !kgen.paramref<T>>() {
  // CHECK: variant<f32, f64> = <#pop.variant<:f32 2.5{{0+}}e+00, 0>>
  %0 = kgen.param.constant: variant<f32, f64> = <#pop.variant<:f32 2.5, 0>>
  // CHECK: variant<T, U> = <#pop.variant<:!kgen.paramref<T> value, 0>>
  %1 = kgen.param.constant: variant<T, U> = <#pop.variant<:!kgen.paramref<T> value, 0>>
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

// CHECK: f0 = #pop<fmf reassoc>
// CHECK: f1 = #pop<fmf nnan|ninf|reassoc>
"enums.op"() {
  f0 = #pop<fmf reassoc>,
  f1 = #pop<fmf reassoc|ninf|nnan>
} : () -> ()
