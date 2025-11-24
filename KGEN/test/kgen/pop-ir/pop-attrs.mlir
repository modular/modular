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
kgen.generator @array_constants<T: type, A: !kgen.param<T>>() {
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

// CHECK: f0 = #pop.simd_and<#kgen.unknown : !pop.simd<4, si32>, #pop<simd -1> : !pop.simd<4, si32>> : !pop.simd<4, si32>
"simd_and.attr"() { f0 = #pop.simd_and<#kgen.unknown : !pop.simd<4, si32>, #pop<simd -1> : !pop.simd<4, si32>> } : () -> ()

// CHECK: f0 = #pop.cast<#kgen.unknown : !pop.scalar<si32>> : !pop.scalar<ui32>
"pop_cast.op"() { f0 = #pop.cast< #kgen.unknown : !pop.scalar<si32>> : !pop.scalar<ui32> } : () -> ()

// CHECK: f0 = #pop.cast_from_builtin<#kgen.unknown : si64> : !pop.scalar<si64>
"pop_cast_from_builtin.op"() { f0 = #pop.cast_from_builtin<#kgen.unknown : si64> : !pop.scalar<si64> } : () -> ()

// CHECK: f0 = #pop.cast_to_builtin<#kgen.unknown : !pop.scalar<f16>> : f16
"pop_cast_to_builtin.op"() { f0 = #pop.cast_to_builtin<#kgen.unknown: !pop.scalar<f16>> : f16 } : () -> ()

// CHECK: f0 = #pop.simd_splat<#kgen.unknown : !pop.scalar<f16>> : !pop.simd<4, f16>
"pop_simd_splat.op"() { f0 = #pop.simd_splat<#kgen.unknown : !pop.scalar<f16>> : !pop.simd<4, f16> } : () -> ()

// CHECK: f0 = #pop.dtype_to_ui8<*?> : ui8
"pop_dtype_to_ui8.op"() { f0 = #pop.dtype_to_ui8<*?> : ui8 } : () -> ()

// CHECK: f0 = #pop.dtype_from_ui8<#kgen.unknown : ui8> : !kgen.dtype
"pop_dtype_from_ui8.op"() { f0 = #pop.dtype_from_ui8<#kgen.unknown : ui8> : !kgen.dtype } : () -> ()

// CHECK: f0 = #pop.simd_xor<#kgen.unknown : !pop.simd<4, si32>, #pop<simd -1> : !pop.simd<4, si32>> : !pop.simd<4, si32>
"simd_xor.attr"() { f0 = #pop.simd_xor<#kgen.unknown : !pop.simd<4, si32>, #pop<simd -1> : !pop.simd<4, si32>> } : () -> ()

// CHECK: f0 = #pop.simd_cmp<eq, #kgen.unknown : !pop.simd<4, si32>, #pop<simd -1> : !pop.simd<4, si32>> : !pop.simd<4, bool>
"simd_cmp.attr"() { f0 = #pop.simd_cmp<eq, #kgen.unknown : !pop.simd<4, si32>, #pop<simd -1> : !pop.simd<4, si32>> : !pop.simd<4, bool> } : () -> ()

// CHECK: f0 = #pop.simd_reduce_or<#kgen.unknown : !pop.simd<4, si32>> : !pop.scalar<si32>
"simd_reduce_or.attr"() { f0 = #pop.simd_reduce_or<#kgen.unknown : !pop.simd<4, si32>> : !pop.scalar<si32> } : () -> ()
