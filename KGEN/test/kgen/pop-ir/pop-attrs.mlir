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
  // CHECK: param_list<si32> = <[1, value]>
  kgen.param.constant: param_list<si32> = <[1, value]>
  // CHECK: param_list<T> = <[]>
  kgen.param.constant: param_list<T> = <[]>
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

// CHECK: f0 = #pop.simd_shl<#kgen.unknown : !pop.simd<4, si32>, #pop<simd 1> : !pop.simd<4, si32>> : !pop.simd<4, si32>
"simd_shl.attr"() { f0 = #pop.simd_shl<#kgen.unknown : !pop.simd<4, si32>, #pop<simd 1> : !pop.simd<4, si32>> } : () -> ()

// CHECK: f0 = #pop.simd_shr<#kgen.unknown : !pop.simd<4, si32>, #pop<simd 1> : !pop.simd<4, si32>> : !pop.simd<4, si32>
"simd_shr.attr"() { f0 = #pop.simd_shr<#kgen.unknown : !pop.simd<4, si32>, #pop<simd 1> : !pop.simd<4, si32>> } : () -> ()

// CHECK: f0 = #pop.variadic_to_array<:param_list<index> v> : !pop.array<#kgen.param_list.size<:param_list<index> v>, index>
"variadic_to_array.attr"() { f0 = #pop.variadic_to_array<:!kgen.param_list<index> #kgen.param.decl.ref<"v">>
      : !pop.array<#kgen.param_list.size<:!kgen.param_list<index> #kgen.param.decl.ref<"v">>, index> } : () -> ()

// CHECK: f0 = #pop.array<1, 2, 3> : !pop.array<3, index>
"variadic_to_array_fold.attr"() { f0 = #pop.variadic_to_array<:!kgen.param_list<index> #kgen.param_list<1, 2, 3> > : !pop.array<3, index> } : () -> ()

// CHECK:      a0 = #pop<simd 0>
// CHECK-SAME: a1 = #pop<simd 1>
// CHECK-SAME: a2 = #pop<simd 0>
// CHECK-SAME: a3 = #pop<simd 1>
// CHECK-SAME: a4 = #pop<simd -1>
// CHECK-SAME: a5 = #pop<simd 4294967295>
// CHECK-SAME: a6 = #pop<simd -1>
// CHECK-SAME: a7 = #pop<simd 255>
// CHECK-SAME: a8 = #pop<simd -1>
// CHECK-SAME: a9 = #pop<simd 65535>

// CHECK-SAME: b0 = #pop<simd -1>
// CHECK-SAME: b1 = #pop<simd 18446744073709551615>

// CHECK-SAME: c0 = #pop<simd -1>
// CHECK-SAME: c1 = #pop<simd 255>
// CHECK-SAME: c2 = #pop<simd true>
// CHECK-SAME: c3 = #pop<simd true>
// CHECK-SAME: c4 = #pop<simd true>
// CHECK-SAME: c5 = #pop<simd false>

// CHECK-SAME: d0 = #pop<simd "1">
// CHECK-SAME: d1 = #pop<simd "1">

// CHECK-SAME: e0 = #pop<simd 255>

// CHECK-SAME: z6 = #pop<simd -1>
// CHECK-SAME: z7 = #pop<simd 18446744073709551615>
// CHECK-SAME: z8 = #pop<simd -1>
// CHECK-SAME: z9 = #pop<simd 340282366920938463463374607431768211455>
"literal_converts"() {
    a0 = #pop.int_literal_convert<0> : !pop.scalar<si32>,
    a1 = #pop.int_literal_convert<1> : !pop.scalar<si32>,
    a2 = #pop.int_literal_convert<0> : !pop.scalar<ui32>,
    a3 = #pop.int_literal_convert<1> : !pop.scalar<ui32>,
    a4 = #pop.int_literal_convert<-1> : !pop.scalar<si32>,
    a5 = #pop.int_literal_convert<-1> : !pop.scalar<ui32>,
    a6 = #pop.int_literal_convert<-1> : !pop.scalar<si8>,
    a7 = #pop.int_literal_convert<-1> : !pop.scalar<ui8>,
    a8 = #pop.int_literal_convert<-1> : !pop.scalar<si16>,
    a9 = #pop.int_literal_convert<-1> : !pop.scalar<ui16>,

    b0 = #pop.int_literal_convert<-1> : !pop.scalar<si64>,
    b1 = #pop.int_literal_convert<-1> : !pop.scalar<ui64>,

    c0 = #pop.int_literal_convert<65535> : !pop.scalar<si8>,
    c1 = #pop.int_literal_convert<65535> : !pop.scalar<ui8>,
    c2 = #pop.int_literal_convert<65535> : !pop.scalar<bool>,
    c3 = #pop.int_literal_convert<65534> : !pop.scalar<bool>,
    c4 = #pop.int_literal_convert<1> : !pop.scalar<bool>,
    c5 = #pop.int_literal_convert<0> : !pop.scalar<bool>,

    d0 = #pop.int_literal_convert<1> : !pop.scalar<f32>,
    d1 = #pop.int_literal_convert<1> : !pop.scalar<f64>,

    e0 = #pop.int_literal_convert<-1> : !pop.simd<4, ui8>,

    z6 = #pop.int_literal_convert<-1> : !pop.scalar<index>,
    z7 = #pop.int_literal_convert<-1> : !pop.scalar<uindex>,
    z8 = #pop.int_literal_convert<-1> : !pop.scalar<si128>,
    z9 = #pop.int_literal_convert<-1> : !pop.scalar<ui128>
} : () -> ()
