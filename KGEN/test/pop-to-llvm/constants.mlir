// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

// COM: Checking the LLVMIR is easier since the constants are collapsed.

// CHECK-LABEL: @array_constant
kgen.func @array_constant() -> !pop.array<2, i32> {
  // CHECK-NEXT: [2 x i32] [i32 1, i32 2]
  %0 = kgen.param.constant: !pop.array<2, i32> = <#pop.array<1, 2>>
  kgen.return %0 : !pop.array<2, i32>
}

// CHECK-LABEL: @struct_constant
kgen.func @struct_constant() -> !pop.struct<array<1, i32>, struct<i32, i32>> {
  // CHECK-NEXT: { [1 x i32], { i32, i32 } }
  // CHECK-SAME: { [1 x i32] [i32 1], { i32, i32 } { i32 2, i32 3 } }
  %0 = kgen.param.constant: !pop.struct<array<1, i32>, struct<i32, i32>> =
    <#pop.struct<#pop.array<1>, #pop.struct<2, 3>>>
  kgen.return %0 : !pop.struct<array<1, i32>, struct<i32, i32>>
}

// CHECK-LABEL: @simd_constant
kgen.func @simd_constant() -> (!pop.simd<2, bool>, !pop.simd<2, si8>, !pop.scalar<bf16>) {
  // CHECK-NEXT: { <2 x i1>, <2 x i8>, bfloat }
  // CHECK-SAME: { <2 x i1> <i1 true, i1 false>, <2 x i8> <i8 -3, i8 3>, bfloat 0xR3FA0 }
  %0 = kgen.param.constant: !pop.simd<2, bool> = <#pop.simd<true, false>>
  %1 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<-3, 3>>
  %2 = kgen.param.constant: !pop.scalar<bf16> = <#pop.simd<"1.25">>
  kgen.return %0, %1, %2 : !pop.simd<2, bool>, !pop.simd<2, si8>, !pop.scalar<bf16>
}

// CHECK-LABEL: @variant_constant_0
kgen.func @variant_constant_0() -> !pop.variant<i32> {
  // CHECK-NEXT: { [1 x i64], i1 } { [1 x i64] [i64 1], i1 false }
  %0 = kgen.param.constant: !pop.variant<i32> = <#pop.variant<:i32 1>>
  kgen.return %0 : !pop.variant<i32>
}

// CHECK-LABEL: @variant_constant_1
kgen.func @variant_constant_1() -> !pop.variant<struct<i32, i64, i32>, struct<f64, f32>> {
  // CHECK-NEXT: { [2 x i64], i1 } { [2 x i64] [i64 8589934593, i64 12884901888], i1 false }
  %0 = kgen.param.constant: !pop.variant<struct<i32, i64, i32>, struct<f64, f32>> = <#pop.variant<:!pop.struct<i32, i64, i32> #pop.struct<1, 2, 3>>>
  kgen.return %0 : !pop.variant<struct<i32, i64, i32>, struct<f64, f32>>
}

// CHECK-LABEL: @variant_constant_2
kgen.func @variant_constant_2() -> !pop.variant<i1, i2, i3, i4, i5, i6> {
  // CHECK-NEXT: { [1 x i64], i3 } { [1 x i64] [i64 1], i3 3 }
  %0 = kgen.param.constant: !pop.variant<i1, i2, i3, i4, i5, i6> = <#pop.variant<:i4 1>>
  kgen.return %0 : !pop.variant<i1, i2, i3, i4, i5, i6>
}
