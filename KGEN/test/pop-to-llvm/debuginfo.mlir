// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' -mlir-print-debuginfo %s | FileCheck %s

// Test proper handling of debug types.
!arrayTest = !pop.array<2, array<3, array<4, simd<8, bool>>>>
!closureTest = !pop.closure<(index, ui8) -> ()>
!coroutineTest = !pop.coroutine<() -> (i8, si8)>
!packTest = !pop.pack<[index,
                       ui8, i8, si8,
                       ui16, i16, si16,
                       ui32, i32, si32,
                       ui64, i64, si64,
                       ui128, i128, si128,
                       f16, f32, f64, f80, f128, bf16,
                       !pop.array<5, f32>]>
!pointerTest = !pop.pointer<scalar<bool>>
!voidPointerTest = !pop.pointer<scalar<invalid>>
!scalarTest = !pop.scalar<bool>
!simdTest = !pop.simd<8, ui32>
!structTest = !pop.struct<scalar<bool>, array<5, array<4, simd<8, si32>>>, struct<pointer<scalar<bool>>, array<4, simd<8, si32>>>>

// CHECK-DAG: !basic = !debuginfo.basic<bool {sizeInBits = 8, alignInBits = 8, encoding = DW_ATE_boolean}>
// CHECK-DAG: !basic1 = !debuginfo.basic<index {sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_signed}>
// CHECK-DAG: !basic2 = !debuginfo.basic<ui8 {sizeInBits = 8, alignInBits = 1, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic3 = !debuginfo.basic<i8 {sizeInBits = 8, alignInBits = 1, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic4 = !debuginfo.basic<si8 {sizeInBits = 8, alignInBits = 1, encoding = DW_ATE_signed}>
// CHECK-DAG: !basic5 = !debuginfo.basic<ui16 {sizeInBits = 16, alignInBits = 2, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic6 = !debuginfo.basic<i16 {sizeInBits = 16, alignInBits = 2, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic7 = !debuginfo.basic<si16 {sizeInBits = 16, alignInBits = 2, encoding = DW_ATE_signed}>
// CHECK-DAG: !basic8 = !debuginfo.basic<ui32 {sizeInBits = 32, alignInBits = 4, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic9 = !debuginfo.basic<i32 {sizeInBits = 32, alignInBits = 4, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic10 = !debuginfo.basic<si32 {sizeInBits = 32, alignInBits = 4, encoding = DW_ATE_signed}>
// CHECK-DAG: !basic11 = !debuginfo.basic<ui64 {sizeInBits = 64, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic12 = !debuginfo.basic<i64 {sizeInBits = 64, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic13 = !debuginfo.basic<si64 {sizeInBits = 64, alignInBits = 8, encoding = DW_ATE_signed}>
// CHECK-DAG: !basic14 = !debuginfo.basic<ui128 {sizeInBits = 128, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic15 = !debuginfo.basic<i128 {sizeInBits = 128, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic16 = !debuginfo.basic<si128 {sizeInBits = 128, alignInBits = 8, encoding = DW_ATE_signed}>
// CHECK-DAG: !basic17 = !debuginfo.basic<f16 {sizeInBits = 16, alignInBits = 2, encoding = DW_ATE_float}>
// CHECK-DAG: !basic18 = !debuginfo.basic<f32 {sizeInBits = 32, alignInBits = 4, encoding = DW_ATE_float}>
// CHECK-DAG: !basic19 = !debuginfo.basic<f64 {sizeInBits = 64, alignInBits = 8, encoding = DW_ATE_float}>
// CHECK-DAG: !basic20 = !debuginfo.basic<f80 {sizeInBits = 80, alignInBits = 16, encoding = DW_ATE_float}>
// CHECK-DAG: !basic21 = !debuginfo.basic<f128 {sizeInBits = 128, alignInBits = 16, encoding = DW_ATE_float}>
// CHECK-DAG: !basic22 = !debuginfo.basic<bf16 {sizeInBits = 16, alignInBits = 16, encoding = DW_ATE_float}>
// CHECK-DAG: !basic23 = !debuginfo.basic<ui32 {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !basic24 = !debuginfo.basic<si32 {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_signed}>
// CHECK-DAG: !unspecified = !debuginfo.unspecified<"void">
// CHECK-DAG: !array = !debuginfo.array<5 x !basic18>
// CHECK-DAG: !member = !debuginfo.member<m0: !basic1>
// CHECK-DAG: !member1 = !debuginfo.member<m1: !basic2>
// CHECK-DAG: !member2 = !debuginfo.member<m2: !basic3>
// CHECK-DAG: !member3 = !debuginfo.member<m3: !basic4>
// CHECK-DAG: !member4 = !debuginfo.member<m4: !basic5>
// CHECK-DAG: !member5 = !debuginfo.member<m5: !basic6>
// CHECK-DAG: !member6 = !debuginfo.member<m6: !basic7>
// CHECK-DAG: !member7 = !debuginfo.member<m7: !basic8>
// CHECK-DAG: !member8 = !debuginfo.member<m8: !basic9>
// CHECK-DAG: !member9 = !debuginfo.member<m9: !basic10>
// CHECK-DAG: !member10 = !debuginfo.member<m10: !basic11>
// CHECK-DAG: !member11 = !debuginfo.member<m11: !basic12>
// CHECK-DAG: !member12 = !debuginfo.member<m12: !basic13>
// CHECK-DAG: !member13 = !debuginfo.member<m13: !basic14>
// CHECK-DAG: !member14 = !debuginfo.member<m14: !basic15>
// CHECK-DAG: !member15 = !debuginfo.member<m15: !basic16>
// CHECK-DAG: !member16 = !debuginfo.member<m16: !basic17>
// CHECK-DAG: !member17 = !debuginfo.member<m17: !basic18>
// CHECK-DAG: !member18 = !debuginfo.member<m18: !basic19>
// CHECK-DAG: !member19 = !debuginfo.member<m19: !basic20>
// CHECK-DAG: !member20 = !debuginfo.member<m20: !basic21>
// CHECK-DAG: !member21 = !debuginfo.member<m21: !basic22>
// CHECK-DAG: !member22 = !debuginfo.member<m0: !basic>
// CHECK-DAG: !ptr = !debuginfo.ptr<!basic {sizeInBits = 64, alignInBits = 64}>
// CHECK-DAG: !ptr1 = !debuginfo.ptr<!unspecified {sizeInBits = 64, alignInBits = 64}>
// CHECK-DAG: !subroutine = !debuginfo.subroutine<(!basic1, !basic2) -> (): DW_CC_normal>
// CHECK-DAG: !subroutine1 = !debuginfo.subroutine<() -> (!basic3, !basic4): DW_CC_normal>
// CHECK-DAG: !vector = !debuginfo.vector<8 x !basic>
// CHECK-DAG: !vector1 = !debuginfo.vector<8 x !basic23>
// CHECK-DAG: !vector2 = !debuginfo.vector<8 x !basic24>
// CHECK-DAG: !array1 = !debuginfo.array<4 x !vector>
// CHECK-DAG: !array2 = !debuginfo.array<4 x !vector2>
// CHECK-DAG: !member23 = !debuginfo.member<m22: !array>
// CHECK-DAG: !member24 = !debuginfo.member<m0: !ptr>
// CHECK-DAG: !array3 = !debuginfo.array<3 x !array1>
// CHECK-DAG: !array4 = !debuginfo.array<5 x !array2>
// CHECK-DAG: !member25 = !debuginfo.member<m1: !array2>
// CHECK-DAG: !struct = !debuginfo.struct<pack(!member, !member1, !member2, !member3, !member4, !member5, !member6, !member7, !member8, !member9, !member10, !member11, !member12, !member13, !member14, !member15, !member16, !member17, !member18, !member19, !member20, !member21, !member23)>
// CHECK-DAG: !array5 = !debuginfo.array<2 x !array3>
// CHECK-DAG: !member26 = !debuginfo.member<m1: !array4>
// CHECK-DAG: !struct1 = !debuginfo.struct<struct(!member24, !member25)>
// CHECK-DAG: !member27 = !debuginfo.member<m2: !struct1>
// CHECK-DAG: !struct2 = !debuginfo.struct<struct(!member22, !member26, !member27)>
// CHECK-DAG: !subroutine2 = !debuginfo.subroutine<(!array5, !subroutine, !subroutine1, !struct, !ptr, !ptr1, !basic, !vector1, !struct2) -> (): DW_CC_normal>

!test = !debuginfo.subroutine<(!debuginfo.unresolved<!arrayTest>,
                               !debuginfo.unresolved<!closureTest>,
                               !debuginfo.unresolved<!coroutineTest>,
                               !debuginfo.unresolved<!packTest>,
                               !debuginfo.unresolved<!pointerTest>,
                               !debuginfo.unresolved<!voidPointerTest>,
                               !debuginfo.unresolved<!scalarTest>,
                               !debuginfo.unresolved<!simdTest>,
                               !debuginfo.unresolved<!structTest>) -> (): DW_CC_normal>

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_C,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !test

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="i64:64:64", simd_bit_width=128>} {
  kgen.func @foo() {
    kgen.return loc(fused<#subprogram>["foo.mlir":10:10])
  } loc(fused<#subprogram>["foo.mlir":10:10])
}
