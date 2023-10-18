// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' -mlir-print-debuginfo %s | FileCheck %s

// Test proper handling of debug types.
!arrayTest = !pop.array<2, array<3, array<4, simd<8, bool>>>>
!coroutineTest = !pop.coroutine<() -> (i8, si8)>
!packTest = !pop.pack<[index,
                       ui8, i8, si8,
                       ui16, i16, si16,
                       ui32, i32, si32,
                       ui64, i64, si64,
                       ui128, i128, si128,
                       f16, f32, f64, f80, f128, bf16,
                       !pop.array<5, f32>]>
!pointerTest = !kgen.pointer<scalar<bool>>
!voidPointerTest = !kgen.pointer<scalar<invalid>>
!scalarTest = !pop.scalar<bool>
!simdTest = !pop.simd<8, ui32>
!structTest = !pop.struct<scalar<bool>, array<5, array<4, simd<8, si32>>>, struct<pointer<scalar<bool>>, array<4, simd<8, si32>>>>

// CHECK-DAG: ![[BASIC:.*]] = !debuginfo.basic<bool {sizeInBits = 1, alignInBits = 1, encoding = DW_ATE_boolean}>
// CHECK-DAG: ![[BASIC1:.*]] = !debuginfo.basic<index {sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[BASIC2:.*]] = !debuginfo.basic<ui8 {sizeInBits = 8, alignInBits = 1, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC3:.*]] = !debuginfo.basic<i8 {sizeInBits = 8, alignInBits = 1, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC4:.*]] = !debuginfo.basic<si8 {sizeInBits = 8, alignInBits = 1, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[BASIC5:.*]] = !debuginfo.basic<ui16 {sizeInBits = 16, alignInBits = 2, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC6:.*]] = !debuginfo.basic<i16 {sizeInBits = 16, alignInBits = 2, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC7:.*]] = !debuginfo.basic<si16 {sizeInBits = 16, alignInBits = 2, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[BASIC8:.*]] = !debuginfo.basic<ui32 {sizeInBits = 32, alignInBits = 4, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC9:.*]] = !debuginfo.basic<i32 {sizeInBits = 32, alignInBits = 4, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC10:.*]] = !debuginfo.basic<si32 {sizeInBits = 32, alignInBits = 4, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[BASIC11:.*]] = !debuginfo.basic<ui64 {sizeInBits = 64, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC12:.*]] = !debuginfo.basic<i64 {sizeInBits = 64, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC13:.*]] = !debuginfo.basic<si64 {sizeInBits = 64, alignInBits = 8, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[BASIC14:.*]] = !debuginfo.basic<ui128 {sizeInBits = 128, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC15:.*]] = !debuginfo.basic<i128 {sizeInBits = 128, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC16:.*]] = !debuginfo.basic<si128 {sizeInBits = 128, alignInBits = 8, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[BASIC17:.*]] = !debuginfo.basic<f16 {sizeInBits = 16, alignInBits = 2, encoding = DW_ATE_float}>
// CHECK-DAG: ![[BASIC18:.*]] = !debuginfo.basic<f32 {sizeInBits = 32, alignInBits = 4, encoding = DW_ATE_float}>
// CHECK-DAG: ![[BASIC19:.*]] = !debuginfo.basic<f64 {sizeInBits = 64, alignInBits = 8, encoding = DW_ATE_float}>
// CHECK-DAG: ![[BASIC20:.*]] = !debuginfo.basic<f80 {sizeInBits = 80, alignInBits = 16, encoding = DW_ATE_float}>
// CHECK-DAG: ![[BASIC21:.*]] = !debuginfo.basic<f128 {sizeInBits = 128, alignInBits = 16, encoding = DW_ATE_float}>
// CHECK-DAG: ![[BASIC22:.*]] = !debuginfo.basic<bf16 {sizeInBits = 16, alignInBits = 16, encoding = DW_ATE_float}>
// CHECK-DAG: ![[BASIC23:.*]] = !debuginfo.basic<ui32 {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC24:.*]] = !debuginfo.basic<si32 {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_signed}>
// CHECK-DAG: !unspecified = !debuginfo.unspecified<"void">
// CHECK-DAG: !array = !debuginfo.array<5 x ![[BASIC18]]>
// CHECK-DAG: !member = !debuginfo.member<m0: ![[BASIC1]]>
// CHECK-DAG: ![[MEMBER1:.*]] = !debuginfo.member<m1: ![[BASIC2]]>
// CHECK-DAG: ![[MEMBER2:.*]] = !debuginfo.member<m2: ![[BASIC3]]>
// CHECK-DAG: ![[MEMBER3:.*]] = !debuginfo.member<m3: ![[BASIC4]]>
// CHECK-DAG: ![[MEMBER4:.*]] = !debuginfo.member<m4: ![[BASIC5]]>
// CHECK-DAG: ![[MEMBER5:.*]] = !debuginfo.member<m5: ![[BASIC6]]>
// CHECK-DAG: ![[MEMBER6:.*]] = !debuginfo.member<m6: ![[BASIC7]]>
// CHECK-DAG: ![[MEMBER7:.*]] = !debuginfo.member<m7: ![[BASIC8]]>
// CHECK-DAG: ![[MEMBER8:.*]] = !debuginfo.member<m8: ![[BASIC9]]>
// CHECK-DAG: ![[MEMBER9:.*]] = !debuginfo.member<m9: ![[BASIC10]]>
// CHECK-DAG: ![[MEMBER10:.*]] = !debuginfo.member<m10: ![[BASIC11]]>
// CHECK-DAG: ![[MEMBER11:.*]] = !debuginfo.member<m11: ![[BASIC12]]>
// CHECK-DAG: ![[MEMBER12:.*]] = !debuginfo.member<m12: ![[BASIC13]]>
// CHECK-DAG: ![[MEMBER13:.*]] = !debuginfo.member<m13: ![[BASIC14]]>
// CHECK-DAG: ![[MEMBER14:.*]] = !debuginfo.member<m14: ![[BASIC15]]>
// CHECK-DAG: ![[MEMBER15:.*]] = !debuginfo.member<m15: ![[BASIC16]]>
// CHECK-DAG: ![[MEMBER16:.*]] = !debuginfo.member<m16: ![[BASIC17]]>
// CHECK-DAG: ![[MEMBER17:.*]] = !debuginfo.member<m17: ![[BASIC18]]>
// CHECK-DAG: ![[MEMBER18:.*]] = !debuginfo.member<m18: ![[BASIC19]]>
// CHECK-DAG: ![[MEMBER19:.*]] = !debuginfo.member<m19: ![[BASIC20]]>
// CHECK-DAG: ![[MEMBER20:.*]] = !debuginfo.member<m20: ![[BASIC21]]>
// CHECK-DAG: ![[MEMBER21:.*]] = !debuginfo.member<m21: ![[BASIC22]]>
// CHECK-DAG: ![[MEMBER22:.*]] = !debuginfo.member<m0: ![[BASIC]]>
// CHECK-DAG: !ptr = !debuginfo.ptr<![[BASIC]] {sizeInBits = 64, alignInBits = 64}>
// CHECK-DAG: !ptr1 = !debuginfo.ptr<!unspecified {sizeInBits = 64, alignInBits = 64}>
// CHECK-DAG: !subroutine = !debuginfo.subroutine<() -> (![[BASIC3]], ![[BASIC4]]): DW_CC_normal>
// CHECK-DAG: !vector = !debuginfo.vector<8 x ![[BASIC]]>
// CHECK-DAG: !vector1 = !debuginfo.vector<8 x ![[BASIC23]]>
// CHECK-DAG: !vector2 = !debuginfo.vector<8 x ![[BASIC24]]>
// CHECK-DAG: !array1 = !debuginfo.array<4 x !vector>
// CHECK-DAG: !array2 = !debuginfo.array<4 x !vector2>
// CHECK-DAG: ![[MEMBER23:.*]] = !debuginfo.member<m22: !array>
// CHECK-DAG: ![[MEMBER24:.*]] = !debuginfo.member<m0: !ptr>
// CHECK-DAG: !ptr2 = !debuginfo.ptr<!subroutine {sizeInBits = 64, alignInBits = 64}>
// CHECK-DAG: !array3 = !debuginfo.array<3 x !array1>
// CHECK-DAG: !array4 = !debuginfo.array<5 x !array2>
// CHECK-DAG: ![[MEMBER25:.*]] = !debuginfo.member<m1: !array2>
// CHECK-DAG: !struct = !debuginfo.struct<pack(!member, ![[MEMBER1]], ![[MEMBER2]], ![[MEMBER3]], ![[MEMBER4]], ![[MEMBER5]], ![[MEMBER6]], ![[MEMBER7]], ![[MEMBER8]], ![[MEMBER9]], ![[MEMBER10]], ![[MEMBER11]], ![[MEMBER12]], ![[MEMBER13]], ![[MEMBER14]], ![[MEMBER15]], ![[MEMBER16]], ![[MEMBER17]], ![[MEMBER18]], ![[MEMBER19]], ![[MEMBER20]], ![[MEMBER21]], ![[MEMBER23]])>
// CHECK-DAG: !array5 = !debuginfo.array<2 x !array3>
// CHECK-DAG: ![[MEMBER26:.*]] = !debuginfo.member<m1: !array4>
// CHECK-DAG: !struct1 = !debuginfo.struct<struct(![[MEMBER24]], ![[MEMBER25]])>
// CHECK-DAG: ![[MEMBER27:.*]] = !debuginfo.member<m2: !struct1>
// CHECK-DAG: !struct2 = !debuginfo.struct<struct(![[MEMBER22]], ![[MEMBER26]], ![[MEMBER27]])>
// CHECK-DAG: !subroutine1 = !debuginfo.subroutine<(!array5, !ptr2, !struct, !ptr, !ptr1, ![[BASIC]], !vector1, !struct2) -> (): DW_CC_normal>

!test = !debuginfo.subroutine<(!debuginfo.unresolved<!arrayTest>,
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
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !test

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="i64:64:64", simd_bit_width=128>} {
  kgen.func @foo() {
    kgen.return loc(fused<#subprogram>["foo.mlir":10:10])
  } loc(fused<#subprogram>["foo.mlir":10:10])
}
