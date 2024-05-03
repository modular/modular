// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' -mlir-print-debuginfo %s | FileCheck %s

// Test proper handling of debug types.
!arrayTest = !pop.array<2, array<3, array<4, simd<8, bool>>>>
!coroutineTest = !co.routine<() -> (index, index)>
!packTest = !kgen.pack<[index, !pop.array<5, f32>]>
!pointerTest = !kgen.pointer<scalar<bool>>
!voidPointerTest = !kgen.pointer<scalar<invalid>>
!scalarTest = !pop.scalar<bool>
!simdTest = !pop.simd<8, ui32>
!structTest = !kgen.struct<(scalar<bool>, array<5, array<4, simd<8, si32>>>, struct<(pointer<scalar<bool>>, array<4, simd<8, si32>>)>)>
!variantTest = !kgen.variant<!simdTest, !pointerTest>
!signatureTest = !kgen.signature<(!simdTest) -> !pointerTest>

// CHECK-DAG: ![[BASIC:.*]] = !debuginfo.basic<kgen.dtype.bool {sizeInBits = 8, alignInBits = 8, encoding = DW_ATE_boolean}>
// CHECK-DAG: ![[BASIC1:.*]] = !debuginfo.basic<index {sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[UNSPECIFIED:.*]] = !debuginfo.unspecified<"kgen.dtype.invalid">
// CHECK-DAG: ![[ARRAY:.*]] = !debuginfo.array<5 x !{{.*}}>
// CHECK-DAG: ![[MEMBER:.*]] = !debuginfo.member<m0: ![[BASIC1]]>
// CHECK-DAG: ![[MEMBER1:.*]] = !debuginfo.member<m1: !{{.*}}>
// CHECK-DAG: ![[VECTOR_BASIC:.*]] = !debuginfo.vector<1 x ![[BASIC]] {name = "!pop.scalar<bool>"}>
// CHECK-DAG: ![[PTR:.*]] = !debuginfo.ptr<![[VECTOR_BASIC]] {sizeInBits = 64, alignInBits = 64}>
// CHECK-DAG: ![[VECTOR_UNSPECIFIED:.*]] = !debuginfo.vector<1 x ![[UNSPECIFIED]] {name = "!pop.scalar<invalid>"}>
// CHECK-DAG: ![[PTR1:.*]] = !debuginfo.ptr<![[VECTOR_UNSPECIFIED]] {sizeInBits = 64, alignInBits = 64}>
// CHECK-DAG: ![[SUBROUTINE:.*]] = !debuginfo.subroutine<() -> (![[BASIC1]], ![[BASIC1]]): DW_CC_normal>
// CHECK-DAG: ![[VECTOR:.*]] = !debuginfo.vector<8 x ![[BASIC]] {name = "!pop.simd<8, bool>"}>
// CHECK-DAG: ![[VECTOR1:.*]] = !debuginfo.vector<8 x !{{.*}} {name = "!pop.simd<8, ui32>"}>
// CHECK-DAG: ![[VECTOR2:.*]] = !debuginfo.vector<8 x !{{.*}} {name = "!pop.simd<8, si32>"}>
// CHECK-DAG: ![[ARRAY1:.*]] = !debuginfo.array<4 x ![[VECTOR]]>
// CHECK-DAG: ![[ARRAY2:.*]] = !debuginfo.array<4 x ![[VECTOR2]]>
// CHECK-DAG: ![[MEMBER24:.*]] = !debuginfo.member<m0: ![[PTR]]>
// CHECK-DAG: ![[MEMBER22:.*]] = !debuginfo.member<m0: ![[VECTOR_BASIC]]>
// CHECK-DAG: ![[PTR2:.*]] = !debuginfo.ptr<![[SUBROUTINE]] {sizeInBits = 64, alignInBits = 64}>
// CHECK-DAG: ![[ARRAY3:.*]] = !debuginfo.array<3 x ![[ARRAY1]]>
// CHECK-DAG: ![[ARRAY4:.*]] = !debuginfo.array<5 x ![[ARRAY2]]>
// CHECK-DAG: ![[MEMBER25:.*]] = !debuginfo.member<m1: ![[ARRAY2]]>
// CHECK-DAG: ![[STRUCT:.*]] = !debuginfo.struct<"!kgen.pack<[index, array<5, f32>]>"(![[MEMBER]], ![[MEMBER1]])>
// CHECK-DAG: ![[ARRAY5:.*]] = !debuginfo.array<2 x ![[ARRAY3]]>
// CHECK-DAG: ![[MEMBER26:.*]] = !debuginfo.member<m1: ![[ARRAY4]]>
// CHECK-DAG: ![[STRUCT1:.*]] = !debuginfo.struct<"!kgen.struct<(pointer<scalar<bool>>, array<4, simd<8, si32>>)>"(![[MEMBER24]], ![[MEMBER25]])>
// CHECK-DAG: ![[MEMBER27:.*]] = !debuginfo.member<m2: ![[STRUCT1]]>
// CHECK-DAG: ![[STRUCT2:.*]] = !debuginfo.struct<"!kgen.struct<(scalar<bool>, array<5, array<4, simd<8, si32>>>, struct<(pointer<scalar<bool>>, array<4, simd<8, si32>>)>)>"(![[MEMBER22]], ![[MEMBER26]], ![[MEMBER27]])>

// CHECK-DAG: ![[BASIC_I1:.*]] = !debuginfo.basic<i1
// CHECK-DAG: ![[DISCR:.*]] = !debuginfo.member<discr: ![[BASIC_I1]]>
// CHECK-DAG: ![[VARIANT1:.*]] = !debuginfo.member<v0: ![[VECTOR1]]>
// CHECK-DAG: ![[VARIANT2:.*]] = !debuginfo.member<v1: ![[PTR]]>
// CHECK-DAG: ![[VARIANT_PART:.*]] = !debuginfo.variant<""(![[VARIANT1]], ![[VARIANT2]]), ![[DISCR]] {sizeInBits = 256, alignInBits = 64}>
// CHECK-DAG: ![[VARIANT_MEMBER:.*]] = !debuginfo.member<"": ![[VARIANT_PART]]>
// CHECK-DAG: ![[VARIANT_STRUCT:.*]] = !debuginfo.struct<"!kgen.variant<simd<8, ui32>, pointer<scalar<bool>>>"(![[VARIANT_MEMBER]], ![[DISCR]])>

// CHECK-DAG: ![[SUBROUTINE:.*]] = !debuginfo.subroutine<(![[VECTOR1]]) -> (![[PTR]]): DW_CC_normal>
// CHECK-DAG: ![[SUBROUTINE_PTR:.*]] = !debuginfo.ptr<![[SUBROUTINE]] {sizeInBits = 64, alignInBits = 64}>

// CHECK-DAG: ![[STRING_DATA:.*]] = !debuginfo.member<data: !ptr
// CHECK-DAG: ![[STRING_SIZE:.*]] = !debuginfo.member<size: !basic
// CHECK-DAG: ![[STRING:.*]] = !debuginfo.struct<"!kgen.string"(![[STRING_DATA]], ![[STRING_SIZE]])>

// CHECK-DAG: ![[NONE:.*]] = !debuginfo.struct<"!kgen.none"()>

// CHECK-DAG: !debuginfo.subroutine<(![[ARRAY5]], ![[PTR2]], ![[STRUCT]], ![[PTR]], ![[PTR1]], ![[VECTOR_BASIC]], ![[VECTOR1]], ![[STRUCT2]], ![[VARIANT_STRUCT]], ![[SUBROUTINE_PTR]], ![[STRING]], ![[NONE]]) -> (): DW_CC_normal>

!test = !debuginfo.subroutine<(
  !debuginfo.unresolved<!arrayTest>,
  !debuginfo.unresolved<!coroutineTest>,
  !debuginfo.unresolved<!packTest>,
  !debuginfo.unresolved<!pointerTest>,
  !debuginfo.unresolved<!voidPointerTest>,
  !debuginfo.unresolved<!scalarTest>,
  !debuginfo.unresolved<!simdTest>,
  !debuginfo.unresolved<!structTest>,
  !debuginfo.unresolved<!variantTest>,
  !debuginfo.unresolved<!signatureTest>,
  !debuginfo.unresolved<!kgen.string>,
  !debuginfo.unresolved<!kgen.none>
) -> (): DW_CC_normal>

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  name = <"foo">,
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
