// RUN: support-dialect-opt %s -convert-debuginfo-to-llvm -mlir-print-debuginfo | FileCheck %s

// CHECK-DAG: #[[BASIC:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
!f32Type = !debuginfo.basic<kgen.dtype.f32 { sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_float }>

// CHECK-DAG: #[[DOUBLE:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "double", sizeInBits = 64, encoding = DW_ATE_float>
!f64Type = !debuginfo.basic<kgen.dtype.f64 { sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_float }>

// CHECK-DAG: #[[ARRAY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "", baseType = #[[BASIC]], sizeInBits = 320, elements = #llvm.di_subrange<count = 10 : i64>>
!arrayType = !debuginfo.array<10 x !f32Type>

// CHECK-DAG: #[[PTR:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #[[BASIC]], sizeInBits = 64, alignInBits = 64>
!pointerType = !debuginfo.ptr<!f32Type {sizeInBits = 64, alignInBits = 64}>

// CHECK-DAG: #[[MEMBER1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "first", baseType = #[[BASIC]], sizeInBits = 32, alignInBits = 32>
!memberType1 = !debuginfo.member<first: !f32Type>

// CHECK-DAG: #[[MEMBER2:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "second", baseType = {{.*}}, sizeInBits = 64, alignInBits = 64, offsetInBits = 64>
!memberType2 = !debuginfo.member<second: !pointerType>

// CHECK-DAG: #[[STRUCT:.*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "Foo", sizeInBits = 128, alignInBits = 64, elements = #[[MEMBER1]], #[[MEMBER2]]>
!structType = !debuginfo.struct<"Foo"(!memberType1, !memberType2)>

// CHECK-DAG: #[[SUBROUTINE:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[BASIC]], #[[DOUBLE]], #[[ARRAY]], #[[STRUCT]]>
!subroutineType = !debuginfo.subroutine<(!f64Type, !arrayType, !structType) -> (!f32Type): DW_CC_normal>

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = <"foo">,
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !subroutineType

module attributes {M.target_info = #M.target<triple="nvptx64-nvidia-cuda", arch="nvptx64", features="", data_layout="", simd_bit_width=64>} {

func.func private @foo() loc(fused<#subprogram>["test.mlir":10:10])

}
