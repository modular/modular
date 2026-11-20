// RUN: kgen-opt -lower-kgen-to-llvm -mlir-print-debuginfo %s | FileCheck %s

// Test proper handling of base types.
!f32_ptr = !kgen.pointer<!kgen.scalar<f32>>
!f64 = !kgen.scalar<f64>
!f16 = !kgen.scalar<f16>
!bf16 = !kgen.scalar<bf16>
!f8e5m2 = !kgen.scalar<f8e5m2>

// CHECK-DAG: ![[F32:.*]] = !debuginfo.basic<float {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_float}>
// CHECK-DAG: ![[F32_VEC:.*]] = !debuginfo.vector<1 x ![[F32]] {name = "!kgen.scalar<f32>"}>
// CHECK-DAG: ![[F32_PTR:.*]] = !debuginfo.ptr<![[F32_VEC]] {sizeInBits = 64, alignInBits = 64, addressSpace = 0}>

// CHECK-DAG: ![[F64:.*]] = !debuginfo.basic<double {sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_float}>
// CHECK-DAG: ![[F64_VEC:.*]] = !debuginfo.vector<1 x ![[F64]] {name = "!kgen.scalar<f64>"}>

// CHECK-DAG: ![[UI16:.*]] = !debuginfo.basic<uint16_t {sizeInBits = 16, alignInBits = 16, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[MEMBER_X:.*]] = !debuginfo.member<__x: ![[UI16]]>
// CHECK-DAG: ![[F16:.*]] = !debuginfo.struct<__half(![[MEMBER_X]])>
// CHECK-DAG: ![[F16_VEC:.*]] = !debuginfo.vector<1 x ![[F16]] {name = "!kgen.scalar<f16>"}>

// CHECK-DAG: ![[BF16:.*]] = !debuginfo.struct<__nv_bfloat16(![[MEMBER_X]])>
// CHECK-DAG: ![[BF16_VEC:.*]] = !debuginfo.vector<1 x ![[BF16]] {name = "!kgen.scalar<bf16>"}>

// CHECK-DAG: ![[UI8:.*]] = !debuginfo.basic<uint8_t {sizeInBits = 8, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[MEMBER_X8:.*]] = !debuginfo.member<__x: ![[UI8]]>
// CHECK-DAG: ![[F8e5m2:.*]] = !debuginfo.struct<__nv_fp8_e5m2(![[MEMBER_X8]])>
// CHECK-DAG: ![[F8e5m2_VEC:.*]] = !debuginfo.vector<1 x ![[F8e5m2]] {name = "!kgen.scalar<f8e5m2>"}>

!ui8 = !kgen.scalar<ui8>
!si32 = !kgen.scalar<si32>

// CHECK-DAG: ![[UI8_VEC:.*]] = !debuginfo.vector<1 x ![[UI8]] {name = "!kgen.scalar<ui8>"}>

// CHECK-DAG: ![[SI32:.*]] = !debuginfo.basic<int32_t {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[SI32_VEC:.*]] = !debuginfo.vector<1 x ![[SI32]] {name = "!kgen.scalar<si32>"}>

// CHECK: !debuginfo.subroutine<(
// CHECK-SAME: ![[F32_PTR]]
// CHECK-SAME: ![[F64_VEC]]
// CHECK-SAME: ![[F16_VEC]]
// CHECK-SAME: ![[BF16_VEC]]
// CHECK-SAME: ![[F8e5m2_VEC]]
// CHECK-SAME: ![[UI8_VEC]]
// CHECK-SAME: ![[SI32_VEC]]

!test = !debuginfo.subroutine<(
  !debuginfo.unresolved<!f32_ptr>,
  !debuginfo.unresolved<!f64>,
  !debuginfo.unresolved<!f16>,
  !debuginfo.unresolved<!bf16>,
  !debuginfo.unresolved<!f8e5m2>,
  !debuginfo.unresolved<!ui8>,
  !debuginfo.unresolved<!si32>
) -> (): DW_CC_normal>

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !test

#loc = loc(fused<#subprogram>["test.mlir":10:10])

module attributes {M.target_info = #M.target<triple="nvptx64-nvidia-cuda", arch="nvptx64", features="", data_layout="", simd_bit_width=64>} {
  kgen.func @foo() {
    hlcf.return loc(#loc)
  } loc(#loc)
}
