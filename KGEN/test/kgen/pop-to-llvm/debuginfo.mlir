// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' -mlir-print-debuginfo %s | FileCheck %s

// Test proper handling of debug types.
!arrayTest = !pop.array<2, array<3, array<4, simd<8, bool>>>>
!scalarTest = !pop.scalar<bool>
!simdTest = !pop.simd<8, ui32>
!addressTest = !pop.scalar<address>
!invalidTest = !pop.scalar<invalid>

// CHECK-DAG: ![[BASIC:.*]] = !debuginfo.basic<kgen.dtype.bool {sizeInBits = 8, alignInBits = 8, encoding = DW_ATE_boolean}>
// CHECK-DAG: ![[UI32:.*]] = !debuginfo.basic<kgen.dtype.ui32 {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[ADDRESS:.*]] = !debuginfo.basic<kgen.dtype.address {sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[UNSPECIFIED:.*]] = !debuginfo.unspecified<"kgen.dtype.invalid">
// CHECK-DAG: ![[VECTOR_BASIC:.*]] = !debuginfo.vector<1 x ![[BASIC]] {name = "!pop.scalar<bool>"}>
// CHECK-DAG: ![[VECTOR_UNSPECIFIED:.*]] = !debuginfo.vector<1 x ![[UNSPECIFIED]] {name = "!pop.scalar<invalid>"}>
// CHECK-DAG: ![[VECTOR_ADDR:.*]] = !debuginfo.vector<1 x ![[ADDRESS]] {name = "!pop.scalar<address>"}>
// CHECK-DAG: ![[VECTOR:.*]] = !debuginfo.vector<8 x ![[BASIC]] {name = "!pop.simd<8, bool>"}>
// CHECK-DAG: ![[VECTOR1:.*]] = !debuginfo.vector<8 x ![[UI32]] {name = "!pop.simd<8, ui32>"}>
// CHECK-DAG: ![[ARRAY1:.*]] = !debuginfo.array<4 x ![[VECTOR]]>
// CHECK-DAG: ![[ARRAY3:.*]] = !debuginfo.array<3 x ![[ARRAY1]]>
// CHECK-DAG: ![[ARRAY5:.*]] = !debuginfo.array<2 x ![[ARRAY3]]>

// CHECK-DAG: !debuginfo.subroutine<(![[ARRAY5]], ![[VECTOR_BASIC]], ![[VECTOR1]], ![[VECTOR_ADDR]], ![[VECTOR_UNSPECIFIED]]) -> (): DW_CC_normal>

!test = !debuginfo.subroutine<(
  !debuginfo.unresolved<!arrayTest>,
  !debuginfo.unresolved<!scalarTest>,
  !debuginfo.unresolved<!simdTest>,
  !debuginfo.unresolved<!addressTest>,
  !debuginfo.unresolved<!invalidTest>
) -> (): DW_CC_normal>

#subprogram = #debuginfo.subprogram<name = <"foo">> : !test

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="i64:64:64", simd_bit_width=128>} {
  kgen.func @foo() {
    kgen.return loc(fused<#subprogram>["foo.mlir":10:10])
  } loc(fused<#subprogram>["foo.mlir":10:10])
}
