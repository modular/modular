// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' -mlir-print-debuginfo %s | FileCheck %s

// Test proper handling of debug types.

!scalarTest = !pop.scalar<bool>
!simdTest = !pop.simd<8, ui32>

// CHECK-DAG: !basic = !debuginfo.basic<bool {sizeInBits = 8, alignInBits = 8, encoding = DW_ATE_boolean}>
// CHECK-DAG: !basic1 = !debuginfo.basic<ui32 {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_unsigned}>
// CHECK-DAG: !vector = !debuginfo.vector<8 x !basic1>
// CHECK-DAG: !debuginfo.subroutine<(!basic, !vector) -> (): DW_CC_normal>
!test = !debuginfo.subroutine<(!debuginfo.unresolved<!scalarTest>, !debuginfo.unresolved<!simdTest>) -> (): DW_CC_normal>

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

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_size=8, simd_bit_width=128>} {
  kgen.func @foo() {
    kgen.return
  } loc(fused<#subprogram>["foo.mlir":10:10])
}
