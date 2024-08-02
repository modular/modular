// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' -mlir-print-debuginfo %s | FileCheck %s

// Test proper handling of debug types.
!pointerTest = !kgen.pointer<index>
!voidPointerTest = !kgen.pointer<none>
!structTest = !kgen.struct<(index, struct<(index)>)>
!variantTest = !pop.union<index, index>
!signatureTest = !kgen.signature<(index) -> index>

// CHECK-DAG: ![[INDEX:.*]] = !debuginfo.basic<index {sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_signed}>

// CHECK-DAG: ![[MEMBER0:.*]] = !debuginfo.member<m0: ![[INDEX]]>

// CHECK-DAG: ![[PTR:.*]] = !debuginfo.ptr<![[INDEX]] {sizeInBits = 64, alignInBits = 64, addressSpace = 0}>

// CHECK-DAG: ![[NONE:.*]] = !debuginfo.struct<"!kgen.none"()>
// CHECK-DAG: ![[VOID_PTR:.*]] = !debuginfo.ptr<![[NONE]] {sizeInBits = 64, alignInBits = 64, addressSpace = 0}>

// CHECK-DAG: ![[INNER_STRUCT:.*]] = !debuginfo.struct<"!kgen.struct<(index)>"(![[MEMBER0]])>
// CHECK-DAG: ![[STRUCT_MEMBER:.*]] = !debuginfo.member<m1: ![[INNER_STRUCT]]>
// CHECK-DAG: ![[STRUCT:.*]] = !debuginfo.struct<"!kgen.struct<(index, struct<(index)>)>"(![[MEMBER0]], ![[STRUCT_MEMBER]])>

// CHECK-DAG: ![[VARIANT0:.*]] = !debuginfo.member<v0: ![[INDEX]]>
// CHECK-DAG: ![[VARIANT1:.*]] = !debuginfo.member<v1: ![[INDEX]]>
// CHECK-DAG: ![[VARIANT:.*]] = !debuginfo.variant<""(![[VARIANT0]], ![[VARIANT1]]) {sizeInBits = 64, alignInBits = 64}>

// CHECK-DAG: ![[SUBROUTINE:.*]] = !debuginfo.subroutine<(![[INDEX]]) -> (![[INDEX]]): DW_CC_normal>
// CHECK-DAG: ![[SIGNATURE:.*]] = !debuginfo.ptr<![[SUBROUTINE]] {sizeInBits = 64, alignInBits = 64}>

// CHECK-DAG: ![[CHAR:.*]] = !debuginfo.basic<kgen.dtype.si8 {sizeInBits = 8, alignInBits = 8, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[SIZE_MEMBER:.*]] = !debuginfo.member<size: ![[INDEX]]>
// CHECK-DAG: ![[CHAR_PTR:.*]] = !debuginfo.ptr<![[CHAR]] {sizeInBits = 64, alignInBits = 64}>
// CHECK-DAG: ![[DATA_MEMBER:.*]] = !debuginfo.member<data: ![[CHAR_PTR]]>
// CHECK-DAG: ![[STRING:.*]] = !debuginfo.struct<"!kgen.string"(![[DATA_MEMBER]], ![[SIZE_MEMBER]])>

// CHECK-DAG: !debuginfo.subroutine<(![[PTR]], ![[VOID_PTR]], ![[STRUCT]], ![[VARIANT]], ![[SIGNATURE]], ![[STRING]], ![[NONE]]) -> (): DW_CC_normal>

!test = !debuginfo.subroutine<(
  !debuginfo.unresolved<!pointerTest>,
  !debuginfo.unresolved<!voidPointerTest>,
  !debuginfo.unresolved<!structTest>,
  !debuginfo.unresolved<!variantTest>,
  !debuginfo.unresolved<!signatureTest>,
  !debuginfo.unresolved<!kgen.string>,
  !debuginfo.unresolved<!kgen.none>
) -> (): DW_CC_normal>

#subprogram = #debuginfo.subprogram<name = <"foo">> : !test

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="i64:64:64", simd_bit_width=128>} {
  kgen.func @foo() {
    kgen.return loc(fused<#subprogram>["foo.mlir":10:10])
  } loc(fused<#subprogram>["foo.mlir":10:10])
}
