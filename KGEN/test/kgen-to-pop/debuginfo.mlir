// RUN: kgen-opt %s -lower-kgen-to-pop -allow-unregistered-dialect -split-input-file | FileCheck %s

// Test proper handling of debug types.

// CHECK-DAG: ![[LIST_ELEMENT_TYPE:.*]] = !debuginfo.unresolved<index>
// CHECK-DAG: ![[LIST:.*]] = !debuginfo.array<2 x ![[LIST_ELEMENT_TYPE]]>
!listTest = !kgen.list<index[2]>

// CHECK-DAG: ![[MEMBER_TYPE:.*]] = !debuginfo.unresolved<!pop.array<2, simd<4, f32>>>
// CHECK-DAG: ![[MEMBER:.*]] = !debuginfo.member<data: ![[MEMBER_TYPE]]>
// CHECK-DAG: ![[STRUCT:.*]] = !debuginfo.struct<SmallVector(![[MEMBER]])>
kgen.struct.decl @SmallVector<N, T: type> {
  kgen.struct.field data: !pop.array<N, T>
}
!structTest = !kgen.declref<@SmallVector<N = 2, T:type = !pop.simd<4, f32>>>

// CHECK: "test.types"
"test.types"() {
  // CHECK-SAME: listType = ![[LIST]]
  listType = !debuginfo.unresolved<!listTest>,
  // CHECK-SAME: structType = ![[STRUCT]]
  structType = !debuginfo.unresolved<!structTest>
} : () -> ()

// -----

// Test proper handling of debuginfo operations.

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
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1
> : !debuginfo.unresolved<!kgen.list<i32[3]>>

kgen.func @foo() {
  // CHECK:  %[[LIST:.*]] = kgen.param.constant: !pop.array<3, i32> = <#pop.array<1, 2, 3>>
  // CHECK:  debuginfo.value #local_variable = %[[LIST]] : !pop.array<3, i32>
  %values = kgen.param.constant: list<i32[3]> = <[1, 2, 3]>
  debuginfo.value #local_variable = %values : !kgen.list<i32[3]>
  kgen.return
}
