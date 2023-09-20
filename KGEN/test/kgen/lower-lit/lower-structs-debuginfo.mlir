// RUN: kgen-opt %s -lower-structs -allow-unregistered-dialect -split-input-file | FileCheck %s

// Test proper handling of debug types.

// CHECK-DAG: ![[MEMBER:.*]] = !debuginfo.member<data: !pop.array<2, simd<4, f32>>>
// CHECK-DAG: ![[STRUCT:.*]] = !debuginfo.struct<SmallVector(![[MEMBER]])>
lit.struct.decl @SmallVector<N, T: type> {
  lit.struct.field data: !pop.array<N, T>
}
!structTest = !kgen.declref<@SmallVector<N = 2, T:type = !pop.simd<4, f32>>>

// CHECK: "test.types"
"test.types"() {
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
> : !debuginfo.unresolved<!pop.array<3, i32>>
#local_variable1 = #debuginfo.local_variable<
  scope = #subprogram,
  name = "bar",
  file = #file,
  line = 10,
  arg = 1
> : !debuginfo.unresolved<!pop.array<0, i32>>

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

kgen.func @foo() {
  // CHECK-DAG: %[[LIST:.*]] = kgen.param.constant: array<3, i32> = <[1, 2, 3]>

  // CHECK: debuginfo.value #local_variable = %[[LIST]]
  %values = kgen.param.constant: array<3, i32> = <[1, 2, 3]> loc(#loc)
  debuginfo.value #local_variable = %values : !pop.array<3, i32> loc(#loc)
  // CHECK-DAG: %[[EMPTY:.*]] = kgen.param.constant: array<0, i32> = <[]>
  // CHECK: debuginfo.value #local_variable1 = %[[EMPTY]]
  %empty = kgen.param.constant: array<0, i32> = <[]> loc(#loc)
  debuginfo.value #local_variable1 = %empty : !pop.array<0, i32> loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)
