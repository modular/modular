// RUN: kgen-opt %s -lower-lit-types -allow-unregistered-dialect -split-input-file | FileCheck %s

// Test proper handling of debug types.

// Single-field structs are flattened in debuginfo (until #23914).
// CHECK-DAG: ![[FIELD:.*]] = !debuginfo.unresolved<!pop.array<2, simd<4, f32>>>
lit.struct.decl @SmallVector<N, T: type> register_passable {
  lit.struct.field data: !pop.array<N, T>
}
!structTest = !kgen.declref<@SmallVector<2, :type !pop.simd<4, f32>>>

// CHECK-DAG: ![[MEMBER_A:.*]] = !debuginfo.member<a: !kgen.paramref<Int>>
// CHECK-DAG: ![[MEMBER_B:.*]] = !debuginfo.member<b: !pop.simd<4, f32>>
// CHECK-DAG: ![[COMPLEX_STRUCT:.*]] = !debuginfo.struct<"ComplexStruct[A=Int, B=simd<4, f32>]"(![[MEMBER_A]], ![[MEMBER_B]])>
lit.struct.decl @ComplexStruct<A: type, B: type> {
  lit.struct.field a: !kgen.paramref<A>
  lit.struct.field b: !kgen.paramref<B>
}
!structTestComplex = !kgen.declref<@ComplexStruct<Int, :type !pop.simd<4, f32>>>

// CHECK: ![[COMPLEX_STRUCT_REF:.*]] = !debuginfo.ti.ptr<![[COMPLEX_STRUCT]]>
!structTestComplexRef = !lit.ref<!structTestComplex, *"`mystruct">

// CHECK: "test.types"
"test.types"() {
  // CHECK-SAME: structType = ![[FIELD]]
  structType = !debuginfo.unresolved<!structTest>,
  // CHECK-SAME: structTypeComplex = ![[COMPLEX_STRUCT]]
  structTypeComplex = !debuginfo.unresolved<!structTestComplex>,
  // CHECK-SAME: structTypeComplexRef = ![[COMPLEX_STRUCT_REF]]
  structTypeComplexRef = !debuginfo.unresolved<!structTestComplexRef>
} : () -> ()

// -----

// Test proper handling of debuginfo operations.

#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<!pop.array<3, i32>>
#local_variable1 = #debuginfo.local_variable<scope = #subprogram, name = "bar"> : !debuginfo.unresolved<!pop.array<0, i32>>

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
