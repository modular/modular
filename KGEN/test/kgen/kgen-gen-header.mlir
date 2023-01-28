// RUN: kgen %s -emit -func="someKernel:f32(f32,index)" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=SCALAR

// RUN: kgen %s -emit -func="someBufferKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=BUFFER

// RUN: kgen %s -emit -func="someNDBufferKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=NDBUFFER

// RUN: kgen %s -emit -func="someMetaScalarKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=SCALARMETA

// RUN: kgen %s -emit -func="nestedParametricStruct" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=STRUCT

// RUN: kgen %s -emit -func="litNoneKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=VOID

// RUN: kgen %s -emit -func="listOneElem" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=LISTF32

// RUN: kgen %s -emit -func="oneElemStruct" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=ONESTRUCT

// RUN: kgen %s -emit -func="twoElemStruct" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=TWOSTRUCT

// The following should not generate header files at all:
// RUN: echo "" | kgen - -emit -o /dev/null
// RUN: test ! -f /dev/null.h
// RUN: echo "" | kgen - -emit -o -
// RUN: test ! -f -.h

kgen.func @someKernel(%arg1: f32, %arg2: index) -> f32 {
  kgen.return %arg1 : f32
}
// SCALAR: extern float someKernel_c(float, ssize_t);

kgen.func @someBufferKernel(%a: !pop.struct<pointer<simd<1, invalid>>, index, !kgen.dtype>) -> index {
  %size = pop.struct.get %a[1] : !pop.struct<pointer<simd<1, invalid>>, index, !kgen.dtype>
  kgen.return %size : index
}
// BUFFER: extern ssize_t someBufferKernel_c(void *, ssize_t, uint8_t);


kgen.func @someNDBufferKernel(%a: !pop.struct<pointer<simd<1, invalid>>, index, array<5, index>, !kgen.dtype>) -> index {
  %size = pop.struct.get %a[1] : !pop.struct<pointer<simd<1, invalid>>, index, array<5, index>, !kgen.dtype>
  kgen.return %size : index
}
// NDBUFFER: extern ssize_t someNDBufferKernel_c(void *, ssize_t, ssize_t[5], uint8_t);

kgen.func @someMetaScalarKernel(%arg0: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  kgen.return %arg0 : !pop.simd<1, f32>
}
// SCALARMETA: extern float someMetaScalarKernel_c(float);

lit.struct.decl @Foo<DT:dtype> {
  lit.struct.field value : !pop.scalar<DT>
}

lit.struct.decl @Bar {
  lit.struct.field a : !kgen.declref<@Foo<DT:dtype=f32>>
  lit.struct.field b : !pop.scalar<f64>
}

kgen.func @nestedParametricStruct(%a: !kgen.declref<@Bar>) {
  kgen.return
}
// STRUCT: extern void nestedParametricStruct_c(float, double)


kgen.func @litNoneKernel() -> !kgen.list<i1[0]> {
  %0 = kgen.param.constant: list<i1[0]> = <[]>
  kgen.return %0 : !kgen.list<i1[0]>
}

// VOID: extern void litNoneKernel_c();

kgen.func @listOneElem() -> !kgen.list<f32[1]> {
  %0 = kgen.param.constant: list<f32[1]> = <[1.0]>
  kgen.return %0 : !kgen.list<f32[1]>
}

// LISTF32: extern float listOneElem_c();

kgen.func @oneElemStruct(%arg0: i32) -> !pop.struct<i32> {
  %0 = kgen.param.constant: struct<i32> = <{ 0 }>
  kgen.return %0 : !pop.struct<i32>
}

// ONESTRUCT: extern int32_t oneElemStruct_c(int32_t);

kgen.func @twoElemStruct(%arg0: i32) -> !pop.struct<i32, i32> {
  %0 = kgen.param.constant: struct<i32, i32> = <{ 0, 0 }>
  kgen.return %0 : !pop.struct<i32, i32>
}

// TWOSTRUCT: extern void twoElemStruct_c(int32_t, int32_t *, int32_t *);

kgen.export @someKernel
kgen.export @someBufferKernel
kgen.export @someNDBufferKernel
kgen.export @someMetaScalarKernel
kgen.export @nestedParametricStruct
kgen.export @litNoneKernel
kgen.export @listOneElem
kgen.export @oneElemStruct
kgen.export @twoElemStruct
