// RUN: kgen %s -emit-header -func="someKernel:f32(f32,index)" | FileCheck %s --check-prefixes=SCALAR
// RUN: kgen %s -emit-header -func="someBufferKernel" | FileCheck %s --check-prefixes=BUFFER
// RUN: kgen %s -emit-header -func="someNDBufferKernel" | FileCheck %s --check-prefixes=NDBUFFER
// RUN: kgen %s -emit-header -func="someMetaScalarKernel" | FileCheck %s --check-prefixes=SCALARMETA
// RUN: kgen %s -emit-header -func="nestedParametricStruct" | FileCheck %s --check-prefixes=STRUCT
// RUN: kgen %s -emit-header -func="litNoneKernel" | FileCheck %s --check-prefixes=VOID
// RUN: kgen %s -emit-header -func="oneElemStruct" | FileCheck %s --check-prefixes=ONESTRUCT
// RUN: kgen %s -emit-header -func="twoElemStruct" | FileCheck %s --check-prefixes=TWOSTRUCT
// RUN: kgen %s -emit-header -func="oneVariadic" | FileCheck %s --check-prefixes=ONEVARIADIC
// RUN: kgen %s -emit-header -func="twoVariadic" | FileCheck %s --check-prefixes=TWOVARIADIC

kgen.func @someKernel(%arg1: f32, %arg2: index) -> f32 {
  kgen.return %arg1 : f32
}
// SCALAR: extern float someKernel(float, ssize_t);

kgen.func @someBufferKernel(%a: !pop.struct<pointer<simd<1, invalid>>, index, !kgen.dtype>) -> index {
  %size = pop.struct.extract %a[1] : !pop.struct<pointer<simd<1, invalid>>, index, !kgen.dtype>
  kgen.return %size : index
}
// BUFFER: extern ssize_t someBufferKernel(void *, ssize_t, uint8_t);


kgen.func @someNDBufferKernel(%a: !pop.struct<pointer<simd<1, invalid>>, index, array<5, index>, !kgen.dtype>) -> index {
  %size = pop.struct.extract %a[1] : !pop.struct<pointer<simd<1, invalid>>, index, array<5, index>, !kgen.dtype>
  kgen.return %size : index
}
// NDBUFFER: extern ssize_t someNDBufferKernel(void *, ssize_t, ssize_t[5], uint8_t);

kgen.func @someMetaScalarKernel(%arg0: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  kgen.return %arg0 : !pop.simd<1, f32>
}
// SCALARMETA: extern float someMetaScalarKernel(float);

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
// STRUCT: extern void nestedParametricStruct(float, double)


kgen.func @litNoneKernel() -> !pop.array<0, i1> {
  %0 = kgen.param.constant: array<0, i1> = <[]>
  kgen.return %0 : !pop.array<0, i1>
}

// VOID: extern void litNoneKernel();

kgen.func @oneElemStruct(%arg0: i32) -> !pop.struct<i32> {
  %0 = kgen.param.constant: struct<i32> = <{ 0 }>
  kgen.return %0 : !pop.struct<i32>
}

// ONESTRUCT: extern int32_t oneElemStruct(int32_t);

kgen.func @twoElemStruct(%arg0: i32) -> !pop.struct<i32, i32> {
  %0 = kgen.param.constant: struct<i32, i32> = <{ 0, 0 }>
  kgen.return %0 : !pop.struct<i32, i32>
}

// TWOSTRUCT: extern void twoElemStruct(int32_t, int32_t *, int32_t *);

kgen.func @oneVariadic(%arg0: !kgen.variadic<f32>) -> !pop.struct<i32> {
  %0 = kgen.param.constant: struct<i32> = <{ 0 }>
  kgen.return %0 : !pop.struct<i32>
}

// ONEVARIADIC: extern int32_t oneVariadic(void *, ssize_t);

kgen.func @twoVariadic(%arg0: !kgen.variadic<!pop.struct<i32, i32>>,
                       %arg1: !kgen.variadic<i32>) -> !pop.struct<i32> {
  %0 = kgen.param.constant: struct<i32> = <{ 0 }>
  kgen.return %0 : !pop.struct<i32>
}

// TWOVARIADIC: extern int32_t twoVariadic(void *, ssize_t, void *, ssize_t);

kgen.export @someKernel to C
kgen.export @someBufferKernel to C
kgen.export @someNDBufferKernel to C
kgen.export @someMetaScalarKernel to C
kgen.export @nestedParametricStruct to C
kgen.export @litNoneKernel to C
kgen.export @oneElemStruct to C
kgen.export @twoElemStruct to C
kgen.export @oneVariadic to C
kgen.export @twoVariadic to C
