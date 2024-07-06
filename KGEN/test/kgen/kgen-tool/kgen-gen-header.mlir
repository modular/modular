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

kgen.func export C @someKernel(%arg1: f32, %arg2: index) -> f32 {
  kgen.return %arg1 : f32
}
// SCALAR: extern float someKernel(float, ssize_t);

kgen.func export C @someBufferKernel(%a: !kgen.struct<(pointer<simd<1, invalid>>, index, !kgen.dtype)>) -> index {
  %size = kgen.struct.extract %a[1] : !kgen.struct<(pointer<simd<1, invalid>>, index, !kgen.dtype)>
  kgen.return %size : index
}
// BUFFER: extern ssize_t someBufferKernel(void *, ssize_t, uint8_t);


kgen.func export C @someNDBufferKernel(%a: !kgen.struct<(pointer<simd<1, invalid>>, index, array<5, index>, !kgen.dtype)>) -> index {
  %size = kgen.struct.extract %a[1] : !kgen.struct<(pointer<simd<1, invalid>>, index, array<5, index>, !kgen.dtype)>
  kgen.return %size : index
}
// NDBUFFER: extern ssize_t someNDBufferKernel(void *, ssize_t, ssize_t[5], uint8_t);

kgen.func export C @someMetaScalarKernel(%arg0: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  kgen.return %arg0 : !pop.simd<1, f32>
}
// SCALARMETA: extern float someMetaScalarKernel(float);

lit.struct.decl @Foo<DT:dtype> {
  lit.struct.field value : !pop.scalar<DT>
}

lit.struct.decl @Bar {
  lit.struct.field a : !lit.struct<@Foo<:dtype f32>>
  lit.struct.field b : !pop.scalar<f64>
}

kgen.func export C @nestedParametricStruct(%a: !lit.struct<@Bar>) {
  kgen.return
}
// STRUCT: extern void nestedParametricStruct(float, double)


kgen.func export C @litNoneKernel() -> !pop.array<0, i1> {
  %0 = kgen.param.constant: array<0, i1> = <[]>
  kgen.return %0 : !pop.array<0, i1>
}

// VOID: extern void litNoneKernel();

kgen.func export C @oneElemStruct(%arg0: i32) -> !kgen.struct<(i32)> {
  %0 = kgen.param.constant: struct<(i32)> = <{ 0 }>
  kgen.return %0 : !kgen.struct<(i32)>
}

// ONESTRUCT: extern int32_t oneElemStruct(int32_t);

kgen.func export C @twoElemStruct(%arg0: i32) -> !kgen.struct<(i32, i32)> {
  %0 = kgen.param.constant: struct<(i32, i32)> = <{ 0, 0 }>
  kgen.return %0 : !kgen.struct<(i32, i32)>
}

// TWOSTRUCT: extern void twoElemStruct(int32_t, int32_t *, int32_t *);

kgen.func export C @oneVariadic(%arg0: !kgen.variadic<f32>) -> !kgen.struct<(i32)> {
  %0 = kgen.param.constant: struct<(i32)> = <{ 0 }>
  kgen.return %0 : !kgen.struct<(i32)>
}

// ONEVARIADIC: extern int32_t oneVariadic(void *, ssize_t);

kgen.func export C @twoVariadic(%arg0: !kgen.variadic<!kgen.struct<(i32, i32)>>,
                                %arg1: !kgen.variadic<i32>) -> !kgen.struct<(i32)> {
  %0 = kgen.param.constant: struct<(i32)> = <{ 0 }>
  kgen.return %0 : !kgen.struct<(i32)>
}

// TWOVARIADIC: extern int32_t twoVariadic(void *, ssize_t, void *, ssize_t);
