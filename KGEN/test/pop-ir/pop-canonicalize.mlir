// RUN: kgen-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @abs
kgen.func @abs() -> (!pop.simd<2, si8>, !pop.simd<2, f32>) {
  // CHECK-DAG: <1, 1>
  // CHECK-DAG: <"1.25", "1.25">
  %0 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<-1, 1>>
  %1 = kgen.param.constant: !pop.simd<2, f32> = <#pop.simd<"-1.25", "1.25">>
  %2 = pop.abs %0 : !pop.simd<2, si8>
  %3 = pop.abs %1 : !pop.simd<2, f32>
  kgen.return %2, %3 : !pop.simd<2, si8>, !pop.simd<2, f32>
}

// CHECK-LABEL: @neg
kgen.func @neg() -> (!pop.simd<2, si8>, !pop.simd<2, f32>) {
  // CHECK-DAG: <1, -1>
  // CHECK-DAG: <"1.25", "-1.25">
  %0 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<-1, 1>>
  %1 = kgen.param.constant: !pop.simd<2, f32> = <#pop.simd<"-1.25", "1.25">>
  %2 = pop.neg %0 : !pop.simd<2, si8>
  %3 = pop.neg %1 : !pop.simd<2, f32>
  kgen.return %2, %3 : !pop.simd<2, si8>, !pop.simd<2, f32>
}

// CHECK-LABEL: @add
kgen.func @add() -> (!pop.scalar<si8>, !pop.scalar<f32>) {
  // CHECK-DAG: <4>
  // CHECK-DAG: <"-2.5">
  %0 = kgen.param.constant: !pop.scalar<si8> = <#pop.simd<2>>
  %1 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"-1.25">>
  %2 = pop.add %0, %0 : !pop.scalar<si8>
  %3 = pop.add %1, %1 : !pop.scalar<f32>
  kgen.return %2, %3 : !pop.scalar<si8>, !pop.scalar<f32>
}

// CHECK-LABEL: @sub
kgen.func @sub() -> (!pop.scalar<si8>, !pop.scalar<f32>) {
  // CHECK-DAG: <-2>
  // CHECK-DAG: <"-1.25">
  %0 = kgen.param.constant: !pop.scalar<si8> = <#pop.simd<2>>
  %1 = kgen.param.constant: !pop.scalar<si8> = <#pop.simd<4>>
  %2 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1.25">>
  %3 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2.5">>
  %4 = pop.sub %0, %1 : !pop.scalar<si8>
  %5 = pop.sub %2, %3 : !pop.scalar<f32>
  kgen.return %4, %5 : !pop.scalar<si8>, !pop.scalar<f32>
}

// CHECK-LABEL: @mul
kgen.func @mul() -> (!pop.scalar<si8>, !pop.scalar<f32>) {
  // CHECK-DAG: <4>
  // CHECK-DAG: <"6.25">
  %0 = kgen.param.constant: !pop.scalar<si8> = <#pop.simd<2>>
  %1 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2.5">>
  %2 = pop.mul %0, %0 : !pop.scalar<si8>
  %3 = pop.mul %1, %1 : !pop.scalar<f32>
  kgen.return %2, %3 : !pop.scalar<si8>, !pop.scalar<f32>
}

// CHECK-LABEL: @div
kgen.func @div() -> (!pop.scalar<si4>, !pop.scalar<ui4>, !pop.scalar<f32>) {
  // CHECK-DAG: <si4> = <#pop.simd<-3>
  // CHECK-DAG: <ui4> = <#pop.simd<0>>
  // CHECK-DAG: <"1.25">
  %0 = kgen.param.constant: !pop.scalar<si4> = <#pop.simd<7>>
  %1 = kgen.param.constant: !pop.scalar<si4> = <#pop.simd<-2>>
  %2 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<7>>
  %3 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<-2>>
  %4 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2.5">>
  %5 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2">>
  %6 = pop.div %0, %1 : !pop.scalar<si4>
  %7 = pop.div %2, %3 : !pop.scalar<ui4>
  %8 = pop.div %4, %5 : !pop.scalar<f32>
  kgen.return %6, %7, %8 : !pop.scalar<si4>, !pop.scalar<ui4>, !pop.scalar<f32>
}

// CHECK-LABEL: @div_zero
kgen.func @div_zero() -> (!pop.scalar<si4>, !pop.scalar<f32>) {
  %0 = kgen.param.constant: !pop.scalar<si4> = <#pop.simd<0>>
  %1 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"0">>
  // CHECK: pop.div
  %2 = pop.div %0, %0 : !pop.scalar<si4>
  // CHECK: pop.div
  %3 = pop.div %1, %1 : !pop.scalar<f32>
  kgen.return %2, %3 : !pop.scalar<si4>, !pop.scalar<f32>
}

// CHECK-LABEL: @rem
kgen.func @rem() -> (!pop.scalar<si4>, !pop.scalar<ui4>, !pop.scalar<f32>) {
  // CHECK-DAG: <si4> = <#pop.simd<1>
  // CHECK-DAG: <ui4> = <#pop.simd<7>>
  // CHECK-DAG: <"0.5">
  %0 = kgen.param.constant: !pop.scalar<si4> = <#pop.simd<7>>
  %1 = kgen.param.constant: !pop.scalar<si4> = <#pop.simd<-2>>
  %2 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<7>>
  %3 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<-2>>
  %4 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2.5">>
  %5 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2">>
  %6 = pop.rem %0, %1 : !pop.scalar<si4>
  %7 = pop.rem %2, %3 : !pop.scalar<ui4>
  %8 = pop.rem %4, %5 : !pop.scalar<f32>
  kgen.return %6, %7, %8 : !pop.scalar<si4>, !pop.scalar<ui4>, !pop.scalar<f32>
}

// CHECK-LABEL: @max
kgen.func @max() -> (!pop.scalar<si4>, !pop.scalar<f32>) {
  // CHECK-DAG: <-1>
  // CHECK-DAG: <"2">
  %0 = kgen.param.constant: !pop.scalar<si4> = <#pop.simd<-2>>
  %1 = kgen.param.constant: !pop.scalar<si4> = <#pop.simd<-1>>
  %2 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1.25">>
  %3 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2">>
  %4 = pop.max %0, %1 : !pop.scalar<si4>
  %5 = pop.max %2, %3 : !pop.scalar<f32>
  kgen.return %4, %5 : !pop.scalar<si4>, !pop.scalar<f32>
}

// CHECK-LABEL: @min
kgen.func @min() -> (!pop.scalar<ui4>, !pop.scalar<f32>) {
  // CHECK-DAG: <0>
  // CHECK-DAG: <"-2">
  %0 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<0>>
  %1 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<-1>>
  %2 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1.25">>
  %3 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"-2">>
  %4 = pop.min %0, %1 : !pop.scalar<ui4>
  %5 = pop.min %2, %3 : !pop.scalar<f32>
  kgen.return %4, %5 : !pop.scalar<ui4>, !pop.scalar<f32>
}

// CHECK-LABEL: @shl
kgen.func @shl() -> !pop.scalar<ui4> {
  // CHECK-NEXT: <12>
  %0 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<6>>
  %1 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<1>>
  %2 = pop.shl %0, %1 : !pop.scalar<ui4>
  kgen.return %2 : !pop.scalar<ui4>
}

// CHECK-LABEL: @shr
kgen.func @shr() -> (!pop.scalar<ui4>, !pop.scalar<si4>) {
  // CHECK-DAG: <3>
  // CHECK-DAG: <-4>
  %0 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<7>>
  %1 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<1>>
  %2 = kgen.param.constant: !pop.scalar<si4> = <#pop.simd<-7>>
  %3 = kgen.param.constant: !pop.scalar<si4> = <#pop.simd<1>>
  %4 = pop.shr %0, %1 : !pop.scalar<ui4>
  %5 = pop.shr %2, %3 : !pop.scalar<si4>
  kgen.return %4, %5 : !pop.scalar<ui4>, !pop.scalar<si4>
}

// CHECK-LABEL: @copysign
kgen.func @copysign() -> !pop.scalar<f32> {
  // CHECK-NEXT: <"-1.25">
  %0 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1.25">>
  %1 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"-2">>
  %2 = pop.copysign %0, %1 : !pop.scalar<f32>
  kgen.return %2 : !pop.scalar<f32>
}

// CHECK-LABEL: @fma
kgen.func @fma() -> (!pop.scalar<si8>, !pop.scalar<f32>) {
  // CHECK-DAG: <6>
  // CHECK-DAG: <"8.75">
  %0 = kgen.param.constant: !pop.scalar<si8> = <#pop.simd<2>>
  %1 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2.5">>
  %2 = pop.fma %0, %0, %0 : !pop.scalar<si8>
  %3 = pop.fma %1, %1, %1 : !pop.scalar<f32>
  kgen.return %2, %3 : !pop.scalar<si8>, !pop.scalar<f32>
}

// CHECK-LABEL: @index_folds
kgen.func @index_folds() -> (!pop.scalar<index>, !pop.scalar<index>) {
  // COM: Index folds go through the same path as integer folds. We just need to
  // check that ops can fold for index dtypes and do not fold when the results
  // differ between 64-bit and 32-bit arithmetic.
  // CHECK-DAG: %[[DNF_LHS:.*]] = kgen{{.*}}<4294967298>
  // CHECK-DAG: %[[DNF_RHS:.*]] = kgen{{.*}}<2>
  // CHECK-DAG: %[[FOLDED:.*]] = kgen{{.*}}<4294967297>
  // CHECK: %[[R2:.*]] = pop.div %[[DNF_LHS]], %[[DNF_RHS]]
  // CHECK-NEXT: return %[[FOLDED]], %[[R2]]
  %0 = kgen.param.constant: !pop.scalar<index> = <#pop.simd<8589934594>>
  %1 = kgen.param.constant: !pop.scalar<index> = <#pop.simd<4294967298>>
  %2 = kgen.param.constant: !pop.scalar<index> = <#pop.simd<2>>
  %3 = pop.div %0, %2 : !pop.scalar<index>
  %4 = pop.div %1, %2 : !pop.scalar<index>
  kgen.return %3, %4 : !pop.scalar<index>, !pop.scalar<index>
}

// CHECK-LABEL: @struct_construct
kgen.func @struct_construct() -> !pop.struct<si4, ui4> {
  // CHECK-NEXT: constant: !pop.struct<si4, ui4> = <#pop.struct<-3, 7>>
  %0 = kgen.param.constant: si4 = <-3>
  %1 = kgen.param.constant: ui4 = <7>
  %2 = pop.struct.construct(%0, %1) : !pop.struct<si4, ui4>
  kgen.return %2 : !pop.struct<si4, ui4>
}

// CHECK-LABEL: @struct_get
kgen.func @struct_get() -> si4 {
  // CHECK-NEXT: constant: si4 = <-3>
  %0 = kgen.param.constant: !pop.struct<si4, ui4> = <#pop.struct<-3, 7>>
  %1 = pop.struct.get %0[0] : !pop.struct<si4, ui4>
  kgen.return %1 : si4
}

// CHECK-LABEL: @struct_replace
kgen.func @struct_replace() -> !pop.struct<si4, ui4> {
  // CHECK-NEXT: constant: !pop.struct<si4, ui4> = <#pop.struct<-5, 7>>
  %0 = kgen.param.constant: si4 = <-5>
  %1 = kgen.param.constant: !pop.struct<si4, ui4> = <#pop.struct<-3, 7>>
  %2 = pop.struct.replace %0, %1[0] : !pop.struct<si4, ui4>
  kgen.return %2 : !pop.struct<si4, ui4>
}

// CHECK-LABEL: @array_create
kgen.func @array_create() -> !pop.array<2, index> {
  // CHECK-NEXT: constant: !pop.array<2, index> = <#pop.array<0, 0>>
  %idx0 = index.constant 0
  %0 = pop.array.create [%idx0, %idx0] : !pop.array<2, index>
  kgen.return %0 : !pop.array<2, index>
}

// CHECK-LABEL: @array_repeat
kgen.func @array_repeat() -> !pop.array<3, index> {
  // CHECK-NEXT: constant: !pop.array<3, index> = <#pop.array<0, 1, 0>>
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = pop.array.repeat [%idx0, %idx1] : !pop.array<3, index>
  kgen.return %0 : !pop.array<3, index>
}

// CHECK-LABEL: @array_get
kgen.func @array_get() -> index {
  // CHECK-NEXT: constant = <1>
  %0 = kgen.param.constant: !pop.array<2, index> = <#pop.array<0, 1>>
  %1 = pop.array.get %0[1] : !pop.array<2, index>
  kgen.return %1 : index
}

// CHECK-LABEL: @array_replace
kgen.func @array_replace() -> !pop.array<2, index> {
  // CHECK-NEXT: constant: !pop.array<2, index> = <#pop.array<0, 1>>
  %0 = kgen.param.constant: !pop.array<2, index> = <#pop.array<0, 0>>
  %1 = index.constant 1
  %2 = pop.array.replace %1, %0[1] : !pop.array<2, index>
  kgen.return %2 : !pop.array<2, index>
}

// CHECK-LABEL: @variant_create
kgen.func @variant_create() -> !pop.variant<si4, ui4> {
  // CHECK-NEXT: constant: !pop.variant<si4, ui4> = <#pop.variant<:ui4 7>>
  %0 = kgen.param.constant: ui4 = <7>
  %1 = pop.variant.create %0 : ui4 -> !pop.variant<si4, ui4>
  kgen.return %1 : !pop.variant<si4, ui4>
}

// CHECK-LABEL: @variant_is
kgen.func @variant_is() -> i1 {
  // CHECK-NEXT: constant: i1 = <1>
  %0 = kgen.param.constant: !pop.variant<si4, ui4> = <#pop.variant<:ui4 7>>
  %1 = pop.variant.is ui4, %0 : !pop.variant<si4, ui4>
  kgen.return %1 : i1
}

// CHECK-LABEL: @variant_get
kgen.func @variant_get() -> ui4 {
  // CHECK-NEXT: constant: ui4 = <7>
  %0 = kgen.param.constant: !pop.variant<si4, ui4> = <#pop.variant<:ui4 7>>
  %1 = pop.variant.get %0 : !pop.variant<si4, ui4> as ui4
  kgen.return %1 : ui4
}

// CHECK-LABEL: @variant_get_ub
kgen.func @variant_get_ub() -> si4 {
  // CHECK: pop.variant.get
  %0 = kgen.param.constant: !pop.variant<si4, ui4> = <#pop.variant<:ui4 7>>
  %1 = pop.variant.get %0 : !pop.variant<si4, ui4> as si4
  kgen.return %1 : si4
}

// CHECK-LABEL: @variant_create_get
kgen.func @variant_create_get(%a: i32) -> i32 {
  %0 = pop.variant.create %a : i32 -> !pop.variant<i32, f32>
  %1 = pop.variant.get %0 : !pop.variant<i32, f32> as i32
  // CHECK: return %arg0
  kgen.return %1 : i32
}

// CHECK-LABEL: @list_get
kgen.func @list_get() -> i32 {
  // CHECK-NEXT: constant: i32 = <2>
  %0 = kgen.param.constant: list<i32[2]> = <[1, 2]>
  %1 = pop.list.get %0[1] : <i32[2]>
  kgen.return %1 : i32
}

// CHECK-LABEL: @list_create
kgen.func @list_create() -> !kgen.list<i32[2]> {
  // CHECK-NEXT: constant: list<i32[2]> = <[1, 2]>
  %0 = kgen.param.constant: i32 = <1>
  %1 = kgen.param.constant: i32 = <2>
  %2 = pop.list.create(%0, %1) : <i32[2]>
  kgen.return %2 : !kgen.list<i32[2]>
}

// CHECK-LABEL: @list_get_create
kgen.func @list_get_create(%arg0: i32, %arg1: i32) -> i32 {
  %0 = pop.list.create(%arg0, %arg1) : <i32[2]>
  %1 = pop.list.get %0[1] : <i32[2]>
  // CHECK-NEXT: return %arg1
  kgen.return %1 : i32
}
