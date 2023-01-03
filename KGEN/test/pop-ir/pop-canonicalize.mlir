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

// CHECK-LABEL: @cmp_eq
kgen.func @cmp_eq() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <false, true>
  // CHECK-DAG: <false>
  %0 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<1, 2>>
  %1 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<-2, 2>>
  %2 = pop.cmp eq(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1">>
  %4 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2">>
  %5 = pop.cmp eq(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_ne
kgen.func @cmp_ne() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <true, false>
  // CHECK-DAG: <true>
  %0 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<1, 2>>
  %1 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<-2, 2>>
  %2 = pop.cmp ne(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1">>
  %4 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2">>
  %5 = pop.cmp ne(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_lt
kgen.func @cmp_lt() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <false, false>
  // CHECK-DAG: <true>
  %0 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<1, 2>>
  %1 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<-2, 2>>
  %2 = pop.cmp lt(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1">>
  %4 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2">>
  %5 = pop.cmp lt(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_gt
kgen.func @cmp_gt() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <true, false>
  // CHECK-DAG: <false>
  %0 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<1, 2>>
  %1 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<-2, 2>>
  %2 = pop.cmp gt(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1">>
  %4 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2">>
  %5 = pop.cmp gt(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_le
kgen.func @cmp_le() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <false, true>
  // CHECK-DAG: <true>
  %0 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<1, 2>>
  %1 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<-2, 2>>
  %2 = pop.cmp le(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1">>
  %4 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2">>
  %5 = pop.cmp le(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_ge
kgen.func @cmp_ge() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <true, true>
  // CHECK-DAG: <false>
  %0 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<1, 2>>
  %1 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<-2, 2>>
  %2 = pop.cmp ge(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1">>
  %4 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2">>
  %5 = pop.cmp ge(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_index
kgen.func @cmp_index() -> !pop.scalar<bool> {
  // CHECK: pop.cmp
  %0 = kgen.param.constant: !pop.scalar<index> = <#pop.simd<4294967296>>
  %1 = kgen.param.constant: !pop.scalar<index> = <#pop.simd<8589934592>>
  %2 = pop.cmp eq(%0, %1) : !pop.scalar<index>
  kgen.return %2 : !pop.scalar<bool>
}

// CHECK-LABEL: @and
kgen.func @and() -> !pop.scalar<ui4> {
  // CHECK-NEXT <1>
  %0 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<7>>
  %1 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<9>>
  %2 = pop.and %0, %1 : !pop.scalar<ui4>
  kgen.return %2 : !pop.scalar<ui4>
}

// CHECK-LABEL: @or
kgen.func @or() -> !pop.scalar<ui4> {
  // CHECK-NEXT <15>
  %0 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<6>>
  %1 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<9>>
  %2 = pop.or %0, %1 : !pop.scalar<ui4>
  kgen.return %2 : !pop.scalar<ui4>
}

// CHECK-LABEL: @xor
kgen.func @xor() -> !pop.scalar<ui4> {
  // CHECK-NEXT <2>
  %0 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<5>>
  %1 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<7>>
  %2 = pop.xor %0, %1 : !pop.scalar<ui4>
  kgen.return %2 : !pop.scalar<ui4>
}

// CHECK-LABEL: @select
kgen.func @select() -> !pop.simd<2, si4> {
  // CHECK-NEXT: <1, 4>
  %0 = kgen.param.constant: !pop.simd<2, si4> = <#pop.simd<1, 3>>
  %1 = kgen.param.constant: !pop.simd<2, si4> = <#pop.simd<2, 4>>
  %2 = kgen.param.constant: !pop.simd<2, bool> = <#pop.simd<true, false>>
  %3 = pop.select %2, %0, %1 : !pop.simd<2, si4>
  kgen.return %3 : !pop.simd<2, si4>
}

// CHECK-LABEL: @bitcast
kgen.func @bitcast() -> (!pop.simd<2, bf16>, !pop.simd<2, f16>) {
  // CHECK-DAG: <"0.125", "8">
  // CHECK-DAG: <"5.9605E-8", "1.1921E-7">
  %0 = kgen.param.constant: !pop.simd<2, si16> = <#pop.simd<1, 2>>
  %1 = kgen.param.constant: !pop.simd<2, f16> = <#pop.simd<"1.5", "2.5">>
  %2 = pop.bitcast %0 : !pop.simd<2, si16> to !pop.simd<2, f16>
  %3 = pop.bitcast %1 : !pop.simd<2, f16> to !pop.simd<2, bf16>
  kgen.return %3, %2 : !pop.simd<2, bf16>, !pop.simd<2, f16>
}

// CHECK-LABEL: @bitcast_size_change
kgen.func @bitcast_size_change() -> (!pop.simd<4, si16>) {
  // CHECK: pop.bitcast
  %0 = kgen.param.constant: !pop.simd<2, si32> = <#pop.simd<1, 2>>
  %1 = pop.bitcast %0 : !pop.simd<2, si32> to !pop.simd<4, si16>
  kgen.return %1 : !pop.simd<4, si16>
}

// CHECK-LABEL: @pointer_bitcast
kgen.func @pointer_bitcast() -> !pop.pointer<si32> {
  // CHECK-NEXT: !pop.pointer<si32> = <#M.pointer<0>>
  %0 = kgen.param.constant: !pop.pointer<si64> = <#M.pointer<0>>
  %1 = pop.pointer.bitcast %0 : !pop.pointer<si64> to !pop.pointer<si32>
  kgen.return %1 : !pop.pointer<si32>
}

// CHECK-LABEL: @cast
kgen.func @cast() -> (
    !pop.scalar<f16>, !pop.scalar<f16>, !pop.scalar<f16>,
    !pop.scalar<ui8>, !pop.scalar<ui8>, !pop.scalar<ui8>,
    !pop.scalar<bool>, !pop.scalar<bool>, !pop.scalar<bool>,
    !pop.scalar<index>, !pop.scalar<index>, !pop.scalar<index>) {
  // CHECK-DAG: %[[C0:.*]] = kgen{{.*}}<"10.125">
  // CHECK-DAG: %[[C1:.*]] = kgen{{.*}}<"500">
  // CHECK-DAG: %[[C2:.*]] = kgen{{.*}}<"1">
  // CHECK-DAG: %[[C3:.*]] = kgen{{.*}}ui8{{.*}}<10>
  // CHECK-DAG: %[[C4:.*]] = kgen{{.*}}ui8{{.*}}<244>
  // CHECK-DAG: %[[C5:.*]] = kgen{{.*}}ui8{{.*}}<1>
  // CHECK-DAG: %[[TRUE:.*]] = kgen{{.*}}<true>
  // CHECK-DAG: %[[C6:.*]] = kgen{{.*}}index{{.*}}<10>
  // CHECK-DAG: %[[C7:.*]] = kgen{{.*}}index{{.*}}<500>
  // CHECK-DAG: %[[C8:.*]] = kgen{{.*}}index{{.*}}<1>
  %0 = kgen.param.constant: !pop.scalar<bf16> = <#pop.simd<"10.125">>
  %1 = kgen.param.constant: !pop.scalar<si32> = <#pop.simd<500>>
  %2 = kgen.param.constant: !pop.scalar<bool> = <#pop.simd<true>>

  %3 = pop.cast %0 : !pop.scalar<bf16> to !pop.scalar<f16>
  %4 = pop.cast %1 : !pop.scalar<si32> to !pop.scalar<f16>
  %5 = pop.cast %2 : !pop.scalar<bool> to !pop.scalar<f16>

  %6 = pop.cast %0 : !pop.scalar<bf16> to !pop.scalar<ui8>
  %7 = pop.cast %1 : !pop.scalar<si32> to !pop.scalar<ui8>
  %8 = pop.cast %2 : !pop.scalar<bool> to !pop.scalar<ui8>

  %9 = pop.cast %0 : !pop.scalar<bf16> to !pop.scalar<bool>
  %10 = pop.cast %1 : !pop.scalar<si32> to !pop.scalar<bool>
  %11 = pop.cast %2 : !pop.scalar<bool> to !pop.scalar<bool>

  %12 = pop.cast %0 : !pop.scalar<bf16> to !pop.scalar<index>
  %13 = pop.cast %1 : !pop.scalar<si32> to !pop.scalar<index>
  %14 = pop.cast %2 : !pop.scalar<bool> to !pop.scalar<index>

  // CHECK-NEXT: return %[[C0]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[TRUE]], %[[TRUE]], %[[TRUE]]
  // CHECK-SAME: %[[C6]], %[[C7]], %[[C8]]
  kgen.return %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14 :
    !pop.scalar<f16>, !pop.scalar<f16>, !pop.scalar<f16>,
    !pop.scalar<ui8>, !pop.scalar<ui8>, !pop.scalar<ui8>,
    !pop.scalar<bool>, !pop.scalar<bool>, !pop.scalar<bool>,
    !pop.scalar<index>, !pop.scalar<index>, !pop.scalar<index>
}

// CHECK-LABEL: @cast_fp_too_big
kgen.func @cast_fp_too_big() -> (!pop.scalar<si2>, !pop.scalar<si2>) {
  // CHECK-DAG: <1>
  // CHECK-DAG: <"7">
  %0 = kgen.param.constant: !pop.scalar<f16> = <#pop.simd<"1.5">>
  %1 = kgen.param.constant: !pop.scalar<f16> = <#pop.simd<"7">>
  %2 = pop.cast %0 : !pop.scalar<f16> to !pop.scalar<si2>
  // CHECK: %[[TOO_BIG:.*]] = pop.cast
  %3 = pop.cast %1 : !pop.scalar<f16> to !pop.scalar<si2>
  // CHECK-NEXT: return %{{.*}}, %[[TOO_BIG]]
  kgen.return %2, %3 : !pop.scalar<si2>, !pop.scalar<si2>
}

// CHECK-LABEL: @simd_extractelement
kgen.func @simd_extractelement() -> (!pop.scalar<si8>) {
  // CHECK-NEXT: <20>
  %idx1 = index.constant 1
  %0 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<10, 20>>
  %1 = pop.simd.extractelement %0[%idx1] : !pop.simd<2, si8>
  kgen.return %1 : !pop.scalar<si8>
}

// CHECK-LABEL: @simd_insertelement
kgen.func @simd_insertelement() -> (!pop.simd<2, si8>) {
  // CHECK-NEXT: <30, 20>
  %idx0 = index.constant 0
  %0 = kgen.param.constant: !pop.simd<2, si8> = <#pop.simd<10, 20>>
  %1 = kgen.param.constant: !pop.scalar<si8> = <#pop.simd<30>>
  %2 = pop.simd.insertelement %1, %0[%idx0] : !pop.simd<2, si8>
  kgen.return %2 : !pop.simd<2, si8>
}

// CHECK-LABEL: @simd_splat
kgen.func @simd_splat() -> !pop.simd<2, si8> {
  // CHECK-NEXT: <2, 2>
  %0 = kgen.param.constant: !pop.scalar<si8> = <#pop.simd<2>>
  %1 = pop.simd.splat %0 : !pop.simd<2, si8>
  kgen.return %1 : !pop.simd<2, si8>
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

// CHECK-LABEL: @index_to_pointer
kgen.func @index_to_pointer() -> (!pop.pointer<i8>, !pop.scalar<address>) {
  // CHECK-DAG: #M.pointer<1>
  // CHECK-DAG: !pop.scalar<address> = <#pop.simd<2>>
  %0 = kgen.param.constant: !pop.scalar<index> = <#pop.simd<1>>
  %1 = kgen.param.constant: !pop.scalar<index> = <#pop.simd<2>>
  %2 = pop.index_to_pointer %0 : !pop.scalar<index> to !pop.pointer<i8>
  %3 = pop.index_to_pointer %1 : !pop.scalar<index> to !pop.scalar<address>
  kgen.return %2, %3 : !pop.pointer<i8>, !pop.scalar<address>
}

// CHECK-LABEL: @pointer_to_index
kgen.func @pointer_to_index() -> (!pop.scalar<index>, !pop.scalar<index>) {
  // CHECK-DAG: <1>
  // CHECK-DAG: <2>
  %0 = kgen.param.constant: !pop.pointer<i8> = <#M.pointer<1>>
  %1 = kgen.param.constant: !pop.scalar<address> = <#pop.simd<2>>
  %2 = pop.pointer_to_index %0 : !pop.pointer<i8> to !pop.scalar<index>
  %3 = pop.pointer_to_index %1 : !pop.scalar<address> to !pop.scalar<index>
  kgen.return %2, %3 : !pop.scalar<index>, !pop.scalar<index>
}
