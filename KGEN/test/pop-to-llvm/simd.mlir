// RUN: kgen-opt -split-input-file -pass-pipeline='kgen.func(lower-pop-to-llvm)' %s | FileCheck %s

// Test trivial vector conversions to LLVM.

!simd = !pop.simd<4, f32>
kgen.func @trivial_conversions(%a: !simd, %b: !simd, %c: !simd, %d: !pop.simd<4, bool>) {
  // CHECK: llvm.intr.fabs
  %0 = pop.abs %a : !simd
  // CHECK: llvm.fneg
  %1 = pop.neg %a : !simd
  // CHECK: llvm.fadd
  %2 = pop.add %a, %b : !simd
  // CHECK: llvm.fsub
  %3 = pop.sub %a, %b : !simd
  // CHECK: llvm.fmul
  %4 = pop.mul %a, %b : !simd
  // CHECK: llvm.intr.copysign
  %5 = pop.copysign %a, %b : !simd
  // CHECK: llvm.intr.fma
  %6 = pop.fma %a, %b, %c : !simd
  // CHECK: llvm.select
  %7 = pop.select %d, %a, %b : !simd
  // CHECK: llvm.intr.floor
  %8 = pop.floor %a : !simd
  // CHECK: llvm.intr.ceil
  %9 = pop.ceil %a : !simd
  kgen.return
}

// -----

// CHECK-LABEL: int_abs_simd
kgen.func @int_abs_simd(%arg0: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false
  %0 = pop.abs %arg0 : !pop.simd<4, si32>
  // CHECK: "llvm.intr.abs"(%{{.*}}, %[[FALSE]])
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: abs_simd
kgen.func @abs_simd(%arg0: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  %0 = pop.abs %arg0 : !pop.simd<4, f32>
  // CHECK: "llvm.intr.fabs"(%{{.*}})
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: int_abs_1xsi32
kgen.func @int_abs_1xsi32(%arg0: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false
  %0 = pop.abs %arg0 : !pop.simd<1, si32>
  // CHECK: "llvm.intr.abs"(%{{.*}}, %[[FALSE]])
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: abs_1xf32
kgen.func @abs_1xf32(%arg0: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  %0 = pop.abs %arg0 : !pop.simd<1, f32>
  // CHECK: "llvm.intr.fabs"(%{{.*}})
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @int_neg_simd
kgen.func @int_neg_simd(%arg0: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(dense<0> : vector<4xi32>)
  %0 = pop.neg %arg0 : !pop.simd<4, si32>
  // CHECK: llvm.sub %[[ZERO]], %{{.*}}
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @neg_simd
kgen.func @neg_simd(%arg0: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  %0 = pop.neg %arg0 : !pop.simd<4, f32>
  // CHECK: llvm.fneg %{{.*}}
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @int_neg_1xsi32
kgen.func @int_neg_1xsi32(%arg0: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
  %0 = pop.neg %arg0 : !pop.simd<1, si32>
  // CHECK: llvm.sub %[[ZERO]], %{{.*}}
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @neg_1xf32
kgen.func @neg_1xf32(%arg0: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  %0 = pop.neg %arg0 : !pop.simd<1, f32>
  // CHECK: llvm.fneg %{{.*}}
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @add_simd
kgen.func @add_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @fadd_simd
kgen.func @fadd_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.fadd
  %0 = pop.add %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @add_1xsi32
kgen.func @add_1xsi32(%arg0: !pop.simd<1, si32>, %arg1: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg1: !pop.simd<1, si32>
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @fadd_1xf32
kgen.func @fadd_1xf32(%arg0: !pop.simd<1, f32>, %arg1: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: llvm.fadd
  %0 = pop.add %arg0, %arg1: !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @sub_simd
kgen.func @sub_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.sub
  %0 = pop.sub %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @fsub_simd
kgen.func @fsub_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.fsub
  %0 = pop.sub %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @sub_1xsi32
kgen.func @sub_1xsi32(%arg0: !pop.simd<1, si32>, %arg1: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: llvm.sub
  %0 = pop.sub %arg0, %arg1: !pop.simd<1, si32>
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @fsub_1xf32
kgen.func @fsub_1xf32(%arg0: !pop.simd<1, f32>, %arg1: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: llvm.fsub
  %0 = pop.sub %arg0, %arg1: !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @mul_simd
kgen.func @mul_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.mul
  %0 = pop.mul %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @fmul_simd
kgen.func @fmul_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.fmul
  %0 = pop.mul %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @mul_1xsi32
kgen.func @mul_1xsi32(%arg0: !pop.simd<1, si32>, %arg1: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: llvm.mul
  %0 = pop.mul %arg0, %arg1: !pop.simd<1, si32>
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @fmul_1xf32
kgen.func @fmul_1xf32(%arg0: !pop.simd<1, f32>, %arg1: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: llvm.fmul
  %0 = pop.mul %arg0, %arg1: !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @div_simd
kgen.func @div_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.sdiv
  %0 = pop.div %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @fdiv_simd
kgen.func @fdiv_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.fdiv
  %0 = pop.div %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @div_1xsi32
kgen.func @div_1xsi32(%arg0: !pop.simd<1, si32>, %arg1: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: llvm.sdiv
  %0 = pop.div %arg0, %arg1: !pop.simd<1, si32>
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @fdiv_1xf32
kgen.func @fdiv_1xf32(%arg0: !pop.simd<1, f32>, %arg1: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: llvm.fdiv
  %0 = pop.div %arg0, %arg1: !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @max_simd
kgen.func @max_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.intr.smax
  %0 = pop.max %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @fmax_simd
kgen.func @fmax_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.intr.maxnum
  %0 = pop.max %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @max_1xsi32
kgen.func @max_1xsi32(%arg0: !pop.simd<1, si32>, %arg1: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: llvm.intr.smax
  %0 = pop.max %arg0, %arg1: !pop.simd<1, si32>
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @fmax_1xf32
kgen.func @fmax_1xf32(%arg0: !pop.simd<1, f32>, %arg1: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: llvm.intr.maxnum
  %0 = pop.max %arg0, %arg1: !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @min_simd
kgen.func @min_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.intr.smin
  %0 = pop.min %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @fmin_simd
kgen.func @fmin_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.intr.minnum
  %0 = pop.min %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @min_1xsi32
kgen.func @min_1xsi32(%arg0: !pop.simd<1, si32>, %arg1: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: llvm.intr.smin
  %0 = pop.min %arg0, %arg1: !pop.simd<1, si32>
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @fmin_1xf32
kgen.func @fmin_1xf32(%arg0: !pop.simd<1, f32>, %arg1: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: llvm.intr.minnum
  %0 = pop.min %arg0, %arg1: !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @constant_simd
kgen.func @constant_simd() -> !pop.simd<2, si32> {
  // CHECK: llvm.mlir.constant(#M.dense_array<0, 0>
  %0 = pop.constant(#M.dense_array<0, 0> : vector<2xsi32>) : !pop.simd<2, si32>
  // CHECK: llvm.mlir.constant(#M.dense_array<0, 0> : vector<2xi32>)
  %1 = pop.constant(0 : ui32) : !pop.simd<2, ui32>
  kgen.return %0 : !pop.simd<2, si32>
}

// -----

// CHECK-LABEL: @constant_simd
kgen.func @constant_simd() -> !pop.simd<2, f32> {
  // CHECK: llvm.mlir.constant(#M.dense_array<1.{{0*}}e+00, 2.{{0*}}e+00>
  %0 = pop.constant(#M.dense_array<1., 2.> : vector<2xf32>) : !pop.simd<2, f32>
  // CHECK: llvm.mlir.constant(#M.dense_array<1.{{0*}}e+00, 1.{{0*}}e+00> : vector<2xf32>)
  %1 = pop.constant(1. : f32) : !pop.simd<2, f32>
  kgen.return %0 : !pop.simd<2, f32>
}

// -----

// CHECK-LABEL: @constant_1xsi64
kgen.func @constant_1xsi64() -> !pop.simd<1, si64> {
  // CHECK: llvm.mlir.constant(42 : i64) : i64
  %x = pop.constant(#M.dense_array<42> : vector<1xsi64>) : !pop.simd<1, si64>
  // CHECK: llvm.mlir.constant(4 : si64) : i64
  %y = pop.constant(4 : si64) : !pop.simd<1, si64>
  kgen.return %x : !pop.simd<1, si64>
}

// -----

// CHECK-LABEL: @constant_1xf32
kgen.func @constant_1xf32() -> !pop.simd<1, f32> {
  // CHECK: llvm.mlir.constant(3.14{{0+}}e+00 : f32) : f32
  %x = pop.constant(#M.dense_array<3.14> : vector<1xf32>) : !pop.simd<1, f32>
  // CHECK: llvm.mlir.constant(3.14{{0+}}e+00 : f32) : f32
  %y = pop.constant(3.14 : f32) : !pop.simd<1, f32>
  kgen.return %x : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @shl_simd
kgen.func @shl_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @shl_simd
kgen.func @shl_simd(%arg0: !pop.simd<4, ui32>, %arg1: !pop.simd<4, ui32>) -> !pop.simd<4, ui32> {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1: !pop.simd<4, ui32>
  kgen.return %0 : !pop.simd<4, ui32>
}

// -----

// CHECK-LABEL: @shl_1xsi32
kgen.func @shl_1xsi32(%arg0: !pop.simd<1, si32>, %arg1: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1: !pop.simd<1, si32>
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @shl_1xui32
kgen.func @shl_1xui32(%arg0: !pop.simd<1, ui32>, %arg1: !pop.simd<1, ui32>) -> !pop.simd<1, ui32> {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1: !pop.simd<1, ui32>
  kgen.return %0 : !pop.simd<1, ui32>
}

// -----

// CHECK-LABEL: @shr_simd
kgen.func @shr_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.ashr
  %0 = pop.shr %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @shr_simd
kgen.func @shr_simd(%arg0: !pop.simd<4, ui32>, %arg1: !pop.simd<4, ui32>) -> !pop.simd<4, ui32> {
  // CHECK: llvm.lshr
  %0 = pop.shr %arg0, %arg1: !pop.simd<4, ui32>
  kgen.return %0 : !pop.simd<4, ui32>
}

// -----

// CHECK-LABEL: @shr_1xsi32
kgen.func @shr_1xsi32(%arg0: !pop.simd<1, si32>, %arg1: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: llvm.ashr
  %0 = pop.shr %arg0, %arg1: !pop.simd<1, si32>
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @shr_1xui32
kgen.func @shr_1xui32(%arg0: !pop.simd<1, ui32>, %arg1: !pop.simd<1, ui32>) -> !pop.simd<1, ui32> {
  // CHECK: llvm.lshr
  %0 = pop.shr %arg0, %arg1: !pop.simd<1, ui32>
  kgen.return %0 : !pop.simd<1, ui32>
}

// -----

// CHECK-LABEL: @copysign_simd
kgen.func @copysign_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.intr.copysign
  %0 = pop.copysign %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @copysign_1xf32
kgen.func @copysign_1xf32(%arg0: !pop.simd<1, f32>, %arg1: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: llvm.intr.copysign
  %0 = pop.copysign %arg0, %arg1: !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @cmp_uint
kgen.func @cmp_uint(%lhs: !pop.simd<4, ui32>, %rhs: !pop.simd<4, ui32>) {
  // CHECK: llvm.icmp "eq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.simd<4, ui32>
  // CHECK: llvm.icmp "ne"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.simd<4, ui32>
  // CHECK: llvm.icmp "ult"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.simd<4, ui32>
  // CHECK: llvm.icmp "ugt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.simd<4, ui32>
  // CHECK: llvm.icmp "ule"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.simd<4, ui32>
  // CHECK: llvm.icmp "uge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.simd<4, ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @cmp_sint
kgen.func @cmp_sint(%lhs: !pop.simd<4, si32>, %rhs: !pop.simd<4, si32>) {
  // CHECK: llvm.icmp "eq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.simd<4, si32>
  // CHECK: llvm.icmp "ne"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.simd<4, si32>
  // CHECK: llvm.icmp "slt"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.simd<4, si32>
  // CHECK: llvm.icmp "sgt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.simd<4, si32>
  // CHECK: llvm.icmp "sle"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.simd<4, si32>
  // CHECK: llvm.icmp "sge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.simd<4, si32>
  kgen.return
}

// -----

// CHECK-LABEL: @cmp_fp
kgen.func @cmp_fp(%lhs: !pop.simd<4, f32>, %rhs: !pop.simd<4, f32>) {
  // CHECK: llvm.fcmp "oeq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.simd<4, f32>
  // CHECK: llvm.fcmp "one"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.simd<4, f32>
  // CHECK: llvm.fcmp "olt"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.simd<4, f32>
  // CHECK: llvm.fcmp "ogt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.simd<4, f32>
  // CHECK: llvm.fcmp "ole"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.simd<4, f32>
  // CHECK: llvm.fcmp "oge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.simd<4, f32>
  kgen.return
}

// -----

// CHECK-LABEL: @cmp_uint
kgen.func @cmp_uint(%lhs: !pop.simd<1, ui32>, %rhs: !pop.simd<1, ui32>) {
  // CHECK: llvm.icmp "eq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.simd<1, ui32>
  // CHECK: llvm.icmp "ne"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.simd<1, ui32>
  // CHECK: llvm.icmp "ult"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.simd<1, ui32>
  // CHECK: llvm.icmp "ugt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.simd<1, ui32>
  // CHECK: llvm.icmp "ule"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.simd<1, ui32>
  // CHECK: llvm.icmp "uge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.simd<1, ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @cmp_sint
kgen.func @cmp_sint(%lhs: !pop.simd<1, si32>, %rhs: !pop.simd<1, si32>) {
  // CHECK: llvm.icmp "eq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.simd<1, si32>
  // CHECK: llvm.icmp "ne"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.simd<1, si32>
  // CHECK: llvm.icmp "slt"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.simd<1, si32>
  // CHECK: llvm.icmp "sgt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.simd<1, si32>
  // CHECK: llvm.icmp "sle"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.simd<1, si32>
  // CHECK: llvm.icmp "sge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.simd<1, si32>
  kgen.return
}

// -----

// CHECK-LABEL: @cmp_fp
kgen.func @cmp_fp(%lhs: !pop.simd<1, f32>, %rhs: !pop.simd<1, f32>) {
  // CHECK: llvm.fcmp "oeq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.simd<1, f32>
  // CHECK: llvm.fcmp "one"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.simd<1, f32>
  // CHECK: llvm.fcmp "olt"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.simd<1, f32>
  // CHECK: llvm.fcmp "ogt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.simd<1, f32>
  // CHECK: llvm.fcmp "ole"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.simd<1, f32>
  // CHECK: llvm.fcmp "oge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.simd<1, f32>
  kgen.return
}

// -----

// CHECK-LABEL: @fma_simd
kgen.func @fma_simd(%arg0: !pop.simd<4, si32>,
                    %arg1: !pop.simd<4, si32>,
                    %arg2: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.mul
  // CHECK: llvm.add
  %0 = pop.fma %arg0, %arg1, %arg2: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}


// -----

// CHECK-LABEL: @fma_simd
kgen.func @fma_simd(%arg0: !pop.simd<4, f32>,
                    %arg1: !pop.simd<4, f32>,
                    %arg2: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.intr.fma
  %0 = pop.fma %arg0, %arg1, %arg2: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @fma_1xsi32
kgen.func @fma_1xsi32(%arg0: !pop.simd<1, si32>,
                      %arg1: !pop.simd<1, si32>,
                      %arg2: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: llvm.mul
  // CHECK: llvm.add
  %0 = pop.fma %arg0, %arg1, %arg2: !pop.simd<1, si32>
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @fma_1xf32
kgen.func @fma_1xf32(%arg0: !pop.simd<1, f32>,
                     %arg1: !pop.simd<1, f32>,
                     %arg2: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: llvm.intr.fma
  %0 = pop.fma %arg0, %arg1, %arg2: !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// CHECK-LABEL: @select_simd
kgen.func @select_simd(%arg0: !pop.simd<4, bool>,
                    %arg1: !pop.simd<4, si32>,
                    %arg2: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.select
  %0 = pop.select %arg0, %arg1, %arg2: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}


// -----

// CHECK-LABEL: @select_simd
kgen.func @select_simd(%arg0: !pop.simd<4, bool>,
                    %arg1: !pop.simd<4, f32>,
                    %arg2: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.select
  %0 = pop.select %arg0, %arg1, %arg2: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @select_1xsi32
kgen.func @select_1xsi32(%arg0: !pop.simd<1, bool>,
                      %arg1: !pop.simd<1, si32>,
                      %arg2: !pop.simd<1, si32>) -> !pop.simd<1, si32> {
  // CHECK: llvm.select
  %0 = pop.select %arg0, %arg1, %arg2: !pop.simd<1, si32>
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @select_1xf32
kgen.func @select_1xf32(%arg0: !pop.simd<1, bool>,
                     %arg1: !pop.simd<1, f32>,
                     %arg2: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: llvm.select
  %0 = pop.select %arg0, %arg1, %arg2: !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @bitcast
kgen.func @bitcast(%a: !pop.simd<1, si32>,
                   %b: !pop.simd<1, ui64>,
                   %c: !pop.simd<4, f64>,
                   %d: !pop.simd<2, f64>) {
  // CHECK: llvm.bitcast %0 : i32 to f32
  %0 = pop.bitcast %a: !pop.simd<1, si32> to !pop.simd<1, f32>

  // CHECK: llvm.bitcast %1 : i64 to i64
  %1 = pop.bitcast %b: !pop.simd<1, ui64> to !pop.simd<1, si64>

  // CHECK: llvm.bitcast %2 : vector<4xf64> to vector<4xi64>
  %2 = pop.bitcast %c: !pop.simd<4, f64> to !pop.simd<4, si64>

  // CHECK: llvm.bitcast %3 : vector<2xf64> to vector<4xf32>
  %3 = pop.bitcast %d: !pop.simd<2, f64> to !pop.simd<4, f32>

  // CHECK: llvm.bitcast %1 : i64 to vector<2xf32>
  %4 = pop.bitcast %b: !pop.simd<1, ui64> to !pop.simd<2, f32>

  kgen.return
}

// -----

// CHECK-LABEL: @simd_splat_scalar_to_2xf32
kgen.func @simd_splat_scalar_to_2xf32(%a: !pop.simd<1, f32>) -> !pop.simd<2, f32> {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 :
  // CHECK: %[[VECTOR:.*]] = llvm.insertelement %[[E:.*]], %[[UNDEF]][%[[ZERO]] : i32] : vector<2xf32>
  // CHECK: %[[RESULT:.*]] = llvm.shufflevector %[[VECTOR]], %[[UNDEF]] [0, 0] : vector<2xf32>
  // CHECK: unrealized_conversion_cast %[[RESULT]]
  %0 = pop.simd.splat %a : !pop.simd<2, f32>
  kgen.return %0 : !pop.simd<2, f32>
}

// -----

// CHECK-LABEL: @simd_splat_scalar_to_1xf32
kgen.func @simd_splat_scalar_to_1xf32(%a: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: %[[F32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, f32> to f32
  // CHECK: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[F32_VAL]] : f32 to !pop.simd<1, f32>
  %0 = pop.simd.splat %a : !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @simd_extractelement
kgen.func @simd_extractelement(%vec: !pop.simd<4, f32>, %idx: index) -> !pop.simd<1, f32> {
  // CHECK: %[[VEC:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[IDX:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[SCALAR:.*]] = llvm.extractelement %[[VEC]][%[[IDX]] : {{.*}}] : vector<4xf32>
  // CHECK: unrealized_conversion_cast %[[SCALAR]]
  %0 = pop.simd.extractelement %vec[%idx] : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @simd_extractelement_1xf32
kgen.func @simd_extractelement_1xf32(%vec: !pop.simd<1, f32>, %idx: index) -> !pop.simd<1, f32> {
  // CHECK: %[[F32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, f32> to f32
  // CHECK: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[F32_VAL]] : f32 to !pop.simd<1, f32>
  %0 = pop.simd.extractelement %vec[%idx] : !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @simd_insertelement
kgen.func @simd_insertelement(%val: !pop.simd<1, f32>, %vec: !pop.simd<4, f32>, %idx: index) -> !pop.simd<4, f32> {
  // CHECK: %[[VEC:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[VAL:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[IDX:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[RES:.*]] = llvm.insertelement %[[VAL]], %[[VEC]][%[[IDX]] : {{.*}}] : vector<4xf32>
  // CHECK: unrealized_conversion_cast %[[RES]]
  %0 = pop.simd.insertelement %val, %vec[%idx] : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @simd_insertelement_1xf32
kgen.func @simd_insertelement_1xf32(%val: !pop.simd<1, f32>, %vec: !pop.simd<1, f32>, %idx: index) -> !pop.simd<1, f32> {
  // CHECK: %[[F32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, f32> to f32
  // CHECK: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[F32_VAL]] : f32 to !pop.simd<1, f32>
  %0 = pop.simd.insertelement %val, %vec[%idx] : !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @simd_shuffle
kgen.func @simd_shuffle(%a: !pop.simd<2, f32>, %b: !pop.simd<2, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.shufflevector %{{.*}}, %{{.*}} [2, 3, 1, 0]
  %0 = pop.simd.shuffle %a, %b [2, 3, 1, 0] : !pop.simd<2, f32> -> !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @simd_shuffle_1xf32
kgen.func @simd_shuffle_1xf32(%a: !pop.simd<1, f32>, %b: !pop.simd<1, f32>) -> (!pop.simd<2, f32>, !pop.simd<1, f32>) {
  // CHECK: %[[F32VAL0:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, f32> to f32
  // CHECK: %[[F32VAL1:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, f32> to f32
  // CHECK: %[[VECVAL0_0:.*]] = llvm.mlir.undef : vector<2xf32>
  // CHECK: %[[CONST0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[VECVAL0_1:.*]] = llvm.insertelement %[[F32VAL1]], %[[VECVAL0_0]][%[[CONST0_0]] : i32]
  // CHECK: %[[CONST0_1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[VECVAL0_2:.*]] = llvm.insertelement %[[F32VAL0]], %[[VECVAL0_1]][%[[CONST0_1]] : i32]
  // CHECK: builtin.unrealized_conversion_cast %[[VECVAL0_2]] : vector<2xf32> to !pop.simd<2, f32>
  %0 = pop.simd.shuffle %a, %b [1, 0] : !pop.simd<1, f32> -> !pop.simd<2, f32>

  // CHECK: %[[VECVAL1_0:.*]] = llvm.mlir.undef : vector<1xf32>
  // CHECK: %[[CONST1_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[VECVAL1_1:.*]] = llvm.insertelement %[[F32VAL1]], %[[VECVAL1_0]][%[[CONST1_0]] : i32]
  // CHECK: builtin.unrealized_conversion_cast %[[VECVAL1_1]] : vector<1xf32> to !pop.simd<1, f32>
  %1 = pop.simd.shuffle %a, %b [1] : !pop.simd<1, f32> -> !pop.simd<1, f32>

  kgen.return %0, %1 : !pop.simd<2, f32>, !pop.simd<1, f32>
}

// -----

// CHECK-LABEL: @simd_load_store
kgen.func @simd_load_store(%i: index, %p0: !pop.pointer<simd<4, f32>>) {
  // CHECK: llvm.getelementptr %{{.*}} : (!llvm.ptr<vector<4xf32>>, {{.*}}) -> !llvm.ptr<vector<4xf32>>
  %0 = pop.offset %p0[%i] : !pop.pointer<simd<4, f32>>
  // CHECK: llvm.load
  %1 = pop.load %0 : !pop.pointer<simd<4, f32>>
  // CHECK: llvm.store
  pop.store %1, %p0 : !pop.pointer<simd<4, f32>>
  kgen.return
}

// -----

// CHECK-LABEL: @simd_reduce_add
kgen.func @simd_reduce_add(%a: !pop.simd<2, f32>,
                           %b: !pop.simd<2, si32>,
                           %c: !pop.simd<2, ui32>) -> (!pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>) {
  // CHECK: llvm.intr.vector.reduce.fadd
  %0 = pop.simd.reduce.add %a : !pop.simd<2, f32>
  // CHECK: llvm.intr.vector.reduce.add
  %1 = pop.simd.reduce.add %b : !pop.simd<2, si32>
  // CHECK: llvm.intr.vector.reduce.add
  %2 = pop.simd.reduce.add %c : !pop.simd<2, ui32>
  kgen.return %0, %1, %2: !pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>
}

// -----

// CHECK-LABEL: @simd_reduce_add_1xf32
kgen.func @simd_reduce_add_1xf32(%a: !pop.simd<1, f32>,
                                 %b: !pop.simd<1, si32>,
                                 %c: !pop.simd<1, ui32>) -> (!pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>) {
  // CHECK: %[[F32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, f32> to f32
  // CHECK: %[[SI32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, si32> to i32
  // CHECK: %[[UI32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, ui32> to i32
  // CHECK: %[[RESULT0:.*]] = builtin.unrealized_conversion_cast %[[F32_VAL]] : f32 to !pop.simd<1, f32>
  // CHECK: %[[RESULT1:.*]] = builtin.unrealized_conversion_cast %[[SI32_VAL]] : i32 to !pop.simd<1, si32>
  // CHECK: %[[RESULT2:.*]] = builtin.unrealized_conversion_cast %[[UI32_VAL]] : i32 to !pop.simd<1, ui32>
  %0 = pop.simd.reduce.add %a : !pop.simd<1, f32>
  %1 = pop.simd.reduce.add %b : !pop.simd<1, si32>
  %2 = pop.simd.reduce.add %c : !pop.simd<1, ui32>
  kgen.return %0, %1, %2: !pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>
}

// -----

// CHECK-LABEL: @simd_reduce_mul
kgen.func @simd_reduce_mul(%a: !pop.simd<2, f32>,
                           %b: !pop.simd<2, si32>,
                           %c: !pop.simd<2, ui32>) -> (!pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>) {
  // CHECK: llvm.intr.vector.reduce.fmul
  %0 = pop.simd.reduce.mul %a : !pop.simd<2, f32>
  // CHECK: llvm.intr.vector.reduce.mul
  %1 = pop.simd.reduce.mul %b : !pop.simd<2, si32>
  // CHECK: llvm.intr.vector.reduce.mul
  %2 = pop.simd.reduce.mul %c : !pop.simd<2, ui32>
  kgen.return %0, %1, %2: !pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>
}

// -----

// CHECK-LABEL: @simd_reduce_mul_1xf32
kgen.func @simd_reduce_mul_1xf32(%a: !pop.simd<1, f32>,
                                 %b: !pop.simd<1, si32>,
                                 %c: !pop.simd<1, ui32>) -> (!pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>) {
  // CHECK: %[[F32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, f32> to f32
  // CHECK: %[[SI32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, si32> to i32
  // CHECK: %[[UI32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, ui32> to i32
  // CHECK: %[[RESULT0:.*]] = builtin.unrealized_conversion_cast %[[F32_VAL]] : f32 to !pop.simd<1, f32>
  // CHECK: %[[RESULT1:.*]] = builtin.unrealized_conversion_cast %[[SI32_VAL]] : i32 to !pop.simd<1, si32>
  // CHECK: %[[RESULT2:.*]] = builtin.unrealized_conversion_cast %[[UI32_VAL]] : i32 to !pop.simd<1, ui32>
  %0 = pop.simd.reduce.mul %a : !pop.simd<1, f32>
  %1 = pop.simd.reduce.mul %b : !pop.simd<1, si32>
  %2 = pop.simd.reduce.mul %c : !pop.simd<1, ui32>
  kgen.return %0, %1, %2: !pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>
}

// -----

// CHECK-LABEL: @simd_reduce_max
kgen.func @simd_reduce_max(%a: !pop.simd<2, f32>,
                           %b: !pop.simd<2, si32>,
                           %c: !pop.simd<2, ui32>) -> (!pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>) {
  // CHECK: llvm.intr.vector.reduce.fmax
  %0 = pop.simd.reduce.max %a : !pop.simd<2, f32>
  // CHECK: llvm.intr.vector.reduce.smax
  %1 = pop.simd.reduce.max %b : !pop.simd<2, si32>
  // CHECK: llvm.intr.vector.reduce.umax
  %2 = pop.simd.reduce.max %c : !pop.simd<2, ui32>
  kgen.return %0, %1, %2: !pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>
}

// -----

// CHECK-LABEL: @simd_reduce_max_1xf32
kgen.func @simd_reduce_max_1xf32(%a: !pop.simd<1, f32>,
                                 %b: !pop.simd<1, si32>,
                                 %c: !pop.simd<1, ui32>) -> (!pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>) {
  // CHECK: %[[F32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, f32> to f32
  // CHECK: %[[SI32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, si32> to i32
  // CHECK: %[[UI32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, ui32> to i32
  // CHECK: %[[RESULT0:.*]] = builtin.unrealized_conversion_cast %[[F32_VAL]] : f32 to !pop.simd<1, f32>
  // CHECK: %[[RESULT1:.*]] = builtin.unrealized_conversion_cast %[[SI32_VAL]] : i32 to !pop.simd<1, si32>
  // CHECK: %[[RESULT2:.*]] = builtin.unrealized_conversion_cast %[[UI32_VAL]] : i32 to !pop.simd<1, ui32>
  %0 = pop.simd.reduce.max %a : !pop.simd<1, f32>
  %1 = pop.simd.reduce.max %b : !pop.simd<1, si32>
  %2 = pop.simd.reduce.max %c : !pop.simd<1, ui32>
  kgen.return %0, %1, %2: !pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>
}

// -----

// CHECK-LABEL: @simd_reduce_min
kgen.func @simd_reduce_min(%a: !pop.simd<2, f32>,
                           %b: !pop.simd<2, si32>,
                           %c: !pop.simd<2, ui32>) -> (!pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>) {
  // CHECK: llvm.intr.vector.reduce.fmin
  %0 = pop.simd.reduce.min %a : !pop.simd<2, f32>
  // CHECK: llvm.intr.vector.reduce.smin
  %1 = pop.simd.reduce.min %b : !pop.simd<2, si32>
  // CHECK: llvm.intr.vector.reduce.umin
  %2 = pop.simd.reduce.min %c : !pop.simd<2, ui32>
  kgen.return %0, %1, %2: !pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>
}

// -----

// CHECK-LABEL: @simd_reduce_min_1xf32
kgen.func @simd_reduce_min_1xf32(%a: !pop.simd<1, f32>,
                                 %b: !pop.simd<1, si32>,
                                 %c: !pop.simd<1, ui32>) -> (!pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>) {
  // CHECK: %[[F32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, f32> to f32
  // CHECK: %[[SI32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, si32> to i32
  // CHECK: %[[UI32_VAL:.*]] = builtin.unrealized_conversion_cast %[[E:..*]] : !pop.simd<1, ui32> to i32
  // CHECK: %[[RESULT0:.*]] = builtin.unrealized_conversion_cast %[[F32_VAL]] : f32 to !pop.simd<1, f32>
  // CHECK: %[[RESULT1:.*]] = builtin.unrealized_conversion_cast %[[SI32_VAL]] : i32 to !pop.simd<1, si32>
  // CHECK: %[[RESULT2:.*]] = builtin.unrealized_conversion_cast %[[UI32_VAL]] : i32 to !pop.simd<1, ui32>
  %0 = pop.simd.reduce.min %a : !pop.simd<1, f32>
  %1 = pop.simd.reduce.min %b : !pop.simd<1, si32>
  %2 = pop.simd.reduce.min %c : !pop.simd<1, ui32>
  kgen.return %0, %1, %2: !pop.simd<1, f32>, !pop.simd<1, si32>, !pop.simd<1, ui32>
}

// -----

// CHECK-LABEL: @pop_gather
// CHECK-SAME: %[[BASE0:.*]]: !pop.simd<2, address>
// CHECK-SAME: %[[MASK0:.*]]: !pop.simd<2, bool>
// CHECK-SAME: %[[PASSTHROUGH0:.*]]: !pop.simd<2, f32>
kgen.func @pop_gather(%base: !pop.simd<2, address>,
                      %mask: !pop.simd<2, bool>,
                      %passthrough: !pop.simd<2, f32>) -> !pop.simd<2, f32> {
  // CHECK-DAG: %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[BASE0]]
  // CHECK-DAG: %[[MASK:.*]] = builtin.unrealized_conversion_cast %[[MASK0]]
  // CHECK-DAG: %[[PASSTHROUGH:.*]] = builtin.unrealized_conversion_cast %[[PASSTHROUGH0]]
  // CHECK: %[[RESULT:.*]] = llvm.intr.masked.gather %[[BASE]], %[[MASK]], %[[PASSTHROUGH]] {alignment = 4 : i32}
  %0 = pop.simd.gather %base[%mask], %passthrough : !pop.simd<2, f32>
  kgen.return %0 : !pop.simd<2, f32>
}

// -----

// CHECK-LABEL: @pop_scatter
// CHECK-SAME: %[[VALUE0:.*]]: !pop.simd<2, f32>
// CHECK-SAME: %[[BASE0:.*]]: !pop.simd<2, address>
// CHECK-SAME: %[[MASK0:.*]]: !pop.simd<2, bool>
kgen.func @pop_scatter(%value: !pop.simd<2, f32>,
                       %base: !pop.simd<2, address>,
                       %mask: !pop.simd<2, bool>) {
  // CHECK-DAG: %[[VALUE:.*]] = builtin.unrealized_conversion_cast %[[VALUE0]]
  // CHECK-DAG: %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[BASE0]]
  // CHECK-DAG: %[[MASK:.*]] = builtin.unrealized_conversion_cast %[[MASK0]]
  // CHECK: llvm.intr.masked.scatter %[[VALUE]], %[[BASE]], %[[MASK]]
  pop.simd.scatter %value, %base[%mask] : !pop.simd<2, f32>
  kgen.return
}
