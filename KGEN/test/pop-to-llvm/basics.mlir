// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

// CHECK-LABEL: @abs
kgen.func @abs(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.intr.fabs
  %0 = pop.abs %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @abs
kgen.func @abs(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(false
  // CHECK: "llvm.intr.abs"(%{{.*}}, %[[ZERO]]
  %0 = pop.abs %arg0 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// -----

// CHECK-LABEL: @neg
kgen.func @neg(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.fneg
  %0 = pop.neg %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @neg
// CHECK-SAME: %[[ARG0:.*]]:
kgen.func @neg(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(0 :
  // CHECK: llvm.sub %[[LHS]], %[[RHS]]
  %0 = pop.neg %arg0 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}


// -----

// CHECK-LABEL: @neg
// CHECK-SAME: %[[ARG0:.*]]:
kgen.func @neg(%arg0: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(0 :
  // CHECK: llvm.sub %[[LHS]], %[[RHS]]
  %0 = pop.neg %arg0 : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// -----

// CHECK-LABEL: @add
kgen.func @add(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg0 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// -----

// CHECK-LABEL: @add
kgen.func @add(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.fadd
  %0 = pop.add %arg0, %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @add
kgen.func @add(%arg0: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg0 : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// -----

// CHECK-LABEL: @sub
kgen.func @sub(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: llvm.sub
  %0 = pop.sub %arg0, %arg1 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// -----

// CHECK-LABEL: @sub
kgen.func @sub(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.fsub
  %0 = pop.sub %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}


// -----

// CHECK-LABEL: @sub
kgen.func @sub(%arg0: !pop.scalar<index>, %arg1: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK: llvm.sub
  %0 = pop.sub %arg0, %arg1 : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// -----

// CHECK-LABEL: @mul
kgen.func @mul(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: llvm.mul
  %0 = pop.mul %arg0, %arg0 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// -----

// CHECK-LABEL: @mul
kgen.func @mul(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.fmul
  %0 = pop.mul %arg0, %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @max
kgen.func @max(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: llvm.intr.smax
  %0 = pop.max %arg0, %arg1 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// -----

// CHECK-LABEL: @max
kgen.func @max(%arg0: !pop.scalar<index>, %arg1: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK: llvm.intr.smax
  %0 = pop.max %arg0, %arg1 : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// -----

// CHECK-LABEL: @max
kgen.func @max(%arg0: !pop.scalar<ui32>, %arg1: !pop.scalar<ui32>) -> !pop.scalar<ui32> {
  // CHECK: llvm.intr.umax
  %0 = pop.max %arg0, %arg1 : !pop.scalar<ui32>
  kgen.return %0 : !pop.scalar<ui32>
}

// -----

// CHECK-LABEL: @max
kgen.func @max(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.intr.maxnum
  %0 = pop.max %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}


// -----

// CHECK-LABEL: @min
kgen.func @min(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: llvm.intr.smin
  %0 = pop.min %arg0, %arg1 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// -----

// CHECK-LABEL: @min
kgen.func @min(%arg0: !pop.scalar<ui32>, %arg1: !pop.scalar<ui32>) -> !pop.scalar<ui32> {
  // CHECK: llvm.intr.umin
  %0 = pop.min %arg0, %arg1 : !pop.scalar<ui32>
  kgen.return %0 : !pop.scalar<ui32>
}

// -----

// CHECK-LABEL: @min
kgen.func @min(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.intr.minnum
  %0 = pop.min %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----
// CHECK-LABEL: @div
kgen.func @div(%arg0: !pop.scalar<si32>,
                 %arg1: !pop.scalar<ui32>,
                 %arg2: !pop.scalar<f32>) -> (
                  !pop.scalar<si32>,
                  !pop.scalar<ui32>,
                  !pop.scalar<f32>) {
  // CHECK: llvm.sdiv
  %0 = pop.div %arg0, %arg0 : !pop.scalar<si32>
  // CHECK: llvm.udiv
  %1 = pop.div %arg1, %arg1 : !pop.scalar<ui32>
  // CHECK: llvm.fdiv
  %2 = pop.div %arg2, %arg2 : !pop.scalar<f32>
  kgen.return %0, %1, %2 : !pop.scalar<si32>,!pop.scalar<ui32>,!pop.scalar<f32>
}

// -----
// CHECK-LABEL: @rem
kgen.func @rem(%arg0: !pop.scalar<si32>,
               %arg1: !pop.scalar<ui32>,
               %arg2: !pop.scalar<index>,
               %arg3: !pop.scalar<f32>) -> (
                  !pop.scalar<si32>,
                  !pop.scalar<ui32>,
                  !pop.scalar<index>,
                  !pop.scalar<f32>) {
  // CHECK: llvm.srem
  %0 = pop.rem %arg0, %arg0 : !pop.scalar<si32>
  // CHECK: llvm.urem
  %1 = pop.rem %arg1, %arg1 : !pop.scalar<ui32>
  // CHECK: llvm.srem
  %2 = pop.rem %arg2, %arg2 : !pop.scalar<index>
  // CHECK: llvm.frem
  %3 = pop.rem %arg3, %arg3 : !pop.scalar<f32>
  kgen.return %0, %1, %2, %3 : !pop.scalar<si32>,
                               !pop.scalar<ui32>,
                               !pop.scalar<index>,
                               !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @copysign
kgen.func @copysign(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.intr.copysign
  %0 = pop.copysign %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @fma
kgen.func @fma(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.intr.fma
  %0 = pop.fma %arg0, %arg0, %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @fma
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.func @fma(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK: %[[MUL:.*]] = llvm.mul %[[LHS]], %[[LHS]]
  // CHECK: %[[FMA:.*]] = llvm.add %[[MUL]], %[[RHS]]
  // CHECK: builtin.unrealized_conversion_cast %[[FMA]]
  %0 = pop.fma %arg0, %arg0, %arg1 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// -----

// CHECK-LABEL: @select
kgen.func @select(%arg0: !pop.scalar<bool>, %arg1: !pop.scalar<f32>, %arg2: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.select
  %0 = pop.select %arg0, %arg1, %arg2 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @load
kgen.func @load(%p: !pop.pointer<scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: llvm.load
  %0 = pop.load %p : !pop.pointer<scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @load_with_alignment
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
kgen.func @load_with_alignment(%p: !pop.pointer<scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: llvm.load %[[PTR]] {alignment = 128 : i64}
  %0 = pop.load %p align 128 : !pop.pointer<scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @load_with_alignment
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
kgen.func @load_with_alignment(%p: !pop.pointer<scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: llvm.load %[[PTR]]  {alignment = 128 : i64}
  %0 = pop.load %p align 128 : !pop.pointer<scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @store
kgen.func @store(%p: !pop.pointer<scalar<si32>>, %v: !pop.scalar<si32>) {
  // CHECK: llvm.store
  pop.store %v, %p : !pop.pointer<scalar<si32>>
  kgen.return
}

// -----

// CHECK-LABEL: @store_with_alignment
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.func @store_with_alignment(%p: !pop.pointer<scalar<si32>>, %v: !pop.scalar<si32>) {
  // CHECK-DAG: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-DAG: %[[VAL:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK: llvm.store %[[VAL]], %[[PTR]] {alignment = 128 : i64}
  pop.store %v, %p align 128 : !pop.pointer<scalar<si32>>
  kgen.return
}

// -----

// CHECK-LABEL: @store_with_alignment
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.func @store_with_alignment(%p: !pop.pointer<scalar<si32>>, %v: !pop.scalar<si32>) {
  // CHECK-DAG: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-DAG: %[[VAL:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK: llvm.store %[[VAL]], %[[PTR]]  {alignment = 128 : i64}
  pop.store %v, %p align 128 : !pop.pointer<scalar<si32>>
  kgen.return
}

// -----

// CHECK-LABEL: @offset
kgen.func @offset(%p: !pop.pointer<scalar<f32>>, %i: index) -> !pop.pointer<scalar<f32>> {
  // CHECK: llvm.getelementptr %{{.*}}[{{.*}}]
  %0 = pop.offset %p[%i] : !pop.pointer<scalar<f32>>
  kgen.return %0 : !pop.pointer<scalar<f32>>
}

// -----

// CHECK-LABEL: @shifts
kgen.func @shifts(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>, %arg2: !pop.scalar<ui32>, %arg3: !pop.scalar<ui32>) {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1 : !pop.scalar<si32>
  // CHECK: llvm.ashr
  %1 = pop.shr %arg0, %arg1 : !pop.scalar<si32>
  // CHECKL llvm.lshr
  %2 = pop.shr %arg2, %arg3 : !pop.scalar<ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @shifts
kgen.func @shifts(%arg0: !pop.scalar<index>, %arg1: !pop.scalar<index>) {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1 : !pop.scalar<index>
  kgen.return
}

// -----

// CHECK-LABEL: @simd_shift
kgen.func @simd_shift(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>, %arg2: !pop.simd<4, ui32>, %arg3: !pop.simd<4, ui32>) {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1 : !pop.simd<4, si32>
  // CHECK: llvm.ashr
  %1 = pop.shr %arg0, %arg1 : !pop.simd<4, si32>
  // CHECKL llvm.lshr
  %2 = pop.shr %arg2, %arg3 : !pop.simd<4, ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @shifts
kgen.func @shifts(%arg0: !pop.scalar<index>, %arg1: !pop.scalar<index>) {
  // CHECK: llvm.ashr
  %0 = pop.shr %arg0, %arg1 : !pop.scalar<index>
  kgen.return
}

// -----

// CHECK-LABEL: @cmp_bool
kgen.func @cmp_bool(%lhs: !pop.scalar<bool>, %rhs: !pop.scalar<bool>) {
  // CHECK: llvm.icmp "eq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.scalar<bool>
  // CHECK: llvm.icmp "ne"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.scalar<bool>
  // CHECK: llvm.icmp "ult"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.scalar<bool>
  // CHECK: llvm.icmp "ugt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.scalar<bool>
  // CHECK: llvm.icmp "ule"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.scalar<bool>
  // CHECK: llvm.icmp "uge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.scalar<bool>
  kgen.return
}

// -----

// CHECK-LABEL: @cmp_uint
kgen.func @cmp_uint(%lhs: !pop.scalar<ui32>, %rhs: !pop.scalar<ui32>) {
  // CHECK: llvm.icmp "eq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.scalar<ui32>
  // CHECK: llvm.icmp "ne"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.scalar<ui32>
  // CHECK: llvm.icmp "ult"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.scalar<ui32>
  // CHECK: llvm.icmp "ugt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.scalar<ui32>
  // CHECK: llvm.icmp "ule"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.scalar<ui32>
  // CHECK: llvm.icmp "uge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.scalar<ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @cmp_sint
kgen.func @cmp_sint(%lhs: !pop.scalar<si32>, %rhs: !pop.scalar<si32>) {
  // CHECK: llvm.icmp "eq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.scalar<si32>
  // CHECK: llvm.icmp "ne"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.scalar<si32>
  // CHECK: llvm.icmp "slt"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.scalar<si32>
  // CHECK: llvm.icmp "sgt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.scalar<si32>
  // CHECK: llvm.icmp "sle"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.scalar<si32>
  // CHECK: llvm.icmp "sge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.scalar<si32>
  kgen.return
}

// -----

// CHECK-LABEL: @cmp_fp
kgen.func @cmp_fp(%lhs: !pop.scalar<f32>, %rhs: !pop.scalar<f32>) {
  // CHECK: llvm.fcmp "oeq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.scalar<f32>
  // CHECK: llvm.fcmp "one"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.scalar<f32>
  // CHECK: llvm.fcmp "olt"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.scalar<f32>
  // CHECK: llvm.fcmp "ogt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.scalar<f32>
  // CHECK: llvm.fcmp "ole"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.scalar<f32>
  // CHECK: llvm.fcmp "oge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.scalar<f32>
  kgen.return
}

// -----

// CHECK-LABEL: @cmp_index
kgen.func @cmp_index(%lhs: !pop.scalar<index>, %rhs: !pop.scalar<index>) {
  // CHECK: llvm.icmp "eq"
  %0 = pop.cmp eq(%lhs, %rhs) : !pop.scalar<index>
  // CHECK: llvm.icmp "ne"
  %1 = pop.cmp ne(%lhs, %rhs) : !pop.scalar<index>
  // CHECK: llvm.icmp "slt"
  %2 = pop.cmp lt(%lhs, %rhs) : !pop.scalar<index>
  // CHECK: llvm.icmp "sgt"
  %3 = pop.cmp gt(%lhs, %rhs) : !pop.scalar<index>
  // CHECK: llvm.icmp "sle"
  %4 = pop.cmp le(%lhs, %rhs) : !pop.scalar<index>
  // CHECK: llvm.icmp "sge"
  %5 = pop.cmp ge(%lhs, %rhs) : !pop.scalar<index>
  kgen.return
}


// -----

// CHECK-LABEL: @and
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<bool>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<bool>
kgen.func @and(%lhs: !pop.scalar<bool>, %rhs: !pop.scalar<bool>) -> !pop.scalar<bool> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.and %[[LHS]], %[[RHS]] : i1
  %0 = pop.and %lhs, %rhs : !pop.scalar<bool>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<bool>
}

// -----

// CHECK-LABEL: @and
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<si8>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<si8>
kgen.func @and(%lhs: !pop.scalar<si8>, %rhs: !pop.scalar<si8>) -> !pop.scalar<si8> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.and %[[LHS]], %[[RHS]] : i8
  %0 = pop.and %lhs, %rhs : !pop.scalar<si8>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<si8>
}

// -----

// CHECK-LABEL: @and
// CHECK-SAME: %[[LHS0:.*]]: !pop.simd<4, si32>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.simd<4, si32>
kgen.func @and(%lhs: !pop.simd<4, si32>, %rhs: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.and %[[LHS]], %[[RHS]] : vector<4xi32>
  %0 = pop.and %lhs, %rhs : !pop.simd<4, si32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @and
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<index>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<index>
kgen.func @and(%lhs: !pop.scalar<index>, %rhs: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.and %[[LHS]], %[[RHS]]
  %0 = pop.and %lhs, %rhs : !pop.scalar<index>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<index>
}

// -----

// CHECK-LABEL: @or
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<bool>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<bool>
kgen.func @or(%lhs: !pop.scalar<bool>, %rhs: !pop.scalar<bool>) -> !pop.scalar<bool> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.or %[[LHS]], %[[RHS]] : i1
  %0 = pop.or %lhs, %rhs : !pop.scalar<bool>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<bool>
}

// -----

// CHECK-LABEL: @or
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<si8>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<si8>
kgen.func @or(%lhs: !pop.scalar<si8>, %rhs: !pop.scalar<si8>) -> !pop.scalar<si8> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.or %[[LHS]], %[[RHS]] : i8
  %0 = pop.or %lhs, %rhs : !pop.scalar<si8>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<si8>
}

// -----

// CHECK-LABEL: @or
// CHECK-SAME: %[[LHS0:.*]]: !pop.simd<4, si32>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.simd<4, si32>
kgen.func @or(%lhs: !pop.simd<4, si32>, %rhs: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.or %[[LHS]], %[[RHS]] : vector<4xi32>
  %0 = pop.or %lhs, %rhs : !pop.simd<4, si32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @or
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<index>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<index>
kgen.func @or(%lhs: !pop.scalar<index>, %rhs: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.or %[[LHS]], %[[RHS]]
  %0 = pop.or %lhs, %rhs : !pop.scalar<index>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<index>
}

// -----

// CHECK-LABEL: @xor
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<bool>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<bool>
kgen.func @xor(%lhs: !pop.scalar<bool>, %rhs: !pop.scalar<bool>) -> !pop.scalar<bool> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.xor %[[LHS]], %[[RHS]] : i1
  %0 = pop.xor %lhs, %rhs : !pop.scalar<bool>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<bool>
}

// -----

// CHECK-LABEL: @xor
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<si8>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<si8>
kgen.func @xor(%lhs: !pop.scalar<si8>, %rhs: !pop.scalar<si8>) -> !pop.scalar<si8> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.xor %[[LHS]], %[[RHS]] : i8
  %0 = pop.xor %lhs, %rhs : !pop.scalar<si8>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<si8>
}

// -----

// CHECK-LABEL: @xor
// CHECK-SAME: %[[LHS0:.*]]: !pop.simd<4, si32>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.simd<4, si32>
kgen.func @xor(%lhs: !pop.simd<4, si32>, %rhs: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.xor %[[LHS]], %[[RHS]] : vector<4xi32>
  %0 = pop.xor %lhs, %rhs : !pop.simd<4, si32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

// CHECK-LABEL: @xor
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<index>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<index>
kgen.func @xor(%lhs: !pop.scalar<index>, %rhs: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.xor %[[LHS]], %[[RHS]]
  %0 = pop.xor %lhs, %rhs : !pop.scalar<index>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<index>
}


// -----

// CHECK-LABEL: @cmp_simd
kgen.func @cmp_simd(%lhs: !pop.simd<4, f32>, %rhs: !pop.simd<4, f32>) -> !pop.simd<4, bool> {
  // CHECK: llvm.fcmp {{.*}} : vector<4xf32>
  %0 = pop.cmp lt(%lhs, %rhs) : !pop.simd<4, f32>
  // CHECK: vector<4xi1>
  kgen.return %0 : !pop.simd<4, bool>
}

// -----

// CHECK-LABEL: @pointer_to_index
kgen.func @pointer_to_index(%a: !pop.pointer<scalar<f32>>, %b: !pop.pointer<simd<4, si32>>)
    -> (!pop.scalar<index>, !pop.scalar<index>) {
  // CHECK: llvm.ptrtoint
  %0 = pop.pointer_to_index %a : !pop.pointer<scalar<f32>> to !pop.scalar<index>
  // CHECK: llvm.ptrtoint
  %1 = pop.pointer_to_index %b : !pop.pointer<simd<4, si32>> to !pop.scalar<index>
  kgen.return %0, %1 : !pop.scalar<index>, !pop.scalar<index>
}

// -----

// CHECK-LABEL: @index_to_pointer
kgen.func @index_to_pointer(%idx: !pop.scalar<index>) -> (
      !pop.pointer<scalar<f32>>,
      !pop.pointer<simd<4, si32>>) {
  // CHECK: llvm.inttoptr
  %0 = pop.index_to_pointer %idx : !pop.scalar<index> to !pop.pointer<scalar<f32>>
  // CHECK: llvm.inttoptr
  %1 = pop.index_to_pointer %idx : !pop.scalar<index> to !pop.pointer<simd<4, si32>>
  kgen.return %0, %1 :!pop.pointer<scalar<f32>>, !pop.pointer<simd<4, si32>>
}

// -----

// CHECK-LABEL: @address_to_index
kgen.func @address_to_index(%a: !pop.simd<1, address>) -> !pop.scalar<index> {
  // CHECK: llvm.ptrtoint
  %0 = pop.pointer_to_index %a : !pop.simd<1, address> to !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// -----

// CHECK-LABEL: @simd_address_to_index
kgen.func @simd_address_to_index(%a: !pop.simd<4, address>) -> !pop.simd<4, index> {
  // CHECK: llvm.ptrtoint
  %0 = pop.pointer_to_index %a : !pop.simd<4, address> to !pop.simd<4, index>
  kgen.return %0 : !pop.simd<4, index>
}

// -----

// CHECK-LABEL: @index_to_address
kgen.func @index_to_address(%idx: !pop.scalar<index>) -> (!pop.simd<1, address>) {
  // CHECK: llvm.inttoptr
  %0 = pop.index_to_pointer %idx : !pop.scalar<index> to !pop.simd<1, address>
  kgen.return %0 : !pop.simd<1, address>
}

// -----

// CHECK-LABEL: @simd_index_to_address
kgen.func @simd_index_to_address(%idx: !pop.simd<4, index>) -> (!pop.simd<4, address>) {
  // CHECK: llvm.inttoptr
  %0 = pop.index_to_pointer %idx : !pop.simd<4, index> to !pop.simd<4, address>
  kgen.return %0 : !pop.simd<4, address>
}

// -----

// CHECK-LABEL: @lower_raise_cast
kgen.func @lower_raise_cast(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: builtin.unrealized_conversion_cast %arg0 : !pop.scalar<f32> to f32
  %0 = pop.cast_to_builtin %arg0 : !pop.scalar<f32> to f32
  // CHECK: %[[R:.*]] = llvm.fmul
  %1 = llvm.fmul %0, %0 : f32
  // CHECK: builtin.unrealized_conversion_cast %[[R]] : f32 to !pop.scalar<f32>
  %2 = pop.cast_from_builtin %1 : f32 to !pop.scalar<f32>
  kgen.return %2 : !pop.scalar<f32>
}
// -----

// CHECK-LABEL: @cast_to_builtin
kgen.func @cast_to_builtin(%arg0: !pop.scalar<index>) -> index {
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %arg0 : !pop.scalar<index> to i{{64|32}}
  // CHECK: builtin.unrealized_conversion_cast %[[TMP]] : i{{64|32}} to index
  %0 = pop.cast_to_builtin %arg0 : !pop.scalar<index> to index
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: @cast_from_builtin
kgen.func @cast_from_builtin(%arg0: index) -> !pop.scalar<index> {
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %arg0 : index to i{{64|32}}
  // CHECK: builtin.unrealized_conversion_cast %[[TMP]] : i{{64|32}} to !pop.scalar<index>
  %0 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}
// -----

!var = !pop.variant<f32, i64, struct<i8, i8, f64>>

// CHECK-LABEL: @test
kgen.func @test(%a: !var) -> i1 {
  // CHECK: %[[VAR:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[DISCR:.*]] = llvm.extractvalue %[[VAR]][1]
  // CHECK: %[[DISCR_VAL:.*]] = llvm.mlir.constant(0 : i2)
  // CHECK: %[[VAL:.*]] = llvm.icmp "eq" %[[DISCR]], %[[DISCR_VAL]]
  %0 = pop.variant.is f32, %a : !var
  // CHECK: return %[[VAL]]
  kgen.return %0 : i1
}

// -----

// CHECK-LABEL: @one_variant_type
// CHECK: !llvm.struct<(array<1 x i64>, i1)>
kgen.func @one_variant_type(%a: !pop.variant<i32>) -> i1 {
  %0 = pop.variant.is i32, %a : !pop.variant<i32>
  kgen.return %0 : i1
}

// -----

// CHECK-LABEL: @prefetch
kgen.func @prefetch(%p: !pop.pointer<scalar<si32>>) {
  // CHECK-DAG: [[RW:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[LOCALITY:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[CACHETAG:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: "llvm.intr.prefetch"(%{{[0-9]+}}, [[RW]], [[LOCALITY]], [[CACHETAG]])
  pop.prefetch %p (NoLocality, ReadDCache)
    : !pop.pointer<scalar<si32>>
  // CHECK-DAG: [[RW:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: [[LOCALITY:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: [[CACHETAG:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: "llvm.intr.prefetch"(%{{[0-9]+}}, [[RW]], [[LOCALITY]], [[CACHETAG]])
  pop.prefetch %p (LowLocality, WriteDCache)
    : !pop.pointer<scalar<si32>>
  // CHECK-DAG: [[RW:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[LOCALITY:%[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: [[CACHETAG:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: "llvm.intr.prefetch"(%{{[0-9]+}}, [[RW]], [[LOCALITY]], [[CACHETAG]])
  pop.prefetch %p (MediumLocality, ReadICache)
    : !pop.pointer<scalar<si32>>
  // CHECK-DAG: [[RW:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[LOCALITY:%[0-9]+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-DAG: [[CACHETAG:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: "llvm.intr.prefetch"(%{{[0-9]+}}, [[RW]], [[LOCALITY]], [[CACHETAG]])
  pop.prefetch %p (HighLocality, ReadDCache)
    : !pop.pointer<scalar<si32>>
  // CHECK-DAG: [[RW:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[LOCALITY:%[0-9]+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-DAG: [[CACHETAG:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: "llvm.intr.prefetch"(%{{[0-9]+}}, [[RW]], [[LOCALITY]], [[CACHETAG]])
  pop.prefetch %p (VeryHighLocality, ReadDCache)
    : !pop.pointer<scalar<si32>>
  kgen.return
}

// -----

// CHECK-LABEL: @array_create
kgen.func @array_create(%a: i32) -> !pop.array<2, i32> {
  // CHECK: %[[A0:.*]] = llvm.mlir.undef : !llvm.array<2 x i32>
  // CHECK: %[[A1:.*]] = llvm.insertvalue %arg0, %[[A0]][0]
  // CHECK: %[[A2:.*]] = llvm.insertvalue %arg0, %[[A1]][1]
  // CHECK: unrealized_conversion_cast %[[A2]]
  %0 = pop.array.create [%a, %a] : !pop.array<2, i32>
  kgen.return %0 : !pop.array<2, i32>
}

// -----

// CHECK-LABEL: @array_create_issue_4004
kgen.func @array_create_issue_4004() -> !pop.array<3, index> {
  // CHECK: %[[VAL0:.*]] = index.constant 64
  // CHECK: %[[VAL:.*]] = builtin.unrealized_conversion_cast %[[VAL0]]
  %val = index.constant 64
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.array<3 x i64>
  // CHECK: %[[A0:.*]] = llvm.insertvalue %[[VAL]], %[[UNDEF]][0] : !llvm.array<3 x i64>
  // CHECK: %[[A1:.*]] = llvm.insertvalue %[[VAL]], %[[A0]][1] : !llvm.array<3 x i64>
  // CHECK: %[[A2:.*]] = llvm.insertvalue %[[VAL]], %[[A1]][2] : !llvm.array<3 x i64>
  %arry = pop.array.create [%val, %val, %val] : !pop.array<3, index>
  // CHECK: %[[ARRY:.*]] = builtin.unrealized_conversion_cast %[[A2]]
  // CHECK: kgen.return %[[ARRY]]
  kgen.return %arry : !pop.array<3, index>
}


// -----

// CHECK-LABEL: @array_repeat0
kgen.func @array_repeat0(%a: i32, %b: i32) -> !pop.array<3, i32> {
  // CHECK: llvm.insertvalue %arg0, %{{.*}}[0]
  // CHECK: llvm.insertvalue %arg1, %{{.*}}[1]
  // CHECK: llvm.insertvalue %arg0, %{{.*}}[2]
  %0 = pop.array.repeat [%a, %b] : !pop.array<3, i32>
  kgen.return %0 : !pop.array<3, i32>
}

// -----

// CHECK-LABEL: @array_repeat1
kgen.func @array_repeat1(%a: i32, %b: i32) -> !pop.array<1, i32> {
  // CHECK: llvm.insertvalue %arg0, %{{.*}}[0]
  %0 = pop.array.repeat [%a, %b] : !pop.array<1, i32>
  kgen.return %0 : !pop.array<1, i32>
}

// -----

// CHECK-LABEL: @array_get_replace
kgen.func @array_get_replace(%a: !pop.array<2, i32>) -> !pop.array<2, i32> {
  // CHECK: llvm.extractvalue %{{.*}}[0]
  %0 = pop.array.get %a[0] : !pop.array<2, i32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1]
  %1 = pop.array.replace %0, %a[1] : !pop.array<2, i32>
  kgen.return %1 : !pop.array<2, i32>
}

// CHECK-LABEL: @array_gep
kgen.func @array_gep(%a: !pop.pointer<array<2, i32>>, %i: index) -> !pop.pointer<i32> {
  // CHECK: llvm.getelementptr %{{.*}}[0, %{{.*}}] : (!llvm.ptr<array<2 x i32>>, {{.*}}) -> !llvm.ptr<i32>
  %0 = pop.array.gep %a[%i] : <array<2, i32>>
  kgen.return %0 : !pop.pointer<i32>
}

// -----

// CHECK-LABEL: @memcpy
kgen.func @memcpy(%dest: !pop.pointer<scalar<si32>>,
                  %src: !pop.pointer<scalar<si32>>,
                  %size: index) {
  // CHECK: %[[DEST_CAST:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: %[[SRC_CAST:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: %[[SIZE_CAST:.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK: %[[VOLATILE:.*]] = llvm.mlir.constant(false) : i1
  // CHECK:  "llvm.intr.memcpy"(%[[DEST_CAST]], %[[SRC_CAST]], %[[SIZE_CAST]], %[[VOLATILE]]) : (!llvm.ptr<i32>, !llvm.ptr<i32>, i64, i1) -> ()
  pop.memcpy %dest, %src, %size : !pop.pointer<scalar<si32>>
  kgen.return
}


// -----

// CHECK-LABEL: @memcpy_volatile
kgen.func @memcpy_volatile(%dest: !pop.pointer<scalar<si32>>,
                           %src: !pop.pointer<scalar<si32>>,
                           %size: index) {
  // CHECK: %[[DEST_CAST:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: %[[SRC_CAST:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: %[[SIZE_CAST:.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK: %[[VOLATILE:.*]] = llvm.mlir.constant(true) : i1
  // CHECK:  "llvm.intr.memcpy"(%[[DEST_CAST]], %[[SRC_CAST]], %[[SIZE_CAST]], %[[VOLATILE]]) : (!llvm.ptr<i32>, !llvm.ptr<i32>, i64, i1) -> ()
  pop.memcpy volatile %dest, %src, %size : !pop.pointer<scalar<si32>>
  kgen.return
}

// -----

// CHECK-LABEL: @memcpy_inline
kgen.func @memcpy_inline(%dest: !pop.pointer<scalar<f32>>,
                         %src: !pop.pointer<scalar<f32>>,
                         %size: index) {
  // CHECK: %[[DEST_CAST:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: %[[SRC_CAST:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: %[[SIZE_CAST:.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK: %[[VOLATILE:.*]] = llvm.mlir.constant(false) : i1
  // CHECK:  "llvm.intr.memcpy.inline"(%[[DEST_CAST]], %[[SRC_CAST]], %[[SIZE_CAST]], %[[VOLATILE]]) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i1) -> ()
  pop.memcpy inline %dest, %src, %size : !pop.pointer<scalar<f32>>
  kgen.return
}

// -----

// CHECK-LABEL: @memset
kgen.func @memset(%dest: !pop.pointer<scalar<si32>>,
                  %val: !pop.scalar<ui8>,
                  %size: index) {
  // CHECK: %[[DEST_CAST:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: %[[VAL_CAST:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: %[[SIZE_CAST:.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK: %[[VOLATILE:.*]] = llvm.mlir.constant(false) : i1
  // CHECK:  "llvm.intr.memset"(%[[DEST_CAST]], %[[VAL_CAST]], %[[SIZE_CAST]], %[[VOLATILE]]) : (!llvm.ptr<i32>, i8, i64, i1) -> ()
  pop.memset %dest, %val, %size : !pop.pointer<scalar<si32>>
  kgen.return
}


// -----

// CHECK-LABEL: @memset_volatile
kgen.func @memset_volatile(%dest: !pop.pointer<scalar<si32>>,
                           %val: !pop.scalar<ui8>,
                           %size: index) {
  // CHECK: %[[DEST_CAST:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: %[[VAL_CAST:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: %[[SIZE_CAST:.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK: %[[VOLATILE:.*]] = llvm.mlir.constant(true) : i1
  // CHECK:  "llvm.intr.memset"(%[[DEST_CAST]], %[[VAL_CAST]], %[[SIZE_CAST]], %[[VOLATILE]]) : (!llvm.ptr<i32>, i8, i64, i1) -> ()
  pop.memset volatile %dest, %val, %size : !pop.pointer<scalar<si32>>
  kgen.return
}

// -----

// CHECK-LABEL: @call_intrinsic
kgen.func @call_intrinsic(%inp: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: %[[INP_CAST:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: %[[RESULT:.*]] = llvm.call_intrinsic "llvm.round"(%[[INP_CAST]]) : (f32) -> f32
  // CHECK: %[[RES_CAST:.*]] = builtin.unrealized_conversion_cast %[[RESULT]]
  %0 = pop.call_llvm_intrinsic "llvm.round"(%inp) : (!pop.scalar<f32>) -> !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @call_void_intrinsic
kgen.func @call_void_intrinsic(%arg0: !pop.scalar<si64>,
                               %arg1: !pop.pointer<si8>) {
  // CHECK: %[[ARG0_CAST:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: %[[ARG1_CAST:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: llvm.call_intrinsic "llvm.lifetime.start"(%[[ARG0_CAST]], %[[ARG1_CAST]]) : (i64, !llvm.ptr<i8>) -> ()
  pop.call_llvm_intrinsic "llvm.lifetime.start"(%arg0, %arg1) :
    (!pop.scalar<si64>, !pop.pointer<si8>) -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @inline_asm
kgen.func @inline_asm(
    %arg0: !pop.scalar<si32>,
    %arg1: !pop.scalar<si64>) {
  // CHECK: llvm.inline_asm asm_dialect = att "bswap $0", "=r,r" %0 : (i32) -> i8
  %0 = pop.inline_asm "bswap $0", "=r,r" %arg0 : (!pop.scalar<si32>) -> i8
  // CHECK: llvm.inline_asm asm_dialect = att "something", "anotherthing" %0, %1 : (i32, i64) -> i8
  %1 = pop.inline_asm "something", "anotherthing" %arg0, %arg1 :
    (!pop.scalar<si32>, !pop.scalar<si64>) -> i8
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "something", "anotherthing" %0, %1 : (i32, i64) -> i8
  %2 = pop.inline_asm side_effecting "something", "anotherthing" %arg0, %arg1 :
    (!pop.scalar<si32>, !pop.scalar<si64>) -> i8
  // CHECK: llvm.inline_asm is_align_stack asm_dialect = att "something", "anotherthing" %0, %1 : (i32, i64) -> i8
  %3 = pop.inline_asm stack_aligned "something", "anotherthing" %arg0, %arg1 :
    (!pop.scalar<si32>, !pop.scalar<si64>) -> i8
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @atomic_cmpxchg
kgen.func @atomic_cmpxchg(%ptr: !pop.pointer<scalar<index>>,
                          %cmp: !pop.scalar<index>,
                          %new: !pop.scalar<index>) {
  // CHECK: llvm.cmpxchg {{.*}} monotonic monotonic
  %0 = pop.atomic.cmpxchg %ptr, %cmp, %new monotonic monotonic :
                    !pop.pointer<scalar<index>>

  // CHECK: llvm.cmpxchg {{.*}} acq_rel monotonic
  %1 = pop.atomic.cmpxchg %ptr, %cmp, %new acq_rel monotonic :
                    !pop.pointer<scalar<index>>

  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @atomic_rmw
kgen.func @atomic_rmw(%ptr0: !pop.pointer<scalar<index>>,
                      %val0: !pop.scalar<index>,
                      %ptr1: !pop.pointer<scalar<f32>>,
                      %val1: !pop.scalar<f32>,
                      %ptr2: !pop.pointer<scalar<ui32>>,
                      %val2: !pop.scalar<ui32>) {
  // CHECK: llvm.atomicrmw add {{.*}} monotonic
  %0 = pop.atomic.rmw add(%ptr0, %val0) monotonic : !pop.pointer<scalar<index>>
  // CHECK: llvm.atomicrmw sub {{.*}} monotonic
  %1 = pop.atomic.rmw sub(%ptr0, %val0) monotonic : !pop.pointer<scalar<index>>
  // CHECK: llvm.atomicrmw _xor {{.*}} monotonic
  %2 = pop.atomic.rmw xor(%ptr0, %val0) monotonic : !pop.pointer<scalar<index>>
  // CHECK: llvm.atomicrmw min {{.*}} monotonic
  %3 = pop.atomic.rmw min(%ptr0, %val0) monotonic : !pop.pointer<scalar<index>>
  // CHECK: llvm.atomicrmw max {{.*}} monotonic
  %4 = pop.atomic.rmw max(%ptr0, %val0) monotonic : !pop.pointer<scalar<index>>

  // CHECK: llvm.atomicrmw fadd {{.*}} monotonic
  %5 = pop.atomic.rmw add(%ptr1, %val1) monotonic : !pop.pointer<scalar<f32>>

  // CHECK: llvm.atomicrmw umax {{.*}} monotonic
  %6 = pop.atomic.rmw max(%ptr2, %val2) monotonic : !pop.pointer<scalar<ui32>>
  kgen.return
}

// -----

// CHECK-LABEL: @variadic_create
// CHECK-SAME: %[[A0:.*]]: i32
kgen.func @variadic_create(%a: i32) {
  // CHECK: %[[ALLOCA_SIZE:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_SIZE]] x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr<i32>
  // CHECK: llvm.intr.lifetime.start 8, %[[ALLOCA]] : !llvm.ptr<i32>
  // CHECK: %[[GEPI0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[ALLOCA]][0, %[[GEPI0]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
  // CHECK: llvm.store %[[A0]], %[[GEP0]] : !llvm.ptr<i32>
  // CHECK: %[[GEPI1:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[ALLOCA]][0, %[[GEPI1]]]
  // CHECK: llvm.store %[[A0]], %[[GEP1]]
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[STRUCT1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i64)>
  // CHECK: %[[STRUCT2:.*]] = llvm.insertvalue %[[ALLOCA]], %[[STRUCT1]][0]
  // CHECK: llvm.insertvalue %[[SIZE]], %[[STRUCT2]][1]
  // CHECK: llvm.intr.lifetime.end 8, %[[ALLOCA]]
  %0 = pop.variadic.create [%a, %a] : !pop.variadic<i32>
  kgen.return
}

// -----

// CHECK-LABEL: @variadic_create_empty
kgen.func @variadic_create_empty() {
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i32>, i64)>
  %0 = pop.variadic.create [] : !pop.variadic<i32>
  kgen.return
}

// -----

// CHECK-LABEL: @variadic_create_index
kgen.func @variadic_create_index() {
  %0 = index.constant 64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i64>, i64)>
  %1 = pop.variadic.create [%0] : !pop.variadic<index>
  kgen.return
}

// -----

// CHECK-LABEL: @variadic_size
kgen.func @variadic_size(%arg0: !pop.variadic<f32>) -> index {
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, i64)>
  %0 = pop.variadic.size %arg0 : !pop.variadic<f32>
  kgen.return %0 : index
}
