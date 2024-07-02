// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @neg_f32
kgen.func @neg_f32(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.fneg %0 {fastmathFlags = #llvm.fastmath<contract>}
  %0 = pop.neg %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @neg_si32
// CHECK-SAME: %[[ARG0:.*]]:
kgen.func @neg_si32(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-DAG: %[[LHS:.*]] = llvm.mlir.constant(0 :
  // CHECK: llvm.sub %[[LHS]], %[[RHS]]
  %0 = pop.neg %arg0 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}


// CHECK-LABEL: @neg_index
// CHECK-SAME: %[[ARG0:.*]]:
kgen.func @neg_index(%arg0: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-DAG: %[[LHS:.*]] = llvm.mlir.constant(0 :
  // CHECK: llvm.sub %[[LHS]], %[[RHS]]
  %0 = pop.neg %arg0 : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// CHECK-LABEL: @neg_simd_int
kgen.func @neg_simd_int(%arg0: !pop.simd<2, index>,
                        %arg1: !pop.simd<2, ui8>,
                        %arg2: !pop.simd<2, si64>) {
  // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(dense<0> : vector<2xi64>) : vector<2xi64>
  // CHECK:             = llvm.sub [[ZERO]], {{.*}}  : vector<2xi64>
  %0 = pop.neg %arg0 : !pop.simd<2, index>
  // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(dense<0> : vector<2xi8>) : vector<2xi8>
  // CHECK:             = llvm.sub [[ZERO]], {{.*}}  : vector<2xi8>
  %1 = pop.neg %arg1 : !pop.simd<2, ui8>
  // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(dense<0> : vector<2xi64>) : vector<2xi64>
  // CHECK:             = llvm.sub [[ZERO]], {{.*}}  : vector<2xi64>
  %2 = pop.neg %arg2 : !pop.simd<2, si64>
  kgen.return
}

// CHECK-LABEL: @add_si32
kgen.func @add_si32(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg0 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// CHECK-LABEL: @add_f32
kgen.func @add_f32(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.fadd %0, %0 {fastmathFlags = #llvm.fastmath<contract>}
  %0 = pop.add %arg0, %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @add_index
kgen.func @add_index(%arg0: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg0 : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// CHECK-LABEL: @sub_si32
kgen.func @sub_si32(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: llvm.sub
  %0 = pop.sub %arg0, %arg1 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// CHECK-LABEL: @sub_f32
kgen.func @sub_f32(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: llvm.fsub [[ARG0]], [[ARG1]] {fastmathFlags = #llvm.fastmath<contract>}
  %0 = pop.sub %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}


// CHECK-LABEL: @sub_index
kgen.func @sub_index(%arg0: !pop.scalar<index>, %arg1: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK: llvm.sub
  %0 = pop.sub %arg0, %arg1 : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// CHECK-LABEL: @mul_si32
kgen.func @mul_si32(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: llvm.mul
  %0 = pop.mul %arg0, %arg0 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// CHECK-LABEL: @mul_f32
kgen.func @mul_f32(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.fmul %0, %0 {fastmathFlags = #llvm.fastmath<contract>}
  %0 = pop.mul %arg0, %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @max_si32
kgen.func @max_si32(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: llvm.intr.smax
  %0 = pop.max %arg0, %arg1 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// CHECK-LABEL: @max_index
kgen.func @max_index(%arg0: !pop.scalar<index>, %arg1: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK: llvm.intr.smax
  %0 = pop.max %arg0, %arg1 : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// CHECK-LABEL: @max_ui32
kgen.func @max_ui32(%arg0: !pop.scalar<ui32>, %arg1: !pop.scalar<ui32>) -> !pop.scalar<ui32> {
  // CHECK: llvm.intr.umax
  %0 = pop.max %arg0, %arg1 : !pop.scalar<ui32>
  kgen.return %0 : !pop.scalar<ui32>
}

// CHECK-LABEL: @max_f32
kgen.func @max_f32(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: llvm.intr.maxnum([[ARG0]], [[ARG1]]) {fastmathFlags = #llvm.fastmath<contract>}
  %0 = pop.max %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}


// CHECK-LABEL: @min_si32
kgen.func @min_si32(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK: llvm.intr.smin
  %0 = pop.min %arg0, %arg1 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// CHECK-LABEL: @min_ui32
kgen.func @min_ui32(%arg0: !pop.scalar<ui32>, %arg1: !pop.scalar<ui32>) -> !pop.scalar<ui32> {
  // CHECK: llvm.intr.umin
  %0 = pop.min %arg0, %arg1 : !pop.scalar<ui32>
  kgen.return %0 : !pop.scalar<ui32>
}

// CHECK-LABEL: @min_f32
kgen.func @min_f32(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: llvm.intr.minnum([[ARG0]], [[ARG1]]) {fastmathFlags = #llvm.fastmath<contract>}
  %0 = pop.min %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

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
  // CHECK: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<contract>}
  %2 = pop.div %arg2, %arg2 : !pop.scalar<f32>
  kgen.return %0, %1, %2 : !pop.scalar<si32>,!pop.scalar<ui32>,!pop.scalar<f32>
}

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
  // CHECK: llvm.frem %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<contract>}
  %3 = pop.rem %arg3, %arg3 : !pop.scalar<f32>
  kgen.return %0, %1, %2, %3 : !pop.scalar<si32>,
                               !pop.scalar<ui32>,
                               !pop.scalar<index>,
                               !pop.scalar<f32>
}

// CHECK-LABEL: @fma_f32
kgen.func @fma_f32(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.intr.fma
  %0 = pop.fma %arg0, %arg0, %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @fma_si32
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.func @fma_si32(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK: %[[MUL:.*]] = llvm.mul %[[LHS]], %[[LHS]]
  // CHECK: %[[FMA:.*]] = llvm.add %[[MUL]], %[[RHS]]
  // CHECK: builtin.unrealized_conversion_cast %[[FMA]]
  %0 = pop.fma %arg0, %arg0, %arg1 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// CHECK-LABEL: @simd_select
kgen.func @simd_select(%arg0: !pop.simd<4, bool>, %arg1: !pop.simd<4, f32>, %arg2: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK-DAG: [[ARG2:%.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK: llvm.select [[ARG0]], [[ARG1]], [[ARG2]] {fastmathFlags = #llvm.fastmath<contract>}
  %0 = pop.simd.select %arg0, %arg1, %arg2 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @load
kgen.func @load(%p: !kgen.pointer<scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: llvm.load
  %0 = pop.load %p : !kgen.pointer<scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @load_with_alignment
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
kgen.func @load_with_alignment(%p: !kgen.pointer<scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: llvm.load %[[PTR]]  {alignment = 128 : i64}
  %0 = pop.load %p align<128> : !kgen.pointer<scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @store
kgen.func @store(%p: !kgen.pointer<scalar<si32>>, %v: !pop.scalar<si32>) {
  // CHECK: llvm.store
  pop.store %v, %p : !kgen.pointer<scalar<si32>>
  kgen.return
}

// CHECK-LABEL: @store_with_alignment
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.func @store_with_alignment(%p: !kgen.pointer<scalar<si32>>, %v: !pop.scalar<si32>) {
  // CHECK-DAG: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-DAG: %[[VAL:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK: llvm.store %[[VAL]], %[[PTR]] {alignment = 128 : i64}
  pop.store %v, %p align<128> : !kgen.pointer<scalar<si32>>
  kgen.return
}

// CHECK-LABEL: @offset
kgen.func @offset(%p: !kgen.pointer<scalar<f32>>, %i: index) -> !kgen.pointer<scalar<f32>> {
  // CHECK: llvm.getelementptr %{{.*}}[{{.*}}]
  %0 = pop.offset %p[%i] : !kgen.pointer<scalar<f32>>
  kgen.return %0 : !kgen.pointer<scalar<f32>>
}

// CHECK-LABEL: @pop_select
kgen.func @pop_select(%arg0: i1, %arg1: !kgen.struct<(f32)>, %arg2: !kgen.struct<(f32)>) -> !kgen.struct<(f32)> {
  // CHECK: llvm.select %arg0, {{.*}}, {{.*}} {fastmathFlags = #llvm.fastmath<contract>} : i1, !llvm.struct<(f32)>
  %0 = pop.select %arg0, %arg1, %arg2 : !kgen.struct<(f32)>
  kgen.return %0 : !kgen.struct<(f32)>
}

// CHECK-LABEL: @shifts
kgen.func @shifts(
    %arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>,
    %arg2: !pop.scalar<ui32>, %arg3: !pop.scalar<ui32>,
    %arg4: !pop.scalar<index>, %arg5: !pop.scalar<index>) {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1 : !pop.scalar<si32>
  // CHECK: llvm.ashr
  %1 = pop.shr %arg0, %arg1 : !pop.scalar<si32>
  // CHECKL llvm.lshr
  %2 = pop.shr %arg2, %arg3 : !pop.scalar<ui32>
  // CHECK: llvm.shl
  %3 = pop.shl %arg4, %arg5 : !pop.scalar<index>
  // CHECK: llvm.ashr
  %4 = pop.shr %arg4, %arg5 : !pop.scalar<index>
  kgen.return
}

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

// CHECK-LABEL: @cmp_fp
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.func @cmp_fp(%lhs: !pop.scalar<f32>, %rhs: !pop.scalar<f32>) {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: llvm.fcmp "oeq" [[ARG0]], [[ARG1]] {fastmathFlags = #llvm.fastmath<contract>}
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


// CHECK-LABEL: @and_bool
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<bool>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<bool>
kgen.func @and_bool(%lhs: !pop.scalar<bool>, %rhs: !pop.scalar<bool>) -> !pop.scalar<bool> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.and %[[LHS]], %[[RHS]] : i1
  %0 = pop.and %lhs, %rhs : !pop.scalar<bool>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<bool>
}

// CHECK-LABEL: @and_si8
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<si8>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<si8>
kgen.func @and_si8(%lhs: !pop.scalar<si8>, %rhs: !pop.scalar<si8>) -> !pop.scalar<si8> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.and %[[LHS]], %[[RHS]] : i8
  %0 = pop.and %lhs, %rhs : !pop.scalar<si8>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<si8>
}

// CHECK-LABEL: @and_simd
// CHECK-SAME: %[[LHS0:.*]]: !pop.simd<4, si32>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.simd<4, si32>
kgen.func @and_simd(%lhs: !pop.simd<4, si32>, %rhs: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.and %[[LHS]], %[[RHS]] : vector<4xi32>
  %0 = pop.and %lhs, %rhs : !pop.simd<4, si32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @and_index
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<index>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<index>
kgen.func @and_index(%lhs: !pop.scalar<index>, %rhs: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.and %[[LHS]], %[[RHS]]
  %0 = pop.and %lhs, %rhs : !pop.scalar<index>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<index>
}

// CHECK-LABEL: @or_bool
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<bool>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<bool>
kgen.func @or_bool(%lhs: !pop.scalar<bool>, %rhs: !pop.scalar<bool>) -> !pop.scalar<bool> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.or %[[LHS]], %[[RHS]] : i1
  %0 = pop.or %lhs, %rhs : !pop.scalar<bool>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<bool>
}

// CHECK-LABEL: @or_si8
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<si8>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<si8>
kgen.func @or_si8(%lhs: !pop.scalar<si8>, %rhs: !pop.scalar<si8>) -> !pop.scalar<si8> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.or %[[LHS]], %[[RHS]] : i8
  %0 = pop.or %lhs, %rhs : !pop.scalar<si8>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<si8>
}

// CHECK-LABEL: @or_simd
// CHECK-SAME: %[[LHS0:.*]]: !pop.simd<4, si32>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.simd<4, si32>
kgen.func @or_simd(%lhs: !pop.simd<4, si32>, %rhs: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.or %[[LHS]], %[[RHS]] : vector<4xi32>
  %0 = pop.or %lhs, %rhs : !pop.simd<4, si32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @or
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<index>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<index>
kgen.func @or(%lhs: !pop.scalar<index>, %rhs: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.or %[[LHS]], %[[RHS]]
  %0 = pop.or %lhs, %rhs : !pop.scalar<index>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<index>
}

// CHECK-LABEL: @xor_bool
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<bool>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<bool>
kgen.func @xor_bool(%lhs: !pop.scalar<bool>, %rhs: !pop.scalar<bool>) -> !pop.scalar<bool> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.xor %[[LHS]], %[[RHS]] : i1
  %0 = pop.xor %lhs, %rhs : !pop.scalar<bool>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<bool>
}

// CHECK-LABEL: @xor_si8
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<si8>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<si8>
kgen.func @xor_si8(%lhs: !pop.scalar<si8>, %rhs: !pop.scalar<si8>) -> !pop.scalar<si8> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.xor %[[LHS]], %[[RHS]] : i8
  %0 = pop.xor %lhs, %rhs : !pop.scalar<si8>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<si8>
}

// CHECK-LABEL: @xor_simd
// CHECK-SAME: %[[LHS0:.*]]: !pop.simd<4, si32>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.simd<4, si32>
kgen.func @xor_simd(%lhs: !pop.simd<4, si32>, %rhs: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.xor %[[LHS]], %[[RHS]] : vector<4xi32>
  %0 = pop.xor %lhs, %rhs : !pop.simd<4, si32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @xor_index
// CHECK-SAME: %[[LHS0:.*]]: !pop.scalar<index>,
// CHECK-SAME: %[[RHS0:.*]]: !pop.scalar<index>
kgen.func @xor_index(%lhs: !pop.scalar<index>, %rhs: !pop.scalar<index>) -> !pop.scalar<index> {
  // CHECK-DAG: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS0]]
  // CHECK-DAG: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS0]]
  // CHECK: %[[AND:.*]] = llvm.xor %[[LHS]], %[[RHS]]
  %0 = pop.xor %lhs, %rhs : !pop.scalar<index>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[AND]]
  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : !pop.scalar<index>
}


// CHECK-LABEL: @cmp_simd
kgen.func @cmp_simd(%lhs: !pop.simd<4, f32>, %rhs: !pop.simd<4, f32>) -> !pop.simd<4, bool> {
  // CHECK: llvm.fcmp {{.*}} : vector<4xf32>
  %0 = pop.cmp lt(%lhs, %rhs) : !pop.simd<4, f32>
  // CHECK: vector<4xi1>
  kgen.return %0 : !pop.simd<4, bool>
}

// CHECK-LABEL: @pointer_to_index
kgen.func @pointer_to_index(%a: !kgen.pointer<scalar<f32>>, %b: !kgen.pointer<simd<4, si32>>)
    -> (!pop.scalar<index>, !pop.scalar<index>) {
  // CHECK: llvm.ptrtoint
  %0 = pop.pointer_to_index %a : !kgen.pointer<scalar<f32>> to !pop.scalar<index>
  // CHECK: llvm.ptrtoint
  %1 = pop.pointer_to_index %b : !kgen.pointer<simd<4, si32>> to !pop.scalar<index>
  kgen.return %0, %1 : !pop.scalar<index>, !pop.scalar<index>
}

// CHECK-LABEL: @index_to_pointer
kgen.func @index_to_pointer(%idx: !pop.scalar<index>) -> (
      !kgen.pointer<scalar<f32>>,
      !kgen.pointer<simd<4, si32>>) {
  // CHECK: llvm.inttoptr
  %0 = pop.index_to_pointer %idx : !pop.scalar<index> to !kgen.pointer<scalar<f32>>
  // CHECK: llvm.inttoptr
  %1 = pop.index_to_pointer %idx : !pop.scalar<index> to !kgen.pointer<simd<4, si32>>
  kgen.return %0, %1 :!kgen.pointer<scalar<f32>>, !kgen.pointer<simd<4, si32>>
}

// CHECK-LABEL: @address_to_index
kgen.func @address_to_index(%a: !pop.simd<1, address>) -> !pop.scalar<index> {
  // CHECK: llvm.ptrtoint
  %0 = pop.pointer_to_index %a : !pop.simd<1, address> to !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// CHECK-LABEL: @simd_address_to_index
kgen.func @simd_address_to_index(%a: !pop.simd<4, address>) -> !pop.simd<4, index> {
  // CHECK: llvm.ptrtoint
  %0 = pop.pointer_to_index %a : !pop.simd<4, address> to !pop.simd<4, index>
  kgen.return %0 : !pop.simd<4, index>
}

// CHECK-LABEL: @index_to_address
kgen.func @index_to_address(%idx: !pop.scalar<index>) -> (!pop.simd<1, address>) {
  // CHECK: llvm.inttoptr
  %0 = pop.index_to_pointer %idx : !pop.scalar<index> to !pop.simd<1, address>
  kgen.return %0 : !pop.simd<1, address>
}

// CHECK-LABEL: @simd_index_to_address
kgen.func @simd_index_to_address(%idx: !pop.simd<4, index>) -> (!pop.simd<4, address>) {
  // CHECK: llvm.inttoptr
  %0 = pop.index_to_pointer %idx : !pop.simd<4, index> to !pop.simd<4, address>
  kgen.return %0 : !pop.simd<4, address>
}

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

// CHECK-LABEL: @cast_to_builtin
kgen.func @cast_to_builtin(%arg0: !pop.scalar<index>) -> index {
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %arg0 : !pop.scalar<index> to i{{64|32}}
  // CHECK: builtin.unrealized_conversion_cast %[[TMP]] : i{{64|32}} to index
  %0 = pop.cast_to_builtin %arg0 : !pop.scalar<index> to index
  kgen.return %0 : index
}

// CHECK-LABEL: @cast_from_builtin
kgen.func @cast_from_builtin(%arg0: index) -> !pop.scalar<index> {
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %arg0 : index to i{{64|32}}
  // CHECK: builtin.unrealized_conversion_cast %[[TMP]] : i{{64|32}} to !pop.scalar<index>
  %0 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}

// CHECK-LABEL: @array_create
kgen.func @array_create(%a: i32) -> !pop.array<2, i32> {
  // CHECK: %[[A0:.*]] = llvm.mlir.undef : !llvm.array<2 x i32>
  // CHECK: %[[A1:.*]] = llvm.insertvalue %arg0, %[[A0]][0]
  // CHECK: %[[A2:.*]] = llvm.insertvalue %arg0, %[[A1]][1]
  // CHECK: unrealized_conversion_cast %[[A2]]
  %0 = pop.array.create [%a, %a] : !pop.array<2, i32>
  kgen.return %0 : !pop.array<2, i32>
}

// CHECK-LABEL: @array_create_issue_4004
kgen.func @array_create_issue_4004() -> !pop.array<3, index> {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(64 : i64)
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

// CHECK-LABEL: @array_repeat0
kgen.func @array_repeat0(%a: i32, %b: i32) -> !pop.array<3, i32> {
  // CHECK: llvm.insertvalue %arg0, %{{.*}}[0]
  // CHECK: llvm.insertvalue %arg1, %{{.*}}[1]
  // CHECK: llvm.insertvalue %arg0, %{{.*}}[2]
  %0 = pop.array.repeat [%a, %b] : !pop.array<3, i32>
  kgen.return %0 : !pop.array<3, i32>
}

// CHECK-LABEL: @call_intrinsic
kgen.func @call_intrinsic(%inp: !pop.scalar<f32>) -> (!pop.scalar<f32>, !pop.scalar<f32>) {
  // CHECK: %[[INP_CAST:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK: %[[RESULT_1:.*]] = llvm.call_intrinsic "llvm.round"(%[[INP_CAST]]) : (f32) -> f32
  // CHECK-SAME: fastmathFlags = #llvm.fastmath<nnan, reassoc>
  // CHECK: %[[RES_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[RESULT_1]]
  %0 = pop.call_llvm_intrinsic "llvm.round", (%inp) {fastmathFlags = #pop<fmf reassoc|nnan>} : (!pop.scalar<f32>) -> !pop.scalar<f32>
  // CHECK: %[[RESULT_2:.*]] = llvm.call_intrinsic "llvm.round"(%[[INP_CAST]]) : (f32) -> f32
  // CHECK-SAME: fastmathFlags = #llvm.fastmath<fast>
  // CHECK: %[[RES_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[RESULT_2]]
  %1 = pop.call_llvm_intrinsic "llvm.round", (%inp) {fastmathFlags = #pop<fmf fast>} : (!pop.scalar<f32>) -> !pop.scalar<f32>
  kgen.return %0, %1 : !pop.scalar<f32>, !pop.scalar<f32>
}

// CHECK-LABEL: @call_void_intrinsic
kgen.func @call_void_intrinsic(%arg0: !pop.scalar<si64>,
                               %arg1: !kgen.pointer<si8>) {
  // CHECK-DAG: %[[ARG0_CAST:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: %[[ARG1_CAST:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: llvm.call_intrinsic "llvm.lifetime.start"(%[[ARG0_CAST]], %[[ARG1_CAST]]) : (i64, !llvm.ptr) -> ()
  pop.call_llvm_intrinsic "llvm.lifetime.start", (%arg0, %arg1) :
    (!pop.scalar<si64>, !kgen.pointer<si8>) -> ()
  kgen.return
}

// CHECK-LABEL: @parametric_inline_asm
kgen.generator @parametric_inline_asm<asm: string, constraints: string>() {
  // CHECK-NEXT: inline_asm asm, constraints, ()
  pop.inline_asm asm, constraints, () : () -> ()
  kgen.return
}

// CHECK-LABEL: @inline_asm
kgen.func @inline_asm(
    %arg0: !pop.scalar<si32>,
    %arg1: !pop.scalar<si64>) {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %arg1

  // CHECK: llvm.inline_asm asm_dialect = att "bswap $0", "=r,r" [[ARG0]] : (i32) -> i8
  %0 = pop.inline_asm "bswap $0", "=r,r", (%arg0) : (!pop.scalar<si32>) -> i8
  // CHECK: llvm.inline_asm asm_dialect = att "something", "anotherthing" [[ARG0]], [[ARG1]] : (i32, i64) -> i8
  %1 = pop.inline_asm "something", "anotherthing", (%arg0, %arg1) :
    (!pop.scalar<si32>, !pop.scalar<si64>) -> i8
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "something", "anotherthing" [[ARG0]], [[ARG1]] : (i32, i64) -> i8
  %2 = pop.inline_asm side_effecting "something", "anotherthing", (%arg0, %arg1) :
    (!pop.scalar<si32>, !pop.scalar<si64>) -> i8
  // CHECK: llvm.inline_asm is_align_stack asm_dialect = att "something", "anotherthing" [[ARG0]], [[ARG1]] : (i32, i64) -> i8
  %3 = pop.inline_asm stack_aligned "something", "anotherthing", (%arg0, %arg1) :
    (!pop.scalar<si32>, !pop.scalar<si64>) -> i8
  // CHECK: [[R:%.*]] = llvm.inline_asm asm_dialect = att "two_results", "two_results" [[ARG0]] : (i32) -> !llvm.struct<(i32, i32)>
  // CHECK: llvm.extractvalue [[R]][0] : !llvm.struct<(i32, i32)>
  // CHECK: llvm.extractvalue [[R]][1] : !llvm.struct<(i32, i32)>
  %4:2 = pop.inline_asm "two_results", "two_results", (%arg0) : (!pop.scalar<si32>) -> (!pop.scalar<si32>, !pop.scalar<si32>)
  // CHECK: llvm.inline_asm asm_dialect = att "no_results", "no_results" [[ARG0]] : (i32) -> ()
  pop.inline_asm "no_results", "no_results", (%arg0) : (!pop.scalar<si32>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @atomic_cmpxchg
kgen.func @atomic_cmpxchg(%ptr: !kgen.pointer<scalar<index>>,
                          %cmp: !pop.scalar<index>,
                          %new: !pop.scalar<index>) {
  // CHECK: llvm.cmpxchg {{.*}} monotonic monotonic
  %0 = pop.atomic.cmpxchg %ptr, %cmp, %new monotonic monotonic :
                    !kgen.pointer<scalar<index>>

  // CHECK: llvm.cmpxchg {{.*}} acq_rel monotonic
  %1 = pop.atomic.cmpxchg %ptr, %cmp, %new acq_rel monotonic :
                    !kgen.pointer<scalar<index>>

  kgen.return
}

// CHECK-LABEL: kgen.func @atomic_rmw
kgen.func @atomic_rmw(%ptr0: !kgen.pointer<scalar<index>>,
                      %val0: !pop.scalar<index>,
                      %ptr1: !kgen.pointer<scalar<f32>>,
                      %val1: !pop.scalar<f32>,
                      %ptr2: !kgen.pointer<scalar<ui32>>,
                      %val2: !pop.scalar<ui32>) {
  // CHECK: llvm.atomicrmw add {{.*}} monotonic
  %0 = pop.atomic.rmw add(%ptr0, %val0) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: llvm.atomicrmw sub {{.*}} monotonic
  %1 = pop.atomic.rmw sub(%ptr0, %val0) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: llvm.atomicrmw _xor {{.*}} monotonic
  %2 = pop.atomic.rmw xor(%ptr0, %val0) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: llvm.atomicrmw min {{.*}} monotonic
  %3 = pop.atomic.rmw min(%ptr0, %val0) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: llvm.atomicrmw max {{.*}} monotonic
  %4 = pop.atomic.rmw max(%ptr0, %val0) monotonic : !kgen.pointer<scalar<index>>

  // CHECK: llvm.atomicrmw fadd {{.*}} monotonic
  %5 = pop.atomic.rmw add(%ptr1, %val1) monotonic : !kgen.pointer<scalar<f32>>

  // CHECK: llvm.atomicrmw umax {{.*}} monotonic
  %6 = pop.atomic.rmw max(%ptr2, %val2) monotonic : !kgen.pointer<scalar<ui32>>
  kgen.return
}

// CHECK-LABEL: kgen.func @stack_lifetimes
kgen.func @stack_lifetimes() {
  // CHECK-NEXT: [[ONE:%.*]] = llvm.mlir.constant(1 : i64)
  // CHECK-NEXT: [[ALLOC:%.*]] = llvm.alloca [[ONE]] x i64
  // CHECK-NEXT: llvm.intr.lifetime.end 8, [[ALLOC]]
  %0 = pop.stack_allocation 1 x index marked
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: kgen.func @multi_lifetimes
kgen.func @multi_lifetimes() {
  // CHECK: [[A0:%.*]] = llvm.alloca {{.*}} x i64
  // CHECK-NEXT: llvm.intr.lifetime.end 8, [[A0]]
  // CHECK: [[A1:%.*]] = llvm.alloca {{.*}} x i32
  // CHECK-NEXT: llvm.intr.lifetime.end 4, [[A1]]
  %0 = pop.stack_allocation 1 x i64 marked
  %1 = pop.stack_allocation 1 x i32 marked
  // CHECK-NEXT: lifetime.start 8, [[A0]]
  // CHECK-NEXT: lifetime.start 4, [[A1]]
  pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<i64>, !kgen.pointer<i32>
  // CHECK-NEXT: lifetime.end 8, [[A0]]
  // CHECK-NEXT: lifetime.end 4, [[A1]]
  pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<i64>, !kgen.pointer<i32>
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @variadic_create
// CHECK-SAME: %[[A0:.*]]: i24, %[[B0:.*]]: i24
kgen.func @variadic_create(%a: i24, %b: i24) {
  // CHECK: %[[ALLOCA_SIZE:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_SIZE]] x i24 {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK: llvm.intr.lifetime.start 8, %[[ALLOCA]] : !llvm.ptr
  // CHECK: %[[GEPI0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[ALLOCA]][%[[GEPI0]]] : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK: llvm.store %[[A0]], %[[GEP0]] : i24, !llvm.ptr
  // CHECK: %[[GEPI1:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[ALLOCA]][%[[GEPI1]]]
  // CHECK: llvm.store %[[B0]], %[[GEP1]]
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[STRUCT1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  // CHECK: %[[STRUCT2:.*]] = llvm.insertvalue %[[ALLOCA]], %[[STRUCT1]][0]
  // CHECK: llvm.insertvalue %[[SIZE]], %[[STRUCT2]][1]
  // CHECK: llvm.intr.lifetime.end 8, %[[ALLOCA]]
  %0 = pop.variadic.create [%a, %b] : !kgen.variadic<i24>
  kgen.return
}

// CHECK-LABEL: @variadic_splat
// CHECK-SAME: %[[A0:.*]]: i24
kgen.func @variadic_splat(%a: i24) {
  // CHECK: %[[ALLOCA_SIZE:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_SIZE]] x i24 {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK: llvm.intr.lifetime.start 8, %[[ALLOCA]] : !llvm.ptr
  // CHECK: %[[GEPI0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[ALLOCA]][%[[GEPI0]]] : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK: llvm.store %[[A0]], %[[GEP0]] : i24, !llvm.ptr
  // CHECK: %[[GEPI1:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[ALLOCA]][%[[GEPI1]]]
  // CHECK: llvm.store %[[A0]], %[[GEP1]]
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[STRUCT1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  // CHECK: %[[STRUCT2:.*]] = llvm.insertvalue %[[ALLOCA]], %[[STRUCT1]][0]
  // CHECK: llvm.insertvalue %[[SIZE]], %[[STRUCT2]][1]
  // CHECK: llvm.intr.lifetime.end 8, %[[ALLOCA]]
  %0 = pop.variadic.splat 2, %a : !kgen.variadic<i24>
  kgen.return
}

// CHECK-LABEL: @variadic_create_empty
kgen.func @variadic_create_empty() {
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %0 = pop.variadic.create [] : !kgen.variadic<i32>
  kgen.return
}

// CHECK-LABEL: @variadic_create_index
kgen.func @variadic_create_index() {
  %0 = index.constant 64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %1 = pop.variadic.create [%0] : !kgen.variadic<index>
  kgen.return
}

// CHECK-LABEL: @variadic_size
kgen.func @variadic_size(%arg0: !kgen.variadic<f32>) -> index {
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, i64)>
  %0 = pop.variadic.size %arg0 : !kgen.variadic<f32>
  kgen.return %0 : index
}

// CHECK-LABEL: @variadic_get
kgen.func @variadic_get(%arg0: !kgen.variadic<f32>) -> f32 {
  %0 = index.constant 3
  // CHECK: %[[ALLOCA:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr, i64)>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][%{{[0-9]+}}]
  // CHECK: %[[LOAD:.*]] = llvm.load %[[GEP]]
  %1 = pop.variadic.get %arg0[%0] : !kgen.variadic<f32>
  kgen.return %1 : f32
}

// CHECK-LABEL: @extract_size
kgen.func @extract_size(%a: !kgen.string) ->  index {
  // CHECK: unrealized_conversion_cast %arg0 : !kgen.string to !llvm.struct<(ptr, i64)>
  // CHECK: llvm.extractvalue %0[1] : !llvm.struct<(ptr, i64)>
  %1 = pop.string.size %a
  kgen.return %1: index
}

// CHECK-LABEL: @extract_addr
kgen.func @extract_addr(%a: !kgen.string) -> !kgen.pointer<scalar<si8>> {
  // CHECK: llvm.extractvalue %0[0] : !llvm.struct<(ptr, i64)>
  %1 = pop.string.address %a
  kgen.return %1: !kgen.pointer<scalar<si8>>
}

// CHECK-LABEL: kgen.func @fence
kgen.func @fence() {
  // CHECK: llvm.fence acquire
  pop.fence acquire
  // CHECK: llvm.fence syncscope("agent") seq_cst
  pop.fence syncscope("agent") seq_cst
  // CHECK: llvm.fence syncscope("singlethread") acq_rel
  pop.fence syncscope("singlethread") acq_rel
  kgen.return
}

}
