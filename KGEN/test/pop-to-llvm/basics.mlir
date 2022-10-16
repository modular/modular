// RUN: kgen-opt -split-input-file -pass-pipeline='kgen.func(lower-pop-to-llvm)' %s | FileCheck %s

// CHECK-LABEL: @constant
kgen.func @constant() -> !pop.scalar<f32> {
  // CHECK: llvm.mlir.constant(1.{{0+}}e+00 : f32) : f32
  %cst0 = pop.constant(1.0 : f32) : !pop.scalar<f32>
  kgen.return %cst0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @constant
kgen.func @constant() -> !pop.scalar<si64> {
  // CHECK: llvm.mlir.constant(1 : si64) : i64
  %cst0 = pop.constant(1 : si64) : !pop.scalar<si64>
  kgen.return %cst0 : !pop.scalar<si64>
}

// -----

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
kgen.func @load(%p: !pop.pointer<!pop.scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: llvm.load
  %0 = pop.load %p : !pop.pointer<!pop.scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @load_with_alignment
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
kgen.func @load_with_alignment(%p: !pop.pointer<!pop.scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: llvm.load %[[PTR]] {alignment = 128 : i64}
  %0 = pop.load %p align 128 : !pop.pointer<!pop.scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @load_with_alignment
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
kgen.func @load_with_alignment(%p: !pop.pointer<!pop.scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: llvm.load %[[PTR]]  {alignment = 128 : i64}
  %0 = pop.load %p align 128 : !pop.pointer<!pop.scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @store
kgen.func @store(%p: !pop.pointer<!pop.scalar<si32>>, %v: !pop.scalar<si32>) {
  // CHECK: llvm.store
  pop.store %v, %p : !pop.pointer<!pop.scalar<si32>>
  kgen.return
}

// -----

// CHECK-LABEL: @store_with_alignment
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.func @store_with_alignment(%p: !pop.pointer<!pop.scalar<si32>>, %v: !pop.scalar<si32>) {
  // CHECK-DAG: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-DAG: %[[VAL:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK: llvm.store %[[VAL]], %[[PTR]] {alignment = 128 : i64}
  pop.store %v, %p align 128 : !pop.pointer<!pop.scalar<si32>>
  kgen.return
}

// -----

// CHECK-LABEL: @store_with_alignment
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.func @store_with_alignment(%p: !pop.pointer<!pop.scalar<si32>>, %v: !pop.scalar<si32>) {
  // CHECK-DAG: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-DAG: %[[VAL:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK: llvm.store %[[VAL]], %[[PTR]]  {alignment = 128 : i64}
  pop.store %v, %p align 128 : !pop.pointer<!pop.scalar<si32>>
  kgen.return
}

// -----

// CHECK-LABEL: @offset
kgen.func @offset(%p: !pop.pointer<!pop.scalar<f32>>, %i: index) -> !pop.pointer<!pop.scalar<f32>> {
  // CHECK: llvm.getelementptr %{{.*}}[{{.*}}]
  %0 = pop.offset %p[%i] : !pop.pointer<!pop.scalar<f32>>
  kgen.return %0 : !pop.pointer<!pop.scalar<f32>>
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

// CHECK-LABEL: @cmp_simd
kgen.func @cmp_simd(%lhs: !pop.simd<4, f32>, %rhs: !pop.simd<4, f32>) -> !pop.simd<4, bool> {
  // CHECK: llvm.fcmp {{.*}} : vector<4xf32>
  %0 = pop.cmp lt(%lhs, %rhs) : !pop.simd<4, f32>
  // CHECK: vector<4xi1>
  kgen.return %0 : !pop.simd<4, bool>
}

// -----

// CHECK-LABEL: @pointer_to_index
kgen.func @pointer_to_index(%a: !pop.pointer<!pop.scalar<f32>>,
                            %b: !pop.pointer<!pop.simd<4, si32>>) -> (index, index) {
  // CHECK: llvm.ptrtoint
  %0 = pop.pointer_to_index %a : !pop.pointer<!pop.scalar<f32>>
  // CHECK: llvm.ptrtoint
  %1 = pop.pointer_to_index %b : !pop.pointer<!pop.simd<4, si32>>
  kgen.return %0, %1 : index, index
}

// -----

// CHECK-LABEL: @index_to_pointer
kgen.func @index_to_pointer(%idx: index) -> (
      !pop.pointer<!pop.scalar<f32>>,
      !pop.pointer<!pop.simd<4, si32>>) {
  // CHECK: llvm.inttoptr
  %0 = pop.index_to_pointer %idx : !pop.pointer<!pop.scalar<f32>>
  // CHECK: llvm.inttoptr
  %1 = pop.index_to_pointer %idx : !pop.pointer<!pop.simd<4, si32>>
  kgen.return %0, %1 :!pop.pointer<!pop.scalar<f32>>, !pop.pointer<!pop.simd<4, si32>>
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

!var = !pop.variant<f32, i64, !pop.struct<i8, i8, f64>>

// CHECK-LABEL: @create
// CHECK-SAME: %[[A:.*]]: i64
kgen.func public @create(%a: i64) -> !var {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[MEM:.*]] = llvm.alloca %[[ONE]] x !llvm.array<2 x i64>
  // CHECK: %[[PTR:.*]] = llvm.bitcast %[[MEM]] : !llvm.ptr<array<2 x i64>> to !llvm.ptr<i64>
  // CHECK-NEXT: llvm.intr.lifetime.start 1, %[[MEM]]
  // CHECK: llvm.store %[[A]], %[[PTR]]
  // CHECK: %[[CONTENT:.*]] = llvm.load %[[MEM]]
  // CHECK-NEXT: llvm.intr.lifetime.end 1, %[[MEM]]
  // CHECK: %[[S0:.*]] = llvm.mlir.undef : !llvm.struct<(array<2 x i64>, i2)>
  // CHECK: %[[S1:.*]] = llvm.insertvalue %[[CONTENT]], %[[S0]][0]
  // CHECK: %[[DISCR:.*]] = llvm.mlir.constant(1 : i2)
  // CHECK: %[[S2:.*]] = llvm.insertvalue %[[DISCR]], %[[S1]][1]
  %0 = pop.variant.create %a : i64 -> !var
  // CHECK: unrealized_conversion_cast %[[S2]]
  kgen.return %0 : !var
}

// CHECK-LABEL: @test
kgen.func public @test(%a: !var) -> i1 {
  // CHECK: %[[VAR:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[DISCR:.*]] = llvm.extractvalue %[[VAR]][1]
  // CHECK: %[[DISCR_VAL:.*]] = llvm.mlir.constant(0 : i2)
  // CHECK: %[[VAL:.*]] = llvm.icmp "eq" %[[DISCR]], %[[DISCR_VAL]]
  %0 = pop.variant.is f32, %a : !var
  // CHECK: return %[[VAL]]
  kgen.return %0 : i1
}

// CHECK-LABEL: @bitcast
kgen.func public @bitcast(%a: !var) -> f32 {
  // CHECK: %[[VAR:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[MEM:.*]] = llvm.alloca %[[ONE]]
  // CHECK: %[[CONTENT:.*]] = llvm.extractvalue %0[0]
  // CHECK: llvm.intr.lifetime.start 1, %[[MEM]]
  // CHECK: llvm.store %[[CONTENT]], %[[MEM]]
  // CHECK: %[[PTR:.*]] = llvm.bitcast %[[MEM]] : !llvm.ptr<array<2 x i64>> to !llvm.ptr<f32>
  // CHECK: %[[RESULT:.*]] = llvm.load %[[PTR]]
  // CHECK: llvm.intr.lifetime.end 1, %[[MEM]]
  %0 = pop.variant.get %a : !var as f32
  // CHECK: return %[[RESULT]]
  kgen.return %0 : f32
}

// -----

kgen.func @use(%a: !pop.variant<i32, i64>) {
  kgen.return
}

// CHECK-LABEL: @hoist_alloca
kgen.func public @hoist_alloca(%a: i32, %ub: index) {
  // CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK-NEXT: %[[PTR:.*]] = llvm.alloca %[[ONE]] x !llvm.array<1 x i64>
  %0 = index.constant 0
  %1 = index.constant 1
  // CHECK: scf.for
  scf.for %i = %0 to %ub step %1 {
    // CHECK: llvm.bitcast %[[PTR]] : !llvm.ptr<array<1 x i64>> to !llvm.ptr<i32>
    %2 = pop.variant.create %a : i32 -> !pop.variant<i32, i64>
    kgen.call @use(%2) : (!pop.variant<i32, i64>) -> ()
  }
  kgen.return
}

// -----

kgen.func @use(%a: i32) {
  kgen.return
}

// CHECK-LABEL: @hoist_alloca
kgen.func public @hoist_alloca(%a: !pop.variant<i32, i64>, %ub: index) {
  // CHECK: builtin.unrealized_conversion_cast
  // CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK-NEXT: %[[PTR:.*]] = llvm.alloca %[[ONE]] x !llvm.array<1 x i64>
  %0 = index.constant 0
  %1 = index.constant 1
  // CHECK: scf.for
  scf.for %i = %0 to %ub step %1 {
    // CHECK: llvm.bitcast %[[PTR]] : !llvm.ptr<array<1 x i64>> to !llvm.ptr<i32>
    %2 = pop.variant.get %a : !pop.variant<i32, i64> as i32
    kgen.call @use(%2) : (i32) -> ()
  }
  kgen.return
}

// -----

// CHECK-LABEL: @prefetch
kgen.func @prefetch(%p: !pop.pointer<!pop.scalar<si32>>) {
  // CHECK-DAG: [[RW:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[LOCALITY:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[CACHETAG:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: "llvm.intr.prefetch"(%{{[0-9]+}}, [[RW]], [[LOCALITY]], [[CACHETAG]])
  pop.prefetch %p (NoLocality, ReadDCache)
    : !pop.pointer<!pop.scalar<si32>>
  // CHECK-DAG: [[RW:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: [[LOCALITY:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: [[CACHETAG:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: "llvm.intr.prefetch"(%{{[0-9]+}}, [[RW]], [[LOCALITY]], [[CACHETAG]])
  pop.prefetch %p (LowLocality, WriteDCache)
    : !pop.pointer<!pop.scalar<si32>>
  // CHECK-DAG: [[RW:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[LOCALITY:%[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: [[CACHETAG:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: "llvm.intr.prefetch"(%{{[0-9]+}}, [[RW]], [[LOCALITY]], [[CACHETAG]])
  pop.prefetch %p (MediumLocality, ReadICache)
    : !pop.pointer<!pop.scalar<si32>>
  // CHECK-DAG: [[RW:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[LOCALITY:%[0-9]+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-DAG: [[CACHETAG:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: "llvm.intr.prefetch"(%{{[0-9]+}}, [[RW]], [[LOCALITY]], [[CACHETAG]])
  pop.prefetch %p (HighLocality, ReadDCache)
    : !pop.pointer<!pop.scalar<si32>>
  // CHECK-DAG: [[RW:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[LOCALITY:%[0-9]+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-DAG: [[CACHETAG:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: "llvm.intr.prefetch"(%{{[0-9]+}}, [[RW]], [[LOCALITY]], [[CACHETAG]])
  pop.prefetch %p (VeryHighLocality, ReadDCache)
    : !pop.pointer<!pop.scalar<si32>>
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

// CHECK-LABEL: @array_repeat0
kgen.func @array_repeat0(%a: i32, %b: i32) -> !pop.array<3, i32> {
  // CHECK: llvm.insertvalue %arg0, %{{.*}}[0]
  // CHECK: llvm.insertvalue %arg1, %{{.*}}[1]
  // CHECK: llvm.insertvalue %arg0, %{{.*}}[2]
  %0 = pop.array.repeat [%a, %b] : !pop.array<3, i32>
  kgen.return %0 : !pop.array<3, i32>
}

// CHECK-LABEL: @array_repeat1
kgen.func @array_repeat1(%a: i32, %b: i32) -> !pop.array<1, i32> {
  // CHECK: llvm.insertvalue %arg0, %{{.*}}[0]
  %0 = pop.array.repeat [%a, %b] : !pop.array<1, i32>
  kgen.return %0 : !pop.array<1, i32>
}

// CHECK-LABEL: @array_get_replace
kgen.func @array_get_replace(%a: !pop.array<2, i32>) -> !pop.array<2, i32> {
  // CHECK: llvm.extractvalue %{{.*}}[0]
  %0 = pop.array.get %a[0] : !pop.array<2, i32>
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1]
  %1 = pop.array.replace %0, %a[1] : !pop.array<2, i32>
  kgen.return %1 : !pop.array<2, i32>
}

// -----

kgen.func @indirect_call(%fn: (i32, i64) -> (f32, f64), %a: i32, %b: i64) -> (f32, f64) {
  // CHECK: %[[FN:.*]] = builtin.unrealized_conversion_cast %arg0 : (i32, i64) -> (f32, f64) to !llvm.ptr<func<struct<(f32, f64)> (i32, i64)>>
  // CHECK: %[[RESULT:.*]] = llvm.call %[[FN]](%arg1, %arg2) : (i32, i64) -> !llvm.struct<(f32, f64)>
  // CHECK: %[[R0:.*]] = llvm.extractvalue %[[RESULT]][0]
  // CHECK: %[[R1:.*]] = llvm.extractvalue %[[RESULT]][1]
  %0:2 = pop.indirect_call %fn(%a, %b) : (i32, i64) -> (f32, f64)
  // CHECK: return %[[R0]], %[[R1]] : f32, f64
  kgen.return %0#0, %0#1 : f32, f64
}

// -----

// CHECK-LABEL: @memcpy
// CHECK-SAME: %[[DEST:.*]]: !pop.pointer<!pop.scalar<si32>>
// CHECK-SAME: %[[SRC:.*]]: !pop.pointer<!pop.scalar<f32>>
// CHECK-SAME: %[[SIZE:.*]]: index
kgen.func @memcpy(%dest: !pop.pointer<!pop.scalar<si32>>,
                  %src: !pop.pointer<!pop.scalar<f32>>,
                  %size: index) {
  // CHECK: %[[DEST_CAST:.*]] = builtin.unrealized_conversion_cast %[[DEST]]
  // CHECK: %[[SRC_CAST:.*]] = builtin.unrealized_conversion_cast %[[SRC]]
  // CHECK: %[[SIZE_CAST:.*]] = builtin.unrealized_conversion_cast %[[SIZE]]
  // CHECK: %[[VOLATILE:.*]] = llvm.mlir.constant(false) : i1
  // CHECK:  "llvm.intr.memcpy"(%[[DEST_CAST]], %[[SRC_CAST]], %[[SIZE_CAST]], %[[VOLATILE]]) : (!llvm.ptr<i32>, !llvm.ptr<f32>, i64, i1) -> ()
  pop.memcpy %dest, %src, %size : !pop.pointer<!pop.scalar<f32>> to !pop.pointer<!pop.scalar<si32>>
  kgen.return
}


// -----

// CHECK-LABEL: @memcpy_volatile
// CHECK-SAME: %[[DEST:.*]]: !pop.pointer<!pop.scalar<si32>>
// CHECK-SAME: %[[SRC:.*]]: !pop.pointer<!pop.scalar<f32>>
// CHECK-SAME: %[[SIZE:.*]]: index
kgen.func @memcpy_volatile(%dest: !pop.pointer<!pop.scalar<si32>>,
                           %src: !pop.pointer<!pop.scalar<f32>>,
                           %size: index) {
  // CHECK: %[[DEST_CAST:.*]] = builtin.unrealized_conversion_cast %[[DEST]]
  // CHECK: %[[SRC_CAST:.*]] = builtin.unrealized_conversion_cast %[[SRC]]
  // CHECK: %[[SIZE_CAST:.*]] = builtin.unrealized_conversion_cast %[[SIZE]]
  // CHECK: %[[VOLATILE:.*]] = llvm.mlir.constant(true) : i1
  // CHECK:  "llvm.intr.memcpy"(%[[DEST_CAST]], %[[SRC_CAST]], %[[SIZE_CAST]], %[[VOLATILE]]) : (!llvm.ptr<i32>, !llvm.ptr<f32>, i64, i1) -> ()
  pop.memcpy volatile %dest, %src, %size : !pop.pointer<!pop.scalar<f32>> to !pop.pointer<!pop.scalar<si32>>
  kgen.return
}

// -----

// CHECK-LABEL: @memcpy_inline
// CHECK-SAME: %[[DEST:.*]]: !pop.pointer<!pop.scalar<si32>>
// CHECK-SAME: %[[SRC:.*]]: !pop.pointer<!pop.scalar<f32>>
// CHECK-SAME: %[[SIZE:.*]]: index
kgen.func @memcpy_inline(%dest: !pop.pointer<!pop.scalar<si32>>,
                         %src: !pop.pointer<!pop.scalar<f32>>,
                         %size: index) {
  // CHECK: %[[DEST_CAST:.*]] = builtin.unrealized_conversion_cast %[[DEST]]
  // CHECK: %[[SRC_CAST:.*]] = builtin.unrealized_conversion_cast %[[SRC]]
  // CHECK: %[[SIZE_CAST:.*]] = builtin.unrealized_conversion_cast %[[SIZE]]
  // CHECK: %[[VOLATILE:.*]] = llvm.mlir.constant(false) : i1
  // CHECK:  "llvm.intr.memcpy.inline"(%[[DEST_CAST]], %[[SRC_CAST]], %[[SIZE_CAST]], %[[VOLATILE]]) : (!llvm.ptr<i32>, !llvm.ptr<f32>, i64, i1) -> ()
  pop.memcpy inline %dest, %src, %size : !pop.pointer<!pop.scalar<f32>> to !pop.pointer<!pop.scalar<si32>>
  kgen.return
}
