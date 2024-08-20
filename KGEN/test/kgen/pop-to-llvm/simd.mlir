// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

// Test trivial vector conversions to LLVM.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @trivial_conversions(%a: !pop.simd<4, f32>, %b: !pop.simd<4, f32>, %c: !pop.simd<4, f32>, %d: !pop.simd<4, bool>) {
  // CHECK: llvm.fneg
  %0 = pop.neg %a : !pop.simd<4, f32>
  // CHECK: llvm.fadd
  %1 = pop.add %a, %b : !pop.simd<4, f32>
  // CHECK: llvm.fsub
  %2 = pop.sub %a, %b : !pop.simd<4, f32>
  // CHECK: llvm.fmul
  %3 = pop.mul %a, %b : !pop.simd<4, f32>
  // CHECK: llvm.intr.fma
  %4 = pop.fma %a, %b, %c : !pop.simd<4, f32>
  // CHECK: llvm.select
  %5 = pop.simd.select %d, %a, %b : !pop.simd<4, f32>
  kgen.return
}

// CHECK-LABEL: @int_neg_simd
kgen.func @int_neg_simd(%arg0: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(dense<0> : vector<4xi32>)
  %0 = pop.neg %arg0 : !pop.simd<4, si32>
  // CHECK: llvm.sub %[[ZERO]], %{{.*}}
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @neg_simd
kgen.func @neg_simd(%arg0: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  %0 = pop.neg %arg0 : !pop.simd<4, f32>
  // CHECK: llvm.fneg %{{.*}}
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @add_simd_si32
kgen.func @add_simd_si32(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @add_simd_index
kgen.func @add_simd_index(%arg0: !pop.simd<4, index>, %arg1: !pop.simd<4, index>) -> !pop.simd<4, index> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg1: !pop.simd<4, index>
  kgen.return %0 : !pop.simd<4, index>
}

// CHECK-LABEL: @fadd_simd
kgen.func @fadd_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.fadd
  %0 = pop.add %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @sub_simd
kgen.func @sub_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.sub
  %0 = pop.sub %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @fsub_simd
kgen.func @fsub_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.fsub
  %0 = pop.sub %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @mul_simd
kgen.func @mul_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.mul
  %0 = pop.mul %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @fmul_simd
kgen.func @fmul_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.fmul
  %0 = pop.mul %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @div_simd
kgen.func @div_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.sdiv
  %0 = pop.div %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @fdiv_simd
kgen.func @fdiv_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.fdiv
  %0 = pop.div %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @max_simd
kgen.func @max_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.intr.smax
  %0 = pop.max %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @fmax_simd
kgen.func @fmax_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.intr.maxnum
  %0 = pop.max %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @min_simd
kgen.func @min_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.intr.smin
  %0 = pop.min %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @fmin_simd
kgen.func @fmin_simd(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.intr.minnum
  %0 = pop.min %arg0, %arg1: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @shl_simd_si32
kgen.func @shl_simd_si32(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @shl_simd_ui32
kgen.func @shl_simd_ui32(%arg0: !pop.simd<4, ui32>, %arg1: !pop.simd<4, ui32>) -> !pop.simd<4, ui32> {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1: !pop.simd<4, ui32>
  kgen.return %0 : !pop.simd<4, ui32>
}

// CHECK-LABEL: @shr_simd
kgen.func @shr_simd_si32(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.ashr
  %0 = pop.shr %arg0, %arg1: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @shr_simd
kgen.func @shr_simd_ui32(%arg0: !pop.simd<4, ui32>, %arg1: !pop.simd<4, ui32>) -> !pop.simd<4, ui32> {
  // CHECK: llvm.lshr
  %0 = pop.shr %arg0, %arg1: !pop.simd<4, ui32>
  kgen.return %0 : !pop.simd<4, ui32>
}

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

// CHECK-LABEL: @fma_simd_si32
kgen.func @fma_simd_si32(%arg0: !pop.simd<4, si32>,
                    %arg1: !pop.simd<4, si32>,
                    %arg2: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.mul
  // CHECK: llvm.add
  %0 = pop.fma %arg0, %arg1, %arg2: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}


// CHECK-LABEL: @fma_simd_f32
kgen.func @fma_simd_f32(%arg0: !pop.simd<4, f32>,
                    %arg1: !pop.simd<4, f32>,
                    %arg2: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.intr.fma
  %0 = pop.fma %arg0, %arg1, %arg2: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @select_simd_si32
kgen.func @select_simd_si32(%arg0: !pop.simd<4, bool>,
                    %arg1: !pop.simd<4, si32>,
                    %arg2: !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // CHECK: llvm.select
  %0 = pop.simd.select %arg0, %arg1, %arg2: !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}


// CHECK-LABEL: @select_simd_f32
kgen.func @select_simd_f32(%arg0: !pop.simd<4, bool>,
                    %arg1: !pop.simd<4, f32>,
                    %arg2: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.select
  %0 = pop.simd.select %arg0, %arg1, %arg2: !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @bitcast
kgen.func @bitcast(%a: !pop.scalar<si32>,
                   %b: !pop.scalar<ui64>,
                   %c: !pop.simd<4, f64>,
                   %d: !pop.simd<2, f64>) {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK-DAG: [[ARG2:%.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK-DAG: [[ARG3:%.*]] = builtin.unrealized_conversion_cast %arg3

  // CHECK: llvm.bitcast [[ARG0]] : i32 to f32
  %0 = pop.bitcast %a: !pop.scalar<si32> to !pop.scalar<f32>

  // CHECK: llvm.bitcast [[ARG1]] : i64 to i64
  %1 = pop.bitcast %b: !pop.scalar<ui64> to !pop.scalar<si64>

  // CHECK: llvm.bitcast [[ARG2]] : vector<4xf64> to vector<4xi64>
  %2 = pop.bitcast %c: !pop.simd<4, f64> to !pop.simd<4, si64>

  // CHECK: llvm.bitcast [[ARG3]] : vector<2xf64> to vector<4xf32>
  %3 = pop.bitcast %d: !pop.simd<2, f64> to !pop.simd<4, f32>

  // CHECK: llvm.bitcast [[ARG1]] : i64 to vector<2xf32>
  %4 = pop.bitcast %b: !pop.scalar<ui64> to !pop.simd<2, f32>

  // CHECK: %[[B0:.*]] = llvm.bitcast [[ARG1]] : i64 to vector<64xi1>
  %5 = pop.bitcast %b: !pop.scalar<ui64> to !pop.simd<64, bool>

  // CHECK: llvm.bitcast %[[B0]] : vector<64xi1> to f64
  %6 = pop.bitcast %5: !pop.simd<64, bool> to !pop.simd<1, f64>

  kgen.return
}

// CHECK-LABEL: @simd_splat_scalar_to_2xf32
kgen.func @simd_splat_scalar_to_2xf32(%a: !pop.scalar<f32>) -> !pop.simd<2, f32> {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 :
  // CHECK: %[[VECTOR:.*]] = llvm.insertelement %[[E:.*]], %[[UNDEF]][%[[ZERO]] : i32] : vector<2xf32>
  // CHECK: %[[RESULT:.*]] = llvm.shufflevector %[[VECTOR]], %[[UNDEF]] [0, 0] : vector<2xf32>
  // CHECK: unrealized_conversion_cast %[[RESULT]]
  %0 = pop.simd.splat %a : !pop.simd<2, f32>
  kgen.return %0 : !pop.simd<2, f32>
}

// CHECK-LABEL: @simd_extractelement
kgen.func @simd_extractelement(%vec: !pop.simd<4, f32>, %idx: index) -> !pop.scalar<f32> {
  // CHECK-DAG: %[[VEC:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: %[[IDX:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: %[[SCALAR:.*]] = llvm.extractelement %[[VEC]][%[[IDX]] : {{.*}}] : vector<4xf32>
  // CHECK: unrealized_conversion_cast %[[SCALAR]]
  %0 = pop.simd.extractelement %vec[%idx] : !pop.simd<4, f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @simd_insertelement
kgen.func @simd_insertelement(%val: !pop.scalar<f32>, %vec: !pop.simd<4, f32>, %idx: index) -> !pop.simd<4, f32> {
  // CHECK-DAG: %[[VAL:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: %[[VEC:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK-DAG: %[[IDX:.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK: %[[RES:.*]] = llvm.insertelement %[[VAL]], %[[VEC]][%[[IDX]] : {{.*}}] : vector<4xf32>
  // CHECK: unrealized_conversion_cast %[[RES]]
  %0 = pop.simd.insertelement %val, %vec[%idx] : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @simd_insertelement_1xf32
// CHECK-SAME: (%[[ARG0:[[:alnum:]]+]]:
kgen.func @simd_insertelement_1xf32(%val: !pop.scalar<f32>, %vec: !pop.scalar<f32>, %idx: index) -> !pop.scalar<f32> {
  // CHECK-NEXT: kgen.return %[[ARG0]]
  %0 = pop.simd.insertelement %val, %vec[%idx] : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @simd_shuffle
kgen.func @simd_shuffle(%a: !pop.simd<2, f32>, %b: !pop.simd<2, f32>) -> !pop.simd<4, f32> {
  // CHECK: llvm.shufflevector %{{.*}}, %{{.*}} [2, 3, 1, 0]
  %0 = pop.simd.shuffle <2, f32> %a, %b -> <4, f32> :array<4,index> [2, 3, 1, 0]
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @simd_shuffle_1xf32
kgen.func @simd_shuffle_1xf32(%a: !pop.scalar<f32>, %b: !pop.scalar<f32>) -> (!pop.simd<2, f32>, !pop.scalar<f32>) {
  // CHECK-DAG: %[[F32VAL0:.*]] = builtin.unrealized_conversion_cast %arg0 : !pop.scalar<f32> to f32
  // CHECK-DAG: %[[F32VAL1:.*]] = builtin.unrealized_conversion_cast %arg1 : !pop.scalar<f32> to f32
  // CHECK: %[[VECVAL0_0:.*]] = llvm.mlir.undef : vector<2xf32>
  // CHECK: %[[CONST0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[VECVAL0_1:.*]] = llvm.insertelement %[[F32VAL1]], %[[VECVAL0_0]][%[[CONST0_0]] : i32]
  // CHECK: %[[CONST0_1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[VECVAL0_2:.*]] = llvm.insertelement %[[F32VAL0]], %[[VECVAL0_1]][%[[CONST0_1]] : i32]
  // CHECK: builtin.unrealized_conversion_cast %[[VECVAL0_2]] : vector<2xf32> to !pop.simd<2, f32>
  %0 = pop.simd.shuffle <1, f32> %a, %b -> <2, f32> :array<2,index> [1, 0]

  // CHECK: %[[VECVAL1_0:.*]] = llvm.mlir.undef : vector<1xf32>
  // CHECK: %[[CONST1_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[VECVAL1_1:.*]] = llvm.insertelement %[[F32VAL1]], %[[VECVAL1_0]][%[[CONST1_0]] : i32]
  // CHECK: builtin.unrealized_conversion_cast %[[VECVAL1_1]] : vector<1xf32> to !pop.scalar<f32>
  %1 = pop.simd.shuffle <1, f32> %a, %b -> <1, f32> :array<1,index> [1]

  kgen.return %0, %1 : !pop.simd<2, f32>, !pop.scalar<f32>
}

// CHECK-LABEL: @simd_load_store
kgen.func @simd_load_store(%i: index, %p0: !kgen.pointer<simd<4, f32>>) {
  // CHECK: llvm.getelementptr %{{.*}} : (!llvm.ptr, {{.*}}) -> !llvm.ptr
  %0 = pop.offset %p0[%i] : !kgen.pointer<simd<4, f32>>
  // CHECK: llvm.load
  %1 = pop.load %0 : !kgen.pointer<simd<4, f32>>
  // CHECK: llvm.store
  pop.store %1, %p0 : !kgen.pointer<simd<4, f32>>
  kgen.return
}

}
