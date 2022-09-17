// RUN: kgen-opt -split-input-file -pass-pipeline='kgen.func(convert-pop-to-llvm)' %s | FileCheck %s

// Test trivial vector conversions to LLVM.

!simd = !meta.simd<4, f32>
kgen.func @trivial_conversions(%a: !simd, %b: !simd, %c: !simd, %d: !meta.simd<4, bool>) {
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
  kgen.return
}

// -----

// CHECK-LABEL: int_abs_simd
kgen.func @int_abs_simd(%arg0: !meta.simd<4, si32>) -> !meta.simd<4, si32> {
  // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false
  %0 = pop.abs %arg0 : !meta.simd<4, si32>
  // CHECK: "llvm.intr.abs"(%{{.*}}, %[[FALSE]])
  kgen.return %0 : !meta.simd<4, si32>
}

// -----

// CHECK-LABEL: @int_neg_simd
kgen.func @int_neg_simd(%arg0: !meta.simd<4, si32>) -> !meta.simd<4, si32> {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(dense<0> : vector<4xi32>)
  %0 = pop.neg %arg0 : !meta.simd<4, si32>
  // CHECK: llvm.sub %[[ZERO]], %{{.*}}
  kgen.return %0 : !meta.simd<4, si32>
}

// -----

// CHECK-LABEL: @constant_simd
kgen.func @constant_simd() -> !meta.simd<2, si32> {
  // CHECK: llvm.mlir.constant(dense<0>
  %0 = pop.constant(dense<0> : vector<2xsi32>) : !meta.simd<2, si32>
  kgen.return %0 : !meta.simd<2, si32>
}

// -----

// CHECK-LABEL: @constant_simd
kgen.func @constant_simd() -> !meta.simd<2, f32> {
  // CHECK: llvm.mlir.constant(dense<[1.{{0*}}e+00, 2.{{0*}}e+00]>
  %0 = pop.constant(dense<[1., 2.]> : vector<2xf32>) : !meta.simd<2, f32>
  kgen.return %0 : !meta.simd<2, f32>
}

// -----

// CHECK-LABEL: @simd_splat
kgen.func @simd_splat(%a: !meta.scalar<f32>) -> !meta.simd<2, f32> {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 :
  // CHECK: %[[VECTOR:.*]] = llvm.insertelement %[[E:.*]], %[[UNDEF]][%[[ZERO]] : i32] : vector<2xf32>
  // CHECK: %[[RESULT:.*]] = llvm.shufflevector %[[VECTOR]], %[[UNDEF]] [0, 0] : vector<2xf32>
  // CHECK: unrealized_conversion_cast %[[RESULT]]
  %0 = pop.simd.splat %a : !meta.simd<2, f32>
  kgen.return %0 : !meta.simd<2, f32>
}

// -----

// CHECK-LABEL: @simd_splat
kgen.func @simd_splat(%a: !meta.scalar<f32>) -> !meta.simd<1, f32> {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 :
  // CHECK: %[[VECTOR:.*]] = llvm.insertelement %[[E:.*]], %[[UNDEF]][%[[ZERO]] : i32] : vector<1xf32>
  // CHECK: unrealized_conversion_cast %[[VECTOR]]
  %0 = pop.simd.splat %a : !meta.simd<1, f32>
  kgen.return %0 : !meta.simd<1, f32>
}

// -----

// CHECK-LABEL: @simd_extractelement
kgen.func @simd_extractelement(%vec: !meta.simd<4, f32>, %idx: index) -> !meta.scalar<f32> {
  // CHECK: %[[VEC:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[IDX:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[SCALAR:.*]] = llvm.extractelement %[[VEC]][%[[IDX]] : {{.*}}] : vector<4xf32>
  // CHECK: unrealized_conversion_cast %[[SCALAR]]
  %0 = pop.simd.extractelement %vec[%idx] : !meta.simd<4, f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @simd_insertelement
kgen.func @simd_insertelement(%val: !meta.scalar<f32>, %vec: !meta.simd<4, f32>, %idx: index) -> !meta.simd<4, f32> {
  // CHECK: %[[VEC:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[VAL:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[IDX:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[RES:.*]] = llvm.insertelement %[[VAL]], %[[VEC]][%[[IDX]] : {{.*}}] : vector<4xf32>
  // CHECK: unrealized_conversion_cast %[[RES]]
  %0 = pop.simd.insertelement %val, %vec[%idx] : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// -----

// CHECK-LABEL: @simd_shuffle
kgen.func @simd_shuffle(%a: !meta.simd<2, f32>, %b: !meta.simd<2, f32>) -> !meta.simd<4, f32> {
  // CHECK: llvm.shufflevector %{{.*}}, %{{.*}} [2, 3, 1, 0]
  %0 = pop.simd.shuffle %a, %b [2, 3, 1, 0] : !meta.simd<2, f32> -> !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// -----

// CHECK-LABEL: @simd_load_store
kgen.func @simd_load_store(%i: index, %p0: !meta.pointer<!meta.simd<4, f32>>) {
  // CHECK: llvm.getelementptr %{{.*}} : (!llvm.ptr<vector<4xf32>>, {{.*}}) -> !llvm.ptr<vector<4xf32>>
  %0 = pop.offset %p0[%i] : !meta.pointer<!meta.simd<4, f32>>
  // CHECK: llvm.load
  %1 = pop.load %0 : !meta.pointer<!meta.simd<4, f32>>
  // CHECK: llvm.store
  pop.store %1, %p0 : !meta.pointer<!meta.simd<4, f32>>
  kgen.return
}

// -----

// CHECK-LABEL: @simd_reduce_add
kgen.func @simd_reduce_add(%a: !meta.simd<2, f32>,
                           %b: !meta.simd<2, si32>,
                           %c: !meta.simd<2, ui32>) {
  // CHECK: llvm.intr.vector.reduce.fadd
  %0 = pop.simd.reduce.add %a : !meta.simd<2, f32>
  // CHECK: llvm.intr.vector.reduce.add
  %1 = pop.simd.reduce.add %b : !meta.simd<2, si32>
  // CHECK: llvm.intr.vector.reduce.add
  %2 = pop.simd.reduce.add %c : !meta.simd<2, ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @simd_reduce_mul
kgen.func @simd_reduce_mul(%a: !meta.simd<2, f32>,
                           %b: !meta.simd<2, si32>,
                           %c: !meta.simd<2, ui32>) {
  // CHECK: llvm.intr.vector.reduce.fmul
  %0 = pop.simd.reduce.mul %a : !meta.simd<2, f32>
  // CHECK: llvm.intr.vector.reduce.mul
  %1 = pop.simd.reduce.mul %b : !meta.simd<2, si32>
  // CHECK: llvm.intr.vector.reduce.mul
  %2 = pop.simd.reduce.mul %c : !meta.simd<2, ui32>
  kgen.return
}


// -----

// CHECK-LABEL: @simd_reduce_max
kgen.func @simd_reduce_max(%a: !meta.simd<2, f32>,
                           %b: !meta.simd<2, si32>,
                           %c: !meta.simd<2, ui32>) {
  // CHECK: llvm.intr.vector.reduce.fmax
  %0 = pop.simd.reduce.max %a : !meta.simd<2, f32>
  // CHECK: llvm.intr.vector.reduce.smax
  %1 = pop.simd.reduce.max %b : !meta.simd<2, si32>
  // CHECK: llvm.intr.vector.reduce.umax
  %2 = pop.simd.reduce.max %c : !meta.simd<2, ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @simd_reduce_min
kgen.func @simd_reduce_min(%a: !meta.simd<2, f32>,
                           %b: !meta.simd<2, si32>,
                           %c: !meta.simd<2, ui32>) {
  // CHECK: llvm.intr.vector.reduce.fmin
  %0 = pop.simd.reduce.min %a : !meta.simd<2, f32>
  // CHECK: llvm.intr.vector.reduce.smin
  %1 = pop.simd.reduce.min %b : !meta.simd<2, si32>
  // CHECK: llvm.intr.vector.reduce.umin
  %2 = pop.simd.reduce.min %c : !meta.simd<2, ui32>
  kgen.return
}
