// RUN: mo-opt %s --gc-pipeline=constant-fold=false --mlir-print-ir-after=mogg-to-mgp -o %t.mlir 2>&1 | tee %t.mgp
// RUN: mt --execute --result-output-style=full %t.mlir 2>&1 | tee %t.out
// RUN: FileCheck %s < %t.out
// RUN: FileCheck %s --check-prefix=CHECK-MGP < %t.mgp
// Verify that the parallel body contains multiple kernels before lowering.
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s --check-prefix=CHECK-MOGG
// Verify lowering works with fusion disabled (multiple kernel groups).
// RUN: mo-opt %s --gc-pipeline="constant-fold=false fuse-jit-kernels=false" -o /dev/null


// Tests for elementwise fusion patterns inside mo.parallel.
// All tests use f32 [2,2] matrices.
// Input 1: [[1, 0], [0, -1]]
// Input 2: [[0, 1], [-1, 0]]


// --- Negative + relu with f32: elementwise chain ---
// relu(neg(x)):
//   neg([[1,0],[0,-1]]) = [[-1,0],[0,1]], relu = [[0,0],[0,1]]
//   neg([[0,1],[-1,0]]) = [[0,-1],[1,0]], relu = [[0,0],[1,0]]
mo.graph @parallel_neg_relu_f32() -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %a = mo.constant <#M.dense_array<1.0, 0.0, 0.0, -1.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %b = mo.constant <#M.dense_array<0.0, 1.0, -1.0, 0.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> (!mo.bundle<[!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>]>)
  %res_bundle = mo.parallel (%arg0) in (%bundle : !mo.bundle<[!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>]>)
         -> (!mo.bundle<[!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>]>) {
    %neg = mo.negative(%arg0) : !mo.tensor<[2, 2], f32>
    %relu = mo.relu(%neg) : !mo.tensor<[2, 2], f32>
    mo.yield %relu : !mo.tensor<[2, 2], f32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>]>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %res_0, %res_1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK: 'parallel_neg_relu_f32' returned tensor<2x2xf32>
// CHECK-SAME: [0.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00, 1.{{0*}}e+00]
// CHECK: , tensor<2x2xf32>
// CHECK-SAME: [0.{{0*}}e+00, 0.{{0*}}e+00, 1.{{0*}}e+00, 0.{{0*}}e+00]


// --- Three chained ops: negative + relu + negative ---
// neg(relu(neg(x))):
//   neg([[1,0],[0,-1]]) = [[-1,0],[0,1]]
//   relu = [[0,0],[0,1]]
//   neg = [[0,0],[0,-1]]
//
//   neg([[0,1],[-1,0]]) = [[0,-1],[1,0]]
//   relu = [[0,0],[1,0]]
//   neg = [[0,0],[-1,0]]
mo.graph @parallel_three_ops() -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %a = mo.constant <#M.dense_array<1.0, 0.0, 0.0, -1.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %b = mo.constant <#M.dense_array<0.0, 1.0, -1.0, 0.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> (!mo.bundle<[!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>]>)
  %res_bundle = mo.parallel (%arg0) in (%bundle : !mo.bundle<[!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>]>)
         -> (!mo.bundle<[!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>]>) {
    %neg1 = mo.negative(%arg0) : !mo.tensor<[2, 2], f32>
    %relu = mo.relu(%neg1) : !mo.tensor<[2, 2], f32>
    %neg2 = mo.negative(%relu) : !mo.tensor<[2, 2], f32>
    mo.yield %neg2 : !mo.tensor<[2, 2], f32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>]>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %res_0, %res_1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// neg(0.0) = -0.0 in IEEE 754, so zeros become negative zero.
// CHECK: 'parallel_three_ops' returned tensor<2x2xf32>
// CHECK-SAME: [-0.{{0*}}e+00, -0.{{0*}}e+00, -0.{{0*}}e+00, -1.{{0*}}e+00]
// CHECK: , tensor<2x2xf32>
// CHECK-SAME: [-0.{{0*}}e+00, -0.{{0*}}e+00, -1.{{0*}}e+00, -0.{{0*}}e+00]

// Both graphs produce fused compiled regions with async dispatch.
// CHECK-MGP: mogg.routine.async
// CHECK-MGP:   mogg.call.async @{{__[A-Za-z0-9_]+}}({{.*}})
// CHECK-MGP:   mogg.call.async @{{__[A-Za-z0-9_]+}}({{.*}})
// CHECK-MGP:   mogg.async.join
// CHECK-MGP:   mogg.async.ready

// CHECK-MGP: mogg.routine.async
// CHECK-MGP:   mogg.call.async @{{__[A-Za-z0-9_]+}}({{.*}})
// CHECK-MGP:   mogg.call.async @{{__[A-Za-z0-9_]+}}({{.*}})
// CHECK-MGP:   mogg.async.join
// CHECK-MGP:   mogg.async.ready

// Verify that before lowering, the parallel body has multiple kernels.
// neg+relu = 2 kernels:
// CHECK-MOGG: mo.parallel
// CHECK-MOGG:   mogg.routine
// CHECK-MOGG:   mogg.routine
// CHECK-MOGG:   mo.yield
// neg+relu+neg = 3 kernels:
// CHECK-MOGG: mo.parallel
// CHECK-MOGG:   mogg.routine
// CHECK-MOGG:   mogg.routine
// CHECK-MOGG:   mogg.routine
// CHECK-MOGG:   mo.yield
