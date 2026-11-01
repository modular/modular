// RUN: mo-opt %s --gc-pipeline=constant-fold=false --mlir-print-ir-after=mogg-to-mgp -o %t.mlir 2>&1 | tee %t.mgp
// RUN: mt --execute --result-output-style=full %t.mlir 2>&1 | tee %t.out
// RUN: FileCheck %s < %t.out
// RUN: FileCheck %s --check-prefix=CHECK-MGP < %t.mgp


// Input 1: [[1, 0], [0, -1]]
// Input 2: [[0, 1], [-1, 0]]


// --- Matmul alone on GPU ---
mo.graph @parallel_gpu_matmul() -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %a = mo.constant <#M.dense_array<1.0, 0.0, 0.0, -1.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, gpu:0>
  %b = mo.constant <#M.dense_array<0.0, 1.0, -1.0, 0.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, gpu:0>
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>) -> (!mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>]>)
  %res_bundle = mo.parallel (%arg0) in (%bundle : !mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>]>)
         -> (!mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>]>) {
    %mm = mo.matmul(%arg0, %arg0) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
    mo.yield %mm : !mo.tensor<[2, 2], f32, gpu:0>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>]>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>)
  %ch0 = mo.chain.create()
  %r0, %ch1 = mo.transfer[%ch0] %res_0 : !mo.tensor<[2, 2], f32, gpu:0> to <"cpu", 0>
  %r1, %ch2 = mo.transfer[%ch1] %res_1 : !mo.tensor<[2, 2], f32, gpu:0> to <"cpu", 0>
  mo.output %r0, %r1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK: 'parallel_gpu_matmul' returned tensor<2x2xf32>
// CHECK-SAME: [1.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00, 1.{{0*}}e+00]
// CHECK: , tensor<2x2xf32>
// CHECK-SAME: [-1.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00, -1.{{0*}}e+00]


// --- Matmul + relu on GPU: epilogue fusion ---
mo.graph @parallel_gpu_matmul_relu() -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %a = mo.constant <#M.dense_array<1.0, 0.0, 0.0, -1.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, gpu:0>
  %b = mo.constant <#M.dense_array<0.0, 1.0, -1.0, 0.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, gpu:0>
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>) -> (!mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>]>)
  %res_bundle = mo.parallel (%arg0) in (%bundle : !mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>]>)
         -> (!mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>]>) {
    %mm = mo.matmul(%arg0, %arg0) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
    %relu = mo.relu(%mm) : !mo.tensor<[2, 2], f32, gpu:0>
    mo.yield %relu : !mo.tensor<[2, 2], f32, gpu:0>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>]>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>)
  %ch0 = mo.chain.create()
  %r0, %ch1 = mo.transfer[%ch0] %res_0 : !mo.tensor<[2, 2], f32, gpu:0> to <"cpu", 0>
  %r1, %ch2 = mo.transfer[%ch1] %res_1 : !mo.tensor<[2, 2], f32, gpu:0> to <"cpu", 0>
  mo.output %r0, %r1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK: 'parallel_gpu_matmul_relu' returned tensor<2x2xf32>
// CHECK-SAME: [1.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00, 1.{{0*}}e+00]
// CHECK: , tensor<2x2xf32>
// CHECK-SAME: [0.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00]

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
