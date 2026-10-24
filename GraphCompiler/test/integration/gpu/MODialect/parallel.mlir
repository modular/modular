// RUN: mo-opt %s --gc-pipeline=constant-fold=false --mlir-print-ir-after=mogg-to-mgp -o %t.mlir 2>&1 | tee %t.mgp
// RUN: mt --execute --result-output-style=full %t.mlir 2>&1 | tee %t.out
// RUN: FileCheck %s < %t.out
// RUN: FileCheck %s --check-prefix=CHECK-MGP < %t.mgp


// Test for mo.parallel op on GPU
mo.graph @parallel_gpu_op() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant <#M.dense_array<1, -2, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32, gpu:0>
  %const2 = mo.constant <#M.dense_array<-9, 8, -7> : tensor<3xsi32>> : !mo.tensor<[3], si32, gpu:0>
  %bundle = mo.tensor.bundle(%const1, %const2) : (!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>) -> (!mo.bundle<[!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>]>)
  %res_bundle = mo.parallel (%arg0) in (%bundle : !mo.bundle<[!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>]>)
         -> (!mo.bundle<[!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>]>) {
    %1 = mo.relu(%arg0) : !mo.tensor<[3], si32, gpu:0>
    mo.yield %1 : !mo.tensor<[3], si32, gpu:0>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>]>) -> (!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>)
  // Transfer GPU results back to CPU for output
  %ch0 = mo.chain.create()
  %res0_host, %ch1 = mo.transfer[%ch0] %res_0 : !mo.tensor<[3], si32, gpu:0> to <"cpu", 0>
  %res1_host, %ch2 = mo.transfer[%ch1] %res_1 : !mo.tensor<[3], si32, gpu:0> to <"cpu", 0>
  mo.output %res0_host, %res1_host : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}

// CHECK: 'parallel_gpu_op' returned tensor<3xsi32> [1, 0, 3]
// CHECK: , tensor<3xsi32> [0, 8, 0]


// Test mo.parallel with multiple kernels on GPU: negative then relu.
mo.graph @parallel_gpu_neg_relu() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant <#M.dense_array<1, -2, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32, gpu:0>
  %const2 = mo.constant <#M.dense_array<-9, 8, -7> : tensor<3xsi32>> : !mo.tensor<[3], si32, gpu:0>
  %bundle = mo.tensor.bundle(%const1, %const2) : (!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>) -> (!mo.bundle<[!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>]>)
  %res_bundle = mo.parallel (%arg0) in (%bundle : !mo.bundle<[!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>]>)
         -> (!mo.bundle<[!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>]>) {
    %neg = mo.negative(%arg0) : !mo.tensor<[3], si32, gpu:0>
    %relu = mo.relu(%neg) : !mo.tensor<[3], si32, gpu:0>
    mo.yield %relu : !mo.tensor<[3], si32, gpu:0>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>]>) -> (!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>)
  %ch0 = mo.chain.create()
  %res0_host, %ch1 = mo.transfer[%ch0] %res_0 : !mo.tensor<[3], si32, gpu:0> to <"cpu", 0>
  %res1_host, %ch2 = mo.transfer[%ch1] %res_1 : !mo.tensor<[3], si32, gpu:0> to <"cpu", 0>
  mo.output %res0_host, %res1_host : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}

// neg([1,-2,3])=[-1,2,-3] -> relu=[0,2,0]; neg([-9,8,-7])=[9,-8,7] -> relu=[9,0,7]
// CHECK: 'parallel_gpu_neg_relu' returned tensor<3xsi32> [0, 2, 0]
// CHECK: , tensor<3xsi32> [9, 0, 7]

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
