// RUN: mo-opt %s --gc-pipeline=constant-fold=false --mlir-print-ir-after=mogg-to-mgp -o %t.mlir 2>&1 | tee %t.mgp
// RUN: mt --execute --result-output-style=full %t.mlir 2>&1 | tee %t.out
// RUN: FileCheck %s < %t.out
// RUN: FileCheck %s --check-prefix=CHECK-MGP < %t.mgp

// Execute two relu ops
mo.graph @sequential_relu() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant <#M.dense_array<1, -2, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %const2 = mo.constant <#M.dense_array<-9, 8, -7> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %res1 = mo.relu(%const1) : !mo.tensor<[3], si32>
  %res2 = mo.relu(%const2) : !mo.tensor<[3], si32>
  mo.output %res1, %res2 : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}

// CHECK: 'sequential_relu' returned tensor<3xsi32> [1, 0, 3]
// CHECK: , tensor<3xsi32> [0, 8, 0]


// Test for mo.parallel op
mo.graph @parallel_op() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant <#M.dense_array<1, -2, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %const2 = mo.constant <#M.dense_array<-9, 8, -7> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %bundle = mo.tensor.bundle(%const1, %const2) : (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>)
  %res_bundle = mo.parallel (%arg0) in (%bundle : !mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>)
         -> (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>) {
    %1 = mo.relu(%arg0) : !mo.tensor<[3], si32>
    mo.yield %1 : !mo.tensor<[3], si32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>) -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>)
  mo.output %res_0, %res_1 : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}


// Test mo.parallel with multiple kernels: add with captured constant + relu.
mo.graph @parallel_add_relu() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant <#M.dense_array<1, -2, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %const2 = mo.constant <#M.dense_array<-9, 8, -7> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %bias = mo.constant <#M.dense_array<-5, -5, -5> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %bundle = mo.tensor.bundle(%const1, %const2) : (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>)
  %res_bundle = mo.parallel (%arg0) in (%bundle : !mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>)
         -> (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>) {
    %added = mo.add(%arg0, %bias) : !mo.tensor<[3], si32>
    %relu = mo.relu(%added) : !mo.tensor<[3], si32>
    mo.yield %relu : !mo.tensor<[3], si32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>) -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>)
  mo.output %res_0, %res_1 : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}

// CHECK: 'parallel_add_relu' returned tensor<3xsi32> [0, 0, 0]
// CHECK: , tensor<3xsi32> [0, 3, 0]


// Test mo.parallel where the kernel uses the same input twice.
// The optimizer strength-reduces add(%arg0, %arg0) to mul(%arg0, 2), which
// introduces a captured constant and exercises the captured-value path.
mo.graph @parallel_self_add_relu() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant <#M.dense_array<1, -2, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %const2 = mo.constant <#M.dense_array<-9, 8, -7> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %bundle = mo.tensor.bundle(%const1, %const2) : (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>)
  %res_bundle = mo.parallel (%arg0) in (%bundle : !mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>)
         -> (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>) {
    %added = mo.add(%arg0, %arg0) : !mo.tensor<[3], si32>
    %relu = mo.relu(%added) : !mo.tensor<[3], si32>
    mo.yield %relu : !mo.tensor<[3], si32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>) -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>)
  mo.output %res_0, %res_1 : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}

// add(x,x) = 2*x, then relu:
// [1,-2,3]*2=[2,-4,6] -> relu=[2,0,6]; [-9,8,-7]*2=[-18,16,-14] -> relu=[0,16,0]
// CHECK: 'parallel_self_add_relu' returned tensor<3xsi32> [2, 0, 6]
// CHECK: , tensor<3xsi32> [0, 16, 0]


// Test mo.parallel with a diamond dependence pattern: relu and negative are
// independent branches that both consume the input, and add merges them.
//   relu(X)   neg(X)
//       \      /
//      add(B, C)
// [1,-2,3]:  relu=[1,0,3], neg=[-1,2,-3], add=[0,2,0]
// [-9,8,-7]: relu=[0,8,0], neg=[9,-8,7],  add=[9,0,7]
mo.graph @parallel_diamond() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant <#M.dense_array<1, -2, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %const2 = mo.constant <#M.dense_array<-9, 8, -7> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %bundle = mo.tensor.bundle(%const1, %const2) : (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>)
  %res_bundle = mo.parallel (%arg0) in (%bundle : !mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>)
         -> (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>) {
    %r = mo.relu(%arg0) : !mo.tensor<[3], si32>
    %n = mo.negative(%arg0) : !mo.tensor<[3], si32>
    %sum = mo.add(%r, %n) : !mo.tensor<[3], si32>
    mo.yield %sum : !mo.tensor<[3], si32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], si32>, !mo.tensor<[3], si32>]>) -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>)
  mo.output %res_0, %res_1 : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}

// CHECK: 'parallel_diamond' returned tensor<3xsi32> [0, 2, 0]
// CHECK: , tensor<3xsi32> [9, 0, 7]

// Each parallel graph lowers to a fused compiled region with async dispatch:
// two call.stub.async (one per launch), an async.join, and an async.ready.

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
