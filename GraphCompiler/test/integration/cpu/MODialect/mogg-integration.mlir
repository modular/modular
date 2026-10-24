// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones %t.mlir | FileCheck %s

// This test is for fully e2e testing jit compiled kernels in the normal
// pipeline. The other mogg_ tests are to test individual specific
// subcomponents of mogg.

mo.graph @abs() -> (!mo.tensor<[3, 6], f32>) {
  %cst1 = mo.constant <#M.dense_array<1.500000e+00, -2.000000e+00, 2.500000e+00, -3.000000e+00, 3.500000e+00, 4.000000e+00, -4.500000e+00, 5.000000e+00, 5.500000e+00, -6.000000e+00, 6.500000e+00, -7.000000e+00, 7.500000e+00, -8.000000e+00, 8.5000000e+00, -9.000000e+00, 9.500000e+00, -1.000000e+01> : tensor<18xf32>> : !mo.tensor<[3, 6], f32>
  %1 = mo.abs(%cst1) : !mo.tensor<[3, 6], f32>
  mo.output %1 : !mo.tensor<[3, 6], f32>
}

// CHECK-IR-LABEL: mgp.model @abs
// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'abs':
// CHECK: 'abs' returned tensor<3x6xf32> [1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00, 6.000000e+00, 6.500000e+00, 7.000000e+00, 7.500000e+00, 8.000000e+00, 8.500000e+00, 9.000000e+00, 9.500000e+00, 1.000000e+01]


mo.graph @pow() -> (!mo.tensor<[3, 6], f32>) {
  %cst1 = mo.constant <#M.dense_array<50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00, 50.000000e+00> : tensor<18xf32>> : !mo.tensor<[3, 6], f32>
  %cst2 = mo.constant <#M.dense_array<2.500000e+00> : tensor<1xf32>> : !mo.tensor<[], f32>
  %bcast = mo.static.broadcast_to(%cst2) : !mo.tensor<[], f32> -> !mo.tensor<[3, 6], f32>
  %1 = mo.pow(%cst1 : !mo.tensor<[3, 6], f32>, %bcast : !mo.tensor<[3, 6], f32>) : !mo.tensor<[3, 6], f32>
  mo.output %1 : !mo.tensor<[3, 6], f32>
}

// CHECK-IR-LABEL: mgp.model @pow
// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'pow':
// CHECK: 'pow' returned tensor<3x6xf32> [1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04, 1.76776{{[[:digit:]]+}}e+04]


mo.graph @pow_int() -> (!mo.tensor<[3, 6], si32>) {
  %cst1 = mo.constant <#M.dense_array<3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3> : tensor<18xsi32>> : !mo.tensor<[3, 6], si32>
  %cst2 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %bcast = mo.static.broadcast_to(%cst2) : !mo.tensor<[], si32> -> !mo.tensor<[3, 6], si32>
  %1 = mo.pow(%cst1 : !mo.tensor<[3, 6], si32>, %bcast : !mo.tensor<[3, 6], si32>) : !mo.tensor<[3, 6], si32>
  mo.output %1 : !mo.tensor<[3, 6], si32>
}

// CHECK-IR-LABEL: mgp.model @pow_int
// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'pow_int':
// CHECK: pow_int' returned tensor<3x6xsi32> [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

mo.graph @batch_matmul() -> (!mo.tensor<[2, 2, 2], f32>) {
  %cst1 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<2x2x2xf32>> : !mo.tensor<[2, 2, 2], f32>
  %cst2 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<2x2x2xf32>> : !mo.tensor<[2, 2, 2], f32>
  %1 = mo.batch_matmul(%cst1, %cst2) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>) -> (!mo.tensor<[2, 2, 2], f32>)
  mo.output %1 : !mo.tensor<[2, 2, 2], f32>

}

// CHECK-IR-LABEL: mgp.model @batch_matmul
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.batch_matmul{{.*}}"
// CHECK-LABEL:--- Running 'batch_matmul':
// CHECK: 'batch_matmul' returned tensor<2x2x2xf32> [7.000000e+00, 1.000000e+01, 1.500000e+01, 2.200000e+01, 6.700000e+01, 7.800000e+01, 9.100000e+01, 1.060000e+02]

mo.graph @matmul(%arg0: !mo.tensor<[3, 3], f32>) -> (!mo.tensor<[3, 3], f32>) {
  %A = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0> : tensor<3x3xf32>} : () -> !mo.tensor<[3, 3], f32>
  %output = mo.matmul(%A, %arg0) : (!mo.tensor<[3, 3], f32>, !mo.tensor<[3, 3], f32>) -> !mo.tensor<[3, 3], f32>
  mo.output %output : !mo.tensor<[3, 3], f32>
}

// CHECK-IR-LABEL: mgp.model @matmul
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.matmul{{.*}}"
// CHECK-LABEL:--- Running 'matmul':
// CHECK: 'matmul' returned tensor<3x3xf32> [6.000000e+00, 6.000000e+00, 6.000000e+00, 1.500000e+01, 1.500000e+01, 1.500000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01]
