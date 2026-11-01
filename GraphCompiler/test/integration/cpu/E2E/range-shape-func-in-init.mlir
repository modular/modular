// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir --mlir-print-ir-after=mgp-exec-invariant-code-motion 2>&1 | FileCheck --check-prefix=CHECK-IR %s
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// Check shapeof(range) + mo.range was moved to init
// CHECK-IR-LABEL: mgp.model @shape_func_consts
// CHECK-IR: fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"shapeof(mo.range)
// CHECK-IR: fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.range
// CHECK-IR: } execute (

// CHECK-LABEL: Running 'shape_func_consts':
// CHECK: 'shape_func_consts' returned tensor<8xf32> [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]

mo.graph @shape_func_consts() -> !mo.tensor<[D0], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %1 = mo.constant <#M.dense_array<0.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %2 = mo.constant <#M.dense_array<8.000000e+00> : !M.array<1xf32>> : !mo.tensor<[], f32>
  mo.invoke_shape_func["mo.range"]<() -> D0>(%1, %2, %0) [dataDeptTensors = [0 : i32, 1 : i32, 2 : i32]] : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
  %3 = mo.range(%1, %2, %0) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D0], f32>
  mo.output %3 : !mo.tensor<[D0], f32>
}
