// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir

mo.graph @add(%arg0: !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) isSubgraph {
  %1 = mo.constant.external [name = "foo", align = 16, device = #M.device_ref<"cpu", 0>, isPlaceholder = true] : !mo.tensor<[5], f32>
  %add = mo.add(%arg0, %1) : !mo.tensor<[5], f32>
  mo.output %add : !mo.tensor<[5], f32>
}

mo.graph @combined() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[5], f32>
  %add = mo.call @add(%0) [prefix = "prefix"] : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  mo.output %add : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'combined' returned tensor<5xf32> [2.000000e+00, 3.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+01]
