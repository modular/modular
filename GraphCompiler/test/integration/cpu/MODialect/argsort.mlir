// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// `mx.argsort` takes its tensor arguments as role + DenseTensor traits. A
// package build does not exercise the generated argument struct, so this graph
// is what catches a migration that broke kernel-parameter inference or the
// `TileTensorable` projection.
//
// The sorted values are a constant with distinct entries so the permutation is
// unique, multiplied by the all-ones graph input so the data reaching the
// kernel comes from a runtime buffer rather than a folded constant.

// CHECK: 'argsort_ascending' returned tensor<5xsi64> [3, 0, 2, 1, 4]
mo.graph @argsort_ascending(%ones : !mo.tensor<[5], f32>) -> !mo.tensor<[5], si64> {
  %x = mo.constant <#M.dense_array<-1.0, 3.0, 2.0, -2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %inp = mo.mul(%ones, %x) : !mo.tensor<[5], f32>
  %out = mo.custom [
    symbol = "mx.argsort",
    device = #M.device_ref<"cpu", 0>,
    parameters = { ascending = true }
  ] (%inp) : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], si64>
  mo.output %out : !mo.tensor<[5], si64>
}

// CHECK: 'argsort_descending' returned tensor<5xsi64> [4, 1, 2, 0, 3]
mo.graph @argsort_descending(%ones : !mo.tensor<[5], f32>) -> !mo.tensor<[5], si64> {
  %x = mo.constant <#M.dense_array<-1.0, 3.0, 2.0, -2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %inp = mo.mul(%ones, %x) : !mo.tensor<[5], f32>
  %out = mo.custom [
    symbol = "mx.argsort",
    device = #M.device_ref<"cpu", 0>,
    parameters = { ascending = false }
  ] (%inp) : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], si64>
  mo.output %out : !mo.tensor<[5], si64>
}
