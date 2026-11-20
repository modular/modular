// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones --input-shapes=1x1 --input-shapes="" %t.mlir | FileCheck %s


// CHECK: tensor<1x1xsi64> [1]
// CHECK-NEXT: , tensor<1x1xsi64> [0]
mo.graph @top_k_dyn<D0, D1>(%in: !mo.tensor<[D0, D1], si64>, %K: !mo.tensor<[], si64>) -> (!mo.tensor<[D0, D1], si64>, !mo.tensor<[D0, D1], si64>) {
  %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %sorted = "mo.constant"() {value = #M.dense_array<1> : tensor<1xi1>} : () -> !mo.tensor<[], bool>
  %values, %indices = mo.top_k(%in, %K, %axis, %sorted) : (
    !mo.tensor<[D0, D1], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[D0, D1], si64>, !mo.tensor<[D0, D1], si64>)
  mo.output %values, %indices : !mo.tensor<[D0, D1], si64>, !mo.tensor<[D0, D1], si64>
}
