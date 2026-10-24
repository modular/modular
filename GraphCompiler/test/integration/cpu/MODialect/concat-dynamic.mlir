// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=0 --result-output-style=full %t.mlir | FileCheck %s

// CHECK: --- Running 'concat_empty_dynamic':
// CHECK-NEXT: 'concat_empty_dynamic' returned tensor<0xsi32> []
mo.graph @concat_empty_dynamic<D0 -> D1>(%arg: !mo.tensor<[D0], si32>) -> !mo.tensor<[D1], si32> {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %out = mo.concat<() -> D2>[0](%arg, %arg) : (
    !mo.tensor<[D0], si32>, !mo.tensor<[D0], si32>
  ) -> !mo.tensor<[D2], si32>
  mo.output<D2> %out : !mo.tensor<[D2], si32>
}

// CHECK: --- Running 'concat_empty_from_upstream':
// CHECK-NEXT: 'concat_empty_from_upstream' returned tensor<0xsi64> []
mo.graph @concat_empty_from_upstream<D0 -> D1>(%arg: !mo.tensor<[D0], si32>) -> !mo.tensor<[D1], si64> {
  %arr = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %empty = mo.gather<() -> D2>(%arr, %arg) [axis = 0] : (
    !mo.tensor<[1], si64>, !mo.tensor<[D0], si32>
  ) -> !mo.tensor<[D2], si64>
  %out = mo.concat<() -> D3>[0](%empty, %empty) : (
    !mo.tensor<[D2], si64>, !mo.tensor<[D2], si64>
  ) -> !mo.tensor<[D3], si64>
  mo.output<D3> %out : !mo.tensor<[D3], si64>
}
