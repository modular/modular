// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=compact --num-threads=4 %t.mlir | FileCheck %s

mo.graph @batch_matmul_big(%x: !mo.tensor<[2048,128,64], f32>, %y: !mo.tensor<[2048,64,128], f32>) -> (!mo.tensor<[2048,128,128], f32>) {
  %output = mo.batch_matmul(%x, %y) : (!mo.tensor<[2048,128,64], f32>, !mo.tensor<[2048,64,128], f32>) -> (!mo.tensor<[2048,128,128], f32>)
  mo.output %output : !mo.tensor<[2048,128,128], f32>
}
// CHECK{LITERAL}: 'batch_matmul_big' returned tensor([[[64.0000, 64.0000, 64.0000, ..., 64.0000, 64.0000, 64.0000],

mo.graph @batch_matmul_big_mixed_precision(%x: !mo.tensor<[2048,128,64], ui8>, %y: !mo.tensor<[2048,64,128], si8>) -> (!mo.tensor<[2048,128,128], si32>) {
  %output = mo.batch_matmul(%x, %y) : (!mo.tensor<[2048,128,64], ui8>, !mo.tensor<[2048,64,128], si8>) -> (!mo.tensor<[2048,128,128], si32>)
  mo.output %output : !mo.tensor<[2048,128,128], si32>
}
// CHECK{LITERAL}: 'batch_matmul_big_mixed_precision' returned tensor([[[64, 64, 64, ..., 64, 64, 64],
