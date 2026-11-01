// REQUIRES: x86_64-linux
// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s


mo.graph @matmul_u8_s8_s32(%input: !mo.tensor<[4, 2], ui8>) -> !mo.tensor<[4, 4], si32> {

  %weights = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6, 7, 8> : tensor<2x4xsi8>> : !mo.tensor<[2, 4], si8, {layout = #mo.layout<KN>}>
  %packed_weights = mo.layout.transform(%weights)
    [kgenParams = {a_type = ui8, a_shape = dense<[4,2]> : vector<2xi64>, b_type = si8, b_shape = dense<[2,4]> : vector<2xi64>, c_type = si32, c_shape = dense<[4,4]> : vector<2xi64>}] :
    !mo.tensor<[2, 4], si8, {layout = #mo.layout<KN>}> ->
    !mo.tensor<[2, 4], si8, {layout = #mo.layout<KNkni>}>

  %matmul = mo.matmul(%input, %packed_weights) : (
    !mo.tensor<[4, 2], ui8>, !mo.tensor<[2, 4], si8, {layout = #mo.layout<KNkni>}>
  ) -> !mo.tensor<[4, 4], si32>

  mo.output %matmul : !mo.tensor<[4, 4], si32>
}
// CHECK{LITERAL}: 'matmul_u8_s8_s32' returned tensor<4x4xsi32> [6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12]
