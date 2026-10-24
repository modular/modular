// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @random_uniform() -> !mo.tensor<[1, 1, 7, 8], f64> {
  %shape = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %lowerBound = mo.constant <#M.dense_array<0.0> : tensor<1xf64>> : !mo.tensor<[], f64>
  %upperBound = mo.constant <#M.dense_array<100.0> : tensor<1xf64>> : !mo.tensor<[], f64>
  %seed = mo.constant <#M.dense_array<2> : tensor<1xui64>> : !mo.tensor<[1], ui64, gpu:0>
  %output = mo.random.uniform(%shape, %lowerBound, %upperBound, %seed) : (!mo.tensor<[4], si64>, !mo.tensor<[], f64>, !mo.tensor<[], f64>, !mo.tensor<[1], ui64, gpu:0>) -> !mo.tensor<[1, 1, 7, 8], f64, gpu:0>

  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %output : !mo.tensor<[1, 1, 7, 8], f64, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[1, 1, 7, 8], f64>
}

// CHECK-LABEL: --- Running 'random_uniform':
// CHECK-NEXT: 'random_uniform' returned tensor<1x1x7x8xf64> [{{((-?[[:digit:]]\.[[:digit:]]+e[+-][[:digit:]]+(, )?){56})}}
