// RUN: mt --execute --result-output-style=full %s | FileCheck %s

// CHECK-LABEL: --- Running 'while_op':
// CHECK-NEXT{LITERAL}: 'while_op' returned tensor<f32> [-3.900000e+01]

mo.graph @while_op() -> (!mo.tensor<[], f32>) {
  %x = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %i = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[], si64>
  %y = mo.constant <#M.dense_array<4.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %res:2 = mo.while (%i as %ii: !mo.tensor<[], si64>, %x as %xi: !mo.tensor<[], f32>) {
    %max = mo.constant <#M.dense_array<10> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %cond = mo.greater(%max, %ii) : (!mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[], bool>
    mo.while.condition(%cond) %ii, %xi : !mo.tensor<[], si64>, !mo.tensor<[], f32>
  } do {
    %one = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %gt = mo.greater(%xi, %y) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
    %new_x = mo.if(%gt : !mo.tensor<[], bool>) -> !mo.tensor<[], f32> {
      %r = mo.add(%xi, %y) : !mo.tensor<[], f32>
      mo.yield %r : !mo.tensor<[], f32>
    } else {
      %r = mo.sub(%xi, %y) : !mo.tensor<[], f32>
      mo.yield %r : !mo.tensor<[], f32>
    }
    // Canonicalization will replace this constant with %i.
    %zero = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %tmp = mo.add(%ii, %zero) : !mo.tensor<[], si64>
    %new_i = mo.add(%tmp, %one) : !mo.tensor<[], si64>
    mo.yield %new_i, %new_x : !mo.tensor<[], si64>, !mo.tensor<[], f32>
  }
  mo.output %res#1 : !mo.tensor<[], f32>
}
