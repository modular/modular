// RUN: mo-opt --mo-constant-fold --verify-diagnostics
mo.graph @crashes() -> !mo.tensor<[3], si32> {
  %data = mo.constant <#M.dense_array<1,1,1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  // expected-error@below {{[ConstantFold] setup for model crashes_constant_subgraphs returned error: This is an error}}
  %out = mo.custom [symbol = "mo.test.return_error"](%data) : (!mo.tensor<[3], si32>) -> !mo.tensor<[3], si32>
  mo.output %out : !mo.tensor<[3], si32>
}
