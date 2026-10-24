// RUN: mt --execute  --model-inputs=ones --input-shapes=1x3 --input-shapes=3x4 --result-output-style=full %s | FileCheck --enable-var-scope %s

// CHECK-LABEL: --- Running 'matmul':
// CHECK-NEXT: 'matmul' returned tensor<1x4xf32> [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]
mo.graph @matmul<D0, D1, D2, D3>(%X: !mo.tensor<[D0, D1], f32>, %Y: !mo.tensor<[D2, D3], f32> ) -> !mo.tensor<[D0, D3], f32>
{
  kgen.param.declare chain: !mo.chain =
    <apply(:(!kgen.scalar<bool>) ->
    !mo.chain "mo.assert"{message = "D1 != D2" : !kgen.string}, identical(:scalar<si64> D1, D2))>

  %chain = mosh.param.to_value: !mo.chain = <chain>
  %Xp, %Yp = mo.guard[%chain](%X, %Y) : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>

  %Z = mo.matmul(%Xp, %Yp) :
   (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D3], f32>

  mo.output %Z: !mo.tensor<[D0, D3], f32>
}
