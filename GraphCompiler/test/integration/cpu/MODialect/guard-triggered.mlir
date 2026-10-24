// RUN: mo-opt --gc-pipeline %s -o %t.mlir
// RUN: mt --execute  --model-inputs=ones --input-shapes=1x3 --input-shapes=40x4 --result-output-style=full %t.mlir > %t.output || true

// using the rather laborious specification of passes above because
// predicate-assert-deps pass doesn't currently handle mo.guard ops and I don't
// want any interactions with that pass in this test

// using the ... || true construct on the 2nd RUN line since lit seems to
// consider commands that return non-zero as an entire test failure

// CHECK-LABEL: --- Running 'matmul':
// CHECK-NEXT: 'matmul' returned <error: D1 != D2>
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
