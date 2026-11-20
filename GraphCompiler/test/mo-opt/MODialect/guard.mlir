// RUN: mo-opt --canonicalize %s | FileCheck --enable-var-scope %s

// CHECK-LABEL: mo.graph @matmul_assert
// CHECK-SAME: <[[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D0]], [[D3]]], f32> {
// CHECK-NEXT: kgen.param.declare chain: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[[D1]] != [[D2]]" : !kgen.string}, identical(:scalar<si64> [[D1]], [[D2]]))>
// CHECK-NEXT: [[V0:[^ ]*]] = mosh.param.to_value: !mo.chain = <chain>
// CHECK-NEXT: [[V1:[^ ]*]]:2 = mo.guard[[[V0]]] ([[VARG0]], [[VARG1]]) : !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.matmul([[V1]]#0, [[V1]]#1) : (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D0]], [[D3]]], f32>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[[[D0]], [[D3]]], f32>
mo.graph @matmul_assert<D0, D1, D2, D3>(%X: !mo.tensor<[D0, D1], f32>, %Y: !mo.tensor<[D2, D3], f32> ) -> !mo.tensor<[D0, D3], f32>
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


// CHECK-LABEL: mo.graph @various_types
// CHECK-SAME: ([[VARG0:[^ ]*]]: !mo.tensor<[20, 30], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[30, 50], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5], si64>, [[VARG3:[^ ]*]]: !mo.scalar<f64>, [[VARG4:[^ ]*]]: !mo.chain) -> (!mo.tensor<[20, 30], f32>, !mo.tensor<[30, 50], f32>, !mo.tensor<[5], si64>, !mo.scalar<f64>) {
// CHECK-NEXT: [[V0:[^ ]*]]:4 = mo.guard[[[VARG4]]] ([[VARG0]], [[VARG1]], [[VARG2]], [[VARG3]]) : !mo.tensor<[20, 30], f32>, !mo.tensor<[30, 50], f32>, !mo.tensor<[5], si64>, !mo.scalar<f64>
// CHECK-NEXT: mo.output [[V0]]#0, [[V0]]#1, [[V0]]#2, [[V0]]#3 : !mo.tensor<[20, 30], f32>, !mo.tensor<[30, 50], f32>, !mo.tensor<[5], si64>, !mo.scalar<f64>
mo.graph @various_types(%X: !mo.tensor<[20, 30], f32>,
                        %Y: !mo.tensor<[30, 50], f32>,
                        %tensor3: !mo.tensor<[5], si64>,
                        %scalar: !mo.scalar<f64>,
                        %chain: !mo.chain ) ->
                    (!mo.tensor<[20, 30], f32>, !mo.tensor<[30, 50], f32>, !mo.tensor<[5], si64>, !mo.scalar<f64>)
{
  %Xp, %Yp, %tensor3p, %scalarp = mo.guard[%chain](%X, %Y, %tensor3, %scalar) :
   !mo.tensor<[20, 30], f32>,
   !mo.tensor<[30, 50], f32>,
   !mo.tensor<[5], si64>,
   !mo.scalar<f64>

  mo.output %Xp, %Yp, %tensor3p, %scalarp : !mo.tensor<[20, 30], f32>, !mo.tensor<[30, 50], f32>, !mo.tensor<[5], si64>, !mo.scalar<f64>
}


// Test that duplicate inputs are deduplicated during canonicalization
// CHECK-LABEL: mo.graph @guard_duplicate_inputs
// CHECK-SAME: (%[[ARG0:[^ ]*]]: !mo.tensor<[], f32>, %[[ARG1:[^ ]*]]: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) {
mo.graph @guard_duplicate_inputs(%arg0: !mo.tensor<[], f32>, %arg1: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) {
  %ch = mo.chain.create()

  // CHECK: %[[V0:[^ ]*]]:2 = mo.guard[{{.*}}] (%[[ARG0]], %[[ARG1]]) : !mo.tensor<[], f32>, !mo.tensor<[], f32>
  %a0, %a1, %a2, %a3 = mo.guard[%ch](%arg0, %arg1, %arg0, %arg1) : !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
  // CHECK-NEXT: mo.output %[[V0]]#0, %[[V0]]#1, %[[V0]]#0, %[[V0]]#1 : !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
  mo.output %a0, %a1, %a2, %a3 : !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
}
