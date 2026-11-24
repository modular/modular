// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @broadcast_shape
// CHECK-DAG: kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(identical(:scalar<si64> D0, 1), identical(:scalar<si64> D0, 50)))>
// CHECK-DAG: kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(identical(:scalar<si64> add(D0, D1), 1), identical(:scalar<si64> add(D0, D1), 80)))>
// CHECK: [[V0:%.*]] = mo.static.broadcast_to(%arg0) : !mo.tensor<[], si64> -> !mo.tensor<[D0, D1, 4, cond(identical(:scalar<si64> D0, 1), 50, D0), 70, cond(identical(:scalar<si64> add(D0, D1), 1), 80, add(D0, D1))], si64>
// CHECK: mo.output{{.*}} [[V0]]
mo.graph @broadcast_shape<D0, D1 -> D2, D3, D4, D5, D6, D7>(%input: !mo.tensor<[], si64>) -> !mo.tensor<[D2, D3, D4, D5, D6, D7], si64> {
  %shape1 = mo.constant <#M.dense_array<50, 70, 80>: tensor<3xsi64>> : !mo.tensor<[3], si64>
  %shape2_params = mosh.param.to_value: !mosh.ape = <[D0, D1, 4, D0, 70, add(D0, D1)]>
  %shape2 = mo.shape.to_tensor(%shape2_params) -> !mo.tensor<[6], si64>
  %bcast_shape = mo.broadcast_shape(%shape1, %shape2): (!mo.tensor<[3], si64>, !mo.tensor<[6], si64>) -> !mo.tensor<[6], si64>
  %res = mo.broadcast_to<() -> D8, D9, D10, D11, D12, D13>(%input, %bcast_shape) : (!mo.tensor<[], si64>, !mo.tensor<[6], si64>) -> !mo.tensor<[D8, D9, D10, D11, D12, D13], si64>
  mo.output<D8, D9, D10, D11, D12, D13> %res : !mo.tensor<[D8, D9, D10, D11, D12, D13], si64>
}
