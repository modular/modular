// RUN: mo-opt %s --mo-propagate-shapes --canonicalize --verify-diagnostics

// Fails because dim index 2 in output we try to reconcile dim 50 with 40
mo.graph @broadcast_shape_static_error<D0, D1 -> D2, D3, D4, D5, D6>(%input: !mo.tensor<[], si64>) -> !mo.tensor<[D2, D3, D4, D5, D6], si64> {
  %shape1 = mo.constant <#M.dense_array<50, 70, 80>: tensor<3xsi64>> : !mo.tensor<[3], si64>
  %shape2_params = mosh.param.to_value: !mosh.ape = <[D0, 4, 40, 70, 80]>
  %shape2 = mo.shape.to_tensor(%shape2_params) -> !mo.tensor<[5], si64>

  %bcast_shape = mo.broadcast_shape(%shape1, %shape2): (!mo.tensor<[3], si64>, !mo.tensor<[5], si64>) -> !mo.tensor<[5], si64>
  // expected-error@+1 {{[broadcast_shape] dimensions are not broadcastable}}
  %res = mo.broadcast_to<() -> D7, D8, D9, D10, D11>(%input, %bcast_shape) : (!mo.tensor<[], si64>, !mo.tensor<[5], si64>) -> !mo.tensor<[D7, D8, D9, D10, D11], si64>

  mo.output<D7, D8, D9, D10, D11> %res : !mo.tensor<[D7, D8, D9, D10, D11], si64>
}
