// RUN: mo-opt %s --mo-propagate-shapes --canonicalize --verify-diagnostics


mo.graph @gather_1d_indices<N1, N2 -> out0, out1>(
  %input: !mo.tensor<[], si64>,
  %dummy: !mo.tensor<[N1, 10, N2], f32>
) -> !mo.tensor<[out0, out1], si64> {

  %indices = mo.constant <#M.dense_array<-1, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, 10, N2], f32>) -> !mo.tensor<[3], si64>
  %shape = mo.gather(%dummy_shape, %indices) [axis = 0] : (
    !mo.tensor<[3], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2], si64>

  // expected-error@+1 {{[gather] index out of bound: 3 for input size: 3}}
  %res = mo.broadcast_to<() -> D0, D1>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D0, D1], si64>

  mo.output<D0, D1> %res : !mo.tensor<[D0, D1], si64>
}
