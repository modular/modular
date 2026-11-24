// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-raise-back-contiguous-views | FileCheck %s

// A prefix of a static input along one dim, with a symbolic extent, raises to
// mogg.slice. The stop operand is the output extent. Any other slice with a
// symbolic extent stays a map.iter.elementwise.

// CHECK-LABEL: mo.graph @symbolicStopLeadingDim
// CHECK-NOT:     map.iter.elementwise
// CHECK:         [[START:%.*]] = mo.constant <#M.dense_array<0> : !M.array<1xsi64>>
// CHECK:         [[EXTENT:%.*]] = mosh.param.to_value{{.*}}<R>
// CHECK:         [[STOP:%.*]] = mo.index.to_tensor([[EXTENT]]) -> !mo.tensor<[], si64>
// CHECK:         mogg.slice(%arg0, [[START]], [[STOP]]) : (!mo.tensor<[4, 512], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[R, 512], f32>
mo.graph @symbolicStopLeadingDim<R -> OR>(
    %input: !mo.tensor<[4, 512], f32>,
    %stop: !mo.tensor<[2], si64>
) -> !mo.tensor<[OR, 512], f32> {
  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[4, 512], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[R, 512], f32>
  mo.output<R> %res : !mo.tensor<[R, 512], f32>
}

// The input's leading dim has extent 1, so it is the sliced dim even though
// it is not the first input dim with extent other than 1.

// CHECK-LABEL: mo.graph @symbolicStopOneRow
// CHECK-NOT:     map.iter.elementwise
// CHECK:         mogg.slice(%arg0, %{{.*}}, %{{.*}}) : (!mo.tensor<[1, 512], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[R, 512], f32>
mo.graph @symbolicStopOneRow<R -> OR>(
    %input: !mo.tensor<[1, 512], f32>,
    %stop: !mo.tensor<[2], si64>
) -> !mo.tensor<[OR, 512], f32> {
  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[1, 512], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[R, 512], f32>
  mo.output<R> %res : !mo.tensor<[R, 512], f32>
}

// The symbolic dim is not the leading dim, so the rows of the result are not
// contiguous in the input.

// CHECK-LABEL: mo.graph @symbolicInnerDimNotRaised
// CHECK-NOT:     mogg.slice
// CHECK:         map.iter.elementwise<[4, C]>
mo.graph @symbolicInnerDimNotRaised<C -> OC>(
    %input: !mo.tensor<[4, 512], f32>,
    %stop: !mo.tensor<[2], si64>
) -> !mo.tensor<[4, OC], f32> {
  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[4, 512], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[4, C], f32>
  mo.output<C> %res : !mo.tensor<[4, C], f32>
}

// CHECK-LABEL: mo.graph @symbolicStopNonzeroStartNotRaised
// CHECK-NOT:     mogg.slice
// CHECK:         map.iter.elementwise<[R, 512]>
mo.graph @symbolicStopNonzeroStartNotRaised<R -> OR>(
    %input: !mo.tensor<[4, 512], f32>,
    %stop: !mo.tensor<[2], si64>
) -> !mo.tensor<[OR, 512], f32> {
  %start = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[4, 512], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[R, 512], f32>
  mo.output<R> %res : !mo.tensor<[R, 512], f32>
}

// CHECK-LABEL: mo.graph @symbolicStopStridedNotRaised
// CHECK-NOT:     mogg.slice
// CHECK:         map.iter.elementwise<[R, 512]>
mo.graph @symbolicStopStridedNotRaised<R -> OR>(
    %input: !mo.tensor<[4, 512], f32>,
    %stop: !mo.tensor<[2], si64>
) -> !mo.tensor<[OR, 512], f32> {
  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[4, 512], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[R, 512], f32>
  mo.output<R> %res : !mo.tensor<[R, 512], f32>
}
