// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-to-mogg | FileCheck %s

// mo.invoke_shape_func has no MAP form, so map-to-mogg lowers it directly. Its
// `kgenParams` are ordinary parameters of the shape function and must reach the
// call alongside the declared `output_rank`; forwarding only the rank leaves
// `axis` uninferable at emit time.

// CHECK-LABEL: mo.graph @gatherShapeFuncKgenParams
// CHECK: mogg.call.shape @{{.*}}gather_shape_fn
// CHECK-SAME: [axis = -2 : index, output_rank = 4 : index]
mo.graph @gatherShapeFuncKgenParams<() -> D0, D1, D2, D3>(
    %input: !mo.tensor<[1, 2, 3], f32>,
    %indices: !mo.tensor<[4, 5], f32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {
  mo.invoke_shape_func["mo.gather"]<() -> D4, D5, D6, D7>(%input, %indices)
    [kgenParams = {axis = -2 : index}, dataDeptTensors = []] :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[4, 5], f32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
