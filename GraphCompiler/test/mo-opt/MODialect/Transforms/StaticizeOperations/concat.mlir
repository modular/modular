// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @concat
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>,
mo.graph @concat<N1, N2 -> out0, out1, out2>(
  %input: !mo.tensor<[], si64>,
  %dummy1: !mo.tensor<[N1], f32>,
  %dummy2: !mo.tensor<[N2], f32>
) -> !mo.tensor<[out0, out1, out2], si64> {

  %axis = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %one = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %n1 = mo.shape_of(%dummy1) : (!mo.tensor<[N1], f32>) -> !mo.tensor<[1], si32>
  %n2 = mo.shape_of(%dummy2) : (!mo.tensor<[N2], f32>) -> !mo.tensor<[1], si32>
  %shape = mo.concat[0](%n1, %one, %n2) : (
    !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>
  ) -> !mo.tensor<[3], si32>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[[[N1]], 1, [[N2]]], si64>
  %res = mo.broadcast_to<() -> D0, D1, D2>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[D0, D1, D2], si64>


  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0, D1, D2> %res : !mo.tensor<[D0, D1, D2], si64>
}
