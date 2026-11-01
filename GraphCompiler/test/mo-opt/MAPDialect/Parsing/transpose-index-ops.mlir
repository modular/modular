// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trip map.index.transpose and the map.index.transpose.spec attribute
// it takes. Covers a simple 2D swap and a 3D permutation. The perm rides in as
// a runtime operand (like slice's bounds), even when the spec bakes it as
// literals. Value names are matched with capture variables so changes in the
// printer's SSA numbering don't break the test.

#swapSpec = #map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>
#permute3dSpec = #map.index.transpose.spec<[2, 3, 4] -> [4, 2, 3], perm = [2, 0, 1]>

module {

// 2D swap: out.dim[0] = in.dim[1], out.dim[1] = in.dim[0].
// CHECK-LABEL: mo.graph @transposeSwap
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.elementwise<[3, 2]>
// CHECK-NEXT:    inputs  ([[INPUT:%[^ :,]+]]: !mo.tensor<[2, 3], f32>, [[PERM:%[^ :,]+]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:    outputs ([[OUTARG:%[^ :,]+]]: !mo.tensor<[3, 2], f32>)
// CHECK-NEXT:    index   ([[IDX:%[^ :,]+]]: !map.index<2>) -> !mo.tensor<[3, 2], f32> {
// CHECK-NEXT:   [[IDX_:%[^ ]+]] = map.index.transpose<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>> ([[IDX]], [[PERM]]) : (!map.index<2>, !mo.tensor<[2], si64>) -> !map.index<2>
// CHECK-NEXT:   [[VAL:%[^ ]+]] = map.load [[INPUT]]{{\[}}[[IDX_]]] : !mo.tensor<[2, 3], f32>
// CHECK-NEXT:   map.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[3, 2], f32>
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT]] : !mo.tensor<[3, 2], f32>
mo.graph @transposeSwap(%input: !mo.tensor<[2, 3], f32>, %perm: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 2], f32> {
  %output = map.iter.elementwise<[3, 2]>
      inputs  (%input: !mo.tensor<[2, 3], f32>, %perm: !mo.tensor<[2], si64>)
      outputs (%out: !mo.tensor<[3, 2], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[3, 2], f32>
  {
    %idx_ = map.index.transpose<#swapSpec>(%idx, %perm)
        : (!map.index<2>, !mo.tensor<[2], si64>) -> !map.index<2>
    %val = map.load %input[%idx_] : !mo.tensor<[2, 3], f32>
    map.store %out[%idx] = %val : !mo.tensor<[3, 2], f32>
  }
  mo.output %output : !mo.tensor<[3, 2], f32>
}

// 3D permutation: out.dim[i] = in.dim[perm[i]] for perm = [2, 0, 1].
// CHECK-LABEL: mo.graph @transpose3D
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.elementwise<[4, 2, 3]>
// CHECK-NEXT:    inputs  ([[INPUT2:%[^ :,]+]]: !mo.tensor<[2, 3, 4], f32>, [[PERM2:%[^ :,]+]]: !mo.tensor<[3], si64>)
// CHECK-NEXT:    outputs ([[OUTARG2:%[^ :,]+]]: !mo.tensor<[4, 2, 3], f32>)
// CHECK-NEXT:    index   ([[IDX2:%[^ :,]+]]: !map.index<3>) -> !mo.tensor<[4, 2, 3], f32> {
// CHECK-NEXT:   [[IDX2_:%[^ ]+]] = map.index.transpose<#map.index.transpose.spec<[2, 3, 4] -> [4, 2, 3], perm = [2, 0, 1]>> ([[IDX2]], [[PERM2]]) : (!map.index<3>, !mo.tensor<[3], si64>) -> !map.index<3>
// CHECK-NEXT:   [[VAL2:%[^ ]+]] = map.load [[INPUT2]]{{\[}}[[IDX2_]]] : !mo.tensor<[2, 3, 4], f32>
// CHECK-NEXT:   map.store [[OUTARG2]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[4, 2, 3], f32>
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT2]] : !mo.tensor<[4, 2, 3], f32>
mo.graph @transpose3D(%input: !mo.tensor<[2, 3, 4], f32>, %perm: !mo.tensor<[3], si64>)
    -> !mo.tensor<[4, 2, 3], f32> {
  %output = map.iter.elementwise<[4, 2, 3]>
      inputs  (%input: !mo.tensor<[2, 3, 4], f32>, %perm: !mo.tensor<[3], si64>)
      outputs (%out: !mo.tensor<[4, 2, 3], f32>)
      index   (%idx: !map.index<3>)
      -> !mo.tensor<[4, 2, 3], f32>
  {
    %idx_ = map.index.transpose<#permute3dSpec>(%idx, %perm)
        : (!map.index<3>, !mo.tensor<[3], si64>) -> !map.index<3>
    %val = map.load %input[%idx_] : !mo.tensor<[2, 3, 4], f32>
    map.store %out[%idx] = %val : !mo.tensor<[4, 2, 3], f32>
  }
  mo.output %output : !mo.tensor<[4, 2, 3], f32>
}

}
