// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trip map.index.reshape and the map.index.reshape.spec attribute it
// takes. Covers a rank-reducing case (flatten) and a rank-expanding case
// (unflatten). Value names are matched with capture variables so changes in
// the printer's SSA numbering don't break the test.

#flattenSpec = #map.index.reshape.spec<[4, 6] -> [24]>
#expandSpec = #map.index.reshape.spec<[24] -> [4, 6]>

module {

// Rank-reducing reshape: flatten [4, 6] into [24].
// CHECK-LABEL: mo.graph @reshapeFlatten
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.elementwise<[24]>
// CHECK-NEXT:    inputs  ([[INPUT:%[^ :,]+]]: !mo.tensor<[4, 6], f32>)
// CHECK-NEXT:    outputs ([[OUTARG:%[^ :,]+]]: !mo.tensor<[24], f32>)
// CHECK-NEXT:    index   ([[IDX:%[^ :,]+]]: !map.index<1>) -> !mo.tensor<[24], f32> {
// CHECK-NEXT:   [[IDX_:%[^ ]+]] = map.index.reshape<#map.index.reshape.spec<[4, 6] -> [24]>> ([[IDX]]) : !map.index<1> -> !map.index<2>
// CHECK-NEXT:   [[VAL:%[^ ]+]] = map.load [[INPUT]]{{\[}}[[IDX_]]] : !mo.tensor<[4, 6], f32>
// CHECK-NEXT:   map.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[24], f32>
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT]] : !mo.tensor<[24], f32>
mo.graph @reshapeFlatten(%input: !mo.tensor<[4, 6], f32>)
    -> !mo.tensor<[24], f32> {
  %output = map.iter.elementwise<[24]>
      inputs  (%input: !mo.tensor<[4, 6], f32>)
      outputs (%out: !mo.tensor<[24], f32>)
      index   (%idx: !map.index<1>)
      -> !mo.tensor<[24], f32>
  {
    %idx_ = map.index.reshape<#flattenSpec>(%idx)
        : !map.index<1> -> !map.index<2>
    %val = map.load %input[%idx_] : !mo.tensor<[4, 6], f32>
    map.store %out[%idx] = %val : !mo.tensor<[24], f32>
  }
  mo.output %output : !mo.tensor<[24], f32>
}

// Rank-expanding reshape: unflatten [24] into [4, 6].
// CHECK-LABEL: mo.graph @reshapeExpand
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.elementwise<[4, 6]>
// CHECK-NEXT:    inputs  ([[INPUT2:%[^ :,]+]]: !mo.tensor<[24], f32>)
// CHECK-NEXT:    outputs ([[OUTARG2:%[^ :,]+]]: !mo.tensor<[4, 6], f32>)
// CHECK-NEXT:    index   ([[IDX2:%[^ :,]+]]: !map.index<2>) -> !mo.tensor<[4, 6], f32> {
// CHECK-NEXT:   [[IDX2_:%[^ ]+]] = map.index.reshape<#map.index.reshape.spec<[24] -> [4, 6]>> ([[IDX2]]) : !map.index<2> -> !map.index<1>
// CHECK-NEXT:   [[VAL2:%[^ ]+]] = map.load [[INPUT2]]{{\[}}[[IDX2_]]] : !mo.tensor<[24], f32>
// CHECK-NEXT:   map.store [[OUTARG2]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[4, 6], f32>
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT2]] : !mo.tensor<[4, 6], f32>
mo.graph @reshapeExpand(%input: !mo.tensor<[24], f32>)
    -> !mo.tensor<[4, 6], f32> {
  %output = map.iter.elementwise<[4, 6]>
      inputs  (%input: !mo.tensor<[24], f32>)
      outputs (%out: !mo.tensor<[4, 6], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[4, 6], f32>
  {
    %idx_ = map.index.reshape<#expandSpec>(%idx)
        : !map.index<2> -> !map.index<1>
    %val = map.load %input[%idx_] : !mo.tensor<[24], f32>
    map.store %out[%idx] = %val : !mo.tensor<[4, 6], f32>
  }
  mo.output %output : !mo.tensor<[4, 6], f32>
}

}
