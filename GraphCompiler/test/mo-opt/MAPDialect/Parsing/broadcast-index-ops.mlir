// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trip map.index.broadcast and the map.index.broadcast.spec attribute
// it takes. Covers a rank-preserving case (in-shape dim 1 broadcasts to N)
// and a rank-expanding case (numpy semantics: in has fewer dims than out).
// Value names are matched with capture variables so changes in the printer's
// SSA numbering don't break the test.

#sameRankSpec = #map.index.broadcast.spec<[1, 64] -> [8, 64]>
#rankExpandingSpec = #map.index.broadcast.spec<[64] -> [8, 64]>

module {

// Rank-preserving broadcast: in-shape dim 0 (size 1) broadcasts to 8.
// CHECK-LABEL: mo.graph @broadcastSameRank
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.elementwise<[8, 64]>
// CHECK-NEXT:    inputs  ([[INPUT:%[^ :,]+]]: !mo.tensor<[1, 64], f32>)
// CHECK-NEXT:    outputs ([[OUTARG:%[^ :,]+]]: !mo.tensor<[8, 64], f32>)
// CHECK-NEXT:    index   ([[IDX:%[^ :,]+]]: !map.index<2>) -> !mo.tensor<[8, 64], f32> {
// CHECK-NEXT:   [[IDX_:%[^ ]+]] = map.index.broadcast<#map.index.broadcast.spec<[1, 64] -> [8, 64]>> ([[IDX]]) : !map.index<2> -> !map.index<2>
// CHECK-NEXT:   [[VAL:%[^ ]+]] = map.load [[INPUT]]{{\[}}[[IDX_]]] : !mo.tensor<[1, 64], f32>
// CHECK-NEXT:   map.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[8, 64], f32>
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT]] : !mo.tensor<[8, 64], f32>
mo.graph @broadcastSameRank(%input: !mo.tensor<[1, 64], f32>)
    -> !mo.tensor<[8, 64], f32> {
  %output = map.iter.elementwise<[8, 64]>
      inputs  (%input: !mo.tensor<[1, 64], f32>)
      outputs (%out: !mo.tensor<[8, 64], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[8, 64], f32>
  {
    %idx_ = map.index.broadcast<#sameRankSpec>(%idx)
        : !map.index<2> -> !map.index<2>
    %val = map.load %input[%idx_] : !mo.tensor<[1, 64], f32>
    map.store %out[%idx] = %val : !mo.tensor<[8, 64], f32>
  }
  mo.output %output : !mo.tensor<[8, 64], f32>
}

// Rank-expanding broadcast: in has one fewer dim than out (numpy semantics --
// the leading output dim is entirely replicated, dropped from the projected
// index).
// CHECK-LABEL: mo.graph @broadcastRankExpanding
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.elementwise<[8, 64]>
// CHECK-NEXT:    inputs  ([[INPUT2:%[^ :,]+]]: !mo.tensor<[64], f32>)
// CHECK-NEXT:    outputs ([[OUTARG2:%[^ :,]+]]: !mo.tensor<[8, 64], f32>)
// CHECK-NEXT:    index   ([[IDX2:%[^ :,]+]]: !map.index<2>) -> !mo.tensor<[8, 64], f32> {
// CHECK-NEXT:   [[IDX2_:%[^ ]+]] = map.index.broadcast<#map.index.broadcast.spec<[64] -> [8, 64]>> ([[IDX2]]) : !map.index<2> -> !map.index<1>
// CHECK-NEXT:   [[VAL2:%[^ ]+]] = map.load [[INPUT2]]{{\[}}[[IDX2_]]] : !mo.tensor<[64], f32>
// CHECK-NEXT:   map.store [[OUTARG2]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[8, 64], f32>
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT2]] : !mo.tensor<[8, 64], f32>
mo.graph @broadcastRankExpanding(%input: !mo.tensor<[64], f32>)
    -> !mo.tensor<[8, 64], f32> {
  %output = map.iter.elementwise<[8, 64]>
      inputs  (%input: !mo.tensor<[64], f32>)
      outputs (%out: !mo.tensor<[8, 64], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[8, 64], f32>
  {
    %idx_ = map.index.broadcast<#rankExpandingSpec>(%idx)
        : !map.index<2> -> !map.index<1>
    %val = map.load %input[%idx_] : !mo.tensor<[64], f32>
    map.store %out[%idx] = %val : !mo.tensor<[8, 64], f32>
  }
  mo.output %output : !mo.tensor<[8, 64], f32>
}

}
