// RUN: mo-opt %s --mogg-import-packages --map-flatten-shapes | FileCheck %s

// Index computation left in a body, here a reshape on the store index that
// `LoopInvariantViewMotion` can't hoist, would see a different index after
// flattening: the loop is left alone.

// CHECK-LABEL: mo.graph @store_reshape
// CHECK-NOT:   map.index.collapse.stride
// CHECK:       map.iter.elementwise.body<[4, 6]>
mo.graph @store_reshape(%input: !mo.tensor<[4, 6], f32>) -> !mo.tensor<[24], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[4, 6], f32>)
      outputs (%out : !mo.tensor<[24], f32>) {
    map.iter.elementwise.body<[4, 6]>
        index (%idx: !map.index<2>)
        capturing {%input0 = %input : !mo.tensor<[4, 6], f32>, %out0 = %out : !mo.tensor<[24], f32>} {
      %val = map.load %input0[%idx] : !mo.tensor<[4, 6], f32>
      %idx_ = map.index.reshape<#map.index.reshape.spec<[24] -> [4, 6]>> (%idx)
          : !map.index<2> -> !map.index<1>
      map.store %out0[%idx_] = %val : !mo.tensor<[24], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[24], f32>
}

// The same on a load: a broadcast index op left in the body. Flattening would
// read the capture at the loop's merged index through an op still built for
// the old rank, so the loop is left alone.

// CHECK-LABEL: mo.graph @load_broadcast
// CHECK-NOT:   map.index.collapse.stride
// CHECK:       map.iter.elementwise.body<[8, 64]>
mo.graph @load_broadcast(%input: !mo.tensor<[1, 64], f32>) -> !mo.tensor<[8, 64], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[1, 64], f32>)
      outputs (%out : !mo.tensor<[8, 64], f32>) {
    map.iter.elementwise.body<[8, 64]>
        index (%idx: !map.index<2>)
        capturing {%input0 = %input : !mo.tensor<[1, 64], f32>, %out0 = %out : !mo.tensor<[8, 64], f32>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[1, 64] -> [8, 64]>> (%idx)
          : !map.index<2> -> !map.index<2>
      %val = map.load %input0[%idx_] : !mo.tensor<[1, 64], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[8, 64], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[8, 64], f32>
}
