// RUN: mo-opt %s --map-infer-max-simd-width | FileCheck %s

// An index the body computes from the loop's own, here a reshape the pipeline
// would have hoisted into a view, can't be followed: the kernel runs at width 1.

// CHECK-LABEL: mo.graph @reshaped_index
// CHECK:       map.iter.elementwise.body<[4, 6]> max_simd_width(1)
mo.graph @reshaped_index(%a: !mo.tensor<[3, 8], f32>) -> !mo.tensor<[4, 6], f32> {
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[3, 8], f32>)
      outputs (%out : !mo.tensor<[4, 6], f32>) {
    map.iter.elementwise.body<[4, 6]>
        index (%idx: !map.index<2>)
        capturing {%a0 = %a : !mo.tensor<[3, 8], f32>, %out0 = %out : !mo.tensor<[4, 6], f32>} {
      %j = map.index.reshape<#map.index.reshape.spec<[3, 8] -> [4, 6]>> (%idx) : !map.index<2> -> !map.index<2>
      %v = map.load %a0[%j] : !mo.tensor<[3, 8], f32>
      map.store %out0[%idx] = %v : !mo.tensor<[4, 6], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[4, 6], f32>
}
