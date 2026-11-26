// RUN: mo-opt %s --split-input-file --verify-diagnostics

// map.index.collapse.stride's verifier: the input and result match the
// signature, there is one group per out dim, the groups cover the in-shape,
// and a static out dim is the product of its group's extents.

// A static out dim must be its group's product.
mo.graph @wrongProduct(%x: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32> {
  %0 = map.iter.opaque inputs (%x: !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    // expected-error @below{{out dim 0 is 7, not the product 6 of its group}}
    %1 = map.index.collapse.stride<#map.view.signature<[2, 3, 4] -> [7, 4]>, groups = [2, 1]> (%x) : !mo.tensor<[2, 3, 4], f32> -> !mo.tensor<[7, 4], f32>
  } {device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[2, 3, 4], f32>
}

// -----

// The groups must cover every in dim.
mo.graph @shortGroups(%x: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32> {
  %0 = map.iter.opaque inputs (%x: !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    // expected-error @below{{groups cover 2 of the in-shape's 3 dims}}
    %1 = map.index.collapse.stride<#map.view.signature<[2, 3, 4] -> [6]>, groups = [2]> (%x) : !mo.tensor<[2, 3, 4], f32> -> !mo.tensor<[6], f32>
  } {device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[2, 3, 4], f32>
}

// -----

// One group per out dim.
mo.graph @groupCount(%x: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32> {
  %0 = map.iter.opaque inputs (%x: !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    // expected-error @below{{expected one group per out dim, got 1 for rank 2}}
    %1 = map.index.collapse.stride<#map.view.signature<[2, 3, 4] -> [6, 4]>, groups = [3]> (%x) : !mo.tensor<[2, 3, 4], f32> -> !mo.tensor<[6, 4], f32>
  } {device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[2, 3, 4], f32>
}

// -----

// The result's shape is the signature's out-shape.
mo.graph @resultShape(%x: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32> {
  %0 = map.iter.opaque inputs (%x: !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    // expected-error @below{{result shape}}
    %1 = map.index.collapse.stride<#map.view.signature<[2, 3, 4] -> [6, 4]>, groups = [2, 1]> (%x) : !mo.tensor<[2, 3, 4], f32> -> !mo.tensor<[24], f32>
  } {device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[2, 3, 4], f32>
}
