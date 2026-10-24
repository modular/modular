// RUN: mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages, map-to-mogg)' | FileCheck %s

// A hoisted slice view whose $input is a buffer rather than a tensor -- what
// `LoopInvariantViewMotion` produces once `FuseMutableLoads` has retyped an
// in-place kernel's load target. The view re-addresses mutable storage, so
// `mogg.tensor.create.slice` has to carry `mut = true`; its result stays a
// `!mogg.tensor`, since the view is only loaded from. The tensor case is
// the contrast: same lowering, `mut = false`.

// CHECK-LABEL: mo.graph @sliceStrideOverBuffer
// CHECK:       mogg.tensor.create({{%[^ ]+}}, {{%[^ ]+}})
// CHECK-SAME:    : !mogg.buffer
// CHECK:       mogg.tensor.create.slice
// CHECK-SAME:    mut = true
// CHECK-SAME:    : !mogg.buffer{{$}}
mo.graph @sliceStrideOverBuffer(%buf: !mo.buffer<[10, 64], f32>) -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = map.iter.opaque
    inputs (%buf: !mo.buffer<[10, 64], f32>, %start: !mo.tensor<[2], si64>, %stop: !mo.tensor<[2], si64>, %step: !mo.tensor<[2], si64>)
    outputs (%out: !mo.tensor<[3, 32], f32>) {
    %view = map.index.slice.stride<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> (%buf, %start, %stop, %step) : (!mo.buffer<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 32], f32>
    map.iter.elementwise.body<[3, 32]>
      index (%idx: !map.index<2>)
      capturing {%view0 = %view : !mo.tensor<[3, 32], f32>, %out0 = %out : !mo.tensor<[3, 32], f32>} {
      %val = map.load %view0[%idx] : !mo.tensor<[3, 32], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[3, 32], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// CHECK-LABEL: mo.graph @sliceStrideOverTensor
// CHECK:       mogg.tensor.create.slice
// CHECK-SAME:    mut = false
// CHECK-SAME:    : !mogg.tensor{{$}}
mo.graph @sliceStrideOverTensor(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = map.iter.opaque
    inputs (%input: !mo.tensor<[10, 64], f32>, %start: !mo.tensor<[2], si64>, %stop: !mo.tensor<[2], si64>, %step: !mo.tensor<[2], si64>)
    outputs (%out: !mo.tensor<[3, 32], f32>) {
    %view = map.index.slice.stride<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> (%input, %start, %stop, %step) : (!mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 32], f32>
    map.iter.elementwise.body<[3, 32]>
      index (%idx: !map.index<2>)
      capturing {%view0 = %view : !mo.tensor<[3, 32], f32>, %out0 = %out : !mo.tensor<[3, 32], f32>} {
      %val = map.load %view0[%idx] : !mo.tensor<[3, 32], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[3, 32], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[3, 32], f32>
}
