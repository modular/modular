// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trip map.index.slice and the map.index.slice.spec attribute it takes.
// Covers a fully-static shape/bounds case and a case with symbolic MOSH dims
// (in/out shapes) and dynamic (`?`) start/stop bounds.
// Value names are matched with capture variables so changes in the printer's
// SSA numbering don't break the test.

#staticSpec = #map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>
#dynamicSpec = #map.index.slice.spec<[N, 64] -> [M, 32], starts = [?, 32], stops = [?, 64], steps = [1, 1]>

module {

// Static shape and bounds: verify map.index.slice round-trips unchanged.
// CHECK-LABEL: mo.graph @sliceStatic
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.elementwise<[3, 32]>
// CHECK-NEXT:    inputs  ([[INPUT:%[^ :,]+]]: !mo.tensor<[10, 64], f32>, [[STARTS:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STOPS:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STEPS:%[^ :,]+]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:    outputs ([[OUTARG:%[^ :,]+]]: !mo.tensor<[3, 32], f32>)
// CHECK-NEXT:    index   ([[IDX:%[^ :,]+]]: !map.index<2>) -> !mo.tensor<[3, 32], f32> {
// CHECK-NEXT:   [[IDX_:%[^ ]+]] = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> ([[IDX]], [[STARTS]], [[STOPS]], [[STEPS]]) : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
// CHECK-NEXT:   [[VAL:%[^ ]+]] = map.load [[INPUT]]{{\[}}[[IDX_]]] : !mo.tensor<[10, 64], f32>
// CHECK-NEXT:   map.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[3, 32], f32>
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT]] : !mo.tensor<[3, 32], f32>
mo.graph @sliceStatic(%input: !mo.tensor<[10, 64], f32>,
                      %starts: !mo.tensor<[2], si64>,
                      %stops: !mo.tensor<[2], si64>,
                      %steps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 32], f32> {
  %output = map.iter.elementwise<[3, 32]>
      inputs  (%input: !mo.tensor<[10, 64], f32>,
               %starts: !mo.tensor<[2], si64>,
               %stops: !mo.tensor<[2], si64>,
               %steps: !mo.tensor<[2], si64>)
      outputs (%out: !mo.tensor<[3, 32], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[3, 32], f32>
  {
    %idx_ = map.index.slice<#staticSpec>(%idx, %starts, %stops, %steps)
        : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
    %val = map.load %input[%idx_] : !mo.tensor<[10, 64], f32>
    map.store %out[%idx] = %val : !mo.tensor<[3, 32], f32>
  }
  mo.output %output : !mo.tensor<[3, 32], f32>
}

// Symbolic in/out shape dims (N, M) and dynamic (`?`) starts/stops, matching
// the general case where the mo.slice bounds aren't compile-time constants.
// CHECK-LABEL: mo.graph @sliceDynamic
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.elementwise<[M, 32]>
// CHECK-NEXT:    inputs  ([[INPUT2:%[^ :,]+]]: !mo.tensor<[N, 64], f32>, [[STARTS2:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STOPS2:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STEPS2:%[^ :,]+]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:    outputs ([[OUTARG2:%[^ :,]+]]: !mo.tensor<[M, 32], f32>)
// CHECK-NEXT:    index   ([[IDX2:%[^ :,]+]]: !map.index<2>) -> !mo.tensor<[M, 32], f32> {
// CHECK-NEXT:   [[IDX2_:%[^ ]+]] = map.index.slice<#map.index.slice.spec<[N, 64] -> [M, 32], starts = [?, 32], stops = [?, 64], steps = [1, 1]>> ([[IDX2]], [[STARTS2]], [[STOPS2]], [[STEPS2]]) : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
mo.graph @sliceDynamic<N, M>(
    %input: !mo.tensor<[N, 64], f32>,
    %starts: !mo.tensor<[2], si64>,
    %stops: !mo.tensor<[2], si64>,
    %steps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[M, 32], f32> {
  %output = map.iter.elementwise<[M, 32]>
      inputs  (%input: !mo.tensor<[N, 64], f32>,
               %starts: !mo.tensor<[2], si64>,
               %stops: !mo.tensor<[2], si64>,
               %steps: !mo.tensor<[2], si64>)
      outputs (%out: !mo.tensor<[M, 32], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[M, 32], f32>
  {
    %idx_ = map.index.slice<#dynamicSpec>(%idx, %starts, %stops, %steps)
        : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
    %val = map.load %input[%idx_] : !mo.tensor<[N, 64], f32>
    map.store %out[%idx] = %val : !mo.tensor<[M, 32], f32>
  }
  mo.output %output : !mo.tensor<[M, 32], f32>
}

}
