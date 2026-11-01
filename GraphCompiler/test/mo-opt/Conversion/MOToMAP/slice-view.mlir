// RUN: mo-opt %s --mogg-import-packages --mo-to-map | FileCheck %s

// Verify that mo.slice lowers to a standalone (unfused) map.iter.elementwise over
// the output shape, with a map.index.slice op computing the input index --
// not map.iter.opaque. Each of starts/stops/steps independently shows up as a
// literal in the map.index.slice.spec when it's a compile-time constant (a
// mo.constant operand), and as `?` (ShapedType::kDynamic) when it isn't.

// All of starts/stops/steps are mo.constant: every bound is a literal.

// CHECK-LABEL: mo.graph @sliceStaticBounds
// CHECK-NOT:     map.iter.opaque
// CHECK:         [[OUT:%.*]] = map.iter.elementwise<[3, 32]>
// CHECK:         inputs  ([[INPUT:%.*]]: !mo.tensor<[10, 64], f32>, [[STARTS:%.*]]: !mo.tensor<[2], si64>, [[STOPS:%.*]]: !mo.tensor<[2], si64>, [[STEPS:%.*]]: !mo.tensor<[2], si64>)
// CHECK:         outputs ([[OUTARG:%.*]]: !mo.tensor<[3, 32], f32>)
// CHECK:         index   ([[IDX:%.*]]: !map.index<2>) -> !mo.tensor<[3, 32], f32> {
// CHECK:         [[IDX_:%.*]] = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> ([[IDX]], [[STARTS]], [[STOPS]], [[STEPS]])
// CHECK:         [[VAL:%.*]] = map.load [[INPUT]]{{\[}}[[IDX_]]] : !mo.tensor<[10, 64], f32>
// CHECK:         map.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[3, 32], f32>
mo.graph @sliceStaticBounds(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// All of starts/stops/steps come from graph arguments (not mo.constant):
// every bound is dynamic (`?`).

// CHECK-LABEL: mo.graph @sliceDynamicBounds
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.elementwise<[3, 32]>
// CHECK:         map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [?, ?], stops = [?, ?], steps = [?, ?]>>
mo.graph @sliceDynamicBounds(
    %input: !mo.tensor<[10, 64], f32>,
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>,
    %step: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 32], f32> {
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// starts/stops are mo.constant but step is a graph argument: only the steps
// field of the spec is dynamic, proving each of starts/stops/steps is
// tracked independently (a non-constant step doesn't force starts/stops to
// also be treated as dynamic).

// CHECK-LABEL: mo.graph @sliceNonConstantStep
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.elementwise<[3, 32]>
// CHECK:         map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [?, ?]>>
mo.graph @sliceNonConstantStep(
    %input: !mo.tensor<[10, 64], f32>,
    %step: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// A negative step walks the axis backwards. MO clamps such a slice to
// [-1, dim-1] rather than the forward [0, dim], so the same literal
// normalizes differently depending on the step's sign: here stop=-7 becomes
// 3, and the walk is 9, 7, 5.

// CHECK-LABEL: mo.graph @sliceReverseStep
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.elementwise<[3, 32]>
// CHECK:         map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [9, 32], stops = [3, 64], steps = [-2, 1]>>
mo.graph @sliceReverseStep(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant <#M.dense_array<-1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<-7, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<-2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// Reversing a whole axis needs a stop of -1, one before element 0. Only the
// reverse clamp can produce it: an out-of-range stop lands there instead of
// at 0, which is what the forward clamp would give.

// CHECK-LABEL: mo.graph @sliceReverseWholeAxis
// CHECK:         map.index.slice<#map.index.slice.spec<[10] -> [10], starts = [9], stops = [-1], steps = [-1]>>
mo.graph @sliceReverseWholeAxis(%input: !mo.tensor<[10], f32>) -> !mo.tensor<[10], f32> {
  %start = mo.constant <#M.dense_array<-1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<-20> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %step = mo.constant <#M.dense_array<-1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[10], f32>
  mo.output %res : !mo.tensor<[10], f32>
}
