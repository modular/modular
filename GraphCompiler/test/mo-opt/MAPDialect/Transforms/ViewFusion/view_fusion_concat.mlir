// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s

// A concat lowers to a view foreach around a map.index.concat.select, so it
// fuses like a view: into a reader that loads it at the reader's own index,
// with each producer spliced into the region of the input it computes. A
// select keeps indexing along the loop's vector, so a reader through a
// transform keeps the concat a kernel of its own, and the select is not
// lowered into prologue functors yet, so an opaque consumer loads it.

// Both producers land in their regions, and the reader's relu follows the
// select: one kernel.

// CHECK-LABEL: mo.graph @concatFused
// CHECK:       map.iter.elementwise<[4, 24]>
// CHECK:       index   (%[[IDX:.*]]: !map.index<2>)
// CHECK-NEXT:  %[[V:.*]] = map.index.concat.select<{{.*}}> (%[[IDX]]) : !map.index<2> -> f32 (%[[I0:.*]]: !map.index<2>) {
// CHECK-NEXT:    %[[A:.*]] = map.load %{{.*}}[%[[I0]]] : !mo.tensor<[4, 8], f32>
// CHECK-NEXT:    %[[NEG:.*]] = map.call.elementwise @{{.*}}.Negative(%[[I0]], %[[A]])
// CHECK-NEXT:    map.yield %[[NEG]] : f32
// CHECK-NEXT:  }, (%[[I1:.*]]: !map.index<2>) {
// CHECK-NEXT:    %[[B:.*]] = map.load %{{.*}}[%[[I1]]] : !mo.tensor<[4, 16], f32>
// CHECK-NEXT:    %[[EXP:.*]] = map.call.elementwise @{{.*}}.Exp(%[[I1]], %[[B]])
// CHECK-NEXT:    map.yield %[[EXP]] : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[RELU:.*]] = map.call.elementwise @{{.*}}.ReLU(%[[IDX]], %[[V]])
// CHECK-NEXT:  map.store %{{.*}}[%[[IDX]]] = %[[RELU]]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output
mo.graph @concatFused(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32> {
  %x = mo.negative(%a) : !mo.tensor<[4, 8], f32>
  %y = mo.exp(%b) : !mo.tensor<[4, 16], f32>
  %c = mo.concat[1] (%x, %y) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32>
  %r = mo.relu(%c) : !mo.tensor<[4, 24], f32>
  mo.output %r : !mo.tensor<[4, 24], f32>
}

// The reader loads the concat through a slice of its index, where the select
// could not follow the vector: the concat stays a kernel of its own.

// CHECK-LABEL: mo.graph @concatReadThroughSlice
// CHECK:       %[[CONCAT:.*]] = map.iter.elementwise<[4, 24]>
// CHECK:       map.index.concat.select
// CHECK:       map.iter.elementwise<[4, 10]>
// CHECK:       %[[SIDX:.*]] = map.index.slice
// CHECK-NEXT:  map.load %[[CONCAT]][%[[SIDX]]]
mo.graph @concatReadThroughSlice(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 10], f32> {
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32>
  %start = mo.constant <#M.dense_array<0, 6> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %s = mo.slice(%c, %start, %stop, %step) : (!mo.tensor<[4, 24], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 10], f32>
  %r = mo.relu(%s) : !mo.tensor<[4, 10], f32>
  mo.output %r : !mo.tensor<[4, 10], f32>
}

// The gather's data prologue only loads the concat, which stays a foreach.

// CHECK-LABEL: mo.graph @concatIntoGather
// CHECK:       %[[CONCAT:.*]] = map.iter.elementwise<[3, 8]>
// CHECK:       map.index.concat.select
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%[[CONCAT]]: !mo.tensor<[3, 8], f32>
// CHECK:       map.iter.prologue (%[[PI:.*]] : !map.index<2>) capturing {%[[CAP:.*]] = %[[CONCAT]] : !mo.tensor<[3, 8], f32>} {
// CHECK-NEXT:    map.load %[[CAP]][%[[PI]]]
// CHECK-NEXT:    map.yield
// CHECK:       kernelName = "mo.gather"
mo.graph @concatIntoGather(%a: !mo.tensor<[3, 4], f32>, %b: !mo.tensor<[3, 4], f32>, %indices: !mo.tensor<[2], si32>) -> !mo.tensor<[2, 8], f32> {
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[3, 4], f32>, !mo.tensor<[3, 4], f32>) -> !mo.tensor<[3, 8], f32>
  %g = mo.gather(%c, %indices) [axis = 0] : (!mo.tensor<[3, 8], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 8], f32>
  mo.output %g : !mo.tensor<[2, 8], f32>
}
