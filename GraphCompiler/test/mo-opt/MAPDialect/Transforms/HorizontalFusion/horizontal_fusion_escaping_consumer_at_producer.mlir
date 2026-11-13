// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s

// An escaping consumer whose producer's result something reads in between,
// so the merged kernel can't simply take the consumer's place. If the
// consumer's other operands are all defined before the producer, it takes the
// producer's place; otherwise readers that are only metadata move past the
// consumer.

// The add feeds the cumsum, then the relu: one kernel, ahead of the cumsum,
// stores the add and the relu.
// CHECK-LABEL: mo.graph @add_cumsum_relu
// CHECK:       %[[K:.*]]:2 = map.iter.elementwise<[8]>
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.call.elementwise @{{.*}}.ReLU
// CHECK:       %[[SUM:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[K]]#0
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[SUM]], %[[K]]#1
mo.graph @add_cumsum_relu(%a: !mo.tensor<[8], f32>, %b: !mo.tensor<[8], f32>) -> (!mo.tensor<[8], f32>, !mo.tensor<[8], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[8], f32>
  %k = mo.cumsum(%x) [axis = 0, exclusive = 0, reverse = 0] : (!mo.tensor<[8], f32>) -> !mo.tensor<[8], f32>
  %y = mo.relu(%x) : !mo.tensor<[8], f32>
  mo.output %k, %y : !mo.tensor<[8], f32>, !mo.tensor<[8], f32>
}

// The consumer also reads the cumsum, defined after the producer: neither
// place holds, so the consumer stays a kernel of its own.
// CHECK-LABEL: mo.graph @add_cumsum_mul
// CHECK:       map.iter.elementwise<[8]>
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.iter.opaque
// CHECK:       map.iter.elementwise<[8]>
// CHECK:       map.call.elementwise @{{.*}}.Mul
mo.graph @add_cumsum_mul(%a: !mo.tensor<[8], f32>, %b: !mo.tensor<[8], f32>) -> (!mo.tensor<[8], f32>, !mo.tensor<[8], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[8], f32>
  %k = mo.cumsum(%x) [axis = 0, exclusive = 0, reverse = 0] : (!mo.tensor<[8], f32>) -> !mo.tensor<[8], f32>
  %y = mo.mul(%x, %k) : !mo.tensor<[8], f32>
  mo.output %k, %y : !mo.tensor<[8], f32>, !mo.tensor<[8], f32>
}

// A result returned through a `mo.rebind`, and a window of it whose stops are
// computed after it. The rebind moves past the window, and the merged kernel
// takes its place.
// CHECK-LABEL: mo.graph @add_slice_window_rebind
// CHECK:       %[[K:.*]]:2 = map.iter.elementwise<[B, S, 16]>
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.index.slice.select
// CHECK:       %[[FULL:.*]] = mo.rebind(%[[K]]#0)
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[FULL]], %[[K]]#1
mo.graph @add_slice_window_rebind<B, S>(%a: !mo.tensor<[B, S, 16], f32>, %b: !mo.tensor<[B, S, 16], f32>) -> (!mo.tensor<[B, S, 16], f32>, !mo.tensor<[B, 8, 16], f32>) {
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[B, S, 16], f32>
  %full = mo.rebind(%x) : !mo.tensor<[B, S, 16], f32> -> !mo.tensor<[B, S, 16], f32>
  %shape = mosh.param.to_value: !mosh.ape = <[B, S, 16]>
  %stops = mo.shape.to_tensor(%shape) -> !mo.tensor<[3], si64>
  %window = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[B, S, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[B, 8, 16], f32>
  mo.output %full, %window : !mo.tensor<[B, S, 16], f32>, !mo.tensor<[B, 8, 16], f32>
}

// The add is returned and read by the cumsum in between, and the relu reads
// every other row of it through a slice: the merged kernel takes the add's
// place, ahead of the cumsum, and the relu runs inside a slice select there.
// CHECK-LABEL: mo.graph @add_cumsum_slice_relu
// CHECK:       %[[K:.*]]:2 = map.iter.elementwise<[4, 8, 16]>
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.index.slice.select
// CHECK:       map.call.elementwise @{{.*}}.ReLU
// CHECK:       %[[SUM:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[K]]#0
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[SUM]], %[[K]]#0, %[[K]]#1
mo.graph @add_cumsum_slice_relu(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 8, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %k = mo.cumsum(%x) [axis = 2, exclusive = 0, reverse = 0] : (!mo.tensor<[4, 8, 16], f32>) -> !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 4, 16], f32>
  mo.output %k, %x, %y : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>
}

// The consumer also reads a cumsum of `%a` defined after the add, so the merged
// kernel can't take the add's place; the rebind reading the add in between is
// metadata, but the cumsum of it sits before the consumer, so the rebind can't
// move past it either. The mul stays a kernel of its own.
// CHECK-LABEL: mo.graph @add_rebind_cumsum_mul
// CHECK:       map.iter.elementwise<[8]>
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       mo.rebind
// CHECK:       map.iter.opaque
// CHECK:       map.iter.opaque
// CHECK:       map.iter.elementwise<[8]>
// CHECK:       map.call.elementwise @{{.*}}.Mul
mo.graph @add_rebind_cumsum_mul(%a: !mo.tensor<[8], f32>, %b: !mo.tensor<[8], f32>) -> (!mo.tensor<[8], f32>, !mo.tensor<[8], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[8], f32>
  %r = mo.rebind(%x) : !mo.tensor<[8], f32> -> !mo.tensor<[8], f32>
  %k = mo.cumsum(%r) [axis = 0, exclusive = 0, reverse = 0] : (!mo.tensor<[8], f32>) -> !mo.tensor<[8], f32>
  %c = mo.cumsum(%a) [axis = 0, exclusive = 0, reverse = 0] : (!mo.tensor<[8], f32>) -> !mo.tensor<[8], f32>
  %y = mo.mul(%x, %c) : !mo.tensor<[8], f32>
  mo.output %k, %y : !mo.tensor<[8], f32>, !mo.tensor<[8], f32>
}
