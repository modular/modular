// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-kernels | FileCheck %s

// An index reader, a kernel that reads its element index, must see the index
// of the element in its own output. Fused into a load through a view, it would
// see the consumer's index instead, so it stays unfused.

// CHECK-LABEL: mo.graph @indexReaderThroughSlice
// CHECK:       map.iter.elementwise<[8]>
// CHECK:       map.call.elementwise @{{.*}}.ElemwiseAddWithIndex
// CHECK:       map.iter.elementwise<[4]>
// CHECK:       map.index.slice
// CHECK-NOT:   ElemwiseAddWithIndex
// CHECK:       map.call.elementwise @{{.*}}.Add
mo.graph @indexReaderThroughSlice(%x: !mo.tensor<[8], si32>, %y: !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32> {
  %z = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %start = mo.constant <#M.dense_array<4> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<8> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %s = mo.slice(%z, %start, %stop, %step) : (
    !mo.tensor<[8], si32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[4], si32>
  %res = mo.add(%s, %y) : !mo.tensor<[4], si32>
  mo.output %res : !mo.tensor<[4], si32>
}

// The same when the index reader reads its own input through a view.
// `ViewFuser`, not `ElementwiseFuser`, fuses such a producer.

// CHECK-LABEL: mo.graph @indexReaderOverViewThroughSlice
// CHECK:       map.iter.elementwise<[8]>
// CHECK:       map.index.slice
// CHECK:       map.call.elementwise @{{.*}}.ElemwiseAddWithIndex
// CHECK:       map.iter.elementwise<[4]>
// CHECK:       map.index.slice
// CHECK-NOT:   ElemwiseAddWithIndex
// CHECK:       map.call.elementwise @{{.*}}.Add
mo.graph @indexReaderOverViewThroughSlice(%x: !mo.tensor<[16], si32>, %y: !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32> {
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %start0 = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %start4 = mo.constant <#M.dense_array<4> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<8> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %h = mo.slice(%x, %start0, %stop, %step) : (
    !mo.tensor<[16], si32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[8], si32>
  %z = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%h) : (!mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
  %s = mo.slice(%z, %start4, %stop, %step) : (
    !mo.tensor<[8], si32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[4], si32>
  %res = mo.add(%s, %y) : !mo.tensor<[4], si32>
  mo.output %res : !mo.tensor<[4], si32>
}

// When the consumer reads the index reader at its own index, the element index
// does not change, so the index reader fuses.

// CHECK-LABEL: mo.graph @indexReaderAtLoopIndex
// CHECK:       map.iter.elementwise<[8]>
// CHECK:       map.call.elementwise @{{.*}}.ElemwiseAddWithIndex
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output
mo.graph @indexReaderAtLoopIndex(%x: !mo.tensor<[8], si32>, %y: !mo.tensor<[8], si32>) -> !mo.tensor<[8], si32> {
  %z = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
  %res = mo.add(%z, %y) : !mo.tensor<[8], si32>
  mo.output %res : !mo.tensor<[8], si32>
}

// One consumer reads the index reader at its own index and one through a
// slice. The barrier keeps it in memory for the slice, so a copy in the first
// consumer would only compute it again. It fuses into neither. The tensors are
// on a GPU because a host kernel is never copied.

// CHECK-LABEL: mo.graph @indexReaderNotDuplicated
// CHECK:       map.iter.elementwise<[8]>
// CHECK:       map.call.elementwise @{{.*}}.ElemwiseAddWithIndex
// CHECK:       map.iter.elementwise<[8]>
// CHECK-NOT:   ElemwiseAddWithIndex
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.iter.elementwise<[4]>
// CHECK-NOT:   ElemwiseAddWithIndex
// CHECK:       map.call.elementwise @{{.*}}.Mul
mo.graph @indexReaderNotDuplicated(%x: !mo.tensor<[8], si32, gpu:0>, %y: !mo.tensor<[8], si32, gpu:0>, %w: !mo.tensor<[4], si32, gpu:0>) -> (!mo.tensor<[8], si32, gpu:0>, !mo.tensor<[4], si32, gpu:0>) {
  %z = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[8], si32, gpu:0>) -> !mo.tensor<[8], si32, gpu:0>
  %a = mo.add(%z, %y) : !mo.tensor<[8], si32, gpu:0>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %start = mo.constant <#M.dense_array<4> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<8> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %s = mo.slice(%z, %start, %stop, %step) : (
    !mo.tensor<[8], si32, gpu:0>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[4], si32, gpu:0>
  %b = mo.mul(%s, %w) : !mo.tensor<[4], si32, gpu:0>
  mo.output %a, %b : !mo.tensor<[8], si32, gpu:0>, !mo.tensor<[4], si32, gpu:0>
}
