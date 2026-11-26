// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// A kernel calls its epilogue with vectors along its own output's innermost
// dim, never crossing one of its rows. Once
// a reshape moves onto the epilogue, the consumer's other loads address their
// tensors in the reshaped shape and step by its innermost stride. Where such a
// vector can cross a row of the reshaped shape, the load is right only on a
// tensor laid out row after row across that boundary; otherwise the reshape
// isn't fused and stays its own kernel.

// [3, 8] -> [4, 6]: an 8-wide vector crosses rows of 6, and a broadcast row
// repeats instead of continuing. Not fused.
// CHECK-LABEL: mo.graph @innermost_split_broadcast
// CHECK:       map.iter.opaque
// CHECK-NOT:   map.index.reshape
// CHECK:       map.call.opaque
// CHECK:       map.iter.elementwise<[4, 6]>
// CHECK:       map.index.reshape<#map.index.reshape.spec<[3, 8] -> [4, 6]>>
mo.graph @innermost_split_broadcast(
  %x: !mo.tensor<[3, 8], f32>,
  %b: !mo.tensor<[6], f32>
) -> !mo.tensor<[4, 6], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[3, 8], f32> -> !mo.tensor<[4, 6], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[6], f32> -> !mo.tensor<[4, 6], f32>
  %out = mo.add(%r, %bb) : !mo.tensor<[4, 6], f32>
  mo.output %out : !mo.tensor<[4, 6], f32>
}

// The same split read from a row-major tensor: crossing a row just continues
// into the next one. Fused.
// CHECK-LABEL: mo.graph @innermost_split_row_major
// CHECK:       map.iter.epilogue
// CHECK:       map.index.reshape<#map.index.reshape.spec<[4, 6] -> [3, 8]>>
// CHECK:       map.call.opaque
// CHECK-NOT:   map.iter.elementwise
mo.graph @innermost_split_row_major(
  %x: !mo.tensor<[3, 8], f32>,
  %y: !mo.tensor<[4, 6], f32>
) -> !mo.tensor<[4, 6], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[3, 8], f32> -> !mo.tensor<[4, 6], f32>
  %out = mo.add(%r, %y) : !mo.tensor<[4, 6], f32>
  mo.output %out : !mo.tensor<[4, 6], f32>
}

// [3, 8] -> [3, 8, 1]: the load steps by the size-1 dim's stride, 1 in the
// slice, while its rows are 4 apart. Not fused.
// CHECK-LABEL: mo.graph @trailing_one_slice
// CHECK:       map.iter.opaque
// CHECK-NOT:   map.index.reshape
// CHECK:       map.call.opaque
// CHECK:       map.iter.elementwise<[3, 8, 1]>
// CHECK:       map.index.reshape<#map.index.reshape.spec<[3, 8] -> [3, 8, 1]>>
mo.graph @trailing_one_slice(
  %x: !mo.tensor<[3, 8], f32>,
  %y: !mo.tensor<[3, 8, 4], f32>
) -> !mo.tensor<[3, 8, 1], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[3, 8], f32> -> !mo.tensor<[3, 8, 1], f32>
  %start = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stop = mo.constant <#M.dense_array<3, 8, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %step = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %s = mo.slice(%y, %start, %stop, %step) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[3, 8, 1], f32>
  %out = mo.add(%r, %s) : !mo.tensor<[3, 8, 1], f32>
  mo.output %out : !mo.tensor<[3, 8, 1], f32>
}

// The same reshape with a dense tensor: its size-1 dim's stride is its row's.
// Fused.
// CHECK-LABEL: mo.graph @trailing_one_dense
// CHECK:       map.iter.epilogue
// CHECK:       map.index.reshape<#map.index.reshape.spec<[3, 8, 1] -> [3, 8]>>
// CHECK:       map.call.opaque
// CHECK-NOT:   map.iter.elementwise
mo.graph @trailing_one_dense(
  %x: !mo.tensor<[3, 8], f32>,
  %y: !mo.tensor<[3, 8, 1], f32>
) -> !mo.tensor<[3, 8, 1], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[3, 8], f32> -> !mo.tensor<[3, 8, 1], f32>
  %out = mo.add(%r, %y) : !mo.tensor<[3, 8, 1], f32>
  mo.output %out : !mo.tensor<[3, 8, 1], f32>
}

// A broadcast of size-1 dims: both strides are 0, every lane reads the same
// element. Fused.
// CHECK-LABEL: mo.graph @trailing_one_broadcast
// CHECK:       map.iter.epilogue
// CHECK:       map.index.reshape<#map.index.reshape.spec<[3, 8, 1] -> [3, 8]>>
// CHECK:       map.call.opaque
// CHECK-NOT:   map.iter.elementwise
mo.graph @trailing_one_broadcast(
  %x: !mo.tensor<[3, 8], f32>,
  %b: !mo.tensor<[3, 1, 1], f32>
) -> !mo.tensor<[3, 8, 1], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[3, 8], f32> -> !mo.tensor<[3, 8, 1], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[3, 1, 1], f32> -> !mo.tensor<[3, 8, 1], f32>
  %out = mo.add(%r, %bb) : !mo.tensor<[3, 8, 1], f32>
  mo.output %out : !mo.tensor<[3, 8, 1], f32>
}

// [6, 8] -> [2, 3, 8] splits an outer dim: the innermost extent is still the
// kernel's, so no vector leaves a row, even of a broadcast. Fused.
// CHECK-LABEL: mo.graph @outer_split_broadcast
// CHECK:       map.iter.epilogue
// CHECK:       map.index.reshape<#map.index.reshape.spec<[2, 3, 8] -> [6, 8]>>
// CHECK:       map.call.opaque
// CHECK-NOT:   map.iter.elementwise
mo.graph @outer_split_broadcast(
  %x: !mo.tensor<[6, 8], f32>,
  %b: !mo.tensor<[2, 1, 8], f32>
) -> !mo.tensor<[2, 3, 8], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[6, 8], f32>) -> !mo.tensor<[6, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[6, 8], f32> -> !mo.tensor<[2, 3, 8], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[2, 1, 8], f32> -> !mo.tensor<[2, 3, 8], f32>
  %out = mo.add(%r, %bb) : !mo.tensor<[2, 3, 8], f32>
  mo.output %out : !mo.tensor<[2, 3, 8], f32>
}

// [3, 8] -> [24] merges: a vector never leaves a row of 24. Fused, even with
// a slice.
// CHECK-LABEL: mo.graph @merge_slice
// CHECK:       map.iter.epilogue
// CHECK:       map.index.reshape<#map.index.reshape.spec<[24] -> [3, 8]>>
// CHECK:       map.call.opaque
// CHECK-NOT:   map.iter.elementwise
mo.graph @merge_slice(
  %x: !mo.tensor<[3, 8], f32>,
  %y: !mo.tensor<[32], f32>
) -> !mo.tensor<[24], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[3, 8], f32> -> !mo.tensor<[24], f32>
  %start = mo.constant <#M.dense_array<4> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<28> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %s = mo.slice(%y, %start, %stop, %step) : (!mo.tensor<[32], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[24], f32>
  %out = mo.add(%r, %s) : !mo.tensor<[24], f32>
  mo.output %out : !mo.tensor<[24], f32>
}

// A symbolic innermost extent D that the reshape keeps innermost: the block
// below each dim has D as a factor, so no vector crosses. Fused.
// CHECK-LABEL: mo.graph @symbolic_kept_innermost
// CHECK:       map.iter.epilogue
// CHECK:       map.index.reshape<#map.index.reshape.spec<[1, N, D] -> [N, D]>>
// CHECK:       map.call.opaque
// CHECK-NOT:   map.iter.elementwise
mo.graph @symbolic_kept_innermost<N, D>(
  %x: !mo.tensor<[N, D], f32>,
  %b: !mo.tensor<[D], f32>
) -> !mo.tensor<[1, N, D], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[N, D], f32>) -> !mo.tensor<[N, D], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[N, D], f32> -> !mo.tensor<[1, N, D], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[D], f32> -> !mo.tensor<[1, N, D], f32>
  %out = mo.add(%r, %bb) : !mo.tensor<[1, N, D], f32>
  mo.output %out : !mo.tensor<[1, N, D], f32>
}

// A symbolic innermost extent D moved off the innermost dim: vectors cross the
// size-1 rows, and a slice of [N, D, 2] keeps rows 2 apart. Not fused.
// CHECK-LABEL: mo.graph @symbolic_trailing_one_broadcast
// CHECK:       map.iter.opaque
// CHECK-NOT:   map.index.reshape
// CHECK:       map.call.opaque
// CHECK:       map.iter.elementwise<[N, D, 1]>
mo.graph @symbolic_trailing_one_broadcast<N, D>(
  %x: !mo.tensor<[N, D], f32>,
  %b: !mo.tensor<[N, D, 2], f32>
) -> !mo.tensor<[N, D, 1], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[N, D], f32>) -> !mo.tensor<[N, D], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[N, D], f32> -> !mo.tensor<[N, D, 1], f32>
  %start = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stop = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %step = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %s = mo.slice(%b, %start, %stop, %step) : (!mo.tensor<[N, D, 2], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[N, D, 1], f32>
  %out = mo.add(%r, %s) : !mo.tensor<[N, D, 1], f32>
  mo.output %out : !mo.tensor<[N, D, 1], f32>
}

// [B, N, D] -> [B, N * D] merges the kernel's two inner dims: the boundary
// of the merged row is one of the kernel's, which no vector crosses, even
// with a symbolic D. Fused, even with a broadcast row.
// CHECK-LABEL: mo.graph @symbolic_merge_broadcast
// CHECK:       map.iter.epilogue
// CHECK:       map.index.reshape<#map.index.reshape.spec<[B, mul_no_wrap(D, N)] -> [B, N, D]>>
// CHECK:       map.call.opaque
mo.graph @symbolic_merge_broadcast<B, N, D>(
  %x: !mo.tensor<[B, N, D], f32>,
  %b: !mo.tensor<[N, D], f32>
) -> !mo.tensor<[B, N, D], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[B, N, D], f32>) -> !mo.tensor<[B, N, D], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[B, N, D], f32> -> !mo.tensor<[B, mul_no_wrap(N, D)], f32>
  %b1 = mo.static.reshape(%b) : !mo.tensor<[N, D], f32> -> !mo.tensor<[1, mul_no_wrap(N, D)], f32>
  %bb = mo.static.broadcast_to(%b1) : !mo.tensor<[1, mul_no_wrap(N, D)], f32> -> !mo.tensor<[B, mul_no_wrap(N, D)], f32>
  %sum = mo.add(%r, %bb) : !mo.tensor<[B, mul_no_wrap(N, D)], f32>
  %out = mo.static.reshape(%sum) : !mo.tensor<[B, mul_no_wrap(N, D)], f32> -> !mo.tensor<[B, N, D], f32>
  mo.output %out : !mo.tensor<[B, N, D], f32>
}

// A kernel that reads the element's index derives each lane's position from
// the first lane's, as if the vector stayed in one row of the reshaped shape;
// the index works like a load of a tensor broadcast over the rows. Behind a
// reshape, it isn't fused, whether or not a vector can cross a row.

// [3, 8] -> [4, 6], where an 8-wide vector crosses rows of 6. Not fused.
// CHECK-LABEL: mo.graph @innermost_split_index
// CHECK:       map.iter.opaque
// CHECK-NOT:   map.index.reshape
// CHECK:       map.call.opaque
// CHECK:       map.iter.elementwise<[4, 6]>
// CHECK:       map.index.reshape<#map.index.reshape.spec<[3, 8] -> [4, 6]>>
// CHECK:       map.call.elementwise @{{.*}}ElemwiseAddWithIndex
mo.graph @innermost_split_index(
  %x: !mo.tensor<[3, 8], f32>
) -> !mo.tensor<[4, 6], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[3, 8], f32> -> !mo.tensor<[4, 6], f32>
  %out = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%r) : (!mo.tensor<[4, 6], f32>) -> !mo.tensor<[4, 6], f32>
  mo.output %out : !mo.tensor<[4, 6], f32>
}

// [6, 8] -> [2, 3, 8] only splits the rows, so no vector crosses one; still
// not fused, to keep the rule simple.
// CHECK-LABEL: mo.graph @row_split_index
// CHECK:       map.iter.opaque
// CHECK-NOT:   map.index.reshape
// CHECK:       map.call.opaque
// CHECK:       map.iter.elementwise<[2, 3, 8]>
// CHECK:       map.call.elementwise @{{.*}}ElemwiseAddWithIndex
mo.graph @row_split_index(
  %x: !mo.tensor<[6, 8], f32>
) -> !mo.tensor<[2, 3, 8], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[6, 8], f32>) -> !mo.tensor<[6, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[6, 8], f32> -> !mo.tensor<[2, 3, 8], f32>
  %out = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%r) : (!mo.tensor<[2, 3, 8], f32>) -> !mo.tensor<[2, 3, 8], f32>
  mo.output %out : !mo.tensor<[2, 3, 8], f32>
}

// Without a reshape, the kernel calls the epilogue at its own index, so an
// index-reading kernel is fused as before.
// CHECK-LABEL: mo.graph @no_reshape_index
// CHECK:       map.iter.epilogue
// CHECK:       map.call.elementwise @{{.*}}ElemwiseAddWithIndex
// CHECK:       map.call.opaque
// CHECK-NOT:   map.iter.elementwise
mo.graph @no_reshape_index(
  %x: !mo.tensor<[3, 8], f32>
) -> !mo.tensor<[3, 8], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %out = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%c) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  mo.output %out : !mo.tensor<[3, 8], f32>
}

// A kernel picks its own vector width, which need not divide its innermost
// extent: a CPU matmul runs 16 lanes on rows of 24, then a scalar tail. The one
// thing its vectors never do is cross a row of its output, so only a reshaped
// row that is a whole number of kernel rows is safe for a non-dense tensor.

// [4, 24] -> [12, 8]: rows of 8 split a kernel row, and the largest power of
// two dividing 24 (8) says nothing about the width. Not fused.
// CHECK-LABEL: mo.graph @row_not_multiple_of_kernel_row
// CHECK:       map.iter.opaque
// CHECK-NOT:   map.index.reshape
// CHECK:       map.call.opaque
// CHECK:       map.iter.elementwise<[12, 8]>
mo.graph @row_not_multiple_of_kernel_row(
  %x: !mo.tensor<[4, 24], f32>,
  %b: !mo.tensor<[8], f32>
) -> !mo.tensor<[12, 8], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[4, 24], f32>) -> !mo.tensor<[4, 24], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[4, 24], f32> -> !mo.tensor<[12, 8], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[8], f32> -> !mo.tensor<[12, 8], f32>
  %out = mo.add(%r, %bb) : !mo.tensor<[12, 8], f32>
  mo.output %out : !mo.tensor<[12, 8], f32>
}

// [4, 24] -> [2, 48]: rows of 48 are two kernel rows. Fused.
// CHECK-LABEL: mo.graph @row_multiple_of_kernel_row
// CHECK:       map.iter.epilogue
// CHECK:       map.index.reshape<#map.index.reshape.spec<[2, 48] -> [4, 24]>>
// CHECK:       map.call.opaque
// CHECK-NOT:   map.iter.elementwise
mo.graph @row_multiple_of_kernel_row(
  %x: !mo.tensor<[4, 24], f32>,
  %b: !mo.tensor<[48], f32>
) -> !mo.tensor<[2, 48], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[4, 24], f32>) -> !mo.tensor<[4, 24], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[4, 24], f32> -> !mo.tensor<[2, 48], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[48], f32> -> !mo.tensor<[2, 48], f32>
  %out = mo.add(%r, %bb) : !mo.tensor<[2, 48], f32>
  mo.output %out : !mo.tensor<[2, 48], f32>
}

// A vector crossing a row also wraps every dim below it, size-1 dims included:
// [3, 8] -> [6, 1, 4] reads the bias as [6, 4] does. Not fused.
// CHECK-LABEL: mo.graph @size1_dim_broadcast
// CHECK:       map.iter.opaque
// CHECK-NOT:   map.index.reshape
// CHECK:       map.call.opaque
// CHECK:       map.iter.elementwise<[6, 1, 4]>
mo.graph @size1_dim_broadcast(
  %x: !mo.tensor<[3, 8], f32>,
  %b: !mo.tensor<[4], f32>
) -> !mo.tensor<[6, 1, 4], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[3, 8], f32> -> !mo.tensor<[6, 1, 4], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[4], f32> -> !mo.tensor<[6, 1, 4], f32>
  %out = mo.add(%r, %bb) : !mo.tensor<[6, 1, 4], f32>
  mo.output %out : !mo.tensor<[6, 1, 4], f32>
}

// A load through two index transforms (a slice, then a broadcast): its layout
// is built from the tensor outwards. Not fused, since the broadcast's rows of 6
// split the kernel's rows of 8.
// CHECK-LABEL: mo.graph @slice_then_broadcast
// CHECK:       map.iter.opaque
// CHECK-NOT:   map.index.reshape
// CHECK:       map.call.opaque
// CHECK:       map.iter.elementwise<[4, 6]>
mo.graph @slice_then_broadcast(
  %x: !mo.tensor<[3, 8], f32>,
  %y: !mo.tensor<[10], f32>
) -> !mo.tensor<[4, 6], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[3, 8], f32> -> !mo.tensor<[4, 6], f32>
  %start = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<8> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %s = mo.slice(%y, %start, %stop, %step) : (!mo.tensor<[10], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[6], f32>
  %bb = mo.static.broadcast_to(%s) : !mo.tensor<[6], f32> -> !mo.tensor<[4, 6], f32>
  %out = mo.add(%r, %bb) : !mo.tensor<[4, 6], f32>
  mo.output %out : !mo.tensor<[4, 6], f32>
}

// A second reshape fusion is checked against the kernel's own shape, [2, 16],
// not the first reshape's [4, 8]: the kernel's vectors still run along rows
// of 16. Rows of 8 split them: the bias is not fused.
// CHECK-LABEL: mo.graph @chained_reshapes_broadcast
// CHECK:       map.iter.epilogue
// CHECK:       map.index.reshape<#map.index.reshape.spec<[4, 8] -> [2, 16]>>
// CHECK:       map.call.opaque
// CHECK:       map.iter.elementwise<[2, 2, 8]>
mo.graph @chained_reshapes_broadcast(
  %x: !mo.tensor<[2, 16], f32>,
  %y: !mo.tensor<[4, 8], f32>,
  %b: !mo.tensor<[8], f32>
) -> !mo.tensor<[2, 2, 8], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[2, 16], f32>) -> !mo.tensor<[2, 16], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[2, 16], f32> -> !mo.tensor<[4, 8], f32>
  %a = mo.add(%r, %y) : !mo.tensor<[4, 8], f32>
  %r2 = mo.static.reshape(%a) : !mo.tensor<[4, 8], f32> -> !mo.tensor<[2, 2, 8], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[8], f32> -> !mo.tensor<[2, 2, 8], f32>
  %out = mo.add(%r2, %bb) : !mo.tensor<[2, 2, 8], f32>
  mo.output %out : !mo.tensor<[2, 2, 8], f32>
}

// The same chain with a dense tensor: both fuse, and the second reshape reads
// the first's result, reshape(reshape(idx)).
// CHECK-LABEL: mo.graph @chained_reshapes_dense
// CHECK:       map.iter.epilogue (%{{.*}}, %[[IDX:.*]] : !map.index<2>)
// CHECK:       %[[R1:.*]] = map.index.reshape<#map.index.reshape.spec<[4, 8] -> [2, 16]>> (%[[IDX]])
// CHECK:       map.index.reshape<#map.index.reshape.spec<[2, 2, 8] -> [4, 8]>> (%[[R1]])
// CHECK:       map.call.opaque
// CHECK-NOT:   map.iter.elementwise
mo.graph @chained_reshapes_dense(
  %x: !mo.tensor<[2, 16], f32>,
  %y: !mo.tensor<[4, 8], f32>,
  %z: !mo.tensor<[2, 2, 8], f32>
) -> !mo.tensor<[2, 2, 8], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[2, 16], f32>) -> !mo.tensor<[2, 16], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[2, 16], f32> -> !mo.tensor<[4, 8], f32>
  %a = mo.add(%r, %y) : !mo.tensor<[4, 8], f32>
  %r2 = mo.static.reshape(%a) : !mo.tensor<[4, 8], f32> -> !mo.tensor<[2, 2, 8], f32>
  %out = mo.add(%r2, %z) : !mo.tensor<[2, 2, 8], f32>
  mo.output %out : !mo.tensor<[2, 2, 8], f32>
}

// A transposed operand on the crossed rows: [3, 8] -> [4, 6] crosses rows of
// 6, and the transpose of [6, 4] doesn't lay its rows of 6 out one after the
// other. Not fused.
// CHECK-LABEL: mo.graph @innermost_split_transpose
// CHECK:       map.iter.opaque
// CHECK-NOT:   map.index.reshape
// CHECK:       map.call.opaque
// CHECK:       map.iter.elementwise<[4, 6]>
mo.graph @innermost_split_transpose(
  %x: !mo.tensor<[3, 8], f32>,
  %t: !mo.tensor<[6, 4], f32>
) -> !mo.tensor<[4, 6], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[3, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[3, 8], f32> -> !mo.tensor<[4, 6], f32>
  %tt = mo.transpose(%t, %perm) : (!mo.tensor<[6, 4], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 6], f32>
  %out = mo.add(%r, %tt) : !mo.tensor<[4, 6], f32>
  mo.output %out : !mo.tensor<[4, 6], f32>
}

// The same broadcast with no reshape: the consumer runs in the kernel's own
// shape, so the check doesn't apply and the add is fused.
// CHECK-LABEL: mo.graph @no_reshape_broadcast
// CHECK:       map.iter.epilogue
// CHECK:       map.call.opaque
// CHECK-NOT:   map.iter.elementwise
mo.graph @no_reshape_broadcast(
  %x: !mo.tensor<[4, 6], f32>,
  %b: !mo.tensor<[6], f32>
) -> !mo.tensor<[4, 6], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[4, 6], f32>) -> !mo.tensor<[4, 6], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[6], f32> -> !mo.tensor<[4, 6], f32>
  %out = mo.add(%c, %bb) : !mo.tensor<[4, 6], f32>
  mo.output %out : !mo.tensor<[4, 6], f32>
}
