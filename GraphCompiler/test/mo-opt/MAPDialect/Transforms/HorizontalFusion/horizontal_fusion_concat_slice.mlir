// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-horizontal | FileCheck %s

// A concat whose result escapes and is also sliced: the slice consumer reads
// the concat through `map.index.slice`, so the escaping-reader merge inverts
// that slice with a `map.index.slice.select`. The concat's epilogue then stores
// every element to the concat and, in the select's body, the ones the slice
// selects at the slice's own index -- one kernel, two outputs. The slice's
// bounds tensors stop being kernel inputs: the select carries them statically.

// The KV-cache window update: keep the last 8 rows of a cache extended along a
// dynamic axis. The negative start stays as written, for the select to resolve
// against the concat's runtime extent.
// CHECK-LABEL: mo.graph @concat_slice_window
// CHECK:       %[[OUT:.*]]:2 = map.iter.opaque
// CHECK-NEXT:  inputs (%{{.*}}: !mo.tensor<[B, 8, 16], f32>, %{{.*}}: !mo.tensor<[B, S, 16], f32>)
// CHECK-NEXT:  outputs (%[[O0:[a-z0-9]+]] : !mo.tensor<[B, T, 16], f32>, %[[O1:[a-z0-9]+]] : !mo.tensor<[B, 8, 16], f32>)
// CHECK:       %[[EPI:[a-z0-9]+]] = map.iter.epilogue (%[[VAL:[a-z0-9]+]] : f32, %[[IDX:[a-z0-9]+]] : !map.index<3>) capturing {%[[CAP0:[a-z0-9]+]] = %[[O0]] : !mo.tensor<[B, T, 16], f32>, %[[CAP1:[a-z0-9]+]] = %[[O1]] : !mo.tensor<[B, 8, 16], f32>} {
// CHECK-NEXT:  map.store %[[CAP0]][%[[IDX]]] = %[[VAL]] : !mo.tensor<[B, T, 16], f32>
// CHECK-NEXT:  map.index.slice.select<#map.index.slice.spec<[B, T, 16] -> [B, 8, 16], starts = [0, -8, 0], stops = [?, ?, ?], steps = [1, 1, 1]>> (%[[IDX]]) -> (%[[IOUT:[a-z0-9]+]]) : !map.index<3> {
// CHECK-NEXT:  map.store %[[CAP1]][%[[IOUT]]] = %[[VAL]] : !mo.tensor<[B, 8, 16], f32>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  map.call.opaque @{{.*}}.FusedVariadicInputs[axis = 1 : i64] inputs(%{{.*}}, %{{.*}}) outputs(%[[EPI]])
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output<B, T> %[[OUT]]#0, %[[OUT]]#1
mo.graph @concat_slice_window<B, S, T -> OB, OT>(
  %cache: !mo.tensor<[B, 8, 16], f32>,
  %new: !mo.tensor<[B, S, 16], f32>,
  %stops: !mo.tensor<[3], si64>
) -> (!mo.tensor<[OB, OT, 16], f32>, !mo.tensor<[OB, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%cache, %new) : (!mo.tensor<[B, 8, 16], f32>, !mo.tensor<[B, S, 16], f32>) -> !mo.tensor<[B, T, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[B, T, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[B, 8, 16], f32>
  mo.output<B, T> %concat, %slice : !mo.tensor<[B, T, 16], f32>, !mo.tensor<[B, 8, 16], f32>
}

// The k and v caches of one attention block: two windows over two concats,
// sharing only the slices' bounds constants. The windows must not pair with
// each other on those -- neither loads them, so the pair would save nothing --
// or each would lose the concat it fuses into.
// CHECK-LABEL: mo.graph @concat_slice_kv_windows
// CHECK:       map.iter.opaque
// CHECK:       map.index.slice.select{{.*}} -> (%[[K_IOUT:[a-z0-9]+]]) : !map.index<3> {
// CHECK-NEXT:  map.store %{{.*}}[%[[K_IOUT]]] = %{{.*}} : !mo.tensor<[B, 8, 16], f32>
// CHECK:       map.call.opaque @{{.*}}.FusedVariadicInputs
// CHECK:       map.iter.opaque
// CHECK:       map.index.slice.select{{.*}} -> (%[[V_IOUT:[a-z0-9]+]]) : !map.index<3> {
// CHECK-NEXT:  map.store %{{.*}}[%[[V_IOUT]]] = %{{.*}} : !mo.tensor<[B, 8, 16], f32>
// CHECK:       map.call.opaque @{{.*}}.FusedVariadicInputs
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output
mo.graph @concat_slice_kv_windows<B, S, T -> OB, OT>(
  %k_cache: !mo.tensor<[B, 8, 16], f32>,
  %k_new: !mo.tensor<[B, S, 16], f32>,
  %v_cache: !mo.tensor<[B, 8, 16], f32>,
  %v_new: !mo.tensor<[B, S, 16], f32>,
  %stops: !mo.tensor<[3], si64>
) -> (!mo.tensor<[OB, OT, 16], f32>, !mo.tensor<[OB, 8, 16], f32>, !mo.tensor<[OB, OT, 16], f32>, !mo.tensor<[OB, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %k = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%k_cache, %k_new) : (!mo.tensor<[B, 8, 16], f32>, !mo.tensor<[B, S, 16], f32>) -> !mo.tensor<[B, T, 16], f32>
  %v = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%v_cache, %v_new) : (!mo.tensor<[B, 8, 16], f32>, !mo.tensor<[B, S, 16], f32>) -> !mo.tensor<[B, T, 16], f32>
  %k_win = mo.slice(%k, %starts, %stops, %steps) : (!mo.tensor<[B, T, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[B, 8, 16], f32>
  %v_win = mo.slice(%v, %starts, %stops, %steps) : (!mo.tensor<[B, T, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[B, 8, 16], f32>
  mo.output<B, T> %k, %k_win, %v, %v_win : !mo.tensor<[B, T, 16], f32>, !mo.tensor<[B, 8, 16], f32>, !mo.tensor<[B, T, 16], f32>, !mo.tensor<[B, 8, 16], f32>
}

// All static: the leading dim makes the window non-contiguous, so it is no
// view. MOToMAP resolves the negative start against the static extent.
// CHECK-LABEL: mo.graph @concat_slice_static_window
// CHECK:       map.iter.opaque
// CHECK:       map.store %{{.*}} : !mo.tensor<[4, 9, 16], f32>
// CHECK-NEXT:  map.index.slice.select<#map.index.slice.spec<[4, 9, 16] -> [4, 8, 16], starts = [0, 1, 0], stops = [4, 9, 16], steps = [1, 1, 1]>>{{.*}} -> (%[[IOUT:[a-z0-9]+]]) : !map.index<3> {
// CHECK-NEXT:  map.store %{{.*}}[%[[IOUT]]] = %{{.*}} : !mo.tensor<[4, 8, 16], f32>
// CHECK:       map.call.opaque @{{.*}}.FusedVariadicInputs
// CHECK-NOT:   map.iter.elementwise
mo.graph @concat_slice_static_window(
  %arg0: !mo.tensor<[4, 8, 16], f32>,
  %arg1: !mo.tensor<[4, 1, 16], f32>
) -> (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 9, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%arg0, %arg1) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 1, 16], f32>) -> !mo.tensor<[4, 9, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 8, 16], f32>
  mo.output %concat, %slice : !mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 8, 16], f32>
}

// A strided slice: every other row, from row 1.
// CHECK-LABEL: mo.graph @concat_slice_step
// CHECK:       map.iter.opaque
// CHECK:       map.index.slice.select<#map.index.slice.spec<[4, 8, 16] -> [4, 4, 16], starts = [0, 1, 0], stops = [4, 8, 16], steps = [1, 2, 1]>>{{.*}} -> (%[[IOUT:[a-z0-9]+]]) : !map.index<3> {
// CHECK-NEXT:  map.store %{{.*}}[%[[IOUT]]] = %{{.*}} : !mo.tensor<[4, 4, 16], f32>
// CHECK-NOT:   map.iter.elementwise
mo.graph @concat_slice_step(
  %arg0: !mo.tensor<[4, 6, 16], f32>,
  %arg1: !mo.tensor<[4, 2, 16], f32>
) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 8, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%arg0, %arg1) : (!mo.tensor<[4, 6, 16], f32>, !mo.tensor<[4, 2, 16], f32>) -> !mo.tensor<[4, 8, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  mo.output %concat, %slice : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>
}

// With the slice the concat's only reader, nothing escapes: a plain
// producer/consumer edge, which is output fusion's to take, not this merge's.
// CHECK-LABEL: mo.graph @concat_slice_single_use
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise
mo.graph @concat_slice_single_use(
  %arg0: !mo.tensor<[4, 8, 16], f32>,
  %arg1: !mo.tensor<[4, 1, 16], f32>
) -> !mo.tensor<[4, 8, 16], f32> {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 9, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%arg0, %arg1) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 1, 16], f32>) -> !mo.tensor<[4, 9, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 8, 16], f32>
  mo.output %slice : !mo.tensor<[4, 8, 16], f32>
}

// A contiguous window (no non-unit dim ahead of the sliced one) is an offset
// view of the concat, cheaper than a second store.
// CHECK-LABEL: mo.graph @concat_slice_contiguous
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise
mo.graph @concat_slice_contiguous(
  %arg0: !mo.tensor<[1, 8, 16], f32>,
  %arg1: !mo.tensor<[1, 1, 16], f32>
) -> (!mo.tensor<[1, 9, 16], f32>, !mo.tensor<[1, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<1, 9, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%arg0, %arg1) : (!mo.tensor<[1, 8, 16], f32>, !mo.tensor<[1, 1, 16], f32>) -> !mo.tensor<[1, 9, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[1, 9, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[1, 8, 16], f32>
  mo.output %concat, %slice : !mo.tensor<[1, 9, 16], f32>, !mo.tensor<[1, 8, 16], f32>
}

// The store is vectorized along the innermost dim, so a slice of it stays its
// own kernel.
// CHECK-LABEL: mo.graph @concat_slice_inner_dim
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise
mo.graph @concat_slice_inner_dim(
  %arg0: !mo.tensor<[4, 8, 16], f32>,
  %arg1: !mo.tensor<[4, 1, 16], f32>
) -> (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 9, 8], f32>) {
  %starts = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 9, 8> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%arg0, %arg1) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 1, 16], f32>) -> !mo.tensor<[4, 9, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 9, 8], f32>
  mo.output %concat, %slice : !mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 9, 8], f32>
}

// A reversed slice is not one the select takes.
// CHECK-LABEL: mo.graph @concat_slice_neg_step
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise
mo.graph @concat_slice_neg_step(
  %arg0: !mo.tensor<[4, 8, 16], f32>,
  %arg1: !mo.tensor<[4, 1, 16], f32>
) -> (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 9, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, -10, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, -1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%arg0, %arg1) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 1, 16], f32>) -> !mo.tensor<[4, 9, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 9, 16], f32>
  mo.output %concat, %slice : !mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 9, 16], f32>
}

// Runtime starts and steps leave the select no static bounds to carry.
// CHECK-LABEL: mo.graph @concat_slice_dynamic_bounds
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise
mo.graph @concat_slice_dynamic_bounds(
  %arg0: !mo.tensor<[4, 8, 16], f32>,
  %arg1: !mo.tensor<[4, 1, 16], f32>,
  %starts: !mo.tensor<[3], si64>,
  %stops: !mo.tensor<[3], si64>,
  %steps: !mo.tensor<[3], si64>
) -> (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 8, 16], f32>) {
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%arg0, %arg1) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 1, 16], f32>) -> !mo.tensor<[4, 9, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 8, 16], f32>
  mo.output %concat, %slice : !mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 8, 16], f32>
}

// A window that is also read again: the window's own reader stays out of the
// epilogue that now holds the select. Rebuilt at that epilogue's store index, it
// would read the concat's element at the concat's position rather than the
// window's.
// CHECK-LABEL: mo.graph @concat_slice_window_read_again
// CHECK:       %[[OUT:[a-z0-9]+]]:2 = map.iter.opaque
// CHECK:       map.index.slice.select
// CHECK:       map.call.opaque @{{.*}}.FusedVariadicInputs
// CHECK:       %[[RELU:[a-z0-9]+]] = map.iter.elementwise<[4, 8, 16]>
// CHECK-NEXT:  inputs (%[[OUT]]#1: !mo.tensor<[4, 8, 16], f32>)
// CHECK:       map.call.elementwise @{{.*}}.ReLU
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1, %[[RELU]]
mo.graph @concat_slice_window_read_again(
  %arg0: !mo.tensor<[4, 8, 16], f32>,
  %arg1: !mo.tensor<[4, 1, 16], f32>
) -> (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 9, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%arg0, %arg1) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 1, 16], f32>) -> !mo.tensor<[4, 9, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 8, 16], f32>
  %relu = mo.relu(%slice) : !mo.tensor<[4, 8, 16], f32>
  mo.output %concat, %slice, %relu : !mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 16], f32>
}

// Two windows of one concat: the slices pair first, both loading the concat,
// and the pair, one consumer with two stores, is not one the select takes. The
// concat stays its own kernel and the pair reads it.
// CHECK-LABEL: mo.graph @concat_slice_two_windows
// CHECK:       %[[CONCAT:[0-9]+]] = map.iter.opaque
// CHECK-NOT:   map.index.slice.select
// CHECK:       %[[PAIR:[0-9]+]]:2 = map.iter.elementwise<[4, 8, 16]>
// CHECK-NEXT:  inputs (%[[CONCAT]]: !mo.tensor<[4, 16, 16], f32>
// CHECK-NOT:   map.index.slice.select
// CHECK:       fusion_horizontal
// CHECK:       mo.output %[[CONCAT]], %[[PAIR]]#0, %[[PAIR]]#1
mo.graph @concat_slice_two_windows(
  %arg0: !mo.tensor<[4, 15, 16], f32>,
  %arg1: !mo.tensor<[4, 1, 16], f32>
) -> (!mo.tensor<[4, 16, 16], f32>, !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 16], f32>) {
  %tailStarts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stepStarts = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 16, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %unit = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %step2 = mo.constant <#M.dense_array<1, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%arg0, %arg1) : (!mo.tensor<[4, 15, 16], f32>, !mo.tensor<[4, 1, 16], f32>) -> !mo.tensor<[4, 16, 16], f32>
  %tail = mo.slice(%concat, %tailStarts, %stops, %unit) : (!mo.tensor<[4, 16, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 8, 16], f32>
  %even = mo.slice(%concat, %stepStarts, %stops, %step2) : (!mo.tensor<[4, 16, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 8, 16], f32>
  mo.output %concat, %tail, %even : !mo.tensor<[4, 16, 16], f32>, !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 16], f32>
}
