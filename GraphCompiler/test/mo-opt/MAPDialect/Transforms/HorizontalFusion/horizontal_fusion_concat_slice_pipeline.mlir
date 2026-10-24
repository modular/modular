// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// The concat+slice merge inside the whole fusion pipeline, where every other
// fuser reaches its fixpoint before horizontal fusion sees what is left. The
// window's store still lands beside the concat's own, and no slice kernel
// remains. The declines have their cases in
// `horizontal_fusion_concat_slice.mlir`, but for the work on a window that
// only this pipeline fuses onto the slice, and for an epilogue that only this
// pipeline reshapes.

// The KV-cache window update.
// CHECK-LABEL: mo.graph @concat_slice_window
// CHECK:       %[[OUT:[a-z0-9]+]]:2 = map.iter.opaque
// CHECK:       map.iter.epilogue
// CHECK-NEXT:  map.store %{{.*}} : !mo.tensor<[B, T, 16], f32>
// CHECK-NEXT:  map.index.slice.select<#map.index.slice.spec<[B, T, 16] -> [B, 8, 16], starts = [0, -8, 0], stops = [?, ?, ?], steps = [1, 1, 1]>>{{.*}} -> (%[[IOUT:[a-z0-9]+]]) : !map.index<3> {
// CHECK-NEXT:  map.store %{{.*}}[%[[IOUT]]] = %{{.*}} : !mo.tensor<[B, 8, 16], f32>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  map.call.opaque @{{.*}}.FusedVariadicInputs
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

// The k and v windows, sharing their bounds constants, each fuse into their own
// concat rather than pairing with each other.
// CHECK-LABEL: mo.graph @concat_slice_kv_windows
// CHECK:       %[[K:[a-z0-9]+]]:2 = map.iter.opaque
// CHECK:       map.index.slice.select{{.*}} -> (%[[K_IOUT:[a-z0-9]+]]) : !map.index<3> {
// CHECK-NEXT:  map.store %{{.*}}[%[[K_IOUT]]] = %{{.*}} : !mo.tensor<[B, 8, 16], f32>
// CHECK:       map.call.opaque @{{.*}}.FusedVariadicInputs
// CHECK:       %[[V:[a-z0-9]+]]:2 = map.iter.opaque
// CHECK:       map.index.slice.select{{.*}} -> (%[[V_IOUT:[a-z0-9]+]]) : !map.index<3> {
// CHECK-NEXT:  map.store %{{.*}}[%[[V_IOUT]]] = %{{.*}} : !mo.tensor<[B, 8, 16], f32>
// CHECK:       map.call.opaque @{{.*}}.FusedVariadicInputs
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output<B, T> %[[K]]#0, %[[K]]#1, %[[V]]#0, %[[V]]#1
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

// A cache update whose new rows are one group of heads sliced out of a
// reshaped projection, in bf16, as in an attention block. Prologue fusion first
// folds the reshape and the slice into the concat's input, and the window then
// joins the concat's epilogue: one kernel reads the projection and writes both
// the cache and its window.
// CHECK-LABEL: mo.graph @concat_slice_projection_window
// CHECK:       %[[OUT:[a-z0-9]+]]:2 = map.iter.opaque
// CHECK:       map.iter.prologue
// CHECK:       map.iter.prologue
// CHECK-NEXT:  map.index.slice<#map.index.slice.spec<[B, S, 6, 16] -> [B, S, 2, 16], starts = [0, 0, 4, 0]
// CHECK-NEXT:  map.index.reshape<#map.index.reshape.spec<[B, S, 96] -> [B, S, 6, 16]>>
// CHECK-NEXT:  map.load %{{.*}} : !mo.tensor<[B, S, 96], bf16>
// CHECK:       map.iter.epilogue
// CHECK-NEXT:  map.store %{{.*}} : !mo.tensor<[B, T, 2, 16], bf16>
// CHECK-NEXT:  map.index.slice.select<#map.index.slice.spec<[B, T, 2, 16] -> [B, 8, 2, 16], starts = [0, -8, 0, 0], stops = [?, ?, ?, ?], steps = [1, 1, 1, 1]>>{{.*}} -> (%[[IOUT:[a-z0-9]+]]) : !map.index<4> {
// CHECK-NEXT:  map.store %{{.*}}[%[[IOUT]]] = %{{.*}} : !mo.tensor<[B, 8, 2, 16], bf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  map.call.opaque @{{.*}}.FusedVariadicInputs
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output<B, T> %[[OUT]]#0, %[[OUT]]#1
mo.graph @concat_slice_projection_window<B, S, T -> OB, OT>(
  %cache: !mo.tensor<[B, 8, 2, 16], bf16>,
  %proj: !mo.tensor<[B, S, 96], bf16>,
  %stops: !mo.tensor<[4], si64>
) -> (!mo.tensor<[OB, OT, 2, 16], bf16>, !mo.tensor<[OB, 8, 2, 16], bf16>) {
  %head_starts = mo.constant <#M.dense_array<0, 0, 4, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %head_stops = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 6, 9223372036854775807> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %starts = mo.constant <#M.dense_array<0, -8, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %heads = mo.static.reshape(%proj) : !mo.tensor<[B, S, 96], bf16> -> !mo.tensor<[B, S, 6, 16], bf16>
  %new = mo.slice(%heads, %head_starts, %head_stops, %steps) : (!mo.tensor<[B, S, 6, 16], bf16>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, S, 2, 16], bf16>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%cache, %new) : (!mo.tensor<[B, 8, 2, 16], bf16>, !mo.tensor<[B, S, 2, 16], bf16>) -> !mo.tensor<[B, T, 2, 16], bf16>
  %window = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[B, T, 2, 16], bf16>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, 8, 2, 16], bf16>
  mo.output<B, T> %concat, %window : !mo.tensor<[B, T, 2, 16], bf16>, !mo.tensor<[B, 8, 2, 16], bf16>
}

// A window that is also rectified. Elementwise fusion first puts the relu on
// the slice's kernel, and the merge then moves both into the select, where the
// relu takes the select's index, the window's own, as the slice kernel's loop
// index was: one kernel writes the cache and the rectified window.
// CHECK-LABEL: mo.graph @concat_slice_relu_window
// CHECK:       %[[OUT:[a-z0-9]+]]:2 = map.iter.opaque
// CHECK:       map.iter.epilogue (%[[VAL:[a-z0-9]+]] : f32, %{{[a-z0-9]+}} : !map.index<3>)
// CHECK-NEXT:  map.store %{{.*}} = %[[VAL]] : !mo.tensor<[B, T, 16], f32>
// CHECK-NEXT:  map.index.slice.select<#map.index.slice.spec<[B, T, 16] -> [B, 8, 16], starts = [0, -8, 0], stops = [?, ?, ?], steps = [1, 1, 1]>>{{.*}} -> (%[[IOUT:[a-z0-9]+]]) : !map.index<3> {
// CHECK-NEXT:  %[[RELU:[a-z0-9]+]] = map.call.elementwise @{{.*}}.ReLU(%[[IOUT]], %[[VAL]])
// CHECK-NEXT:  map.store %{{.*}}[%[[IOUT]]] = %[[RELU]] : !mo.tensor<[B, 8, 16], f32>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  map.call.opaque @{{.*}}.FusedVariadicInputs
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output<B, T> %[[OUT]]#0, %[[OUT]]#1
mo.graph @concat_slice_relu_window<B, S, T -> OB, OT>(
  %cache: !mo.tensor<[B, 8, 16], f32>,
  %new: !mo.tensor<[B, S, 16], f32>,
  %stops: !mo.tensor<[3], si64>
) -> (!mo.tensor<[OB, OT, 16], f32>, !mo.tensor<[OB, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%cache, %new) : (!mo.tensor<[B, 8, 16], f32>, !mo.tensor<[B, S, 16], f32>) -> !mo.tensor<[B, T, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[B, T, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[B, 8, 16], f32>
  %relu = mo.relu(%slice) : !mo.tensor<[B, 8, 16], f32>
  mo.output<B, T> %concat, %relu : !mo.tensor<[B, T, 16], f32>, !mo.tensor<[B, 8, 16], f32>
}

// A window added to another input stays its own kernel: the add also loads the
// bias at the loop index, which the merge does not take yet.
// CHECK-LABEL: mo.graph @concat_slice_add_window
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise<[B, 8, 16]>
// CHECK:       map.index.slice<
// CHECK:       map.call.elementwise @{{.*}}.Add
mo.graph @concat_slice_add_window<B, S, T -> OB, OT>(
  %cache: !mo.tensor<[B, 8, 16], f32>,
  %new: !mo.tensor<[B, S, 16], f32>,
  %bias: !mo.tensor<[B, 8, 16], f32>,
  %stops: !mo.tensor<[3], si64>
) -> (!mo.tensor<[OB, OT, 16], f32>, !mo.tensor<[OB, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%cache, %new) : (!mo.tensor<[B, 8, 16], f32>, !mo.tensor<[B, S, 16], f32>) -> !mo.tensor<[B, T, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[B, T, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[B, 8, 16], f32>
  %sum = mo.add(%slice, %bias) : !mo.tensor<[B, 8, 16], f32>
  mo.output<B, T> %concat, %sum : !mo.tensor<[B, T, 16], f32>, !mo.tensor<[B, 8, 16], f32>
}

// A window of a matmul result that is reshaped first stays its own kernel.
// Epilogue fusion moves the reshape and the add into the matmul's epilogue,
// which then stores at a reshaped index. The matmul still hands it vectors
// along its own 64-wide inner dim, each spanning several rows of the
// `[128, 4]` result, so a select, which tests a vector once at its first index,
// would store whole vectors into rows the window skips.
// CHECK-LABEL: mo.graph @matmul_reshape_window
// CHECK:       %[[MM:[a-z0-9]+]] = map.iter.opaque
// CHECK:       map.index.reshape<#map.index.reshape.spec<[128, 4] -> [8, 64]>>
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK:       %[[WIN:[a-z0-9]+]] = map.iter.elementwise<[64, 4]>
// CHECK-NEXT:  inputs (%[[MM]]: !mo.tensor<[128, 4], f32>
// CHECK:       map.index.slice<
// CHECK:       mo.output %[[MM]], %[[WIN]]
mo.graph @matmul_reshape_window(
  %a: !mo.tensor<[8, 16], f32>,
  %b: !mo.tensor<[16, 64], f32>,
  %y: !mo.tensor<[128, 4], f32>
) -> (!mo.tensor<[128, 4], f32>, !mo.tensor<[64, 4], f32>) {
  %starts = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stops = mo.constant <#M.dense_array<128, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[8, 16], f32>, !mo.tensor<[16, 64], f32>) -> !mo.tensor<[8, 64], f32>
  %r = mo.static.reshape(%mm) : !mo.tensor<[8, 64], f32> -> !mo.tensor<[128, 4], f32>
  %s = mo.add(%r, %y) : !mo.tensor<[128, 4], f32>
  %w = mo.slice(%s, %starts, %stops, %steps) : (!mo.tensor<[128, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[64, 4], f32>
  mo.output %s, %w : !mo.tensor<[128, 4], f32>, !mo.tensor<[64, 4], f32>
}
