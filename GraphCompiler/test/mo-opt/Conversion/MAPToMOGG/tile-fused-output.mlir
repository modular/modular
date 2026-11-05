// RUN: mo-opt %s --split-input-file --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg | FileCheck %s
// RUN: mo-opt %s --split-input-file --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-outline-structured-kernel-bodies --canonicalize --cse --map-to-mogg | FileCheck %s --check-prefix=FOREACH

// With `tile_based_fusion = true` the sm_100a `mo.matmul` registration and
// the `tile_rms_norm` test kernel are tile kernels, whose outputs are
// `_FusedComputeOutputTileTensor` / `_FusedOutputTileTensor`, whose epilogues
// lower to `mogg.tile.create.epilogue`.

// CHECK-LABEL: mo.graph @matmul
// CHECK:       mogg.tile.create.epilogue %{{.*}} (%[[VAL:[^ ]+]] : !mogg.tile<f32>, %{{[^ ]+}} : !mogg.tile_coord<2>) capturing {%{{[^ ]+}} = %{{[^ ]+}} : !mogg.tensor} ptr %{{.*}} lane = compute, device = "gpu"
// CHECK-NEXT:    mogg.yield %[[VAL]] : !mogg.tile<f32>
// CHECK:       mogg.call.opaque
// CHECK-NOT:   hasEpilogueFusion = true
module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "", tile_based_fusion = true>}} {
mo.graph @matmul(%a: !mo.tensor<[256, 256], f32, gpu:0>, %b: !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0> {
  %0 = mo.matmul(%a, %b) : (!mo.tensor<[256, 256], f32, gpu:0>, !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0>
  mo.output %0 : !mo.tensor<[256, 256], f32, gpu:0>
}
}

// -----

// CHECK-LABEL: mo.graph @rms_norm
// CHECK:       mogg.tile.create.epilogue {{.*}} lane = store, device = "gpu"
module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "", tile_based_fusion = true>}} {
mo.graph @rms_norm(%x: !mo.tensor<[256, 256], f32, gpu:0>, %gamma: !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0> {
  %0 = mo.custom [symbol = "tile_rms_norm", device = #M.device_ref<"gpu", 0>](%x, %gamma) : (!mo.tensor<[256, 256], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0>
  mo.output %0 : !mo.tensor<[256, 256], f32, gpu:0>
}
}

// -----

// A fused epilogue into a tile output follows the kernel's tile argument, not
// the SIMD fallbacks: the compute tile body yields its value for the kernel to
// store, and the store tile body keeps its store.
// CHECK-LABEL: mo.graph @matmul_add
// CHECK:       mogg.tile.create.epilogue %{{.*}} (%[[VAL:[^ ]+]] : !mogg.tile<f32>, %[[IDX:[^ ]+]] : !mogg.tile_coord<2>) capturing {%{{[^ ]+}} = %{{[^ ]+}} : !mogg.tensor, %[[Z:[^ ]+]] = %{{[^ ]+}} : !mogg.tensor} ptr %{{.*}} lane = compute
// CHECK-NEXT:    %[[ZV:.*]] = mogg.tile.load %[[Z]][%[[IDX]]] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
// CHECK-NEXT:    %[[SUM:.*]] = mogg.call.elementwise @{{.*}}Add{{.*}}(%[[VAL]], %[[ZV]]) {{.*}}: !mogg.tile<f32>, !mogg.tile<f32> -> !mogg.tile<f32>
// CHECK-NEXT:    mogg.yield %[[SUM]] : !mogg.tile<f32>
// CHECK:       mogg.call.opaque {{.*}}hasEpilogueFusion = true
module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "", tile_based_fusion = true>}} {
mo.graph @matmul_add(%a: !mo.tensor<[256, 256], f32, gpu:0>, %b: !mo.tensor<[256, 256], f32, gpu:0>, %z: !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0> {
  %0 = mo.matmul(%a, %b) : (!mo.tensor<[256, 256], f32, gpu:0>, !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0>
  %1 = mo.add(%0, %z) : !mo.tensor<[256, 256], f32, gpu:0>
  mo.output %1 : !mo.tensor<[256, 256], f32, gpu:0>
}
}

// -----

// CHECK-LABEL: mo.graph @rms_norm_mul
// CHECK:       mogg.tile.create.epilogue %{{.*}} (%[[VAL:[^ ]+]] : !mogg.tile<f32>, %[[IDX:[^ ]+]] : !mogg.tile_coord<2>) capturing {%[[OUT:[^ ]+]] = %{{[^ ]+}} : !mogg.tensor, %[[S:[^ ]+]] = %{{[^ ]+}} : !mogg.tensor} ptr %{{.*}} lane = store
// CHECK-NEXT:    %[[SV:.*]] = mogg.tile.load %[[S]][%[[IDX]]]
// CHECK-NEXT:    %[[PROD:.*]] = mogg.call.elementwise @{{.*}}Mul{{.*}}(%[[VAL]], %[[SV]])
// CHECK-NEXT:    mogg.tile.store %[[OUT]][%[[IDX]]] = %[[PROD]] : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "", tile_based_fusion = true>}} {
mo.graph @rms_norm_mul(%x: !mo.tensor<[256, 256], f32, gpu:0>, %gamma: !mo.tensor<[256], f32, gpu:0>, %scale: !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0> {
  %0 = mo.custom [symbol = "tile_rms_norm", device = #M.device_ref<"gpu", 0>](%x, %gamma) : (!mo.tensor<[256, 256], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0>
  %1 = mo.mul(%0, %scale) : !mo.tensor<[256, 256], f32, gpu:0>
  mo.output %1 : !mo.tensor<[256, 256], f32, gpu:0>
}
}

// -----

// Standalone kernels lower once outlined, hence the FOREACH run.
// FOREACH-LABEL: mo.graph @add_mul
// FOREACH:       mogg.call.foreach.tile @{{.*}}foreach_fusion_tile shape(%{{.*}}) context(%{{.*}}) (%[[IDX:[^ ]+]] : !mogg.tile_coord<2>) capturing {{{.*}}} device = "gpu"
// FOREACH:         mogg.tile.load %{{.*}}[%[[IDX]]] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
// FOREACH:         mogg.call.elementwise @{{.*}}Add{{.*}} : !mogg.tile<f32>, !mogg.tile<f32> -> !mogg.tile<f32>
// FOREACH:         mogg.call.elementwise @{{.*}}Mul{{.*}} : !mogg.tile<f32>, !mogg.tile<f32> -> !mogg.tile<f32>
// FOREACH:         mogg.tile.store %{{.*}}[%[[IDX]]] = %{{.*}} : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
// FOREACH-NOT:   mogg.call.foreach @
module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "", tile_based_fusion = true>}} {
mo.graph @add_mul(%x: !mo.tensor<[256, 256], f32, gpu:0>, %y: !mo.tensor<[256, 256], f32, gpu:0>, %z: !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0> {
  %0 = mo.add(%x, %y) : !mo.tensor<[256, 256], f32, gpu:0>
  %1 = mo.mul(%0, %z) : !mo.tensor<[256, 256], f32, gpu:0>
  mo.output %1 : !mo.tensor<[256, 256], f32, gpu:0>
}
}

// -----

// Without tile-based fusion the same kernel keeps the SIMD foreach.
// FOREACH-LABEL: mo.graph @add_mul_simd
// FOREACH:       mogg.call.foreach @
// FOREACH-NOT:   mogg.tile.
module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "">}} {
mo.graph @add_mul_simd(%x: !mo.tensor<[256, 256], f32, gpu:0>, %y: !mo.tensor<[256, 256], f32, gpu:0>, %z: !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0> {
  %0 = mo.add(%x, %y) : !mo.tensor<[256, 256], f32, gpu:0>
  %1 = mo.mul(%0, %z) : !mo.tensor<[256, 256], f32, gpu:0>
  mo.output %1 : !mo.tensor<[256, 256], f32, gpu:0>
}
}
