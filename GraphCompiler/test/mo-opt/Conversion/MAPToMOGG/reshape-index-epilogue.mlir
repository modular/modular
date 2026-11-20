// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg | FileCheck %s

// A reshape fused into a (non-accumulating) fused-output kernel's epilogue
// lowers so the in-epilogue `map.index.reshape` (the store-index transform)
// becomes a `mogg.index.reshape` inside the `mogg.tensor.create.epilogue`
// body -- NOT hoisted, unlike the `.stride` view ops. The bias load and the
// store both read the reshaped [64] store index, while the epilogue's own
// result keeps the producer's [8, 8] shape.

// CHECK-LABEL: mo.graph @custom_reshape_add
// The [64] store-target pointer is handed to the epilogue via get_ptr, so emit
// can build the kernel's fused output over real storage of the epilogue's own
// [8, 8] shape (the store still lands in the [64] capture at the reshaped idx).
// CHECK: mogg.tensor.get_ptr
// CHECK: mogg.tensor.create.epilogue %{{[^ ]+}} (%[[V:[^ ,)]+]] : !mogg.simd<f32>, %[[I:[^ ,)]+]] : !map.index<2>) capturing {%[[CAP0:[^ ,)]+]] = %{{[^ ]+}} : !mogg.tensor, %[[CAP1:[^ ,)]+]] = %{{[^ ]+}} : !mogg.tensor}
// CHECK: %[[SIDX:.*]] = mogg.index.reshape(%[[I]]) <fromShape = {{.*}}[8, 8]{{.*}}, toShape = {{.*}}[64]{{.*}}> : !map.index<2> -> !map.index<1>
// CHECK: %[[BV:.*]] = mogg.tensor.load %[[CAP1]][%[[SIDX]]] : !mogg.tensor, !map.index<1>
// CHECK: %[[ADD:.*]] = mogg.call.elementwise @{{.*}}Add
// CHECK: mogg.tensor.store %[[CAP0]][%[[SIDX]]] = %[[ADD]] : !mogg.tensor, !map.index<1>, !mogg.simd<f32>
// CHECK-NOT: mogg.yield
mo.graph @custom_reshape_add(
  %x: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[64], f32>
) -> !mo.tensor<[64], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[8, 8], f32> -> !mo.tensor<[64], f32>
  %out = mo.add(%r, %z) : !mo.tensor<[64], f32>
  mo.output %out : !mo.tensor<[64], f32>
}

// Unit dims in the reshaped shape: the store-index reshape carries them through
// its `[1, 64, 1]` toShape. Fully static, so it still uses the static
// `mogg.index.reshape` primitive with no runtime shape captures.
// CHECK-LABEL: mo.graph @custom_reshape_add_unit_dims
// CHECK: mogg.tensor.get_ptr
// CHECK: mogg.tensor.create.epilogue
// CHECK: %[[SIDX:.*]] = mogg.index.reshape({{.*}}) <fromShape = {{.*}}[8, 8]{{.*}}, toShape = {{.*}}[1, 64, 1]{{.*}}>
// CHECK: mogg.tensor.store {{.*}}[%[[SIDX]]]
mo.graph @custom_reshape_add_unit_dims(
  %x: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[1, 64, 1], f32>
) -> !mo.tensor<[1, 64, 1], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[8, 8], f32> -> !mo.tensor<[1, 64, 1], f32>
  %out = mo.add(%r, %z) : !mo.tensor<[1, 64, 1], f32>
  mo.output %out : !mo.tensor<[1, 64, 1], f32>
}
