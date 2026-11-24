// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-tmp-decompose-composites --mo-to-map --map-fuse-kernels --map-rcse \
// RUN:   --map-outline-structured-kernel-bodies --map-livm \
// RUN:   --map-flatten-shapes | FileCheck %s

// A norm or a softmax has phases besides its folds: a per-element phase
// (`map.iter.map`) and, after row CSE, a row cache (`map.iter.tile`). Each
// takes the merged index, so the kernel flattens like a fold-only one.
// CHECK-LABEL: mo.graph @softmax
// CHECK:       map.index.collapse.stride<#map.view.signature<[4, 8, 128] -> [32, 128]>, groups = [2, 1]>
// CHECK:       map.iter.rowwise.body<[32, 128], axis = 1>
// CHECK:       map.iter.tile index (%{{.*}}: !map.index<2>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceMax index (%{{.*}}: !map.index<2>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%{{.*}}: !map.index<2>)
// CHECK:       map.iter.map index (%{{.*}}: !map.index<2>)
mo.graph @softmax(%x: !mo.tensor<[4, 8, 128], bf16>) -> !mo.tensor<[4, 8, 128], bf16> {
  %y = mo.composite.reduce.softmax(%x) [axis = 2] : (!mo.tensor<[4, 8, 128], bf16>) -> !mo.tensor<[4, 8, 128], bf16>
  mo.output %y : !mo.tensor<[4, 8, 128], bf16>
}

// CHECK-LABEL: mo.graph @rms_norm
// CHECK-DAG:   map.index.collapse.stride<#map.view.signature<[4, 8, 128] -> [32, 128]>, groups = [2, 1]> (%arg0)
// CHECK:       map.iter.rowwise.body<[32, 128], axis = 1>
// CHECK:       map.iter.tile index (%{{.*}}: !map.index<2>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%{{.*}}: !map.index<2>)
// CHECK:       map.iter.map index (%{{.*}}: !map.index<2>)
mo.graph @rms_norm(%x: !mo.tensor<[4, 8, 128], bf16>, %w: !mo.tensor<[128], bf16>) -> !mo.tensor<[4, 8, 128], bf16> {
  %y = mo.composite.reduce.rms_norm(%x, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[4, 8, 128], bf16>, !mo.tensor<[128], bf16>) -> !mo.tensor<[4, 8, 128], bf16>
  mo.output %y : !mo.tensor<[4, 8, 128], bf16>
}
