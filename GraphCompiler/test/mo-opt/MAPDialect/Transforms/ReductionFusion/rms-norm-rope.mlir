// RUN: mo-opt %s --mo-tmp-decompose-composites --canonicalize | mo-opt --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-kernels --map-rlicm --cse --map-rcse | FileCheck %s

// rms_norm computed in f32 and cast back, then rotate-half RoPE (the split as
// slices, as the pipeline writes it), fuse into one rowwise kernel: fold the
// squares, cache the normalized row in shared memory, and write
// n * cos + rotate(n) * sin, reading each element's partner 128 elements away.
// CHECK-LABEL: mo.graph @rmsNormRope
// CHECK:       %[[R:.*]] = map.iter.rowwise<[8, 256], axis = 1>
// CHECK:         map.iter.reduce
// CHECK:         %[[C:.*]] = map.iter.cache index (%{{.*}}: !map.index<2>) -> bf16 {
// CHECK:         } {shared = true}
// CHECK:         map.iter.map index (%[[J:.*]]: !map.index<2>) {
// CHECK:           map.iter.cache.load %[[C]][%[[J]]] : !map.index<2>, bf16
// CHECK:           map.index.concat.select<#map.index.concat.spec<[8, 128], [8, 128] -> [8, 256], axis = 1>> (%[[J]])
// CHECK:             map.iter.cache.load %[[C]][%[[J]]] offset(128) : !map.index<2>, bf16
// CHECK:             map.iter.cache.load %[[C]][%[[J]]] offset(-128) : !map.index<2>, bf16
// CHECK-NOT:   map.iter.rowwise
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[R]]
mo.graph @rmsNormRope(%x: !mo.tensor<[8, 256], bf16>, %w: !mo.tensor<[256], f32>, %cos: !mo.tensor<[8, 256], bf16>, %sin: !mo.tensor<[8, 256], bf16>) -> !mo.tensor<[8, 256], bf16> {
  %xf = mo.cast(%x) : (!mo.tensor<[8, 256], bf16>) -> !mo.tensor<[8, 256], f32>
  %n = mo.composite.reduce.rms_norm(%xf, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[8, 256], f32>, !mo.tensor<[256], f32>) -> !mo.tensor<[8, 256], f32>
  %nb = mo.cast(%n) : (!mo.tensor<[8, 256], f32>) -> !mo.tensor<[8, 256], bf16>
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 128> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<8, 128> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<8, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %x1 = mo.slice(%nb, %start1, %stop1, %step) : (!mo.tensor<[8, 256], bf16>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 128], bf16>
  %x2 = mo.slice(%nb, %start2, %stop2, %step) : (!mo.tensor<[8, 256], bf16>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 128], bf16>
  %nx2 = mo.negative(%x2) : !mo.tensor<[8, 128], bf16>
  %rot = mo.concat[1] (%nx2, %x1) : (!mo.tensor<[8, 128], bf16>, !mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 256], bf16>
  %a = mo.mul(%nb, %cos) : !mo.tensor<[8, 256], bf16>
  %b = mo.mul(%rot, %sin) : !mo.tensor<[8, 256], bf16>
  %y = mo.add(%a, %b) : !mo.tensor<[8, 256], bf16>
  mo.output %y : !mo.tensor<[8, 256], bf16>
}
