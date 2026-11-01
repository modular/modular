// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// A reshape between a (non-accumulating) fused-output kernel and an add fuses
// into the producer's epilogue: ElementwiseFuser folds the reshape into the
// add, then ReshapeEpilogueFuser relocates it onto the producer's epilogue as
// a swapped `map.index.reshape` on the STORE index. Only the store DESTINATION
// is retyped to [64]; the epilogue RESULT (what the kernel binds) keeps the
// producer's [8, 8] shape. The bias load and store both use the reshaped
// index. (An accumulating producer -- matmul/conv/concat -- is declined, since
// it would touch the null-pointer fused-output slice; see ReshapeEpilogueFuser.)

// CHECK-LABEL: mo.graph @custom_reshape_add
// CHECK:       %[[OUT:.*]] = map.iter.opaque
// CHECK:       outputs (%{{.*}} : !mo.tensor<[64], f32>) {
// CHECK:       map.iter.epilogue (%[[VAL:.*]] : f32, %[[IDX:.*]] : !map.index<2>)
// CHECK:       %[[SIDX:.*]] = map.index.reshape<#map.index.reshape.spec<[64] -> [8, 8]>>
// CHECK:       %[[BV:.*]] = map.load %{{.*}}[%[[SIDX]]] : !mo.tensor<[64], f32>
// CHECK:       %[[ADD:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[VAL]], %[[BV]])
// CHECK:       map.store %{{.*}}[%[[SIDX]]] = %[[ADD]] : !mo.tensor<[64], f32>
// CHECK:       map.call.opaque @{{.*}}MapLowerOneFusedOutput
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"custom__map_lower_one_fused_output"
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]] : !mo.tensor<[64], f32>
mo.graph @custom_reshape_add(
  %x: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[64], f32>
) -> !mo.tensor<[64], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[8, 8], f32> -> !mo.tensor<[64], f32>
  %out = mo.add(%r, %z) : !mo.tensor<[64], f32>
  mo.output %out : !mo.tensor<[64], f32>
}

// A pure reshape (no downstream elementwise op to unlock) is left alone -- a
// no-op relabel in MGP -- so it stays its own view kernel.

// CHECK-LABEL: mo.graph @custom_reshape_only
// CHECK:       map.iter.opaque
// CHECK:       map.iter.elementwise
// CHECK:       map.index.reshape<#map.index.reshape.spec<[8, 8] -> [64]>>
mo.graph @custom_reshape_only(
  %x: !mo.tensor<[8, 8], f32>
) -> !mo.tensor<[64], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[8, 8], f32> -> !mo.tensor<[64], f32>
  mo.output %r : !mo.tensor<[64], f32>
}
