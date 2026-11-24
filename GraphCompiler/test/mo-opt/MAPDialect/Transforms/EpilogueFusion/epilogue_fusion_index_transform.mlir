// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-epilogue | FileCheck %s

// `mo.slice` lowers to a `map.iter.elementwise` over the OUTPUT shape whose body
// transforms the loop index (`map.index.slice`) before loading the input -- see
// MOToMAP's slice lowering. As a consumer it looks fusible: single-result, and
// that result is a tensor. But the splice replaces the load of the producer's
// result with the value the epilogue holds for ITS index, and here the load
// wants `i * step + start`, so fusing would silently turn
// `out[i] = producer[i * step + start]` into `out[i] = producer[i]` (and inline
// `map.index.slice` under `map.iter.epilogue`, whose parent constraint forbids
// it). Eligibility declines when the FUSED operand's load is index-transformed;
// a transform on any other operand stays fusible.

// CHECK-LABEL: mo.graph @no_fuse_slice_consumer
// The kernel keeps its own full-shape output and its plain store epilogue.
// CHECK:       map.iter.opaque
// CHECK:         outputs (%{{.*}} : !mo.tensor<[8], f32>)
// CHECK:       map.iter.epilogue
// CHECK-NOT:     map.index.slice
// CHECK:       map.call.opaque @{{.*}}MapLowerOneFusedOutput
// The slice stays its own kernel, index transform intact.
// CHECK:       map.iter.elementwise<[3]>
// CHECK:       map.index.slice
mo.graph @no_fuse_slice_consumer(%a: !mo.tensor<[8], f32>) -> !mo.tensor<[3], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[8], f32>) -> !mo.tensor<[8], f32>
  %start = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<4> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %res = mo.slice(%c, %start, %stop, %step) : (
    !mo.tensor<[8], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[3], f32>
  mo.output %res : !mo.tensor<[3], f32>
}
