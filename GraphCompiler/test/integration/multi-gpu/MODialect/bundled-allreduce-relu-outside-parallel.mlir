// Variant of bundled-allreduce-fusion.mlir where the relu occurs outside the
// parallel region, applied to each unbundled result individually.  Verifies
// that after the create+merge pipeline pulls the relu into the parallel, the
// MAP fusion pipeline still produces an epilogue-fused kernel.  The create
// and merge passes in isolation are covered by their respective unit tests
// under mo-opt/.../Transforms/.

// RUN: mo-opt %s --mlir-print-local-scope --mogg-import-packages \
// RUN:     --mo-create-parallel-from-unbundle --canonicalize \
// RUN:     --mo-merge-adjacent-parallels \
// RUN:     --mo-pipeline=kernel-decls-pre-populated=true \
// RUN:     --mo-to-map --map-pipeline --map-to-mogg \
// RUN:     | FileCheck %s --check-prefix=CHECK-FUSED

// After create+merge + the MAP fusion pipeline, relu is fused into the
// allreduce as an epilogue, so a single mogg.routine remains inside
// mo.parallel, its fusion_epilogue joining the allreduce and the relu.
// CHECK-FUSED-LABEL: mo.graph @bundled_allreduce_relu_outside
// CHECK-FUSED:      mo.parallel
// CHECK-FUSED:        mogg.routine
// CHECK-FUSED:          fusion_epilogue<[
// CHECK-FUSED-SAME:       "mo.bundled.allreduce.sum"
// CHECK-FUSED-SAME:       "mo.relu"
// CHECK-FUSED:          mogg.tensor.create.epilogue
// CHECK-FUSED:            mogg.call.elementwise @{{.*}}.ReLU
// CHECK-FUSED:          mogg.call.opaque @{{.*}}BundledAllReduceSum

mo.graph @bundled_allreduce_relu_outside(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %in0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0> : tensor<3xf32>> : !mo.tensor<[3], f32, gpu:0>
  %in1 = mo.constant <#M.dense_array<4.0, -5.0, 6.0> : tensor<3xf32>> : !mo.tensor<[3], f32, gpu:1>

  %dt = mo.tensor.bundle(%in0, %in1) : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>) -> (!mo.bundle<[3], f32, gpu:0-1>)
  %ch = mo.chain.create()
  %res_bundle, %out_ch = mo.parallel (%arg, %bch) in (%dt : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>, %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch) -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %peer0, %peer1 = mo.bundled.expand(%arg) : !mo.tensor<[3], f32, gpu:0> -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%peer0, %peer1, %sig0, %sig1, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }

  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[3], f32, gpu:0-1>) -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  %relu_0 = mo.relu(%res_0) : !mo.tensor<[3], f32, gpu:0>
  %relu_1 = mo.relu(%res_1) : !mo.tensor<[3], f32, gpu:1>
  %res0_host, %ch3 = mo.transfer[%out_ch] %relu_0 : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  %res1_host, %ch4 = mo.transfer[%ch3] %relu_1 : !mo.tensor<[3], f32, gpu:1> to <"cpu", 0>
  mo.output %res0_host, %res1_host : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}
