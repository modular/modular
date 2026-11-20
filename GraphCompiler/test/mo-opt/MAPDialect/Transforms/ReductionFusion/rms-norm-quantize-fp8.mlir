// RUN: mo-opt %s --mogg-import-packages --mo-tmp-decompose-composites --mo-to-map --map-fuse-kernels | FileCheck %s

// rms_norm with a per-token FP8 quantize fuses into one rowwise kernel: a sum
// fold, a max fold over the normalized row, the scale and its reciprocal once
// per row, the scale stored per row, and a map phase that writes the quantized
// row.
// CHECK-LABEL: mo.graph @rmsNormQuantize
// CHECK:       map.iter.rowwise<[M, 4096], axis = 1>
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:         %[[MAX:.*]] = map.iter.reduce @builtin_kernels.monoids.ReduceMax
// CHECK:         %[[CAPPED:.*]] = map.call.elementwise @builtin_kernels.elementwise.Min(%{{.*}}, %[[MAX]], %{{.*}})
// CHECK:         %[[SCALE:.*]] = map.call.elementwise @builtin_kernels.elementwise.Div(%{{.*}}, %[[CAPPED]], %{{.*}})
// CHECK:         map.store %{{.*}}[%{{.*}}] = %[[SCALE]] : !mo.tensor<[M, 1], f32, gpu:0>
// CHECK:         %[[INV:.*]] = map.call.elementwise @builtin_kernels.elementwise.Select
// CHECK:         map.iter.map
// CHECK:           map.call.elementwise @builtin_kernels.elementwise.Mul(%{{.*}}, %{{.*}}, %[[INV]])
// CHECK:           map.call.elementwise @builtin_kernels.elementwise.Cast({{.*}}) : !map.index<2>, f32 -> f8E4M3FN
// CHECK-NOT:   map.iter.rowwise
// CHECK:       mo.output
mo.graph @rmsNormQuantize<M>(%x: !mo.tensor<[M, 4096], bf16, gpu:0>, %w: !mo.tensor<[4096], bf16, gpu:0>) -> (!mo.tensor<[M, 4096], f8e4m3fn, gpu:0>, !mo.tensor<[1, M], f32, gpu:0>) {
  %ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32, cpu:0>
  %q:2 = mo.composite.rms_norm_fused_quantize_dynamic_scaled_fp8(%x, %w, %ub) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>} : (!mo.tensor<[M, 4096], bf16, gpu:0>, !mo.tensor<[4096], bf16, gpu:0>, !mo.tensor<[], f32, cpu:0>) -> (!mo.tensor<[M, 4096], f8e4m3fn, gpu:0>, !mo.tensor<[1, M], f32, gpu:0>)
  mo.output %q#0, %q#1 : !mo.tensor<[M, 4096], f8e4m3fn, gpu:0>, !mo.tensor<[1, M], f32, gpu:0>
}
