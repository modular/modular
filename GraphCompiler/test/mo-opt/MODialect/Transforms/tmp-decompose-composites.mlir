// RUN: mo-opt --split-input-file --mo-tmp-decompose-composites %s | FileCheck %s

// With no kernel registered for it, rms_norm becomes the ops it is made of,
// computed as the hand-written kernel computes it: a bf16 input accumulates in
// f32, and `multiply_before_cast` multiplies by the weight before casting back.
// CHECK-LABEL: mo.graph @bf16MultiplyBeforeCast
// CHECK-NOT:     mo.composite.reduce.rms_norm
// CHECK:         %[[XA:.*]] = mo.cast(%arg0) : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32>
// CHECK:         %[[SQ:.*]] = mo.mul(%[[XA]], %[[XA]])
// CHECK:         %[[SUM:.*]] = mo.reduce.add(%[[SQ]]) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
// CHECK:         %[[MS:.*]] = mo.div(%[[SUM]], %{{.*}}) : !mo.tensor<[8, 1], f32>
// CHECK:         %[[V:.*]] = mo.add(%[[MS]], %{{.*}}) : !mo.tensor<[8, 1], f32>
// CHECK:         %[[INV:.*]] = mo.rsqrt(%[[V]]) : !mo.tensor<[8, 1], f32>
// CHECK:         %[[INVB:.*]] = mo.static.broadcast_to(%[[INV]]) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
// CHECK:         %[[NORMED:.*]] = mo.mul(%[[XA]], %[[INVB]]) : !mo.tensor<[8, 128], f32>
// CHECK:         %[[W:.*]] = mo.cast(%arg1) : (!mo.tensor<[128], bf16>) -> !mo.tensor<[128], f32>
// CHECK:         %[[G:.*]] = mo.add(%[[W]], %{{.*}}) : !mo.tensor<[128], f32>
// CHECK:         %[[GB:.*]] = mo.static.broadcast_to(%[[G]]) : !mo.tensor<[128], f32> -> !mo.tensor<[8, 128], f32>
// CHECK:         %[[Y:.*]] = mo.mul(%[[NORMED]], %[[GB]]) : !mo.tensor<[8, 128], f32>
// CHECK:         %[[OUT:.*]] = mo.cast(%[[Y]]) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], bf16>
// CHECK:         mo.output %[[OUT]]
mo.graph @bf16MultiplyBeforeCast(%x: !mo.tensor<[8, 128], bf16>, %w: !mo.tensor<[128], bf16>) -> !mo.tensor<[8, 128], bf16> {
  %y = mo.composite.reduce.rms_norm(%x, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16>, !mo.tensor<[128], bf16>) -> !mo.tensor<[8, 128], bf16>
  mo.output %y : !mo.tensor<[8, 128], bf16>
}

// -----

// Without `multiply_before_cast`, the normalized value is cast back first and
// multiplied by the weight in the input dtype, offset cast to it too.
// CHECK-LABEL: mo.graph @bf16CastBeforeMultiply
// CHECK:         mo.rsqrt
// CHECK:         %[[NORMED:.*]] = mo.mul(%{{.*}}, %{{.*}}) : !mo.tensor<[8, 128], f32>
// CHECK:         %[[OFF:.*]] = mo.cast(%{{.*}}) : (!mo.tensor<[], f32>) -> !mo.tensor<[], bf16>
// CHECK:         %[[OFFB:.*]] = mo.static.broadcast_to(%[[OFF]]) : !mo.tensor<[], bf16> -> !mo.tensor<[128], bf16>
// CHECK:         %[[G:.*]] = mo.add(%arg1, %[[OFFB]]) : !mo.tensor<[128], bf16>
// CHECK:         %[[GB:.*]] = mo.static.broadcast_to(%[[G]])
// CHECK:         %[[N16:.*]] = mo.cast(%[[NORMED]]) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], bf16>
// CHECK:         %[[Y:.*]] = mo.mul(%[[N16]], %[[GB]]) : !mo.tensor<[8, 128], bf16>
// CHECK:         mo.output %[[Y]]
mo.graph @bf16CastBeforeMultiply(%x: !mo.tensor<[8, 128], bf16>, %w: !mo.tensor<[128], bf16>) -> !mo.tensor<[8, 128], bf16> {
  %y = mo.composite.reduce.rms_norm(%x, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[8, 128], bf16>, !mo.tensor<[128], bf16>) -> !mo.tensor<[8, 128], bf16>
  mo.output %y : !mo.tensor<[8, 128], bf16>
}

// -----

// A weight in another dtype is cast to the one it multiplies in.
// CHECK-LABEL: mo.graph @f32Weight
// CHECK:         mo.rsqrt
// CHECK:         %[[W:.*]] = mo.cast(%arg1) : (!mo.tensor<[128], f32>) -> !mo.tensor<[128], bf16>
// CHECK:         mo.add(%[[W]], %{{.*}}) : !mo.tensor<[128], bf16>
mo.graph @f32Weight(%x: !mo.tensor<[8, 128], bf16>, %w: !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], bf16> {
  %y = mo.composite.reduce.rms_norm(%x, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[8, 128], bf16>, !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], bf16>
  mo.output %y : !mo.tensor<[8, 128], bf16>
}

// -----

// An f32 input accumulates in f32: no cast. A symbolic axis length comes from
// the shape; a static one is a constant.
// CHECK-LABEL: mo.graph @f32Dynamic
// CHECK-NOT:     mo.cast(%arg0)
// CHECK:         %[[SQ:.*]] = mo.mul(%arg0, %arg0) : !mo.tensor<[M, N], f32>
// CHECK:         %[[SUM:.*]] = mo.reduce.add(%[[SQ]]) : (!mo.tensor<[M, N], f32>) -> !mo.tensor<[M, 1], f32>
// CHECK:         %[[D:.*]] = mosh.param.to_value = <N>
// CHECK:         %[[DT:.*]] = mo.index.to_tensor(%[[D]])
// CHECK:         %[[DF:.*]] = mo.cast(%[[DT]]) : {{.*}} -> !mo.tensor<[], f32>
// CHECK:         %[[DB:.*]] = mo.static.broadcast_to(%[[DF]]) : !mo.tensor<[], f32> -> !mo.tensor<[M, 1], f32>
// CHECK:         mo.div(%[[SUM]], %[[DB]]) : !mo.tensor<[M, 1], f32>
mo.graph @f32Dynamic<M, N>(%x: !mo.tensor<[M, N], f32>, %w: !mo.tensor<[N], f32>) -> !mo.tensor<[M, N], f32> {
  %y = mo.composite.reduce.rms_norm(%x, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[M, N], f32>, !mo.tensor<[N], f32>) -> !mo.tensor<[M, N], f32>
  mo.output %y : !mo.tensor<[M, N], f32>
}

// -----

// On a GPU, a symbolic axis length is read on the host and copied over.
// CHECK-LABEL: mo.graph @gpuDynamic
// CHECK:         %[[D:.*]] = mosh.param.to_value = <N>
// CHECK:         %[[DT:.*]] = mo.index.to_tensor(%[[D]]) -> !mo.tensor<[], ui64>
// CHECK:         %[[DF:.*]] = mo.cast(%[[DT]]) : (!mo.tensor<[], ui64>) -> !mo.tensor<[], f32>
// CHECK:         %[[C:.*]] = mo.chain.create()
// CHECK:         %[[DG:.*]], %{{.*}} = mo.transfer[%[[C]]] %[[DF]] : !mo.tensor<[], f32> to <"gpu", 0>
// CHECK:         mo.static.broadcast_to(%[[DG]]) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[M, 1], f32, gpu:0>
mo.graph @gpuDynamic<M, N>(%x: !mo.tensor<[M, N], f32, gpu:0>, %w: !mo.tensor<[N], f32, gpu:0>) -> !mo.tensor<[M, N], f32, gpu:0> {
  %y = mo.composite.reduce.rms_norm(%x, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[M, N], f32, gpu:0>, !mo.tensor<[N], f32, gpu:0>) -> !mo.tensor<[M, N], f32, gpu:0>
  mo.output %y : !mo.tensor<[M, N], f32, gpu:0>
}

// -----

// A kernel registered only for another device doesn't keep it on this one.
#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>
mogg.kernel.decl @tpu_rms_norm registration = "mo.composite.reduce.rms_norm" device_info = <"tpu", "", "", ""> execute = #kf struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @otherDevice
// CHECK-NOT:     mo.composite.reduce.rms_norm
// CHECK:         mo.reduce.add
mo.graph @otherDevice(%x: !mo.tensor<[8, 128], f32>, %w: !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], f32> {
  %y = mo.composite.reduce.rms_norm(%x, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[8, 128], f32>, !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// -----

// A target that registers a kernel for rms_norm keeps running it.
#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>
mogg.kernel.decl @plugin_rms_norm registration = "mo.composite.reduce.rms_norm" device_info = <"", "", "", ""> execute = #kf struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @implemented
// CHECK:         mo.composite.reduce.rms_norm
// CHECK-NOT:     mo.reduce.add
mo.graph @implemented(%x: !mo.tensor<[8, 128], f32>, %w: !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], f32> {
  %y = mo.composite.reduce.rms_norm(%x, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[8, 128], f32>, !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}
