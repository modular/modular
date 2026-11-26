// `--mo-pattern-fusion` pulls an allreduce into the scaled sandwich-norm
// boundary after it on every device,
//
//   rms_norm(mul(add(rms_norm(allreduce_sum(x)), r), broadcast(s)))
//
// giving one `mo.composite.distributed.allreduce_rms_norm_fused_scaled_residual_add`,
// where a kernel for it resolves on every device.
// RUN: mo-opt %s --split-input-file --mo-pattern-fusion \
// RUN:   | FileCheck --enable-var-scope %s

!pred = !kgen.generator<!lit.generator<<"label": string, "api": string, "arch": string, "model": string>scalar<bool>>>
#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>
module attributes {mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100">}} {
mogg.kernel.decl @cuda_allreduce_rms_norm_fused_scaled_residual_add registration = "mo.composite.distributed.allreduce_rms_norm_fused_scaled_residual_add" platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred execute = #kf struct_params = [] is_user_defined = false

// Every device carries the whole chain, so one fused op replaces the
// collective and every norm, add and multiply after it. The op takes the
// scale before its broadcast.
// CHECK-LABEL: mo.graph @boundary
// CHECK:         mo.composite.distributed.allreduce_rms_norm_fused_scaled_residual_add
// CHECK-SAME:      !mo.tensor<[1], bf16, gpu:0>, !mo.tensor<[1], bf16, gpu:1>, !mo.chain) ->
// CHECK-NOT:     mo.distributed.allreduce.sum
// CHECK-NOT:     mo.composite.reduce.rms_norm
// CHECK-NOT:     mo.mul
mo.graph @boundary(
  %x0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %x1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %sig0: !mo.buffer<[1], ui8, gpu:0>,
  %sig1: !mo.buffer<[1], ui8, gpu:1>,
  %r0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %r1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %g0a: !mo.tensor<[128], bf16, gpu:0>,
  %g0b: !mo.tensor<[128], bf16, gpu:0>,
  %g1a: !mo.tensor<[128], bf16, gpu:1>,
  %g1b: !mo.tensor<[128], bf16, gpu:1>,
  %s0: !mo.tensor<[1], bf16, gpu:0>,
  %s1: !mo.tensor<[1], bf16, gpu:1>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>) {
  %ch = mo.chain.create()
  %ar:2, %arch = mo.distributed.allreduce.sum(%x0, %x1, %sig0, %sig1, %ch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %n0a = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %add0 = mo.add(%r0, %n0a) : !mo.tensor<[8, 128], bf16, gpu:0>
  %b0 = mo.static.broadcast_to(%s0) : !mo.tensor<[1], bf16, gpu:0> -> !mo.tensor<[8, 128], bf16, gpu:0>
  %sc0 = mo.mul(%add0, %b0) : !mo.tensor<[8, 128], bf16, gpu:0>
  %n0b = mo.composite.reduce.rms_norm(%sc0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %n1a = mo.composite.reduce.rms_norm(%ar#1, %g1a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %add1 = mo.add(%r1, %n1a) : !mo.tensor<[8, 128], bf16, gpu:1>
  %b1 = mo.static.broadcast_to(%s1) : !mo.tensor<[1], bf16, gpu:1> -> !mo.tensor<[8, 128], bf16, gpu:1>
  %sc1 = mo.mul(%add1, %b1) : !mo.tensor<[8, 128], bf16, gpu:1>
  %n1b = mo.composite.reduce.rms_norm(%sc1, %g1b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  mo.output %n0b, %sc0, %n1b, %sc1 : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>
}

// The scale is computed after the collective but doesn't read it, so the
// matcher moves its producer above the collective and fuses.
// CHECK-LABEL: mo.graph @scaleAfterCollective
// CHECK:         mo.add
// CHECK:         mo.add
// CHECK:         mo.composite.distributed.allreduce_rms_norm_fused_scaled_residual_add
// CHECK-NOT:     mo.distributed.allreduce.sum
// CHECK-NOT:     mo.composite.reduce.rms_norm
mo.graph @scaleAfterCollective(
  %x0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %x1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %sig0: !mo.buffer<[1], ui8, gpu:0>,
  %sig1: !mo.buffer<[1], ui8, gpu:1>,
  %r0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %r1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %g0a: !mo.tensor<[128], bf16, gpu:0>,
  %g0b: !mo.tensor<[128], bf16, gpu:0>,
  %g1a: !mo.tensor<[128], bf16, gpu:1>,
  %g1b: !mo.tensor<[128], bf16, gpu:1>,
  %s0: !mo.tensor<[1], bf16, gpu:0>,
  %s1: !mo.tensor<[1], bf16, gpu:1>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>) {
  %ch = mo.chain.create()
  %ar:2, %arch = mo.distributed.allreduce.sum(%x0, %x1, %sig0, %sig1, %ch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %s0x2 = mo.add(%s0, %s0) : !mo.tensor<[1], bf16, gpu:0>
  %n0a = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %add0 = mo.add(%r0, %n0a) : !mo.tensor<[8, 128], bf16, gpu:0>
  %b0 = mo.static.broadcast_to(%s0x2) : !mo.tensor<[1], bf16, gpu:0> -> !mo.tensor<[8, 128], bf16, gpu:0>
  %sc0 = mo.mul(%add0, %b0) : !mo.tensor<[8, 128], bf16, gpu:0>
  %n0b = mo.composite.reduce.rms_norm(%sc0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %s1x2 = mo.add(%s1, %s1) : !mo.tensor<[1], bf16, gpu:1>
  %n1a = mo.composite.reduce.rms_norm(%ar#1, %g1a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %add1 = mo.add(%r1, %n1a) : !mo.tensor<[8, 128], bf16, gpu:1>
  %b1 = mo.static.broadcast_to(%s1x2) : !mo.tensor<[1], bf16, gpu:1> -> !mo.tensor<[8, 128], bf16, gpu:1>
  %sc1 = mo.mul(%add1, %b1) : !mo.tensor<[8, 128], bf16, gpu:1>
  %n1b = mo.composite.reduce.rms_norm(%sc1, %g1b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  mo.output %n0b, %sc0, %n1b, %sc1 : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>
}

// One op carries every device's operands, so when one device's chain
// doesn't match, the collective stays.
// CHECK-LABEL: mo.graph @partialChain
// CHECK:         mo.distributed.allreduce.sum
// CHECK-NOT:     = mo.composite.distributed.allreduce_rms_norm_fused_scaled_residual_add(
mo.graph @partialChain(
  %x0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %x1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %sig0: !mo.buffer<[1], ui8, gpu:0>,
  %sig1: !mo.buffer<[1], ui8, gpu:1>,
  %r0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %g0a: !mo.tensor<[128], bf16, gpu:0>,
  %g0b: !mo.tensor<[128], bf16, gpu:0>,
  %s0: !mo.tensor<[1], bf16, gpu:0>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>) {
  %ch = mo.chain.create()
  %ar:2, %arch = mo.distributed.allreduce.sum(%x0, %x1, %sig0, %sig1, %ch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %n0a = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %add0 = mo.add(%r0, %n0a) : !mo.tensor<[8, 128], bf16, gpu:0>
  %b0 = mo.static.broadcast_to(%s0) : !mo.tensor<[1], bf16, gpu:0> -> !mo.tensor<[8, 128], bf16, gpu:0>
  %sc0 = mo.mul(%add0, %b0) : !mo.tensor<[8, 128], bf16, gpu:0>
  %n0b = mo.composite.reduce.rms_norm(%sc0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  mo.output %n0b, %ar#1 : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>
}
}

// -----

module attributes {mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100">}} {

// No kernel is registered for the composite, so the collective stays.
// CHECK-LABEL: mo.graph @unregistered
// CHECK:         mo.distributed.allreduce.sum
// CHECK-NOT:     = mo.composite.distributed.allreduce_rms_norm_fused_scaled_residual_add(
mo.graph @unregistered(
  %x0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %x1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %sig0: !mo.buffer<[1], ui8, gpu:0>,
  %sig1: !mo.buffer<[1], ui8, gpu:1>,
  %r0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %r1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %g0a: !mo.tensor<[128], bf16, gpu:0>,
  %g0b: !mo.tensor<[128], bf16, gpu:0>,
  %g1a: !mo.tensor<[128], bf16, gpu:1>,
  %g1b: !mo.tensor<[128], bf16, gpu:1>,
  %s0: !mo.tensor<[1], bf16, gpu:0>,
  %s1: !mo.tensor<[1], bf16, gpu:1>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>) {
  %ch = mo.chain.create()
  %ar:2, %arch = mo.distributed.allreduce.sum(%x0, %x1, %sig0, %sig1, %ch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %n0a = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %add0 = mo.add(%r0, %n0a) : !mo.tensor<[8, 128], bf16, gpu:0>
  %b0 = mo.static.broadcast_to(%s0) : !mo.tensor<[1], bf16, gpu:0> -> !mo.tensor<[8, 128], bf16, gpu:0>
  %sc0 = mo.mul(%add0, %b0) : !mo.tensor<[8, 128], bf16, gpu:0>
  %n0b = mo.composite.reduce.rms_norm(%sc0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %n1a = mo.composite.reduce.rms_norm(%ar#1, %g1a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %add1 = mo.add(%r1, %n1a) : !mo.tensor<[8, 128], bf16, gpu:1>
  %b1 = mo.static.broadcast_to(%s1) : !mo.tensor<[1], bf16, gpu:1> -> !mo.tensor<[8, 128], bf16, gpu:1>
  %sc1 = mo.mul(%add1, %b1) : !mo.tensor<[8, 128], bf16, gpu:1>
  %n1b = mo.composite.reduce.rms_norm(%sc1, %g1b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  mo.output %n0b, %sc0, %n1b, %sc1 : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>
}
}
