// `--mo-pattern-fusion` pulls an allreduce into the sandwich-norm boundary
// after it on every device,
//
//   rms_norm(add(rms_norm(allreduce_sum(x)), r))
//
// giving one `mo.composite.distributed.allreduce_rms_norm_fused_residual_add`,
// where a kernel for it resolves on every device.
// RUN: mo-opt %s --split-input-file --mo-pattern-fusion \
// RUN:   | FileCheck --enable-var-scope %s

!pred = !kgen.generator<!lit.generator<<"label": string, "api": string, "arch": string, "model": string>scalar<bool>>>
#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>
module attributes {mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100">}} {
mogg.kernel.decl @cuda_allreduce_rms_norm_fused_residual_add registration = "mo.composite.distributed.allreduce_rms_norm_fused_residual_add" platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred execute = #kf struct_params = [] is_user_defined = false

// Every device carries the whole chain, so one fused op replaces the
// collective and every norm and add after it.
// CHECK-LABEL: mo.graph @boundary
// CHECK:         mo.composite.distributed.allreduce_rms_norm_fused_residual_add
// CHECK-SAME:      {epsilon = 9.99999997E-7, weight_offset = 1.000000e+00,
// CHECK-NOT:     mo.distributed.allreduce.sum
// CHECK-NOT:     mo.composite.reduce.rms_norm
// CHECK-NOT:     mo.add
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
  %g1b: !mo.tensor<[128], bf16, gpu:1>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>) {
  %ch = mo.chain.create()
  %ar:2, %arch = mo.distributed.allreduce.sum(%x0, %x1, %sig0, %sig1, %ch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %n0a = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %add0 = mo.add(%r0, %n0a) : !mo.tensor<[8, 128], bf16, gpu:0>
  %n0b = mo.composite.reduce.rms_norm(%add0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %n1a = mo.composite.reduce.rms_norm(%ar#1, %g1a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %add1 = mo.add(%r1, %n1a) : !mo.tensor<[8, 128], bf16, gpu:1>
  %n1b = mo.composite.reduce.rms_norm(%add1, %g1b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  mo.output %n0b, %add0, %n1b, %add1 : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>
}

// One op carries every device's operands, so when one device's chain
// doesn't match, the collective stays.
// CHECK-LABEL: mo.graph @partialChain
// CHECK:         mo.distributed.allreduce.sum
// CHECK-NOT:     = mo.composite.distributed.allreduce_rms_norm_fused_residual_add(
mo.graph @partialChain(
  %x0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %x1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %sig0: !mo.buffer<[1], ui8, gpu:0>,
  %sig1: !mo.buffer<[1], ui8, gpu:1>,
  %r0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %g0a: !mo.tensor<[128], bf16, gpu:0>,
  %g0b: !mo.tensor<[128], bf16, gpu:0>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>) {
  %ch = mo.chain.create()
  %ar:2, %arch = mo.distributed.allreduce.sum(%x0, %x1, %sig0, %sig1, %ch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %n0a = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %add0 = mo.add(%r0, %n0a) : !mo.tensor<[8, 128], bf16, gpu:0>
  %n0b = mo.composite.reduce.rms_norm(%add0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  mo.output %n0b, %ar#1 : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>
}

// A second norm of the intermediate doesn't block the fusion: the op
// absorbs the first, and the second reads the intermediate from
// `outResidual`.
// CHECK-LABEL: mo.graph @secondNormConsumer
// CHECK:         %{{.*}}:2, %[[RES:[a-zA-Z0-9_]+]]:2, %{{.*}} = mo.composite.distributed.allreduce_rms_norm_fused_residual_add
// CHECK-NOT:     mo.distributed.allreduce.sum
// CHECK-NOT:     mo.add
// CHECK:         mo.composite.reduce.rms_norm(%[[RES]]#0,
// CHECK:         mo.composite.reduce.rms_norm(%[[RES]]#1,
mo.graph @secondNormConsumer(
  %x0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %x1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %sig0: !mo.buffer<[1], ui8, gpu:0>,
  %sig1: !mo.buffer<[1], ui8, gpu:1>,
  %r0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %r1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %g0a: !mo.tensor<[128], bf16, gpu:0>,
  %g0b: !mo.tensor<[128], bf16, gpu:0>,
  %g0c: !mo.tensor<[128], bf16, gpu:0>,
  %g1a: !mo.tensor<[128], bf16, gpu:1>,
  %g1b: !mo.tensor<[128], bf16, gpu:1>,
  %g1c: !mo.tensor<[128], bf16, gpu:1>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>) {
  %ch = mo.chain.create()
  %ar:2, %arch = mo.distributed.allreduce.sum(%x0, %x1, %sig0, %sig1, %ch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %n0a = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %add0 = mo.add(%r0, %n0a) : !mo.tensor<[8, 128], bf16, gpu:0>
  %n0b = mo.composite.reduce.rms_norm(%add0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %n0c = mo.composite.reduce.rms_norm(%add0, %g0c) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %n1a = mo.composite.reduce.rms_norm(%ar#1, %g1a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %add1 = mo.add(%r1, %n1a) : !mo.tensor<[8, 128], bf16, gpu:1>
  %n1b = mo.composite.reduce.rms_norm(%add1, %g1b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %n1c = mo.composite.reduce.rms_norm(%add1, %g1c) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  mo.output %n0b, %n0c, %n1b, %n1c : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>
}

// The first collective's residual is the second collective's normalized
// output, which doesn't exist where the fused op would go, so only the
// second collective fuses.
// CHECK-LABEL: mo.graph @lateResidual
// CHECK:         mo.distributed.allreduce.sum
// CHECK:         mo.composite.distributed.allreduce_rms_norm_fused_residual_add
// CHECK-NOT:     mo.distributed.allreduce.sum
mo.graph @lateResidual(
  %x0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %x1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %y0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %y1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %sig0: !mo.buffer<[1], ui8, gpu:0>,
  %sig1: !mo.buffer<[1], ui8, gpu:1>,
  %g0a: !mo.tensor<[128], bf16, gpu:0>,
  %g0b: !mo.tensor<[128], bf16, gpu:0>,
  %g0c: !mo.tensor<[128], bf16, gpu:0>,
  %g1a: !mo.tensor<[128], bf16, gpu:1>,
  %g1b: !mo.tensor<[128], bf16, gpu:1>,
  %g1c: !mo.tensor<[128], bf16, gpu:1>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>) {
  %ch = mo.chain.create()
  %ar:2, %arch = mo.distributed.allreduce.sum(%x0, %x1, %sig0, %sig1, %ch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %m0 = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %m1 = mo.composite.reduce.rms_norm(%ar#1, %g1a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %br:2, %brch = mo.distributed.allreduce.sum(%y0, %y1, %sig0, %sig1, %arch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %e0 = mo.composite.reduce.rms_norm(%br#0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %e1 = mo.composite.reduce.rms_norm(%br#1, %g1b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %add0 = mo.add(%m0, %e0) : !mo.tensor<[8, 128], bf16, gpu:0>
  %add1 = mo.add(%m1, %e1) : !mo.tensor<[8, 128], bf16, gpu:1>
  %o0 = mo.composite.reduce.rms_norm(%add0, %g0c) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %o1 = mo.composite.reduce.rms_norm(%add1, %g1c) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  mo.output %o0, %o1 : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>
}

// The composite reduces over every device, so an allreduce over groups of
// devices stays.
// CHECK-LABEL: mo.graph @groupedAllReduce
// CHECK:         mo.distributed.allreduce.sum
// CHECK-NOT:     = mo.composite.distributed.allreduce_rms_norm_fused_residual_add(
mo.graph @groupedAllReduce(
  %x0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %x1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %x2: !mo.tensor<[8, 128], bf16, gpu:2>,
  %x3: !mo.tensor<[8, 128], bf16, gpu:3>,
  %sig0: !mo.buffer<[1], ui8, gpu:0>,
  %sig1: !mo.buffer<[1], ui8, gpu:1>,
  %sig2: !mo.buffer<[1], ui8, gpu:2>,
  %sig3: !mo.buffer<[1], ui8, gpu:3>,
  %r0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %r1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %r2: !mo.tensor<[8, 128], bf16, gpu:2>,
  %r3: !mo.tensor<[8, 128], bf16, gpu:3>,
  %g0a: !mo.tensor<[128], bf16, gpu:0>,
  %g0b: !mo.tensor<[128], bf16, gpu:0>,
  %g1a: !mo.tensor<[128], bf16, gpu:1>,
  %g1b: !mo.tensor<[128], bf16, gpu:1>,
  %g2a: !mo.tensor<[128], bf16, gpu:2>,
  %g2b: !mo.tensor<[128], bf16, gpu:2>,
  %g3a: !mo.tensor<[128], bf16, gpu:3>,
  %g3b: !mo.tensor<[128], bf16, gpu:3>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:2>, !mo.tensor<[8, 128], bf16, gpu:2>, !mo.tensor<[8, 128], bf16, gpu:3>, !mo.tensor<[8, 128], bf16, gpu:3>) {
  %ch = mo.chain.create()
  %ar:4, %arch = mo.distributed.allreduce.sum(%x0, %x1, %x2, %x3, %sig0, %sig1, %sig2, %sig3, %ch) [group_size = 2] : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:2>, !mo.tensor<[8, 128], bf16, gpu:3>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.buffer<[1], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:2>, !mo.tensor<[8, 128], bf16, gpu:3>, !mo.chain)
  %n0a = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %add0 = mo.add(%r0, %n0a) : !mo.tensor<[8, 128], bf16, gpu:0>
  %n0b = mo.composite.reduce.rms_norm(%add0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %n1a = mo.composite.reduce.rms_norm(%ar#1, %g1a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %add1 = mo.add(%r1, %n1a) : !mo.tensor<[8, 128], bf16, gpu:1>
  %n1b = mo.composite.reduce.rms_norm(%add1, %g1b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %n2a = mo.composite.reduce.rms_norm(%ar#2, %g2a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:2>, !mo.tensor<[128], bf16, gpu:2>) -> !mo.tensor<[8, 128], bf16, gpu:2>
  %add2 = mo.add(%r2, %n2a) : !mo.tensor<[8, 128], bf16, gpu:2>
  %n2b = mo.composite.reduce.rms_norm(%add2, %g2b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:2>, !mo.tensor<[128], bf16, gpu:2>) -> !mo.tensor<[8, 128], bf16, gpu:2>
  %n3a = mo.composite.reduce.rms_norm(%ar#3, %g3a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:3>, !mo.tensor<[128], bf16, gpu:3>) -> !mo.tensor<[8, 128], bf16, gpu:3>
  %add3 = mo.add(%r3, %n3a) : !mo.tensor<[8, 128], bf16, gpu:3>
  %n3b = mo.composite.reduce.rms_norm(%add3, %g3b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:3>, !mo.tensor<[128], bf16, gpu:3>) -> !mo.tensor<[8, 128], bf16, gpu:3>
  mo.output %n0b, %add0, %n1b, %add1, %n2b, %add2, %n3b, %add3 : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:2>, !mo.tensor<[8, 128], bf16, gpu:2>, !mo.tensor<[8, 128], bf16, gpu:3>, !mo.tensor<[8, 128], bf16, gpu:3>
}
}

// -----

!pred = !kgen.generator<!lit.generator<<"label": string, "api": string, "arch": string, "model": string>scalar<bool>>>
#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>
module attributes {mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "hip", "gfx942", "AMD Instinct MI300X">}} {
mogg.kernel.decl @cuda_allreduce_rms_norm_fused_residual_add registration = "mo.composite.distributed.allreduce_rms_norm_fused_residual_add" platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred execute = #kf struct_params = [] is_user_defined = false

// The kernel doesn't resolve on HIP, so the collective stays.
// CHECK-LABEL: mo.graph @otherDevice
// CHECK:         mo.distributed.allreduce.sum
// CHECK-NOT:     = mo.composite.distributed.allreduce_rms_norm_fused_residual_add(
mo.graph @otherDevice(
  %x0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %x1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %sig0: !mo.buffer<[1], ui8, gpu:0>,
  %sig1: !mo.buffer<[1], ui8, gpu:1>,
  %r0: !mo.tensor<[8, 128], bf16, gpu:0>,
  %r1: !mo.tensor<[8, 128], bf16, gpu:1>,
  %g0a: !mo.tensor<[128], bf16, gpu:0>,
  %g0b: !mo.tensor<[128], bf16, gpu:0>,
  %g1a: !mo.tensor<[128], bf16, gpu:1>,
  %g1b: !mo.tensor<[128], bf16, gpu:1>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>) {
  %ch = mo.chain.create()
  %ar:2, %arch = mo.distributed.allreduce.sum(%x0, %x1, %sig0, %sig1, %ch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %n0a = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %add0 = mo.add(%r0, %n0a) : !mo.tensor<[8, 128], bf16, gpu:0>
  %n0b = mo.composite.reduce.rms_norm(%add0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %n1a = mo.composite.reduce.rms_norm(%ar#1, %g1a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %add1 = mo.add(%r1, %n1a) : !mo.tensor<[8, 128], bf16, gpu:1>
  %n1b = mo.composite.reduce.rms_norm(%add1, %g1b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  mo.output %n0b, %add0, %n1b, %add1 : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>
}
}

// -----

module attributes {mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100">}} {

// No kernel is registered for the composite, so the collective stays.
// CHECK-LABEL: mo.graph @unregistered
// CHECK:         mo.distributed.allreduce.sum
// CHECK-NOT:     = mo.composite.distributed.allreduce_rms_norm_fused_residual_add(
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
  %g1b: !mo.tensor<[128], bf16, gpu:1>
) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>) {
  %ch = mo.chain.create()
  %ar:2, %arch = mo.distributed.allreduce.sum(%x0, %x1, %sig0, %sig1, %ch) : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.chain)
  %n0a = mo.composite.reduce.rms_norm(%ar#0, %g0a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %add0 = mo.add(%r0, %n0a) : !mo.tensor<[8, 128], bf16, gpu:0>
  %n0b = mo.composite.reduce.rms_norm(%add0, %g0b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>) -> !mo.tensor<[8, 128], bf16, gpu:0>
  %n1a = mo.composite.reduce.rms_norm(%ar#1, %g1a) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  %add1 = mo.add(%r1, %n1a) : !mo.tensor<[8, 128], bf16, gpu:1>
  %n1b = mo.composite.reduce.rms_norm(%add1, %g1b) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<1.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[128], bf16, gpu:1>) -> !mo.tensor<[8, 128], bf16, gpu:1>
  mo.output %n0b, %add0, %n1b, %add1 : !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[8, 128], bf16, gpu:1>, !mo.tensor<[8, 128], bf16, gpu:1>
}
}
