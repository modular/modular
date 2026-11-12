// The fusion replaces four ops with one per boundary, so the composite has to
// carry their sources: a warning or a Python stack trace that resolves through
// it would otherwise name the allreduce alone (QA-C1). Locations are not part
// of the declarative assembly format, so the input ops are written generically
// to give each one a name.
// RUN: mo-opt %s --mo-pattern-fusion --mlir-print-debuginfo | FileCheck %s

mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_fused_location(
  %in0: !mo.tensor<[4, 8], bf16, gpu:0>, %in1: !mo.tensor<[4, 8], bf16, gpu:1>,
  %sig0: !mo.buffer<[64], ui8, gpu:0>, %sig1: !mo.buffer<[64], ui8, gpu:1>,
  %res0: !mo.tensor<[4, 8], bf16, gpu:0>, %res1: !mo.tensor<[4, 8], bf16, gpu:1>,
  %gamma0: !mo.tensor<[8], bf16, gpu:0>, %gamma1: !mo.tensor<[8], bf16, gpu:1>,
  %scale_ub: !mo.tensor<[], f32>,
  %c: !mo.chain
) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>,
      !mo.chain) {
  %allred:2, %out_chain = "mo.distributed.allreduce.sum"(%in0, %in1, %sig0, %sig1, %c) : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.buffer<[64], ui8, gpu:0>, !mo.buffer<[64], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.chain) loc("allreduce")
  %add0 = "mo.add"(%res0, %allred#0) : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:0>) -> !mo.tensor<[4, 8], bf16, gpu:0> loc("add0")
  %norm0 = "mo.composite.reduce.rms_norm"(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[8], bf16, gpu:0>) -> !mo.tensor<[4, 8], bf16, gpu:0> loc("norm0")
  %q0:2 = "mo.composite.quantize_dynamic_scaled_fp8"(%norm0, %scale_ub) {group_size_or_per_token = 4 : index} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[2, 4], bf16, gpu:0>) loc("quantize0")
  %add1 = "mo.add"(%res1, %allred#1) : (!mo.tensor<[4, 8], bf16, gpu:1>, !mo.tensor<[4, 8], bf16, gpu:1>) -> !mo.tensor<[4, 8], bf16, gpu:1> loc("add1")
  %norm1 = "mo.composite.reduce.rms_norm"(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[4, 8], bf16, gpu:1>, !mo.tensor<[8], bf16, gpu:1>) -> !mo.tensor<[4, 8], bf16, gpu:1> loc("norm1")
  %q1:2 = "mo.composite.quantize_dynamic_scaled_fp8"(%norm1, %scale_ub) {group_size_or_per_token = 4 : index} : (!mo.tensor<[4, 8], bf16, gpu:1>, !mo.tensor<[], f32>) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:1>, !mo.tensor<[2, 4], bf16, gpu:1>) loc("quantize1")
  mo.output %q0#0, %q1#0, %out_chain : !mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>, !mo.chain
}

// CHECK: #[[ALLRED:loc[0-9]+]] = loc("allreduce")
// CHECK: #[[ADD0:loc[0-9]+]] = loc("add0")
// CHECK: #[[NORM0:loc[0-9]+]] = loc("norm0")
// CHECK: #[[QUANT0:loc[0-9]+]] = loc("quantize0")
// CHECK: #[[ADD1:loc[0-9]+]] = loc("add1")
// CHECK: #[[NORM1:loc[0-9]+]] = loc("norm1")
// CHECK: #[[QUANT1:loc[0-9]+]] = loc("quantize1")

// A norm with another reader keeps its quantize, so that op is still in the
// graph and only the allreduce, adds and norms are the composite's sources. Its
// own location stays on the quantize op.
mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_tapped_location(
  %in0: !mo.tensor<[4, 8], bf16, gpu:0>, %in1: !mo.tensor<[4, 8], bf16, gpu:1>,
  %sig0: !mo.buffer<[64], ui8, gpu:0>, %sig1: !mo.buffer<[64], ui8, gpu:1>,
  %res0: !mo.tensor<[4, 8], bf16, gpu:0>, %res1: !mo.tensor<[4, 8], bf16, gpu:1>,
  %gamma0: !mo.tensor<[8], bf16, gpu:0>, %gamma1: !mo.tensor<[8], bf16, gpu:1>,
  %scale_ub: !mo.tensor<[], f32>,
  %c: !mo.chain
) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>,
      !mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>,
      !mo.chain) {
  %allred:2, %out_chain = "mo.distributed.allreduce.sum"(%in0, %in1, %sig0, %sig1, %c) : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.buffer<[64], ui8, gpu:0>, !mo.buffer<[64], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.chain) loc("allreduce_tapped")
  %add0 = "mo.add"(%res0, %allred#0) : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:0>) -> !mo.tensor<[4, 8], bf16, gpu:0> loc("add0_tapped")
  %norm0 = "mo.composite.reduce.rms_norm"(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[8], bf16, gpu:0>) -> !mo.tensor<[4, 8], bf16, gpu:0> loc("norm0_tapped")
  %q0:2 = "mo.composite.quantize_dynamic_scaled_fp8"(%norm0, %scale_ub) {group_size_or_per_token = 4 : index} : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[2, 4], bf16, gpu:0>) loc("quantize0_tapped")
  %add1 = "mo.add"(%res1, %allred#1) : (!mo.tensor<[4, 8], bf16, gpu:1>, !mo.tensor<[4, 8], bf16, gpu:1>) -> !mo.tensor<[4, 8], bf16, gpu:1> loc("add1_tapped")
  %norm1 = "mo.composite.reduce.rms_norm"(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[4, 8], bf16, gpu:1>, !mo.tensor<[8], bf16, gpu:1>) -> !mo.tensor<[4, 8], bf16, gpu:1> loc("norm1_tapped")
  %q1:2 = "mo.composite.quantize_dynamic_scaled_fp8"(%norm1, %scale_ub) {group_size_or_per_token = 4 : index} : (!mo.tensor<[4, 8], bf16, gpu:1>, !mo.tensor<[], f32>) -> (!mo.tensor<[4, 8], f8e4m3fn, gpu:1>, !mo.tensor<[2, 4], bf16, gpu:1>) loc("quantize1_tapped")
  mo.output %q0#0, %q1#0, %norm0, %norm1, %out_chain : !mo.tensor<[4, 8], f8e4m3fn, gpu:0>, !mo.tensor<[4, 8], f8e4m3fn, gpu:1>, !mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:1>, !mo.chain
}

// CHECK: #[[ALLRED_T:loc[0-9]+]] = loc("allreduce_tapped")
// CHECK: #[[ADD0_T:loc[0-9]+]] = loc("add0_tapped")
// CHECK: #[[NORM0_T:loc[0-9]+]] = loc("norm0_tapped")
// CHECK: #[[ADD1_T:loc[0-9]+]] = loc("add1_tapped")
// CHECK: #[[NORM1_T:loc[0-9]+]] = loc("norm1_tapped")

// The fused locations themselves come after the aliases they are built from.
// The first composite carries every op it replaced: the allreduce first, then
// each device's add, norm and quantize in device order.
// CHECK: loc(fused[#[[ALLRED]], #[[ADD0]], #[[NORM0]], #[[QUANT0]], #[[ADD1]], #[[NORM1]], #[[QUANT1]]])
// The tapped chain's composite is the allreduce, adds and norms only.
// CHECK: loc(fused[#[[ALLRED_T]], #[[ADD0_T]], #[[NORM0_T]], #[[ADD1_T]], #[[NORM1_T]]])
