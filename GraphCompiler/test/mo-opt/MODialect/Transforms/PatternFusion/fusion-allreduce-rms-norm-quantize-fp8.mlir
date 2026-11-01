// `--mo-pattern-fusion` matches an `mo.parallel` whose body is just a
// `mo.bundled.allreduce.sum` followed (outside the parallel) by per-device
// add + rms_norm + quantize_fp8 chains, and rewrites the parallel in one
// step so its body contains the fused
// `mo.composite.bundled.allreduce_add_rms_norm_quant_fp8` op.
// RUN: mo-opt %s --mo-pattern-fusion \
// RUN:   | FileCheck --enable-var-scope %s

// ============================================================================
// Graph 1: allreduce -> add -> rms_norm -> quantize_fp8 (no residual reuse)
// ============================================================================

mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8<total_seq_len>(
  %input0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %input1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %input2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %input3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %sign_buff0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sign_buff1: !mo.buffer<[537919488], ui8, gpu:1>,
  %sign_buff2: !mo.buffer<[537919488], ui8, gpu:2>,
  %sign_buff3: !mo.buffer<[537919488], ui8, gpu:3>,
  %c: !mo.chain,
  %residual0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %residual1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %residual2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %residual3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %gamma0: !mo.tensor<[16384], bf16, gpu:0>,
  %gamma1: !mo.tensor<[16384], bf16, gpu:1>,
  %gamma2: !mo.tensor<[16384], bf16, gpu:2>,
  %gamma3: !mo.tensor<[16384], bf16, gpu:3>
) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>,
      !mo.chain) {
    %input_bundle = mo.tensor.bundle(%input0, %input1, %input2, %input3)
        : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>)
       -> (!mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>)

    %allred_bundle, %out_chain = mo.parallel (%arg, %bch)
        in (%input_bundle : !mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>)
        buffers(%sign_buff0 : !mo.buffer<[537919488], ui8, gpu:0>,
                %sign_buff1 : !mo.buffer<[537919488], ui8, gpu:1>,
                %sign_buff2 : !mo.buffer<[537919488], ui8, gpu:2>,
                %sign_buff3 : !mo.buffer<[537919488], ui8, gpu:3>)
        chain(%c)
        -> (!mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>) {
      %peer0, %peer1, %peer2, %peer3 = mo.bundled.expand(%arg)
          : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
         -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>)
      %out, %ch_out = mo.bundled.allreduce.sum(%peer0, %peer1, %peer2, %peer3,
                                                %sign_buff0, %sign_buff1, %sign_buff2, %sign_buff3, %bch)
          : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
             !mo.buffer<[537919488], ui8, gpu:0>,
             !mo.buffer<[537919488], ui8, gpu:1>,
             !mo.buffer<[537919488], ui8, gpu:2>,
             !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain)
         -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.chain)
      mo.yield %out, %ch_out : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.chain
    }

    %allred0, %allred1, %allred2, %allred3 = mo.tensor.unbundle(%allred_bundle)
        : (!mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>)
       -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>)

    %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>

    // Add + RMSNorm + Quant (GPU:0)
    %add0 = mo.add(%residual0, %allred0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm0 = mo.reduce.rms_norm(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[16384], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm_quant0:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 16384 : index}](%norm0, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)

    // Add + RMSNorm + Quant (GPU:1)
    %add1 = mo.add(%residual1, %allred1) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm1 = mo.reduce.rms_norm(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[16384], bf16, gpu:1>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm_quant1:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 1>, parameters = {group_size_or_per_token = 16384 : index}](%norm1, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[1, total_seq_len], bf16, gpu:1>)

    // Add + RMSNorm + Quant (GPU:2)
    %add2 = mo.add(%residual2, %allred2) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm2 = mo.reduce.rms_norm(%add2, %gamma2) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[16384], bf16, gpu:2>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm_quant2:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 2>, parameters = {group_size_or_per_token = 16384 : index}](%norm2, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[1, total_seq_len], bf16, gpu:2>)

    // Add + RMSNorm + Quant (GPU:3)
    %add3 = mo.add(%residual3, %allred3) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm3 = mo.reduce.rms_norm(%add3, %gamma3) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[16384], bf16, gpu:3>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm_quant3:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 3>, parameters = {group_size_or_per_token = 16384 : index}](%norm3, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.tensor<[1, total_seq_len], bf16, gpu:3>)

    mo.output %norm_quant0#0, %norm_quant1#0, %norm_quant2#0, %norm_quant3#0, %out_chain : !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.chain
}

// The fused parallel takes the original input bundle plus two new bundles
// (residuals and gammas) and produces three output bundles (FP8 outputs,
// scales, post-add residuals) plus the chain.  Its body contains a single
// fused op consuming the captured scale-ub scalar.
// CHECK-LABEL: mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8
// CHECK: mo.tensor.bundle
// CHECK: mo.tensor.bundle
// CHECK: mo.tensor.bundle
// CHECK: mo.parallel
// CHECK:   mo.bundled.expand
// CHECK:   mo.composite.bundled.allreduce_add_rms_norm_quant_fp8
// CHECK-SAME: {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>}
// CHECK:   mo.yield
// CHECK-NOT: mo.bundled.allreduce.sum
// CHECK-NOT: mo.reduce.rms_norm
// CHECK-NOT: mo.quantize_dynamic_scaled_float8

// ============================================================================
// Graph 2: same chain but the residual (add result) has a second use outside
// ============================================================================

mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_residual_use<total_seq_len>(
  %input0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %input1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %input2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %input3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %sign_buff0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sign_buff1: !mo.buffer<[537919488], ui8, gpu:1>,
  %sign_buff2: !mo.buffer<[537919488], ui8, gpu:2>,
  %sign_buff3: !mo.buffer<[537919488], ui8, gpu:3>,
  %c: !mo.chain,
  %residual0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %residual1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %residual2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %residual3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %gamma0: !mo.tensor<[16384], bf16, gpu:0>,
  %gamma1: !mo.tensor<[16384], bf16, gpu:1>,
  %gamma2: !mo.tensor<[16384], bf16, gpu:2>,
  %gamma3: !mo.tensor<[16384], bf16, gpu:3>,
  %other0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %other1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %other2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %other3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>,
      !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
      !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
      !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
      !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
      !mo.chain) {
    %input_bundle = mo.tensor.bundle(%input0, %input1, %input2, %input3)
        : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>)
       -> (!mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>)

    %allred_bundle, %out_chain = mo.parallel (%arg, %bch)
        in (%input_bundle : !mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>)
        buffers(%sign_buff0 : !mo.buffer<[537919488], ui8, gpu:0>,
                %sign_buff1 : !mo.buffer<[537919488], ui8, gpu:1>,
                %sign_buff2 : !mo.buffer<[537919488], ui8, gpu:2>,
                %sign_buff3 : !mo.buffer<[537919488], ui8, gpu:3>)
        chain(%c)
        -> (!mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>) {
      %peer0, %peer1, %peer2, %peer3 = mo.bundled.expand(%arg)
          : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
         -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>)
      %out, %ch_out = mo.bundled.allreduce.sum(%peer0, %peer1, %peer2, %peer3,
                                                %sign_buff0, %sign_buff1, %sign_buff2, %sign_buff3, %bch)
          : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
             !mo.buffer<[537919488], ui8, gpu:0>,
             !mo.buffer<[537919488], ui8, gpu:1>,
             !mo.buffer<[537919488], ui8, gpu:2>,
             !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain)
         -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.chain)
      mo.yield %out, %ch_out : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.chain
    }

    %allred0, %allred1, %allred2, %allred3 = mo.tensor.unbundle(%allred_bundle)
        : (!mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>)
       -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>)

    %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>

    // Add + RMSNorm + Quant + Residual use (GPU:0)
    %add0 = mo.add(%residual0, %allred0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm0 = mo.reduce.rms_norm(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[16384], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm_quant0:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 16384 : index}](%norm0, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
    %out0 = mo.add(%other0, %add0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>

    // Add + RMSNorm + Quant + Residual use (GPU:1)
    %add1 = mo.add(%residual1, %allred1) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm1 = mo.reduce.rms_norm(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[16384], bf16, gpu:1>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm_quant1:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 1>, parameters = {group_size_or_per_token = 16384 : index}](%norm1, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[1, total_seq_len], bf16, gpu:1>)
    %out1 = mo.add(%other1, %add1) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>

    // Add + RMSNorm + Quant + Residual use (GPU:2)
    %add2 = mo.add(%residual2, %allred2) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm2 = mo.reduce.rms_norm(%add2, %gamma2) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[16384], bf16, gpu:2>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm_quant2:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 2>, parameters = {group_size_or_per_token = 16384 : index}](%norm2, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[1, total_seq_len], bf16, gpu:2>)
    %out2 = mo.add(%other2, %add2) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>

    // Add + RMSNorm + Quant + Residual use (GPU:3)
    %add3 = mo.add(%residual3, %allred3) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm3 = mo.reduce.rms_norm(%add3, %gamma3) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[16384], bf16, gpu:3>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm_quant3:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 3>, parameters = {group_size_or_per_token = 16384 : index}](%norm3, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.tensor<[1, total_seq_len], bf16, gpu:3>)
    %out3 = mo.add(%other3, %add3) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>

    mo.output %norm_quant0#0, %norm_quant1#0, %norm_quant2#0, %norm_quant3#0, %out0, %out1, %out2, %out3, %out_chain : !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain
}

// Same fused shape as graph 1, but the post-add residual is read by the
// external `mo.add(%other, %add)` adds.  The fused parallel exposes the
// post-add residuals via its third bundle result; the surviving external
// adds read it through `mo.tensor.unbundle`.
// CHECK-LABEL: mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_residual_use
// CHECK: mo.parallel
// CHECK:   mo.bundled.expand
// CHECK:   mo.composite.bundled.allreduce_add_rms_norm_quant_fp8
// CHECK:   mo.yield
// CHECK: mo.tensor.unbundle
// CHECK: mo.add
// CHECK-NOT: mo.bundled.allreduce.sum
// CHECK-NOT: mo.reduce.rms_norm
// CHECK-NOT: mo.quantize_dynamic_scaled_float8

// ============================================================================
// Graph 3: graph 1 with an `amax_floor` on every quantize op
// ============================================================================

mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_amax_floor<total_seq_len>(
  %input0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %input1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %input2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %input3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %sign_buff0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sign_buff1: !mo.buffer<[537919488], ui8, gpu:1>,
  %sign_buff2: !mo.buffer<[537919488], ui8, gpu:2>,
  %sign_buff3: !mo.buffer<[537919488], ui8, gpu:3>,
  %c: !mo.chain,
  %residual0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %residual1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %residual2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %residual3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %gamma0: !mo.tensor<[16384], bf16, gpu:0>,
  %gamma1: !mo.tensor<[16384], bf16, gpu:1>,
  %gamma2: !mo.tensor<[16384], bf16, gpu:2>,
  %gamma3: !mo.tensor<[16384], bf16, gpu:3>
) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>,
      !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>,
      !mo.chain) {
    %input_bundle = mo.tensor.bundle(%input0, %input1, %input2, %input3)
        : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>)
       -> (!mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>)

    %allred_bundle, %out_chain = mo.parallel (%arg, %bch)
        in (%input_bundle : !mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>)
        buffers(%sign_buff0 : !mo.buffer<[537919488], ui8, gpu:0>,
                %sign_buff1 : !mo.buffer<[537919488], ui8, gpu:1>,
                %sign_buff2 : !mo.buffer<[537919488], ui8, gpu:2>,
                %sign_buff3 : !mo.buffer<[537919488], ui8, gpu:3>)
        chain(%c)
        -> (!mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>) {
      %peer0, %peer1, %peer2, %peer3 = mo.bundled.expand(%arg)
          : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
         -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>)
      %out, %ch_out = mo.bundled.allreduce.sum(%peer0, %peer1, %peer2, %peer3,
                                                %sign_buff0, %sign_buff1, %sign_buff2, %sign_buff3, %bch)
          : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
             !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
             !mo.buffer<[537919488], ui8, gpu:0>,
             !mo.buffer<[537919488], ui8, gpu:1>,
             !mo.buffer<[537919488], ui8, gpu:2>,
             !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain)
         -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.chain)
      mo.yield %out, %ch_out : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.chain
    }

    %allred0, %allred1, %allred2, %allred3 = mo.tensor.unbundle(%allred_bundle)
        : (!mo.bundle<[total_seq_len, 16384], bf16, gpu:0-3>)
       -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
           !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>)

    %scale_ub = mo.constant <#M.dense_array<1.200000e+03> : tensor<f32>> : !mo.tensor<[], f32>

    // Add + RMSNorm + Quant (GPU:0)
    %add0 = mo.add(%residual0, %allred0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm0 = mo.reduce.rms_norm(%add0, %gamma0) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[16384], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm_quant0:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 0>, parameters = {amax_floor = "0.0001", group_size_or_per_token = 16384 : index}](%norm0, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)

    // Add + RMSNorm + Quant (GPU:1)
    %add1 = mo.add(%residual1, %allred1) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm1 = mo.reduce.rms_norm(%add1, %gamma1) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[16384], bf16, gpu:1>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm_quant1:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 1>, parameters = {amax_floor = "0.0001", group_size_or_per_token = 16384 : index}](%norm1, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[1, total_seq_len], bf16, gpu:1>)

    // Add + RMSNorm + Quant (GPU:2)
    %add2 = mo.add(%residual2, %allred2) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm2 = mo.reduce.rms_norm(%add2, %gamma2) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[16384], bf16, gpu:2>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>
    %norm_quant2:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 2>, parameters = {amax_floor = "0.0001", group_size_or_per_token = 16384 : index}](%norm2, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[1, total_seq_len], bf16, gpu:2>)

    // Add + RMSNorm + Quant (GPU:3)
    %add3 = mo.add(%residual3, %allred3) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm3 = mo.reduce.rms_norm(%add3, %gamma3) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[16384], bf16, gpu:3>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>
    %norm_quant3:2 = mo.custom [symbol = "mo.quantize_dynamic_scaled_float8", device = #M.device_ref<"gpu", 3>, parameters = {amax_floor = "0.0001", group_size_or_per_token = 16384 : index}](%norm3, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.tensor<[1, total_seq_len], bf16, gpu:3>)

    mo.output %norm_quant0#0, %norm_quant1#0, %norm_quant2#0, %norm_quant3#0, %out_chain : !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:2>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:3>, !mo.chain
}

// The fused kernel derives the scale with no amax floor, so a quantize op that
// carries one keeps the whole chain unfused.
// CHECK-LABEL: mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_amax_floor
// CHECK-NOT: mo.composite.bundled.allreduce_add_rms_norm_quant_fp8
// CHECK: mo.bundled.allreduce.sum
// CHECK-NOT: mo.composite.bundled.allreduce_add_rms_norm_quant_fp8
// CHECK-COUNT-4: symbol = "mo.quantize_dynamic_scaled_float8"{{.*}}amax_floor = "0.0001"
