// RUN: mo-opt %s --gc-pipeline-pre-jit=constant-fold=true -o %t.mlir
// RUN: FileCheck %s < %t.mlir

// Regression test for GEX-3861. The Qwen3.5 RoPE freqs_cis host-to-device
// transfer must be hoisted into the model init region (uploaded once), not
// left in the per-iteration execute region. The de-hoist was introduced by
// #87980 (broadcast-scalar inlining perturbing the co-fused RoPE kernel's
// operand structure so ExecInvariantCodeMotion could no longer prove the
// transfer liftable) and resolved by #88695 (gating elementwise-fusion
// duplication on memory-traffic cost). If the de-hoist returns, a
// host_to_device lands in the @qwen3_5 execute block and CHECK-NOT fails.
//
// The input is a 2-layer reduction of Qwen3.5-9B (1 linear-attention +
// 1 full-attention block); one linear block is required to trigger the
// fusion duplication. Weights are placeholders; no weight data is needed.

// CHECK-LABEL: mgp.model @qwen3_5 :
// CHECK:       } execute
// CHECK-NOT:   host_to_device
// CHECK:       mef.output

// Type aliases keep the per-line and per-call-signature types readable.
!hidden = !mo.tensor<[total_seq_len, 4096], bf16, gpu:0>
!chain = !mo.chain
!kv_cache = !mo.buffer<[total_num_pages, 2, 1, 256, 4, 256], bf16, gpu:0>
!conv_state = !mo.buffer<[max_slots, 8192, 3], bf16, gpu:0>
!recurrent_state = !mo.buffer<[max_slots, 32, 128, 128], bf16, gpu:0>
!freqs = !mo.tensor<[16, 64], f32, gpu:0>

module {
  mo.graph @qwen3_5<total_seq_len, input_row_offsets_len, return_n_logits, total_num_pages, replica_0_batch_size, replica_0_max_num_pages, replica_0_steps_remaining, batch_size, max_slots, vision_merged_seq_len, total_image_tokens>(%arg0: !mo.tensor<[total_seq_len], si64, gpu:0>, %arg1: !mo.tensor<[input_row_offsets_len], ui32, gpu:0>, %arg2: !mo.tensor<[return_n_logits], si64>, %arg3: !mo.buffer<[269484032], ui8, gpu:0>, %arg4: !kv_cache, %arg5: !mo.tensor<[replica_0_batch_size], ui32, gpu:0>, %arg6: !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32, gpu:0>, %arg7: !mo.tensor<[replica_0_steps_remaining, 2], ui32>, %arg8: !mo.tensor<[4], si64>, %arg9: !mo.tensor<[batch_size], ui32, gpu:0>, %arg10: !conv_state, %arg11: !conv_state, %arg12: !conv_state, %arg13: !recurrent_state, %arg14: !recurrent_state, %arg15: !recurrent_state, %arg16: !mo.tensor<[vision_merged_seq_len, 4096], bf16, gpu:0>, %arg17: !mo.tensor<[total_image_tokens], si32, gpu:0>) -> !hidden attributes {_kernel_library_paths = [], argument_names = ["input0", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9", "input10", "input11", "input12", "input13", "input14", "input15", "input16", "input17"], result_names = ["output0"]} {
    %c0_ui32 = mo.constant <#M.dense_array<0> : tensor<ui32>> : !mo.tensor<[], ui32>
    %f32_one = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
    %f32_16 = mo.constant <#M.dense_array<1.600000e+01> : tensor<f32>> : !mo.tensor<[], f32>
    %f32_zero = mo.constant <#M.dense_array<0.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
    %f64_one = mo.constant <#M.dense_array<1.000000e+00> : tensor<f64>> : !mo.tensor<[], f64>
    %rope_theta = mo.constant <#M.dense_array<1.000000e+07> : tensor<f64>> : !mo.tensor<[], f64>
    %f64_two = mo.constant <#M.dense_array<2.000000e+00> : tensor<f64>> : !mo.tensor<[], f64>
    %f64_64 = mo.constant <#M.dense_array<6.400000e+01> : tensor<f64>> : !mo.tensor<[], f64>
    %f64_zero = mo.constant <#M.dense_array<0.000000e+00> : tensor<f64>> : !mo.tensor<[], f64>
    %vocab_size = mo.constant <#M.dense_array<248320> : tensor<si64>> : !mo.tensor<[], si64, gpu:0>
    %c0_si64 = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64, gpu:0>
    %embed_weight = mo.constant.external [name = "embed_tokens.weight", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[248320, 4096], bf16>
    %chain0 = mo.chain.create()
    %embed_sliced = rmo.slice(%embed_weight) [starts = #mosh<ape[0, 0]> : !mosh.ape, stops = #mosh<ape[248320, 4096]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape] : (!mo.tensor<[248320, 4096], bf16>) -> !mo.tensor<[248320, 4096], bf16>
    %embed_dev, %embed_chain = rmo.mo.transfer[%chain0] %embed_sliced : !mo.tensor<[248320, 4096], bf16> to <"gpu", 0>
    %ge_zero = rmo.greater_equal(%arg0, %c0_si64) : (!mo.tensor<[total_seq_len], si64, gpu:0>, !mo.tensor<[], si64, gpu:0>) -> !mo.tensor<[total_seq_len], bool, gpu:0>
    %ge_vocab = rmo.greater_equal(%arg0, %vocab_size) : (!mo.tensor<[total_seq_len], si64, gpu:0>, !mo.tensor<[], si64, gpu:0>) -> !mo.tensor<[total_seq_len], bool, gpu:0>
    %lt_vocab = rmo.mo.not(%ge_vocab) : (!mo.tensor<[total_seq_len], bool, gpu:0>) -> !mo.tensor<[total_seq_len], bool, gpu:0>
    %in_vocab = rmo.and(%ge_zero, %lt_vocab) : (!mo.tensor<[total_seq_len], bool, gpu:0>, !mo.tensor<[total_seq_len], bool, gpu:0>) -> !mo.tensor<[total_seq_len], bool, gpu:0>
    %ids_shifted = rmo.sub(%arg0, %c0_si64) : (!mo.tensor<[total_seq_len], si64, gpu:0>, !mo.tensor<[], si64, gpu:0>) -> !mo.tensor<[total_seq_len], si64, gpu:0>
    %ids_masked = rmo.mul(%ids_shifted, %in_vocab) : (!mo.tensor<[total_seq_len], si64, gpu:0>, !mo.tensor<[total_seq_len], bool, gpu:0>) -> !mo.tensor<[total_seq_len], si64, gpu:0>
    %embeds = rmo.mo.gather(%embed_dev, %ids_masked) [axis = 0] : (!mo.tensor<[248320, 4096], bf16, gpu:0>, !mo.tensor<[total_seq_len], si64, gpu:0>) -> !hidden
    %mask_col = rmo.reshape(%in_vocab) [newShape = #mosh<ape[total_seq_len, 1]> : !mosh.ape] : (!mo.tensor<[total_seq_len], bool, gpu:0>) -> !mo.tensor<[total_seq_len, 1], bool, gpu:0>
    %mask_bf16 = mo.cast(%mask_col) : (!mo.tensor<[total_seq_len, 1], bool, gpu:0>) -> !mo.tensor<[total_seq_len, 1], bf16, gpu:0>
    %embeds_masked = rmo.mul(%embeds, %mask_bf16) : (!hidden, !mo.tensor<[total_seq_len, 1], bf16, gpu:0>) -> !hidden
    %image_idx_col = rmo.reshape(%arg17) [newShape = #mosh<ape[total_image_tokens, 1]> : !mosh.ape] : (!mo.tensor<[total_image_tokens], si32, gpu:0>) -> !mo.tensor<[total_image_tokens, 1], si32, gpu:0>
    %hidden_in = mo.custom [symbol = "mo.scatter_nd.skip_oob_indices", device = #M.device_ref<"gpu", 0>](%embeds_masked, %arg16, %image_idx_col) : (!hidden, !mo.tensor<[vision_merged_seq_len, 4096], bf16, gpu:0>, !mo.tensor<[total_image_tokens, 1], si32, gpu:0>) -> !hidden
    // RoPE cos/sin (freqs_cis) table: built on host, then transferred to the device.
    %dim_range = rmo.mo.range(%f64_zero, %f64_64, %f64_two) : (!mo.tensor<[], f64>, !mo.tensor<[], f64>, !mo.tensor<[], f64>) -> !mo.tensor<[32], f64>
    %dim_ratio = rmo.div(%dim_range, %f64_64) : (!mo.tensor<[32], f64>, !mo.tensor<[], f64>) -> !mo.tensor<[32], f64>
    %theta_pow = rmo.pow(%rope_theta, %dim_ratio) : (!mo.tensor<[], f64>, !mo.tensor<[32], f64>) -> !mo.tensor<[32], f64>
    %inv_freq = rmo.div(%f64_one, %theta_pow) : (!mo.tensor<[], f64>, !mo.tensor<[32], f64>) -> !mo.tensor<[32], f64>
    %inv_freq_f32 = mo.cast(%inv_freq) : (!mo.tensor<[32], f64>) -> !mo.tensor<[32], f32>
    %positions = rmo.mo.range(%f32_zero, %f32_16, %f32_one) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[16], f32>
    %positions_col = rmo.reshape(%positions) [newShape = #mosh<ape[16, 1]> : !mosh.ape] : (!mo.tensor<[16], f32>) -> !mo.tensor<[16, 1], f32>
    %inv_freq_row = rmo.reshape(%inv_freq_f32) [newShape = #mosh<ape[1, 32]> : !mosh.ape] : (!mo.tensor<[32], f32>) -> !mo.tensor<[1, 32], f32>
    %freqs_outer = rmo.mul(%positions_col, %inv_freq_row) : (!mo.tensor<[16, 1], f32>, !mo.tensor<[1, 32], f32>) -> !mo.tensor<[16, 32], f32>
    %cos = rmo.mo.cos(%freqs_outer) : (!mo.tensor<[16, 32], f32>) -> !mo.tensor<[16, 32], f32>
    %sin = rmo.mo.sin(%freqs_outer) : (!mo.tensor<[16, 32], f32>) -> !mo.tensor<[16, 32], f32>
    %cos_e = rmo.reshape(%cos) [newShape = #mosh<ape[16, 32, 1]> : !mosh.ape] : (!mo.tensor<[16, 32], f32>) -> !mo.tensor<[16, 32, 1], f32>
    %sin_e = rmo.reshape(%sin) [newShape = #mosh<ape[16, 32, 1]> : !mosh.ape] : (!mo.tensor<[16, 32], f32>) -> !mo.tensor<[16, 32, 1], f32>
    %cos_sin = rmo.concat(%cos_e, %sin_e) [axis = -1] : (!mo.tensor<[16, 32, 1], f32>, !mo.tensor<[16, 32, 1], f32>) -> !mo.tensor<[16, 32, 2], f32>
    %freqs_cis = rmo.reshape(%cos_sin) [newShape = #mosh<ape[16, 64]> : !mosh.ape] : (!mo.tensor<[16, 32, 2], f32>) -> !mo.tensor<[16, 64], f32>
    %freqs_cis_dev, %freqs_chain = rmo.mo.transfer[%chain0] %freqs_cis : !mo.tensor<[16, 64], f32> to <"gpu", 0>
    %linear:3 = mo.call @qwen3_5_linear_attention_block(%hidden_in, %arg10, %arg13, %arg9, %arg1, %chain0, %chain0) [prefix = "layers.0."] : (!hidden, !conv_state, !recurrent_state, !mo.tensor<[batch_size], ui32, gpu:0>, !mo.tensor<[input_row_offsets_len], ui32, gpu:0>, !chain, !chain) -> (!hidden, !chain, !chain)
    %full:3 = mo.call @qwen3_5_full_attention_block(%linear#0, %c0_ui32, %arg4, %arg5, %arg6, %arg7, %arg8, %freqs_cis_dev, %arg1, %linear#1, %linear#2) [prefix = "layers.3."] : (!hidden, !mo.tensor<[], ui32>, !kv_cache, !mo.tensor<[replica_0_batch_size], ui32, gpu:0>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32, gpu:0>, !mo.tensor<[replica_0_steps_remaining, 2], ui32>, !mo.tensor<[4], si64>, !freqs, !mo.tensor<[input_row_offsets_len], ui32, gpu:0>, !chain, !chain) -> (!hidden, !chain, !chain)
    mo.output %full#0 : !hidden
  } {counter = 4 : i64}
  mo.graph @qwen3_5_linear_attention_block<total_seq_len, max_slots, batch_size, input_row_offsets_len, return_n_logits, total_num_pages, replica_0_batch_size, replica_0_max_num_pages, replica_0_steps_remaining, vision_merged_seq_len, total_image_tokens>(%arg0: !hidden, %arg1: !conv_state, %arg2: !recurrent_state, %arg3: !mo.tensor<[batch_size], ui32, gpu:0>, %arg4: !mo.tensor<[input_row_offsets_len], ui32, gpu:0>, %arg5: !chain, %arg6: !chain) -> (!hidden, !chain, !chain) isSubgraph attributes {_kernel_library_paths = [], argument_names = ["input0", "input1", "input2", "input3", "input4"], result_names = ["output0", "output1", "output2"]} {
    %gate = rmo.mo.silu(%arg0) : (!hidden) -> !hidden
    %out = rmo.add(%arg0, %gate) : (!hidden, !hidden) -> !hidden
    mo.output %out, %arg5, %arg6 : !hidden, !chain, !chain
  } {counter = 144 : i64}
  mo.graph @qwen3_5_full_attention_block<total_seq_len, total_num_pages, replica_0_batch_size, replica_0_max_num_pages, replica_0_steps_remaining, input_row_offsets_len, return_n_logits, batch_size, max_slots, vision_merged_seq_len, total_image_tokens>(%arg0: !hidden, %arg1: !mo.tensor<[], ui32>, %arg2: !kv_cache, %arg3: !mo.tensor<[replica_0_batch_size], ui32, gpu:0>, %arg4: !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32, gpu:0>, %arg5: !mo.tensor<[replica_0_steps_remaining, 2], ui32>, %arg6: !mo.tensor<[4], si64>, %arg7: !freqs, %arg8: !mo.tensor<[input_row_offsets_len], ui32, gpu:0>, %arg9: !chain, %arg10: !chain) -> (!hidden, !chain, !chain) isSubgraph attributes {_kernel_library_paths = [], argument_names = ["input0", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8"], result_names = ["output0", "output1", "output2"]} {
    %freqs_bf16 = mo.cast(%arg7) : (!freqs) -> !mo.tensor<[16, 64], bf16, gpu:0>
    %qkv = rmo.reshape(%arg0) [newShape = #mosh<ape[total_seq_len, 16, 256]> : !mosh.ape] : (!hidden) -> !mo.tensor<[total_seq_len, 16, 256], bf16, gpu:0>
    // Split the legacy packed max_lengths [steps_remaining, 2] into two rank-1
    // [1] scalar tensors (max_prompt_length, max_cache_length) to match the
    // updated mo.fused_qk_rope.ragged.paged operand signature.
    %max_prompt_2d = rmo.slice(%arg5) [starts = #mosh<ape[0, 0]> : !mosh.ape, stops = #mosh<ape[1, 1]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape] : (!mo.tensor<[replica_0_steps_remaining, 2], ui32>) -> !mo.tensor<[1, 1], ui32>
    %max_prompt_length = rmo.reshape(%max_prompt_2d) [newShape = #mosh<ape[1]> : !mosh.ape] : (!mo.tensor<[1, 1], ui32>) -> !mo.tensor<[1], ui32>
    %max_cache_2d = rmo.slice(%arg5) [starts = #mosh<ape[0, 1]> : !mosh.ape, stops = #mosh<ape[1, 2]> : !mosh.ape, steps = #mosh<ape[1, 1]> : !mosh.ape] : (!mo.tensor<[replica_0_steps_remaining, 2], ui32>) -> !mo.tensor<[1, 1], ui32>
    %max_cache_length = rmo.reshape(%max_cache_2d) [newShape = #mosh<ape[1]> : !mosh.ape] : (!mo.tensor<[1, 1], ui32>) -> !mo.tensor<[1], ui32>
    // -1 marks the pages as packed, so the kernel derives the page-to-page
    // distance from the block shape.
    %page_stride = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<-1> : tensor<1xsi64>} : () -> !mo.tensor<[1], si64>
    %roped:2 = mo.custom [symbol = "mo.fused_qk_rope.ragged.paged", device = #M.device_ref<"gpu", 0>, parameters = {cache_dtype = bf16, interleaved = true}](%qkv, %arg8, %arg2, %page_stride, %arg3, %arg4, %max_prompt_length, %max_cache_length, %freqs_bf16, %arg1, %arg10) : (!mo.tensor<[total_seq_len, 16, 256], bf16, gpu:0>, !mo.tensor<[input_row_offsets_len], ui32, gpu:0>, !kv_cache, !mo.tensor<[1], si64>, !mo.tensor<[replica_0_batch_size], ui32, gpu:0>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32, gpu:0>, !mo.tensor<[1], ui32>, !mo.tensor<[1], ui32>, !mo.tensor<[16, 64], bf16, gpu:0>, !mo.tensor<[], ui32>, !chain) -> (!mo.tensor<[total_seq_len, 16, 256], bf16, gpu:0>, !chain)
    %attn_out = rmo.reshape(%roped#0) [newShape = #mosh<ape[total_seq_len, 4096]> : !mosh.ape] : (!mo.tensor<[total_seq_len, 16, 256], bf16, gpu:0>) -> !hidden
    %out = rmo.add(%arg0, %attn_out) : (!hidden, !hidden) -> !hidden
    mo.output %out, %arg9, %roped#1 : !hidden, !chain, !chain
  } {counter = 608 : i64}
}
