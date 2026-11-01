// RUN: mo-opt %s | FileCheck %s

// Compile-time sniff test: a genuinely larger "part of an LLM" -- not just
// one block copy-pasted N times, but a unique input stem (matmul + gelu),
// a 12x-repeated FFN block (RMSNorm -> gated MLP -> residual) invoked via
// `mo.call` into a shared `mo.graph ... attributes {isSubgraph}` (the
// dialect's own mechanism for "the same pattern repeated" -- no hand-copied
// IR), and a unique output head (RMSNorm -> projection to a different
// output width). Dynamic leading (sequence) dimension `D0` throughout -- see
// `large-transformer-stack-static.mlir` for the all-static sibling. Kernel-
// stub dedup (`MergeIdenticalFuncs`) is expected to collapse the 12
// subgraph-inlined call sites down to one kernel per distinct op in the
// block, while the stem and head (structurally unlike the block and unlike
// each other) stay separate. See README.md in this directory.

// CHECK: mo.graph @ffn_block
// CHECK: mo.graph @transformer_stack

mo.graph @ffn_block<D0>(%x: !mo.tensor<[D0, 256], bf16, gpu:0>, %gamma: !mo.tensor<[256], bf16, gpu:0>, %w_gate: !mo.tensor<[256, 1024], bf16, gpu:0>, %w_up: !mo.tensor<[256, 1024], bf16, gpu:0>, %w_down: !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0> isSubgraph {
  %n = mo.reduce.rms_norm(%x, %gamma) {epsilon = #M.dense_array<9.99999974E-6> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %gate = mo.matmul(%n, %w_gate) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>) -> !mo.tensor<[D0, 1024], bf16, gpu:0>
  %gate_act = mo.gelu(%gate) : !mo.tensor<[D0, 1024], bf16, gpu:0>
  %up = mo.matmul(%n, %w_up) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>) -> !mo.tensor<[D0, 1024], bf16, gpu:0>
  %h = mo.mul(%gate_act, %up) : !mo.tensor<[D0, 1024], bf16, gpu:0>
  %down = mo.matmul(%h, %w_down) : (!mo.tensor<[D0, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %out = mo.add(%x, %down) : !mo.tensor<[D0, 256], bf16, gpu:0>
  mo.output %out : !mo.tensor<[D0, 256], bf16, gpu:0>
}

mo.graph @transformer_stack<D0>(
    %raw: !mo.tensor<[D0, 128], bf16, gpu:0>,
    %w_embed: !mo.tensor<[D0, 256], bf16, gpu:0>,
    %gamma0: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate0: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up0: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down0: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma1: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate1: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up1: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down1: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma2: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate2: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up2: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down2: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma3: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate3: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up3: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down3: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma4: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate4: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up4: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down4: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma5: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate5: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up5: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down5: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma6: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate6: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up6: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down6: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma7: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate7: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up7: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down7: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma8: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate8: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up8: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down8: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma9: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate9: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up9: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down9: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma10: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate10: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up10: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down10: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma11: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate11: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_up11: !mo.tensor<[256, 1024], bf16, gpu:0>,
    %w_down11: !mo.tensor<[1024, 256], bf16, gpu:0>,
    %gamma_final: !mo.tensor<[256], bf16, gpu:0>,
    %w_out: !mo.tensor<[256, 512], bf16, gpu:0>
) -> !mo.tensor<[D0, 512], bf16, gpu:0> {
  // Stem: unique input projection, not repeated.
  %stem0 = mo.matmul(%raw, %w_embed) : (!mo.tensor<[D0, 128], bf16, gpu:0>, !mo.tensor<[D0, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x = mo.gelu(%stem0) : !mo.tensor<[D0, 256], bf16, gpu:0>

  // Repeated middle: 12 calls into the shared ffn_block subgraph.
  %x1 = mo.call @ffn_block(%x, %gamma0, %w_gate0, %w_up0, %w_down0) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x2 = mo.call @ffn_block(%x1, %gamma1, %w_gate1, %w_up1, %w_down1) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x3 = mo.call @ffn_block(%x2, %gamma2, %w_gate2, %w_up2, %w_down2) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x4 = mo.call @ffn_block(%x3, %gamma3, %w_gate3, %w_up3, %w_down3) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x5 = mo.call @ffn_block(%x4, %gamma4, %w_gate4, %w_up4, %w_down4) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x6 = mo.call @ffn_block(%x5, %gamma5, %w_gate5, %w_up5, %w_down5) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x7 = mo.call @ffn_block(%x6, %gamma6, %w_gate6, %w_up6, %w_down6) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x8 = mo.call @ffn_block(%x7, %gamma7, %w_gate7, %w_up7, %w_down7) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x9 = mo.call @ffn_block(%x8, %gamma8, %w_gate8, %w_up8, %w_down8) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x10 = mo.call @ffn_block(%x9, %gamma9, %w_gate9, %w_up9, %w_down9) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x11 = mo.call @ffn_block(%x10, %gamma10, %w_gate10, %w_up10, %w_down10) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %x12 = mo.call @ffn_block(%x11, %gamma11, %w_gate11, %w_up11, %w_down11) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>

  // Head: unique output projection to a different width, not repeated.
  %n_final = mo.reduce.rms_norm(%x12, %gamma_final) {epsilon = #M.dense_array<9.99999974E-6> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %logits = mo.matmul(%n_final, %w_out) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256, 512], bf16, gpu:0>) -> !mo.tensor<[D0, 512], bf16, gpu:0>

  mo.output %logits : !mo.tensor<[D0, 512], bf16, gpu:0>
}
