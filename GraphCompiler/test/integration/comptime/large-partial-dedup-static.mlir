// RUN: mo-opt %s | FileCheck %s

// Compile-time sniff test: same shape as `large-transformer-stack-static.mlir`
// (unique input stem, repeated FFN blocks via `mo.call` into a shared
// `mo.graph ... attributes {isSubgraph}`, unique output head) except the
// last of the 12 repeated calls invokes a *sibling* subgraph
// (`ffn_block_wide`, same op sequence but a wider intermediate size) instead
// of `ffn_block`. Kernel-stub dedup (`MergeIdenticalFuncs`) is
// expected to collapse the 11 `ffn_block` call sites together, while
// `ffn_block_wide`'s differently-shaped kernels stand alone -- dedup, but
// not for the whole module. Static shapes throughout -- see
// `large-partial-dedup-dynamic.mlir` for the dynamic-leading-dim sibling.
// See README.md in this directory.

// CHECK: mo.graph @ffn_block
// CHECK: mo.graph @ffn_block_wide
// CHECK: mo.graph @transformer_stack_partial

mo.graph @ffn_block(%x: !mo.tensor<[128, 256], bf16, gpu:0>, %gamma: !mo.tensor<[256], bf16, gpu:0>, %w_gate: !mo.tensor<[256, 1024], bf16, gpu:0>, %w_up: !mo.tensor<[256, 1024], bf16, gpu:0>, %w_down: !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0> isSubgraph {
  %n = mo.reduce.rms_norm(%x, %gamma) {epsilon = #M.dense_array<9.99999974E-6> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %gate = mo.matmul(%n, %w_gate) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>) -> !mo.tensor<[128, 1024], bf16, gpu:0>
  %gate_act = mo.gelu(%gate) : !mo.tensor<[128, 1024], bf16, gpu:0>
  %up = mo.matmul(%n, %w_up) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>) -> !mo.tensor<[128, 1024], bf16, gpu:0>
  %h = mo.mul(%gate_act, %up) : !mo.tensor<[128, 1024], bf16, gpu:0>
  %down = mo.matmul(%h, %w_down) : (!mo.tensor<[128, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %out = mo.add(%x, %down) : !mo.tensor<[128, 256], bf16, gpu:0>
  mo.output %out : !mo.tensor<[128, 256], bf16, gpu:0>
}

mo.graph @ffn_block_wide(%x: !mo.tensor<[128, 256], bf16, gpu:0>, %gamma: !mo.tensor<[256], bf16, gpu:0>, %w_gate: !mo.tensor<[256, 2048], bf16, gpu:0>, %w_up: !mo.tensor<[256, 2048], bf16, gpu:0>, %w_down: !mo.tensor<[2048, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0> isSubgraph {
  %n = mo.reduce.rms_norm(%x, %gamma) {epsilon = #M.dense_array<9.99999974E-6> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %gate = mo.matmul(%n, %w_gate) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256, 2048], bf16, gpu:0>) -> !mo.tensor<[128, 2048], bf16, gpu:0>
  %gate_act = mo.gelu(%gate) : !mo.tensor<[128, 2048], bf16, gpu:0>
  %up = mo.matmul(%n, %w_up) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256, 2048], bf16, gpu:0>) -> !mo.tensor<[128, 2048], bf16, gpu:0>
  %h = mo.mul(%gate_act, %up) : !mo.tensor<[128, 2048], bf16, gpu:0>
  %down = mo.matmul(%h, %w_down) : (!mo.tensor<[128, 2048], bf16, gpu:0>, !mo.tensor<[2048, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %out = mo.add(%x, %down) : !mo.tensor<[128, 256], bf16, gpu:0>
  mo.output %out : !mo.tensor<[128, 256], bf16, gpu:0>
}

mo.graph @transformer_stack_partial(
    %raw: !mo.tensor<[128, 128], bf16, gpu:0>,
    %w_embed: !mo.tensor<[128, 256], bf16, gpu:0>,
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
    %gamma11_wide: !mo.tensor<[256], bf16, gpu:0>,
    %w_gate11_wide: !mo.tensor<[256, 2048], bf16, gpu:0>,
    %w_up11_wide: !mo.tensor<[256, 2048], bf16, gpu:0>,
    %w_down11_wide: !mo.tensor<[2048, 256], bf16, gpu:0>,
    %gamma_final: !mo.tensor<[256], bf16, gpu:0>,
    %w_out: !mo.tensor<[256, 512], bf16, gpu:0>
) -> !mo.tensor<[128, 512], bf16, gpu:0> {
  // Stem: unique input projection, not repeated.
  %stem0 = mo.matmul(%raw, %w_embed) : (!mo.tensor<[128, 128], bf16, gpu:0>, !mo.tensor<[128, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x = mo.gelu(%stem0) : !mo.tensor<[128, 256], bf16, gpu:0>

  // Repeated middle: 11 calls into ffn_block, collapsible via dedup.
  %x1 = mo.call @ffn_block(%x, %gamma0, %w_gate0, %w_up0, %w_down0) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x2 = mo.call @ffn_block(%x1, %gamma1, %w_gate1, %w_up1, %w_down1) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x3 = mo.call @ffn_block(%x2, %gamma2, %w_gate2, %w_up2, %w_down2) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x4 = mo.call @ffn_block(%x3, %gamma3, %w_gate3, %w_up3, %w_down3) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x5 = mo.call @ffn_block(%x4, %gamma4, %w_gate4, %w_up4, %w_down4) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x6 = mo.call @ffn_block(%x5, %gamma5, %w_gate5, %w_up5, %w_down5) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x7 = mo.call @ffn_block(%x6, %gamma6, %w_gate6, %w_up6, %w_down6) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x8 = mo.call @ffn_block(%x7, %gamma7, %w_gate7, %w_up7, %w_down7) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x9 = mo.call @ffn_block(%x8, %gamma8, %w_gate8, %w_up8, %w_down8) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x10 = mo.call @ffn_block(%x9, %gamma9, %w_gate9, %w_up9, %w_down9) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %x11 = mo.call @ffn_block(%x10, %gamma10, %w_gate10, %w_up10, %w_down10) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[256, 1024], bf16, gpu:0>, !mo.tensor<[1024, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>

  // The 12th call: a differently-shaped sibling subgraph, expected to stay
  // its own separate kernel set (the "not for the whole module" part).
  %x12 = mo.call @ffn_block_wide(%x11, %gamma11_wide, %w_gate11_wide, %w_up11_wide, %w_down11_wide) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>, !mo.tensor<[256, 2048], bf16, gpu:0>, !mo.tensor<[256, 2048], bf16, gpu:0>, !mo.tensor<[2048, 256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>

  // Head: unique output projection to a different width, not repeated.
  %n_final = mo.reduce.rms_norm(%x12, %gamma_final) {epsilon = #M.dense_array<9.99999974E-6> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256], bf16, gpu:0>) -> !mo.tensor<[128, 256], bf16, gpu:0>
  %logits = mo.matmul(%n_final, %w_out) : (!mo.tensor<[128, 256], bf16, gpu:0>, !mo.tensor<[256, 512], bf16, gpu:0>) -> !mo.tensor<[128, 512], bf16, gpu:0>

  mo.output %logits : !mo.tensor<[128, 512], bf16, gpu:0>
}
