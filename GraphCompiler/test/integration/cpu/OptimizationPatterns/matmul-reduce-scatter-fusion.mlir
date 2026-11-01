// COM: Test matmul + reduce_scatter fusion into a single op.
// RUN: mo-opt %s --mo-pattern-fusion='target-device=gpu' | FileCheck %s

// Two matmuls (one per GPU, transpose_b=true) feeding into reduce_scatter on axis 0.
mo.graph @matmul_reduce_scatter_fusion(
  %a0: !mo.tensor<[1024, 4096], bf16, gpu:0>,
  %a1: !mo.tensor<[1024, 4096], bf16, gpu:1>,
  %b0: !mo.tensor<[2048, 4096], bf16, gpu:0, {layout = #mo.layout<NK>}>,
  %b1: !mo.tensor<[2048, 4096], bf16, gpu:1, {layout = #mo.layout<NK>}>,
  %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sig1: !mo.buffer<[537919488], ui8, gpu:1>)
  -> (!mo.tensor<[512, 2048], bf16, gpu:0>,
      !mo.tensor<[512, 2048], bf16, gpu:1>) {
  %chain_in = mo.chain.create()

  // Matmul on GPU:0 -- A[1024,4096] @ B[2048,4096]^T = C[1024,2048]
  %mm0 = mo.matmul(%a0, %b0) [transpose_b = true] : (!mo.tensor<[1024, 4096], bf16, gpu:0>, !mo.tensor<[2048, 4096], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[1024, 2048], bf16, gpu:0>

  // Matmul on GPU:1 -- A[1024,4096] @ B[2048,4096]^T = C[1024,2048]
  %mm1 = mo.matmul(%a1, %b1) [transpose_b = true] : (!mo.tensor<[1024, 4096], bf16, gpu:1>, !mo.tensor<[2048, 4096], bf16, gpu:1, {layout = #mo.layout<NK>}>) -> !mo.tensor<[1024, 2048], bf16, gpu:1>

  // ReduceScatter on axis 0: each GPU gets 1024/2 = 512 rows.
  %rs:2, %out_chain = mo.distributed.reducescatter.sum(%mm0, %mm1, %sig0, %sig1, %chain_in) [axis = 0, group_size = 2] : (!mo.tensor<[1024, 2048], bf16, gpu:0>, !mo.tensor<[1024, 2048], bf16, gpu:1>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[512, 2048], bf16, gpu:0>, !mo.tensor<[512, 2048], bf16, gpu:1>, !mo.chain)

  mo.output %rs#0, %rs#1 : !mo.tensor<[512, 2048], bf16, gpu:0>, !mo.tensor<[512, 2048], bf16, gpu:1>
}

// CHECK-LABEL: mo.graph @matmul_reduce_scatter_fusion
// CHECK: mo.composite.distributed.matmul_reduce_scatter.sum(
// CHECK-NOT: mo.matmul
// CHECK-NOT: mo.distributed.reducescatter.sum

// -----

// Asymmetric pattern: device 0 has add(residual, matmul), device 1 has bare matmul.
// This matches the DeepseekV3/KimiK2.5 TP+EP attention output pattern.
mo.graph @matmul_reduce_scatter_residual_fusion(
  %a0: !mo.tensor<[1024, 4096], bf16, gpu:0>,
  %a1: !mo.tensor<[1024, 4096], bf16, gpu:1>,
  %b0: !mo.tensor<[2048, 4096], bf16, gpu:0, {layout = #mo.layout<NK>}>,
  %b1: !mo.tensor<[2048, 4096], bf16, gpu:1, {layout = #mo.layout<NK>}>,
  %residual: !mo.tensor<[1024, 2048], bf16, gpu:0>,
  %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sig1: !mo.buffer<[537919488], ui8, gpu:1>)
  -> (!mo.tensor<[512, 2048], bf16, gpu:0>,
      !mo.tensor<[512, 2048], bf16, gpu:1>) {
  %chain_in = mo.chain.create()

  // Matmul on GPU:0
  %mm0 = mo.matmul(%a0, %b0) [transpose_b = true] : (!mo.tensor<[1024, 4096], bf16, gpu:0>, !mo.tensor<[2048, 4096], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[1024, 2048], bf16, gpu:0>

  // Residual add on GPU:0 only
  %add0 = mo.add(%residual, %mm0) : !mo.tensor<[1024, 2048], bf16, gpu:0>

  // Matmul on GPU:1 (no residual add)
  %mm1 = mo.matmul(%a1, %b1) [transpose_b = true] : (!mo.tensor<[1024, 4096], bf16, gpu:1>, !mo.tensor<[2048, 4096], bf16, gpu:1, {layout = #mo.layout<NK>}>) -> !mo.tensor<[1024, 2048], bf16, gpu:1>

  // ReduceScatter: device 0 input is add result, device 1 input is bare matmul.
  %rs:2, %out_chain = mo.distributed.reducescatter.sum(%add0, %mm1, %sig0, %sig1, %chain_in) [axis = 0, group_size = 2] : (!mo.tensor<[1024, 2048], bf16, gpu:0>, !mo.tensor<[1024, 2048], bf16, gpu:1>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[512, 2048], bf16, gpu:0>, !mo.tensor<[512, 2048], bf16, gpu:1>, !mo.chain)

  mo.output %rs#0, %rs#1 : !mo.tensor<[512, 2048], bf16, gpu:0>, !mo.tensor<[512, 2048], bf16, gpu:1>
}

// CHECK-LABEL: mo.graph @matmul_reduce_scatter_residual_fusion
// CHECK: mo.composite.distributed.matmul_reduce_scatter.sum(
// CHECK-SAME: has_residual = true
// residual_peer = 0 is the default and is elided from the printed attr-dict.
// CHECK-NOT: mo.add
// CHECK-NOT: mo.matmul
// CHECK-NOT: mo.distributed.reducescatter.sum

// -----

// Non-0-peer residual: device 1 has add(residual, matmul), device 0 has bare
// matmul. The fused op must carry residual_peer = 1 so the residual add is
// applied on the device that actually owns the residual tensor (gpu:1).
mo.graph @matmul_reduce_scatter_residual_peer1_fusion(
  %a0: !mo.tensor<[1024, 4096], bf16, gpu:0>,
  %a1: !mo.tensor<[1024, 4096], bf16, gpu:1>,
  %b0: !mo.tensor<[2048, 4096], bf16, gpu:0, {layout = #mo.layout<NK>}>,
  %b1: !mo.tensor<[2048, 4096], bf16, gpu:1, {layout = #mo.layout<NK>}>,
  %residual: !mo.tensor<[1024, 2048], bf16, gpu:1>,
  %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sig1: !mo.buffer<[537919488], ui8, gpu:1>)
  -> (!mo.tensor<[512, 2048], bf16, gpu:0>,
      !mo.tensor<[512, 2048], bf16, gpu:1>) {
  %chain_in = mo.chain.create()

  // Matmul on GPU:0 (no residual add)
  %mm0 = mo.matmul(%a0, %b0) [transpose_b = true] : (!mo.tensor<[1024, 4096], bf16, gpu:0>, !mo.tensor<[2048, 4096], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[1024, 2048], bf16, gpu:0>

  // Matmul on GPU:1
  %mm1 = mo.matmul(%a1, %b1) [transpose_b = true] : (!mo.tensor<[1024, 4096], bf16, gpu:1>, !mo.tensor<[2048, 4096], bf16, gpu:1, {layout = #mo.layout<NK>}>) -> !mo.tensor<[1024, 2048], bf16, gpu:1>

  // Residual add on GPU:1 only
  %add1 = mo.add(%residual, %mm1) : !mo.tensor<[1024, 2048], bf16, gpu:1>

  // ReduceScatter: device 0 input is bare matmul, device 1 input is add result.
  %rs:2, %out_chain = mo.distributed.reducescatter.sum(%mm0, %add1, %sig0, %sig1, %chain_in) [axis = 0, group_size = 2] : (!mo.tensor<[1024, 2048], bf16, gpu:0>, !mo.tensor<[1024, 2048], bf16, gpu:1>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[512, 2048], bf16, gpu:0>, !mo.tensor<[512, 2048], bf16, gpu:1>, !mo.chain)

  mo.output %rs#0, %rs#1 : !mo.tensor<[512, 2048], bf16, gpu:0>, !mo.tensor<[512, 2048], bf16, gpu:1>
}

// CHECK-LABEL: mo.graph @matmul_reduce_scatter_residual_peer1_fusion
// CHECK: mo.composite.distributed.matmul_reduce_scatter.sum(
// CHECK-SAME: has_residual = true
// CHECK-SAME: residual_peer = 1
// CHECK-NOT: mo.add
// CHECK-NOT: mo.matmul
// CHECK-NOT: mo.distributed.reducescatter.sum
