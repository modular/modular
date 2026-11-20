// COM: Negative case: the matmul + reduce_scatter fusion must NOT fire on
// CPU. The fused kernel only exists for SM100 (B200) today, so on CPU
// the original `matmul + reducescatter` decomposition must stay in the
// graph and run via the existing standalone kernels.
// RUN: mo-opt %s --mo-pattern-fusion='target-device=cpu' | FileCheck %s

mo.graph @matmul_reduce_scatter_no_fusion_on_cpu(
  %a0: !mo.tensor<[1024, 4096], bf16, cpu:0>,
  %a1: !mo.tensor<[1024, 4096], bf16, cpu:1>,
  %b0: !mo.tensor<[2048, 4096], bf16, cpu:0, {layout = #mo.layout<NK>}>,
  %b1: !mo.tensor<[2048, 4096], bf16, cpu:1, {layout = #mo.layout<NK>}>,
  %sig0: !mo.buffer<[537919488], ui8, cpu:0>,
  %sig1: !mo.buffer<[537919488], ui8, cpu:1>)
  -> (!mo.tensor<[512, 2048], bf16, cpu:0>,
      !mo.tensor<[512, 2048], bf16, cpu:1>) {
  %chain_in = mo.chain.create()

  %mm0 = mo.matmul(%a0, %b0) [transpose_b = true] : (!mo.tensor<[1024, 4096], bf16, cpu:0>, !mo.tensor<[2048, 4096], bf16, cpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[1024, 2048], bf16, cpu:0>
  %mm1 = mo.matmul(%a1, %b1) [transpose_b = true] : (!mo.tensor<[1024, 4096], bf16, cpu:1>, !mo.tensor<[2048, 4096], bf16, cpu:1, {layout = #mo.layout<NK>}>) -> !mo.tensor<[1024, 2048], bf16, cpu:1>

  %rs:2, %out_chain = mo.distributed.reducescatter.sum(%mm0, %mm1, %sig0, %sig1, %chain_in) [axis = 0, group_size = 2] : (!mo.tensor<[1024, 2048], bf16, cpu:0>, !mo.tensor<[1024, 2048], bf16, cpu:1>, !mo.buffer<[537919488], ui8, cpu:0>, !mo.buffer<[537919488], ui8, cpu:1>, !mo.chain) -> (!mo.tensor<[512, 2048], bf16, cpu:0>, !mo.tensor<[512, 2048], bf16, cpu:1>, !mo.chain)

  mo.output %rs#0, %rs#1 : !mo.tensor<[512, 2048], bf16, cpu:0>, !mo.tensor<[512, 2048], bf16, cpu:1>
}

// CHECK-LABEL: mo.graph @matmul_reduce_scatter_no_fusion_on_cpu
// CHECK: mo.matmul
// CHECK: mo.matmul
// CHECK: mo.distributed.reducescatter.sum
// CHECK-NOT: mo.composite.distributed.matmul_reduce_scatter.sum
