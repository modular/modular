// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s


module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "">}} {

// A full 2D residual is fused as the composite op's addend.
// CHECK-LABEL: mo.graph @matmul_add_2d
// CHECK-NOT: mo.matmul
// CHECK: %[[OUT:.*]] = mo.composite.matmul_add
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_add_2d(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[65536, 1536], bf16, gpu:0>) -> !mo.tensor<[65536, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  mo.output %1 : !mo.tensor<[65536, 1536], bf16, gpu:0>
}

// A rank-3 residual is reshaped to the matmul's result type and fused.
// CHECK-LABEL: mo.graph @matmul_reshape_add_2d
// CHECK: mo.static.reshape
// CHECK: %[[FUSED:.*]] = mo.composite.matmul_add
// CHECK: %[[OUT:.*]] = mo.static.reshape(%[[FUSED]])
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_reshape_add_2d(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0>, %arg2: !mo.tensor<[32, 2048, 1536], bf16, gpu:0>) -> !mo.tensor<[32, 2048, 1536], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.transpose(%arg1, %0) : (!mo.tensor<[1536, 1536], bf16, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>
  %2 = mo.matmul(%arg0, %1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %3 = mo.static.reshape(%2) : !mo.tensor<[65536, 1536], bf16, gpu:0> -> !mo.tensor<[32, 2048, 1536], bf16, gpu:0>
  %4 = mo.add(%3, %arg2) : !mo.tensor<[32, 2048, 1536], bf16, gpu:0>
  mo.output %4 : !mo.tensor<[32, 2048, 1536], bf16, gpu:0>
}

// CHECK-LABEL: mo.graph @matmul_add_1d
// CHECK-NOT: mo.static.broadcast_to
// CHECK: %[[OUT:.*]] = mo.composite.matmul_add
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_add_1d(%arg0: !mo.tensor<[256, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[8192, 1536], bf16, gpu:0>, %arg2: !mo.tensor<[8192], bf16, gpu:0>) -> !mo.tensor<[256, 8192], bf16, gpu:0> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[8192, 1536], bf16, gpu:0> -> !mo.tensor<[8192, 1536], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %1 = mo.matmul(%arg0, %0) [transpose_b = true] : (!mo.tensor<[256, 1536], bf16, gpu:0>, !mo.tensor<[8192, 1536], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[256, 8192], bf16, gpu:0>
  %2 = mo.static.broadcast_to(%arg2) : !mo.tensor<[8192], bf16, gpu:0> -> !mo.tensor<[256, 8192], bf16, gpu:0>
  %3 = mo.add(%1, %2) : !mo.tensor<[256, 8192], bf16, gpu:0>
  mo.output %3 : !mo.tensor<[256, 8192], bf16, gpu:0>
}

// Negative: matmul has multiple uses — output is used by the add AND returned directly.
// CHECK-LABEL: mo.graph @neg_matmul_multiple_uses
// CHECK: mo.matmul
// CHECK-NOT: mo.composite.matmul_add
// CHECK: mo.add
mo.graph @neg_matmul_multiple_uses(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[65536, 1536], bf16, gpu:0>) -> (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[65536, 1536], bf16, gpu:0>) {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  mo.output %0, %1 : !mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[65536, 1536], bf16, gpu:0>
}

// A 2D residual is fused at M == 1 too.
// CHECK-LABEL: mo.graph @gemv
// CHECK-NOT: mo.matmul
// CHECK: %[[OUT:.*]] = mo.composite.matmul_add
// CHECK: mo.output %[[OUT]]
mo.graph @gemv(%arg0: !mo.tensor<[1, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[1, 1536], bf16, gpu:0>) -> !mo.tensor<[1, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[1, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[1, 1536], bf16, gpu:0>
  mo.output %1 : !mo.tensor<[1, 1536], bf16, gpu:0>
}

// Dynamic M is fused; only N and K have to be static.
// CHECK-LABEL: mo.graph @dynamic_m
// CHECK-NOT: mo.matmul
// CHECK: %[[OUT:.*]] = mo.composite.matmul_add
// CHECK: mo.output %[[OUT]]
mo.graph @dynamic_m(%arg0: !mo.tensor<[dyn_m, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[dyn_m, 1536], bf16, gpu:0>) -> !mo.tensor<[dyn_m, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[dyn_m, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[dyn_m, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[dyn_m, 1536], bf16, gpu:0>
  mo.output %1 : !mo.tensor<[dyn_m, 1536], bf16, gpu:0>
}

// Negative: add's residual input is produced by an elementwise op.
// CHECK-LABEL: mo.graph @neg_elementwise_residual
// CHECK: mo.matmul
// CHECK-NOT: mo.composite.matmul_add
// CHECK: mo.add
mo.graph @neg_elementwise_residual(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[65536, 1536], bf16, gpu:0>) -> !mo.tensor<[65536, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %1 = mo.relu(%arg2) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  %2 = mo.add(%0, %1) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  mo.output %2 : !mo.tensor<[65536, 1536], bf16, gpu:0>
}

// Negative: elementwise op after the add would miss epilogue fusion with fused_matmul_add.
// CHECK-LABEL: mo.graph @neg_downstream_elementwise
// CHECK: mo.matmul
// CHECK-NOT: mo.composite.matmul_add
// CHECK: mo.add
mo.graph @neg_downstream_elementwise(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[65536, 1536], bf16, gpu:0>) -> !mo.tensor<[65536, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  %2 = mo.relu(%1) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  mo.output %2 : !mo.tensor<[65536, 1536], bf16, gpu:0>
}

// A biased linear followed by a residual: the bias is reassociated into the
// residual so the single-addend composite op can take both.
// CHECK-LABEL: mo.graph @bias_then_residual
// CHECK-NOT: mo.matmul
// CHECK: %[[SUM:.*]] = mo.add
// CHECK: %[[OUT:.*]] = mo.composite.matmul_add(%{{.*}}, %{{.*}}, %[[SUM]])
// CHECK: mo.output %[[OUT]]
mo.graph @bias_then_residual(%arg0: !mo.tensor<[65536, 4096], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 4096], bf16, gpu:0>, %arg2: !mo.tensor<[1536], bf16, gpu:0>, %arg3: !mo.tensor<[65536, 1536], bf16, gpu:0>) -> !mo.tensor<[65536, 1536], bf16, gpu:0> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[1536, 4096], bf16, gpu:0> -> !mo.tensor<[1536, 4096], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %1 = mo.matmul(%arg0, %0) [transpose_b = true] : (!mo.tensor<[65536, 4096], bf16, gpu:0>, !mo.tensor<[1536, 4096], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %2 = mo.static.broadcast_to(%arg2) : !mo.tensor<[1536], bf16, gpu:0> -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %3 = mo.add(%1, %2) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  %4 = mo.add(%3, %arg3) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  mo.output %4 : !mo.tensor<[65536, 1536], bf16, gpu:0>
}

// The same chain with the reshape back to rank 3 between the two adds, which
// is how a biased linear inside a transformer block lowers.
// CHECK-LABEL: mo.graph @bias_reshape_then_residual
// CHECK-NOT: mo.matmul
// CHECK: mo.composite.matmul_add
// CHECK: mo.output
mo.graph @bias_reshape_then_residual(%arg0: !mo.tensor<[65536, 4096], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 4096], bf16, gpu:0>, %arg2: !mo.tensor<[1536], bf16, gpu:0>, %arg3: !mo.tensor<[32, 2048, 1536], bf16, gpu:0>) -> !mo.tensor<[32, 2048, 1536], bf16, gpu:0> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[1536, 4096], bf16, gpu:0> -> !mo.tensor<[1536, 4096], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %1 = mo.matmul(%arg0, %0) [transpose_b = true] : (!mo.tensor<[65536, 4096], bf16, gpu:0>, !mo.tensor<[1536, 4096], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %2 = mo.static.broadcast_to(%arg2) : !mo.tensor<[1536], bf16, gpu:0> -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %3 = mo.add(%1, %2) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  %4 = mo.static.reshape(%3) : !mo.tensor<[65536, 1536], bf16, gpu:0> -> !mo.tensor<[32, 2048, 1536], bf16, gpu:0>
  %5 = mo.add(%4, %arg3) : !mo.tensor<[32, 2048, 1536], bf16, gpu:0>
  mo.output %5 : !mo.tensor<[32, 2048, 1536], bf16, gpu:0>
}
}
