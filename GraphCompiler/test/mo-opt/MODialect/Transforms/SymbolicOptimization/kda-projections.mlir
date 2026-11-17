// RUN: mo-opt %s --mo-symbolic-optimize --mo-infer-layouts --mo-pattern-fusion \
// RUN:   --mo-constant-fold | FileCheck %s

// GLM-5.3-Flash's KDA projection pattern, from
// `max/python/max/pipelines/architectures/glm5_next/layers/kda.py`, scaled down
// but with the model's shape relationships: hidden = 32, heads = 2, head_dim =
// 8, so q, k and v are 16 wide, the gate down halves 8 and `b_proj` 2.
//
// `x` feeds six projections, q, k, v, f_a, b and g_a. They merge into one
// 66-wide matmul; q, k and v come back as one 48-wide slice that feeds the
// conv, and f_b/g_b keep projecting their down halves. This is the IR the
// model is expected to produce just before MOToMAP.
//
// CHECK-LABEL: mo.graph @kda_projections
// The six weights concatenate in program order, q, k, v, f_a, b, g_a, and
// pack as one [66, 32] transpose_b operand: 48 + 8 + 2 + 8 columns.
// CHECK: %[[W:.*]] = mogg.concat({{.*}}) : (!mo.tensor<[16, 32], bf16, gpu:0>, !mo.tensor<[16, 32], bf16, gpu:0>, !mo.tensor<[16, 32], bf16, gpu:0>, !mo.tensor<[8, 32], bf16, gpu:0>, !mo.tensor<[2, 32], bf16, gpu:0>, !mo.tensor<[8, 32], bf16, gpu:0>) -> !mo.tensor<[66, 32], bf16, gpu:0>
// CHECK: %[[PACKED:.*]] = mo.rebind(%[[W]]) : !mo.tensor<[66, 32], bf16, gpu:0> -> !mo.tensor<[66, 32], bf16, gpu:0, {layout = #mo.layout<NK>}>
// CHECK: mo.matmul(%arg0, %[[PACKED]]) [transpose_b = true] : (!mo.tensor<[4, 32], bf16, gpu:0>, !mo.tensor<[66, 32], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[4, 66], bf16, gpu:0>
// q, k and v read back as one slice, ready for the conv; f_a, b and g_a are
// one column group each, and f_b/g_b still project the down halves on their
// own. Order-independent: the concat, and so its slice, moves with the
// consumer.
// CHECK-DAG: %[[QKV:.*]] = mo.slice({{.*}}) : (!mo.tensor<[4, 66], bf16, gpu:0>, {{.*}}) -> !mo.tensor<[4, 48], bf16, gpu:0>
// CHECK-DAG: mo.slice({{.*}}) : (!mo.tensor<[4, 66], bf16, gpu:0>, {{.*}}) -> !mo.tensor<[4, 8], bf16, gpu:0>
// CHECK-DAG: mo.static.reshape({{.*}}) : !mo.tensor<[4, 66], bf16, gpu:0> -> !mo.tensor<[1, 4, 66], bf16, gpu:0>
// CHECK-DAG: mo.slice({{.*}}) : (!mo.tensor<[1, 4, 66], bf16, gpu:0>, {{.*}}) -> !mo.tensor<[1, 4, 2], bf16, gpu:0>
// CHECK-DAG: mo.slice({{.*}}) : (!mo.tensor<[4, 66], bf16, gpu:0>, {{.*}}) -> !mo.tensor<[4, 8], bf16, gpu:0>
// CHECK-DAG: mo.matmul({{.*}}, {{.*}}) [transpose_b = true] : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[4, 16], bf16, gpu:0>
// CHECK-DAG: mo.matmul({{.*}}, {{.*}}) [transpose_b = true] : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[4, 16], bf16, gpu:0>
// CHECK: mo.cast(%[[QKV]]) : (!mo.tensor<[4, 48], bf16, gpu:0>) -> !mo.tensor<[4, 48], f32, gpu:0>
// No concat of activation results is left.
// CHECK-NOT: mo.concat[1]

mo.graph @kda_projections(%x: !mo.tensor<[4, 32], bf16, gpu:0>) -> (!mo.tensor<[4, 48], f32, gpu:0>, !mo.tensor<[4, 16], bf16, gpu:0>, !mo.tensor<[4, 16], bf16, gpu:0>, !mo.tensor<[1, 4, 2], bf16, gpu:0>) {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %q_w = mo.constant.external [name = "layer.q_proj.weight", align = 1, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 32], bf16, gpu:0>
  %k_w = mo.constant.external [name = "layer.k_proj.weight", align = 1, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 32], bf16, gpu:0>
  %v_w = mo.constant.external [name = "layer.v_proj.weight", align = 1, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 32], bf16, gpu:0>
  %f_a_w = mo.constant.external [name = "layer.f_a_proj.weight", align = 1, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[8, 32], bf16, gpu:0>
  %b_w = mo.constant.external [name = "layer.b_proj.weight", align = 1, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[2, 32], bf16, gpu:0>
  %g_a_w = mo.constant.external [name = "layer.g_a_proj.weight", align = 1, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[8, 32], bf16, gpu:0>
  %f_b_w = mo.constant.external [name = "layer.f_b_proj.weight", align = 1, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 8], bf16, gpu:0>
  %g_b_w = mo.constant.external [name = "layer.g_b_proj.weight", align = 1, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 8], bf16, gpu:0>

  %q_wt = mo.transpose(%q_w, %perm) : (!mo.tensor<[16, 32], bf16, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[32, 16], bf16, gpu:0>
  %q = mo.matmul(%x, %q_wt) : (!mo.tensor<[4, 32], bf16, gpu:0>, !mo.tensor<[32, 16], bf16, gpu:0>) -> !mo.tensor<[4, 16], bf16, gpu:0>
  %k_wt = mo.transpose(%k_w, %perm) : (!mo.tensor<[16, 32], bf16, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[32, 16], bf16, gpu:0>
  %k = mo.matmul(%x, %k_wt) : (!mo.tensor<[4, 32], bf16, gpu:0>, !mo.tensor<[32, 16], bf16, gpu:0>) -> !mo.tensor<[4, 16], bf16, gpu:0>
  %v_wt = mo.transpose(%v_w, %perm) : (!mo.tensor<[16, 32], bf16, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[32, 16], bf16, gpu:0>
  %v = mo.matmul(%x, %v_wt) : (!mo.tensor<[4, 32], bf16, gpu:0>, !mo.tensor<[32, 16], bf16, gpu:0>) -> !mo.tensor<[4, 16], bf16, gpu:0>

  %f_a_wt = mo.transpose(%f_a_w, %perm) : (!mo.tensor<[8, 32], bf16, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[32, 8], bf16, gpu:0>
  %f_a = mo.matmul(%x, %f_a_wt) : (!mo.tensor<[4, 32], bf16, gpu:0>, !mo.tensor<[32, 8], bf16, gpu:0>) -> !mo.tensor<[4, 8], bf16, gpu:0>
  %f_b_wt = mo.transpose(%f_b_w, %perm) : (!mo.tensor<[16, 8], bf16, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[8, 16], bf16, gpu:0>
  %f_b = mo.matmul(%f_a, %f_b_wt) : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[8, 16], bf16, gpu:0>) -> !mo.tensor<[4, 16], bf16, gpu:0>

  %b_wt = mo.transpose(%b_w, %perm) : (!mo.tensor<[2, 32], bf16, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[32, 2], bf16, gpu:0>
  %b = mo.matmul(%x, %b_wt) : (!mo.tensor<[4, 32], bf16, gpu:0>, !mo.tensor<[32, 2], bf16, gpu:0>) -> !mo.tensor<[4, 2], bf16, gpu:0>
  %b_rows = mo.static.reshape(%b) : !mo.tensor<[4, 2], bf16, gpu:0> -> !mo.tensor<[1, 4, 2], bf16, gpu:0>

  %g_a_wt = mo.transpose(%g_a_w, %perm) : (!mo.tensor<[8, 32], bf16, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[32, 8], bf16, gpu:0>
  %g_a = mo.matmul(%x, %g_a_wt) : (!mo.tensor<[4, 32], bf16, gpu:0>, !mo.tensor<[32, 8], bf16, gpu:0>) -> !mo.tensor<[4, 8], bf16, gpu:0>
  %g_b_wt = mo.transpose(%g_b_w, %perm) : (!mo.tensor<[16, 8], bf16, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[8, 16], bf16, gpu:0>
  %g_b = mo.matmul(%g_a, %g_b_wt) : (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[8, 16], bf16, gpu:0>) -> !mo.tensor<[4, 16], bf16, gpu:0>

  %qkv = mo.concat[1] (%q, %k, %v) : (!mo.tensor<[4, 16], bf16, gpu:0>, !mo.tensor<[4, 16], bf16, gpu:0>, !mo.tensor<[4, 16], bf16, gpu:0>) -> (!mo.tensor<[4, 48], bf16, gpu:0>)
  %qkv_f32 = mo.cast(%qkv) : (!mo.tensor<[4, 48], bf16, gpu:0>) -> !mo.tensor<[4, 48], f32, gpu:0>
  mo.output %qkv_f32, %f_b, %g_b, %b_rows : !mo.tensor<[4, 48], f32, gpu:0>, !mo.tensor<[4, 16], bf16, gpu:0>, !mo.tensor<[4, 16], bf16, gpu:0>, !mo.tensor<[1, 4, 2], bf16, gpu:0>
}
