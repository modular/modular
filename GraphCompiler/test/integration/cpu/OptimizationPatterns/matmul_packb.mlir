// RUN: mo-opt %s --mo-infer-layouts --mo-pattern-fusion | FileCheck %s --check-prefix=CHECK-IR
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s --check-prefix=CHECK-MGP
// RUN: mt --execute --input-shapes=2x2 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s


// Note that the MGP checks are to make sure the `pack_matmul_b_shape_func` call
// got folded away at compile time, so the kernel can use static shape from the
// packed weights.

// CHECK-EXEC-LABEL: --- Running 'matmul_packb_static':
// CHECK-IR-LABEL: matmul_packb_static
// CHECK-MGP-LABEL: mef.callable @matmul_packb_static$exec
mo.graph @matmul_packb_static(%x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[4000, 2], f32>) {
  // CHECK-IR: mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> [[P1:.*]], [[P2:.*]]>(%{{.*}}) [kgenParams = {a_shape = dense<[4000, 2]> : vector<2xi64>, a_type = f32, b_shape = dense<2> : vector<2xi64>, b_type = f32, c_shape = dense<[4000, 2]> : vector<2xi64>, c_type = f32, transpose_in_0 = false}]
  // CHECK-IR: %[[PACKED_WEIGHTS:.*]] = mo.layout.transform(%{{.*}}) [kgenParams = {a_shape = dense<[4000, 2]> : vector<2xi64>, a_type = f32, b_shape = dense<2> : vector<2xi64>, b_type = f32, c_shape = dense<[4000, 2]> : vector<2xi64>, c_type = f32}] : !mo.tensor<[2, 2], f32, {layout = #mo.layout<KN>}> -> !mo.tensor<[[[P1]], [[P2]]], f32, {layout = #mo.layout<KNkni>}>
  // CHECK-IR: mo.matmul({{.*}}, %[[PACKED_WEIGHTS]]) : (!mo.tensor<[4000, 2], f32>
  %w = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %act = mo.static.broadcast_to(%x) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[2000, 2, 2], f32>
  %shape = mo.constant <#M.dense_array<4000, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %act2 = mo.reshape(%act, %shape) : (!mo.tensor<[2000, 2, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[4000, 2], f32>
  // CHECK-MGP-NOT: "pack_matmul_b_shape_func"
  // CHECK-MGP: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.matmul{{.*}}"
  %1 = mo.matmul(%act2, %w) : (!mo.tensor<[4000, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[4000, 2], f32>
  mo.output %1 : !mo.tensor<[4000, 2], f32>
}

// CHECK{LITERAL}: 'matmul_packb_static' returned tensor<4000x2xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00,


// CHECK-EXEC-LABEL: --- Running 'matmul_packb':
// CHECK-IR-LABEL: matmul_packb
// CHECK-MGP-LABEL: mef.callable @matmul_packb$exec
mo.graph @matmul_packb<D0>(%x: !mo.tensor<[D0, 2], f32>) -> (!mo.tensor<[D0, 2], f32>) {
  // CHECK-IR: mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> [[P1:.*]], [[P2:.*]]>(%{{.*}}) [kgenParams = {a_shape = dense<[[[S1:-[0-9]+]], 2]> : vector<2xi64>, a_type = f32, b_shape = dense<2> : vector<2xi64>, b_type = f32, c_shape = dense<[[[S1]], 2]> : vector<2xi64>, c_type = f32, transpose_in_0 = false}]
  // CHECK-IR: %[[PACKED_WEIGHTS:.*]] = mo.layout.transform(%{{.*}}) [kgenParams = {a_shape = dense<[[[S1]], 2]> : vector<2xi64>, a_type = f32, b_shape = dense<2> : vector<2xi64>, b_type = f32, c_shape = dense<[[[S1]], 2]> : vector<2xi64>, c_type = f32}] : !mo.tensor<[2, 2], f32, {layout = #mo.layout<KN>}> -> !mo.tensor<[[[P1]], [[P2]]], f32, {layout = #mo.layout<KNkni>}>
  // CHECK-IR: mo.matmul({{.*}}, %[[PACKED_WEIGHTS]]) : (!mo.tensor<[D0, 2], f32>
  %w = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  // CHECK-MGP-NOT: "pack_matmul_b_shape_func"
  // CHECK-MGP: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.matmul{{.*}}"
  %1 = mo.matmul(%x, %w) : (!mo.tensor<[D0, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[D0, 2], f32>
  mo.output %1 : !mo.tensor<[D0, 2], f32>
}

// CHECK{LITERAL}: 'matmul_packb' returned tensor<2x2xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]


// CHECK-EXEC-LABEL: --- Running 'matmul_transposed_packb':
// CHECK-IR-LABEL: matmul_transposed_packb
// CHECK-MGP-LABEL: mef.callable @matmul_transposed_packb$exec
mo.graph @matmul_transposed_packb<D0>(%x: !mo.tensor<[D0, 2], f32>) -> (!mo.tensor<[D0, 3], f32>) {
  // CHECK-IR: mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> [[P1:.*]], [[P2:.*]]>(%{{.*}}) [kgenParams = {a_shape{{.*}}a_type{{.*}}b_shape = dense<[3, 2]>{{.*}}b_type{{.*}}c_shape{{.*}}c_type{{.*}}transpose_in_0 = true}]
  // CHECK-IR: %[[PACKED_WEIGHTS:.*]] = mo.layout.transform(%{{.*}}) [kgenParams = {a_shape{{.*}}a_type{{.*}}b_shape = dense<[3, 2]>{{.*}}b_type{{.*}}c_shape{{.*}}c_type{{.*}}}] : !mo.tensor<[3, 2], f32, {layout = #mo.layout<NK>}> -> !mo.tensor<[[[P1]], [[P2]]], f32, {layout = #mo.layout<KNkni>}>
  // CHECK-IR: mo.matmul({{.*}}, %[[PACKED_WEIGHTS]]) : (!mo.tensor<[D0, 2], f32>
  %w = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<3x2xf32>> : !mo.tensor<[3, 2], f32>
  %axes = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %0 = mo.transpose(%w, %axes) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  // CHECK-MGP-NOT: "pack_matmul_b_shape_func"
  // CHECK-MGP: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.matmul{{.*}}"
  %1 = mo.matmul(%x, %0) : (!mo.tensor<[D0, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[D0, 3], f32>
  mo.output %1 : !mo.tensor<[D0, 3], f32>
}

// CHECK{LITERAL}: 'matmul_transposed_packb' returned tensor<2x3xf32> [3.000000e+00, 7.000000e+00, 1.100000e+01, 3.000000e+00, 7.000000e+00, 1.100000e+01]


// CHECK-EXEC-LABEL: --- Running 'matmul_pack_reshaped':
// CHECK-IR-LABEL: matmul_pack_reshaped
// CHECK-MGP-LABEL: mef.callable @matmul_pack_reshaped$exec
mo.graph @matmul_pack_reshaped<D0>(%x: !mo.tensor<[D0, 2], f32>) -> (!mo.tensor<[D0, 3], f32>) {
  // CHECK-IR: mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> [[P1:.*]], [[P2:.*]]>(%{{.*}}) [kgenParams = {a_shape{{.*}}a_type{{.*}}b_shape = dense<[2, 3]>{{.*}}b_type{{.*}}c_shape{{.*}}c_type{{.*}}transpose_in_0 = false}]
  // CHECK-IR: %[[PACKED_WEIGHTS:.*]] = mo.layout.transform(%{{.*}}) [kgenParams = {a_shape{{.*}}a_type{{.*}}b_shape = dense<[2, 3]>{{.*}}b_type{{.*}}c_shape{{.*}}c_type{{.*}}}] : !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}> -> !mo.tensor<[[[P1]], [[P2]]], f32, {layout = #mo.layout<KNkni>}>
  // CHECK-IR: mo.matmul({{.*}}, %[[PACKED_WEIGHTS]]) : (!mo.tensor<[D0, 2], f32>
  %w = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[1, 3, 2, 1], f32>
  %perm = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %w_t = mo.transpose(%w, %perm) : (!mo.tensor<[1, 3, 2, 1], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[1, 2, 3, 1], f32>
  %w_rs = mo.static.reshape(%w_t) : !mo.tensor<[1, 2, 3, 1], f32> -> !mo.tensor<[2, 3], f32>
  %1 = mo.matmul(%x, %w_rs) : (!mo.tensor<[D0, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[D0, 3], f32>
  // CHECK-MGP-NOT: "pack_matmul_b_shape_func"
  // CHECK-MGP: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.matmul{{.*}}"
  mo.output %1 : !mo.tensor<[D0, 3], f32>
}

// CHECK{LITERAL}: 'matmul_pack_reshaped' returned tensor<2x3xf32> [3.000000e+00, 7.000000e+00, 1.100000e+01, 3.000000e+00, 7.000000e+00, 1.100000e+01]
