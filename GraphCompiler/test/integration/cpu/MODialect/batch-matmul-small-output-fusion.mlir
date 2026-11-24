// RUN: mo-opt %s --gc-pipeline="trivial-kernel-opt=false constant-fold=false" | FileCheck %s --check-prefix=CHECK-IR
// RUN: mt --execute --result-output-style=full %s | FileCheck %s --check-prefix=CHECK-MT

// The two fusion systems nest the broadcast differently inside the summary, so
// CHECK-IR asserts the batch_matmul/add epilogue that gives the test its name.

mo.graph @batch_matmul_m1_n1() -> (!mo.tensor<[3, 1, 1], f32>) {
  %identity = "mo.constant"() {value = #M.dense_array<1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0> : tensor<3x1x3xf32>} : () -> !mo.tensor<[3, 1, 3], f32>
  %value = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0> : tensor<3x3x1xf32>} : () -> !mo.tensor<[3, 3, 1], f32>
  %one = "mo.constant"() {value = #M.dense_array<1.0> : tensor<1xf32>} : () -> !mo.tensor<[1], f32>
  %x = mo.batch_matmul(%identity, %value) : (!mo.tensor<[3, 1, 3], f32>, !mo.tensor<[3, 3, 1], f32>) -> !mo.tensor<[3, 1, 1], f32>
  %one_bcast = mo.static.broadcast_to(%one) : !mo.tensor<[1], f32> -> !mo.tensor<[3, 1, 1], f32>
  %output = mo.add(%x, %one_bcast): !mo.tensor<[3, 1, 1], f32>
  mo.output %output : !mo.tensor<[3, 1, 1], f32>
}
// CHECK-MT{LITERAL}: 'batch_matmul_m1_n1' returned tensor<3x1x1xf32> [2.000000e+00, 6.000000e+00, 1.000000e+01]
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = "Epilogue({{.*}}mo.batch_matmul{{.*}}mo.add{{.*}}

mo.graph @batch_matmul_m3_n3() -> (!mo.tensor<[3, 3, 3], f32>) {
  %identity = "mo.constant"() {value = #M.dense_array<1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0> : tensor<3x1x3xf32>} : () -> !mo.tensor<[3, 1, 3], f32>
  %value = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0> : tensor<3x3x1xf32>} : () -> !mo.tensor<[3, 3, 1], f32>
  %one = "mo.constant"() {value = #M.dense_array<1.0> : tensor<1xf32>} : () -> !mo.tensor<[1], f32>
  %x = mo.batch_matmul(%value, %identity) : (!mo.tensor<[3, 3, 1], f32>, !mo.tensor<[3, 1, 3], f32>) -> !mo.tensor<[3, 3, 3], f32>
  %one_bcast = mo.static.broadcast_to(%one) : !mo.tensor<[1], f32> -> !mo.tensor<[3, 3, 3], f32>
  %output = mo.add(%x, %one_bcast): !mo.tensor<[3, 3, 3], f32>
  mo.output %output : !mo.tensor<[3, 3, 3], f32>
}
// CHECK-MT{LITERAL}: 'batch_matmul_m3_n3' returned tensor<3x3x3xf32> [2.000000e+00, 1.000000e+00, 1.000000e+00, 3.000000e+00, 1.000000e+00, 1.000000e+00, 4.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 5.000000e+00, 1.000000e+00, 1.000000e+00, 6.000000e+00, 1.000000e+00, 1.000000e+00, 7.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 8.000000e+00, 1.000000e+00, 1.000000e+00, 9.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+01]
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = "Epilogue({{.*}}mo.batch_matmul{{.*}}mo.add{{.*}}

mo.graph @batch_matmul_m1_n1_mixed_precision() -> (!mo.tensor<[3, 1, 1], si32>) {
  %identity = "mo.constant"() {value = #M.dense_array<1, 0, 0, 0, 1, 0, 0, 0, 1> : tensor<3x1x3xui8>} : () -> !mo.tensor<[3, 1, 3], ui8>
  %value = "mo.constant"() {value = #M.dense_array<1, 2, 3, 4, 5, 6, 7, 8, 9> : tensor<3x3x1xsi8>} : () -> !mo.tensor<[3, 3, 1], si8>
  %one = "mo.constant"() {value = #M.dense_array<1> : tensor<1xsi32>} : () -> !mo.tensor<[1], si32>
  %x = mo.batch_matmul(%identity, %value) : (!mo.tensor<[3, 1, 3], ui8>, !mo.tensor<[3, 3, 1], si8>) -> !mo.tensor<[3, 1, 1], si32>
  %one_bcast = mo.static.broadcast_to(%one) : !mo.tensor<[1], si32> -> !mo.tensor<[3, 1, 1], si32>
  %output = mo.add(%x, %one_bcast): !mo.tensor<[3, 1, 1], si32>
  mo.output %output : !mo.tensor<[3, 1, 1], si32>
}
// CHECK-MT{LITERAL}: 'batch_matmul_m1_n1_mixed_precision' returned tensor<3x1x1xsi32> [2, 6, 10]
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = "Epilogue({{.*}}mo.batch_matmul{{.*}}mo.add{{.*}}

mo.graph @batch_matmul_m3_n3_mixed_precision() -> (!mo.tensor<[3, 3, 3], si32>) {
  %identity = "mo.constant"() {value = #M.dense_array<1, 0, 0, 0, 1, 0, 0, 0, 1> : tensor<3x1x3xsi8>} : () -> !mo.tensor<[3, 1, 3], si8>
  %value = "mo.constant"() {value = #M.dense_array<1, 2, 3, 4, 5, 6, 7, 8, 9> : tensor<3x3x1xui8>} : () -> !mo.tensor<[3, 3, 1], ui8>
  %one = "mo.constant"() {value = #M.dense_array<1> : tensor<1xsi32>} : () -> !mo.tensor<[1], si32>
  %x = mo.batch_matmul(%value, %identity) : (!mo.tensor<[3, 3, 1], ui8>, !mo.tensor<[3, 1, 3], si8>) -> !mo.tensor<[3, 3, 3], si32>
  %one_bcast = mo.static.broadcast_to(%one) : !mo.tensor<[1], si32> -> !mo.tensor<[3, 3, 3], si32>
  %output = mo.add(%x, %one_bcast): !mo.tensor<[3, 3, 3], si32>
  mo.output %output : !mo.tensor<[3, 3, 3], si32>
}
// CHECK-MT{LITERAL}: 'batch_matmul_m3_n3_mixed_precision' returned tensor<3x3x3xsi32> [2, 1, 1, 3, 1, 1, 4, 1, 1, 1, 5, 1, 1, 6, 1, 1, 7, 1, 1, 1, 8, 1, 1, 9, 1, 1, 10]
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = "Epilogue({{.*}}mo.batch_matmul{{.*}}mo.add{{.*}}
