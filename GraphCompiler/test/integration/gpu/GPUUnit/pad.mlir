// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir --mlir-print-ir-before=mo-to-map &> %t.mo.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s
// RUN: cat %t.mo.mlir | FileCheck %s --check-prefix=CHECK-IR

// CHECK-IR-LABEL: @pad_constant_1d
// CHECK-IR: mo.pad.constant{{.*}}
// CHECK{LITERAL}: 'pad_constant_1d' returned tensor<6xf32> [5.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 5.000000e+00, 5.000000e+00]
mo.graph @pad_constant_1d() -> (!mo.tensor<[6], f32>) {
  // [1, 2, 3]
  // -->
  // [5, 1, 2, 3, 5, 5]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>> : !mo.tensor<[3], f32, gpu:0>
  %constant = mo.constant <#M.dense_array<5.0> : tensor<f32>> : !mo.tensor<[], f32>
  %paddings = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %res = mo.pad.constant (%input, %paddings, %constant) : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[2], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[6], f32, gpu:0>
  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %res : !mo.tensor<[6], f32, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[6], f32>
}

// CHECK-IR-LABEL: @pad_constant_2d
// CHECK-IR: mo.pad.constant{{.*}}
// CHECK{LITERAL}: 'pad_constant_2d' returned tensor<4x4xf32> [6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00, 6.000000e+00, 6.000000e+00, 3.000000e+00, 4.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00]
mo.graph @pad_constant_2d() -> (!mo.tensor<[4, 4], f32>) {
  // [[1, 2],
  //  [3, 4]]
  // --->
  // [[6, 6, 6, 6],
  //  [6, 1, 2, 6],
  //  [6, 3, 4, 6],
  //  [6, 6, 6, 6]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, gpu:0>
  %constant = mo.constant <#M.dense_array<6.0> : tensor<f32>> : !mo.tensor<[], f32>
  %paddings = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res = mo.pad.constant (%input, %paddings, %constant) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>) -> !mo.tensor<[4, 4], f32, gpu:0>
  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %res : !mo.tensor<[4, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[4, 4], f32>
}

// CHECK-IR-LABEL: @pad_reflect_1d
// CHECK-IR: mo.pad.reflect{{.*}}
// CHECK{LITERAL}: 'pad_reflect_1d' returned tensor<6xf32> [2.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]
mo.graph @pad_reflect_1d() -> (!mo.tensor<[6], f32>) {
  // [1, 2, 3]
  // -->
  // [2, 1, 2, 3, 2, 1]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>> : !mo.tensor<[3], f32, gpu:0>
  %paddings = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %res = mo.pad.reflect (%input, %paddings) : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[6], f32, gpu:0>
  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %res : !mo.tensor<[6], f32, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[6], f32>
}

// CHECK-IR-LABEL: @pad_reflect_2d
// CHECK-IR: mo.pad.reflect{{.*}}
// CHECK{LITERAL}: 'pad_reflect_2d' returned tensor<4x4xf32> [4.000000e+00, 3.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00]
mo.graph @pad_reflect_2d() -> (!mo.tensor<[4, 4], f32>) {
  // [[1, 2],
  //  [3, 4]]
  // --->
  // [[4, 3, 4, 3],
  //  [2, 1, 2, 1],
  //  [4, 3, 4, 3],
  //  [2, 1, 2, 1]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, gpu:0>
  %paddings = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res = mo.pad.reflect (%input, %paddings) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[4, 4], f32, gpu:0>
  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %res : !mo.tensor<[4, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[4, 4], f32>
}

// CHECK-IR-LABEL: @pad_repeat_1d
// CHECK-IR: mo.pad.repeat{{.*}}
// CHECK{LITERAL}: 'pad_repeat_1d' returned tensor<6xf32> [1.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]
mo.graph @pad_repeat_1d() -> (!mo.tensor<[6], f32>) {
  // [1, 2, 3]
  // -->
  // [1, 1, 2, 3, 3, 3]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>> : !mo.tensor<[3], f32, gpu:0>
  %paddings = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %res = mo.pad.repeat (%input, %paddings) : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[6], f32, gpu:0>
  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %res : !mo.tensor<[6], f32, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[6], f32>
}

// CHECK-IR-LABEL: @pad_repeat_2d
// CHECK-IR: mo.pad.repeat{{.*}}
// CHECK{LITERAL}: 'pad_repeat_2d' returned tensor<4x4xf32> [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00]
mo.graph @pad_repeat_2d() -> (!mo.tensor<[4, 4], f32>) {
  // [[1, 2],
  //  [3, 4]]
  // --->
  // [[1, 1, 2, 2],
  //  [1, 1, 2, 2],
  //  [3, 3, 4, 4],
  //  [3, 3, 4, 4]]
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, gpu:0>
  %paddings = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res = mo.pad.repeat (%input, %paddings) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[4, 4], f32, gpu:0>
  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %res : !mo.tensor<[4, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[4, 4], f32>
}
