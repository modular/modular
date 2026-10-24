// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// Generated from numpy:
// np.argmax(input, axis=axis, keepdims=True) where:
// input = np.arange(...)
// For all axes and input / indices shapes in the range:
// input_data_shapes = [(3), (128,), (2, 2)]

mo.graph @arg_max_inshape_3_axis_0_test() -> (!mo.tensor<[1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %res = mo.reduce.arg_max(%data) [axis = 0] : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si64>
  mo.output %res : !mo.tensor<[1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_max_inshape_3_axis_0_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_max{{.*}}

// CHECK: 'arg_max_inshape_3_axis_0_test' returned tensor<1xsi64> [2]

mo.graph @arg_min_inshape_3_axis_0_test() -> (!mo.tensor<[1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %res = mo.reduce.arg_min(%data) [axis = 0] : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si64>
  mo.output %res : !mo.tensor<[1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_min_inshape_3_axis_0_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_min{{.*}}

// CHECK: 'arg_min_inshape_3_axis_0_test' returned tensor<1xsi64> [0]

mo.graph @arg_max_inshape_3_axis_si32_0_test() -> (!mo.tensor<[1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %res = mo.reduce.arg_max(%data) [axis = 0] : (!mo.tensor<[3], si32>) -> !mo.tensor<[1], si64>
  mo.output %res : !mo.tensor<[1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_max_inshape_3_axis_si32_0_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_max{{.*}}

// CHECK: 'arg_max_inshape_3_axis_si32_0_test' returned tensor<1xsi64> [2]

mo.graph @arg_min_inshape_3_axis_0_si32_test() -> (!mo.tensor<[1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %res = mo.reduce.arg_min(%data) [axis = 0] : (!mo.tensor<[3], si32>) -> !mo.tensor<[1], si64>
  mo.output %res : !mo.tensor<[1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_min_inshape_3_axis_0_si32_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_min{{.*}}

// CHECK: 'arg_min_inshape_3_axis_0_si32_test' returned tensor<1xsi64> [0]

mo.graph @arg_max_inshape_128_axis_0_test() -> (!mo.tensor<[1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<128xsi64>> : !mo.tensor<[128], si64>
  %res = mo.reduce.arg_max(%data) [axis = 0] : (!mo.tensor<[128], si64>) -> !mo.tensor<[1], si64>
  mo.output %res : !mo.tensor<[1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_max_inshape_128_axis_0_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_max{{.*}}

// CHECK: 'arg_max_inshape_128_axis_0_test' returned tensor<1xsi64> [127]

mo.graph @arg_min_inshape_128_axis_0_test() -> (!mo.tensor<[1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<128xsi64>> : !mo.tensor<[128], si64>
  %res = mo.reduce.arg_min(%data) [axis = 0] : (!mo.tensor<[128], si64>) -> !mo.tensor<[1], si64>
  mo.output %res : !mo.tensor<[1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_min_inshape_128_axis_0_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_min{{.*}}

// CHECK: 'arg_min_inshape_128_axis_0_test' returned tensor<1xsi64> [0]

mo.graph @arg_max_inshape_1x4_axis_1_test() -> (!mo.tensor<[1, 1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2,3> : tensor<1x4xsi64>> : !mo.tensor<[1, 4], si64>
  %res = mo.reduce.arg_max(%data) [axis = 1] : (!mo.tensor<[1, 4], si64>) -> !mo.tensor<[1, 1], si64>
  mo.output %res : !mo.tensor<[1, 1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_max_inshape_1x4_axis_1_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_max{{.*}}

// CHECK: 'arg_max_inshape_1x4_axis_1_test' returned tensor<1x1xsi64> [3]

mo.graph @arg_min_inshape_1x4_axis_1_test() -> (!mo.tensor<[1, 1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2,3> : tensor<1x4xsi64>> : !mo.tensor<[1, 4], si64>
  %res = mo.reduce.arg_min(%data) [axis = 1] : (!mo.tensor<[1, 4], si64>) -> !mo.tensor<[1, 1], si64>
  mo.output %res : !mo.tensor<[1, 1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_min_inshape_1x4_axis_1_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_min{{.*}}

// CHECK: 'arg_min_inshape_1x4_axis_1_test' returned tensor<1x1xsi64> [0]


mo.graph @arg_max_inshape_2x2_axis_1_test() -> (!mo.tensor<[2, 1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2,3> : tensor<2x2xsi64>> : !mo.tensor<[2, 2], si64>
  %res = mo.reduce.arg_max(%data) [axis = 1] : (!mo.tensor<[2, 2], si64>) -> !mo.tensor<[2, 1], si64>
  mo.output %res : !mo.tensor<[2, 1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_max_inshape_2x2_axis_1_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_max{{.*}}

// CHECK: 'arg_max_inshape_2x2_axis_1_test' returned tensor<2x1xsi64> [1, 1]

mo.graph @arg_min_inshape_2x2_axis_1_test() -> (!mo.tensor<[2, 1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2,3> : tensor<2x2xsi64>> : !mo.tensor<[2, 2], si64>
  %res = mo.reduce.arg_min(%data) [axis = 1] : (!mo.tensor<[2, 2], si64>) -> !mo.tensor<[2, 1], si64>
  mo.output %res : !mo.tensor<[2, 1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_min_inshape_2x2_axis_1_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_min{{.*}}

// CHECK: 'arg_min_inshape_2x2_axis_1_test' returned tensor<2x1xsi64> [0, 0]

mo.graph @arg_max_inshape_32x4_axis_1_test() -> (!mo.tensor<[32, 1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<32x4xsi64>> : !mo.tensor<[32, 4], si64>
  %res = mo.reduce.arg_max(%data) [axis = 1] : (!mo.tensor<[32, 4], si64>) -> !mo.tensor<[32, 1], si64>
  mo.output %res : !mo.tensor<[32, 1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_max_inshape_32x4_axis_1_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_max{{.*}}

// CHECK: 'arg_max_inshape_32x4_axis_1_test' returned tensor<32x1xsi64> [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

mo.graph @arg_min_inshape_32x4_axis_1_test() -> (!mo.tensor<[32, 1], si64>) {
  %data = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<32x4xsi64>> : !mo.tensor<[32, 4], si64>
  %res = mo.reduce.arg_min(%data) [axis = 1] : (!mo.tensor<[32, 4], si64>) -> !mo.tensor<[32, 1], si64>
  mo.output %res : !mo.tensor<[32, 1], si64>
}

// CHECK-IR-LABEL: mgp.model @arg_min_inshape_32x4_axis_1_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.reduce.arg_min{{.*}}

// CHECK: 'arg_min_inshape_32x4_axis_1_test' returned tensor<32x1xsi64> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
