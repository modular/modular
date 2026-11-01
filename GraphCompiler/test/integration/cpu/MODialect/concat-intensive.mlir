// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// Generated from numpy:
// start = 0

// for i in range(10):
//   ins.append(np.arange(start, start + np.prod(a_shape), 1, dtype=np.int32).reshape(a_shape))
//   start += np.prod(a_shape)
// np.concatenate(ins, axis)
// For all axes in the shapes [(2, 2, 4), (1, 2, 1, 1, 2, 1, 4)]

mo.graph @concat_2_2_4_axis0_num10_test() -> (!mo.tensor<[20, 2, 4], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%axis = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[0] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>) -> !mo.tensor<[20, 2, 4], si32>
	mo.output %res : !mo.tensor<[20, 2, 4], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_2_2_4_axis0_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_2_2_4_axis0_num10_test'
// CHECK: 'concat_2_2_4_axis0_num10_test' returned tensor<20x2x4xsi32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]


mo.graph @concat_2_2_4_axis1_num10_test() -> (!mo.tensor<[2, 20, 4], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%axis = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[1] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>) -> !mo.tensor<[2, 20, 4], si32>
	mo.output %res : !mo.tensor<[2, 20, 4], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_2_2_4_axis1_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_2_2_4_axis1_num10_test'
// CHECK: 'concat_2_2_4_axis1_num10_test' returned tensor<2x20x4xsi32> [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55, 64, 65, 66, 67, 68, 69, 70, 71, 80, 81, 82, 83, 84, 85, 86, 87, 96, 97, 98, 99, 100, 101, 102, 103, 112, 113, 114, 115, 116, 117, 118, 119, 128, 129, 130, 131, 132, 133, 134, 135, 144, 145, 146, 147, 148, 149, 150, 151, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63, 72, 73, 74, 75, 76, 77, 78, 79, 88, 89, 90, 91, 92, 93, 94, 95, 104, 105, 106, 107, 108, 109, 110, 111, 120, 121, 122, 123, 124, 125, 126, 127, 136, 137, 138, 139, 140, 141, 142, 143, 152, 153, 154, 155, 156, 157, 158, 159]


mo.graph @concat_2_2_4_axis2_num10_test() -> (!mo.tensor<[2, 2, 40], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%axis = mo.constant <#M.dense_array<2> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[2] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>) -> !mo.tensor<[2, 2, 40], si32>
	mo.output %res : !mo.tensor<[2, 2, 40], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_2_2_4_axis2_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_2_2_4_axis2_num10_test'
// CHECK: 'concat_2_2_4_axis2_num10_test' returned tensor<2x2x40xsi32> [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67, 80, 81, 82, 83, 96, 97, 98, 99, 112, 113, 114, 115, 128, 129, 130, 131, 144, 145, 146, 147, 4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55, 68, 69, 70, 71, 84, 85, 86, 87, 100, 101, 102, 103, 116, 117, 118, 119, 132, 133, 134, 135, 148, 149, 150, 151, 8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59, 72, 73, 74, 75, 88, 89, 90, 91, 104, 105, 106, 107, 120, 121, 122, 123, 136, 137, 138, 139, 152, 153, 154, 155, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63, 76, 77, 78, 79, 92, 93, 94, 95, 108, 109, 110, 111, 124, 125, 126, 127, 140, 141, 142, 143, 156, 157, 158, 159]


mo.graph @concat_2_2_4_axis_neg1_num10_test() -> (!mo.tensor<[2, 2, 40], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<2x2x4xsi32>> : !mo.tensor<[2, 2, 4], si32>
	%axis = mo.constant <#M.dense_array<-1> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[-1] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>, !mo.tensor<[2, 2, 4], si32>) -> !mo.tensor<[2, 2, 40], si32>
	mo.output %res : !mo.tensor<[2, 2, 40], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_2_2_4_axis_neg1_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_2_2_4_axis_neg1_num10_test'
// CHECK: 'concat_2_2_4_axis_neg1_num10_test' returned tensor<2x2x40xsi32> [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67, 80, 81, 82, 83, 96, 97, 98, 99, 112, 113, 114, 115, 128, 129, 130, 131, 144, 145, 146, 147, 4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55, 68, 69, 70, 71, 84, 85, 86, 87, 100, 101, 102, 103, 116, 117, 118, 119, 132, 133, 134, 135, 148, 149, 150, 151, 8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59, 72, 73, 74, 75, 88, 89, 90, 91, 104, 105, 106, 107, 120, 121, 122, 123, 136, 137, 138, 139, 152, 153, 154, 155, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63, 76, 77, 78, 79, 92, 93, 94, 95, 108, 109, 110, 111, 124, 125, 126, 127, 140, 141, 142, 143, 156, 157, 158, 159]


mo.graph @concat_1_2_1_1_2_1_4_axis0_num10_test() -> (!mo.tensor<[10, 2, 1, 1, 2, 1, 4], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%axis = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[0] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>) -> !mo.tensor<[10, 2, 1, 1, 2, 1, 4], si32>
	mo.output %res : !mo.tensor<[10, 2, 1, 1, 2, 1, 4], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_1_2_1_1_2_1_4_axis0_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_1_2_1_1_2_1_4_axis0_num10_test'
// CHECK: 'concat_1_2_1_1_2_1_4_axis0_num10_test' returned tensor<10x2x1x1x2x1x4xsi32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]


mo.graph @concat_1_2_1_1_2_1_4_axis1_num10_test() -> (!mo.tensor<[1, 20, 1, 1, 2, 1, 4], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%axis = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[1] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>) -> !mo.tensor<[1, 20, 1, 1, 2, 1, 4], si32>
	mo.output %res : !mo.tensor<[1, 20, 1, 1, 2, 1, 4], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_1_2_1_1_2_1_4_axis1_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_1_2_1_1_2_1_4_axis1_num10_test'
// CHECK: 'concat_1_2_1_1_2_1_4_axis1_num10_test' returned tensor<1x20x1x1x2x1x4xsi32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]


mo.graph @concat_1_2_1_1_2_1_4_axis2_num10_test() -> (!mo.tensor<[1, 2, 10, 1, 2, 1, 4], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%axis = mo.constant <#M.dense_array<2> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[2] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>) -> !mo.tensor<[1, 2, 10, 1, 2, 1, 4], si32>
	mo.output %res : !mo.tensor<[1, 2, 10, 1, 2, 1, 4], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_1_2_1_1_2_1_4_axis2_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_1_2_1_1_2_1_4_axis2_num10_test'
// CHECK: 'concat_1_2_1_1_2_1_4_axis2_num10_test' returned tensor<1x2x10x1x2x1x4xsi32> [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55, 64, 65, 66, 67, 68, 69, 70, 71, 80, 81, 82, 83, 84, 85, 86, 87, 96, 97, 98, 99, 100, 101, 102, 103, 112, 113, 114, 115, 116, 117, 118, 119, 128, 129, 130, 131, 132, 133, 134, 135, 144, 145, 146, 147, 148, 149, 150, 151, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63, 72, 73, 74, 75, 76, 77, 78, 79, 88, 89, 90, 91, 92, 93, 94, 95, 104, 105, 106, 107, 108, 109, 110, 111, 120, 121, 122, 123, 124, 125, 126, 127, 136, 137, 138, 139, 140, 141, 142, 143, 152, 153, 154, 155, 156, 157, 158, 159]


mo.graph @concat_1_2_1_1_2_1_4_axis3_num10_test() -> (!mo.tensor<[1, 2, 1, 10, 2, 1, 4], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%axis = mo.constant <#M.dense_array<3> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[3] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>) -> !mo.tensor<[1, 2, 1, 10, 2, 1, 4], si32>
	mo.output %res : !mo.tensor<[1, 2, 1, 10, 2, 1, 4], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_1_2_1_1_2_1_4_axis3_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_1_2_1_1_2_1_4_axis3_num10_test'
// CHECK: 'concat_1_2_1_1_2_1_4_axis3_num10_test' returned tensor<1x2x1x10x2x1x4xsi32> [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55, 64, 65, 66, 67, 68, 69, 70, 71, 80, 81, 82, 83, 84, 85, 86, 87, 96, 97, 98, 99, 100, 101, 102, 103, 112, 113, 114, 115, 116, 117, 118, 119, 128, 129, 130, 131, 132, 133, 134, 135, 144, 145, 146, 147, 148, 149, 150, 151, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63, 72, 73, 74, 75, 76, 77, 78, 79, 88, 89, 90, 91, 92, 93, 94, 95, 104, 105, 106, 107, 108, 109, 110, 111, 120, 121, 122, 123, 124, 125, 126, 127, 136, 137, 138, 139, 140, 141, 142, 143, 152, 153, 154, 155, 156, 157, 158, 159]


mo.graph @concat_1_2_1_1_2_1_4_axis4_num10_test() -> (!mo.tensor<[1, 2, 1, 1, 20, 1, 4], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%axis = mo.constant <#M.dense_array<4> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[4] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>) -> !mo.tensor<[1, 2, 1, 1, 20, 1, 4], si32>
	mo.output %res : !mo.tensor<[1, 2, 1, 1, 20, 1, 4], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_1_2_1_1_2_1_4_axis4_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_1_2_1_1_2_1_4_axis4_num10_test'
// CHECK: 'concat_1_2_1_1_2_1_4_axis4_num10_test' returned tensor<1x2x1x1x20x1x4xsi32> [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55, 64, 65, 66, 67, 68, 69, 70, 71, 80, 81, 82, 83, 84, 85, 86, 87, 96, 97, 98, 99, 100, 101, 102, 103, 112, 113, 114, 115, 116, 117, 118, 119, 128, 129, 130, 131, 132, 133, 134, 135, 144, 145, 146, 147, 148, 149, 150, 151, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63, 72, 73, 74, 75, 76, 77, 78, 79, 88, 89, 90, 91, 92, 93, 94, 95, 104, 105, 106, 107, 108, 109, 110, 111, 120, 121, 122, 123, 124, 125, 126, 127, 136, 137, 138, 139, 140, 141, 142, 143, 152, 153, 154, 155, 156, 157, 158, 159]


mo.graph @concat_1_2_1_1_2_1_4_axis5_num10_test() -> (!mo.tensor<[1, 2, 1, 1, 2, 10, 4], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%axis = mo.constant <#M.dense_array<5> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[5] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>) -> !mo.tensor<[1, 2, 1, 1, 2, 10, 4], si32>
	mo.output %res : !mo.tensor<[1, 2, 1, 1, 2, 10, 4], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_1_2_1_1_2_1_4_axis5_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_1_2_1_1_2_1_4_axis5_num10_test'
// CHECK: 'concat_1_2_1_1_2_1_4_axis5_num10_test' returned tensor<1x2x1x1x2x10x4xsi32> [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67, 80, 81, 82, 83, 96, 97, 98, 99, 112, 113, 114, 115, 128, 129, 130, 131, 144, 145, 146, 147, 4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55, 68, 69, 70, 71, 84, 85, 86, 87, 100, 101, 102, 103, 116, 117, 118, 119, 132, 133, 134, 135, 148, 149, 150, 151, 8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59, 72, 73, 74, 75, 88, 89, 90, 91, 104, 105, 106, 107, 120, 121, 122, 123, 136, 137, 138, 139, 152, 153, 154, 155, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63, 76, 77, 78, 79, 92, 93, 94, 95, 108, 109, 110, 111, 124, 125, 126, 127, 140, 141, 142, 143, 156, 157, 158, 159]


mo.graph @concat_1_2_1_1_2_1_4_axis6_num10_test() -> (!mo.tensor<[1, 2, 1, 1, 2, 1, 40], si32>) {
	%in0 = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in1 = mo.constant <#M.dense_array<16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in2 = mo.constant <#M.dense_array<32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in3 = mo.constant <#M.dense_array<48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in4 = mo.constant <#M.dense_array<64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in5 = mo.constant <#M.dense_array<80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in6 = mo.constant <#M.dense_array<96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in7 = mo.constant <#M.dense_array<112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in8 = mo.constant <#M.dense_array<128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in9 = mo.constant <#M.dense_array<144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159> : tensor<1x2x1x1x2x1x4xsi32>> : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%axis = mo.constant <#M.dense_array<6> : tensor<si64>> : !mo.tensor<[], si64>
	%res = mo.concat[6] (%in0,%in1,%in2,%in3,%in4,%in5,%in6,%in7,%in8,%in9) : (!mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>) -> !mo.tensor<[1, 2, 1, 1, 2, 1, 40], si32>
	mo.output %res : !mo.tensor<[1, 2, 1, 1, 2, 1, 40], si32>
}

// CHECK-IR-LABEL: mgp.model @concat_1_2_1_1_2_1_4_axis6_num10_test
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.concat{{.*}}

// CHECK-LABEL: --- Running 'concat_1_2_1_1_2_1_4_axis6_num10_test'
// CHECK: 'concat_1_2_1_1_2_1_4_axis6_num10_test' returned tensor<1x2x1x1x2x1x40xsi32> [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67, 80, 81, 82, 83, 96, 97, 98, 99, 112, 113, 114, 115, 128, 129, 130, 131, 144, 145, 146, 147, 4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55, 68, 69, 70, 71, 84, 85, 86, 87, 100, 101, 102, 103, 116, 117, 118, 119, 132, 133, 134, 135, 148, 149, 150, 151, 8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59, 72, 73, 74, 75, 88, 89, 90, 91, 104, 105, 106, 107, 120, 121, 122, 123, 136, 137, 138, 139, 152, 153, 154, 155, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63, 76, 77, 78, 79, 92, 93, 94, 95, 108, 109, 110, 111, 124, 125, 126, 127, 140, 141, 142, 143, 156, 157, 158, 159]
