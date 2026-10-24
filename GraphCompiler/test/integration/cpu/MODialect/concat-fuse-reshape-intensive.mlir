// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// RUN: mo-opt %s --gc-pipeline="trivial-kernel-opt=false constant-fold=false" -o %t_no_opt.mlir
// RUN: mt --execute --result-output-style=full %t_no_opt.mlir | FileCheck %s

// Generated from numpy:
// Add i to each input
// Reshape each input
// out = np.concatenate(inputs)
// out *= 4

mo.graph @concat_0axis_special_reshape_10_num10_test(%arg0 : !mo.tensor<[10], si32>) -> (!mo.tensor<[10, 10], si32>) {
	%cst_0 = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_0 = mo.static.broadcast_to(%cst_0) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%add_0 = mo.add(%arg0, %bcast_0) : !mo.tensor<[10], si32>
	%in_0 = mo.static.reshape(%add_0) : !mo.tensor<[10], si32> -> !mo.tensor<[1, 10], si32>
	%cst_1 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_1 = mo.static.broadcast_to(%cst_1) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%add_1 = mo.add(%arg0, %bcast_1) : !mo.tensor<[10], si32>
	%in_1 = mo.static.reshape(%add_1) : !mo.tensor<[10], si32> -> !mo.tensor<[1, 10], si32>
	%cst_2 = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_2 = mo.static.broadcast_to(%cst_2) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%add_2 = mo.add(%arg0, %bcast_2) : !mo.tensor<[10], si32>
	%in_2 = mo.static.reshape(%add_2) : !mo.tensor<[10], si32> -> !mo.tensor<[1, 10], si32>
	%cst_3 = mo.constant <#M.dense_array<4> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_3 = mo.static.broadcast_to(%cst_3) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%add_3 = mo.add(%arg0, %bcast_3) : !mo.tensor<[10], si32>
	%in_3 = mo.static.reshape(%add_3) : !mo.tensor<[10], si32> -> !mo.tensor<[1, 10], si32>
	%cst_4 = mo.constant <#M.dense_array<5> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_4 = mo.static.broadcast_to(%cst_4) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%add_4 = mo.add(%arg0, %bcast_4) : !mo.tensor<[10], si32>
	%in_4 = mo.static.reshape(%add_4) : !mo.tensor<[10], si32> -> !mo.tensor<[1, 10], si32>
	%cst_5 = mo.constant <#M.dense_array<6> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_5 = mo.static.broadcast_to(%cst_5) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%add_5 = mo.add(%arg0, %bcast_5) : !mo.tensor<[10], si32>
	%in_5 = mo.static.reshape(%add_5) : !mo.tensor<[10], si32> -> !mo.tensor<[1, 10], si32>
	%cst_6 = mo.constant <#M.dense_array<7> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_6 = mo.static.broadcast_to(%cst_6) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%add_6 = mo.add(%arg0, %bcast_6) : !mo.tensor<[10], si32>
	%in_6 = mo.static.reshape(%add_6) : !mo.tensor<[10], si32> -> !mo.tensor<[1, 10], si32>
	%cst_7 = mo.constant <#M.dense_array<8> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_7 = mo.static.broadcast_to(%cst_7) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%add_7 = mo.add(%arg0, %bcast_7) : !mo.tensor<[10], si32>
	%in_7 = mo.static.reshape(%add_7) : !mo.tensor<[10], si32> -> !mo.tensor<[1, 10], si32>
	%cst_8 = mo.constant <#M.dense_array<9> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_8 = mo.static.broadcast_to(%cst_8) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%add_8 = mo.add(%arg0, %bcast_8) : !mo.tensor<[10], si32>
	%in_8 = mo.static.reshape(%add_8) : !mo.tensor<[10], si32> -> !mo.tensor<[1, 10], si32>
	%cst_9 = mo.constant <#M.dense_array<10> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_9 = mo.static.broadcast_to(%cst_9) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%add_9 = mo.add(%arg0, %bcast_9) : !mo.tensor<[10], si32>
	%in_9 = mo.static.reshape(%add_9) : !mo.tensor<[10], si32> -> !mo.tensor<[1, 10], si32>
	%axis = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
	%concat = mo.concat[0] (%in_0,%in_1,%in_2,%in_3,%in_4,%in_5,%in_6,%in_7,%in_8,%in_9) : (!mo.tensor<[1, 10], si32>, !mo.tensor<[1, 10], si32>, !mo.tensor<[1, 10], si32>, !mo.tensor<[1, 10], si32>, !mo.tensor<[1, 10], si32>, !mo.tensor<[1, 10], si32>, !mo.tensor<[1, 10], si32>, !mo.tensor<[1, 10], si32>, !mo.tensor<[1, 10], si32>, !mo.tensor<[1, 10], si32>) -> !mo.tensor<[10, 10], si32>
	%factor = mo.constant <#M.dense_array<4> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_factor = mo.static.broadcast_to(%factor) : !mo.tensor<[1], si32> -> !mo.tensor<[10, 10], si32>
	%res = mo.mul(%concat, %bcast_factor) : !mo.tensor<[10, 10], si32>
	mo.output %res : !mo.tensor<[10, 10], si32>
}

// CHECK-LABEL: --- Running 'concat_0axis_special_reshape_10_num10_test'
// CHECK: 'concat_0axis_special_reshape_10_num10_test' returned tensor<10x10xsi32> [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]


mo.graph @concat_0axis_special_reshape_2_2_num10_test(%arg0 : !mo.tensor<[2, 2], si32>) -> (!mo.tensor<[10, 2, 2], si32>) {
	%cst_0 = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_0 = mo.static.broadcast_to(%cst_0) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
	%add_0 = mo.add(%arg0, %bcast_0) : !mo.tensor<[2, 2], si32>
	%in_0 = mo.static.reshape(%add_0) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 2, 2], si32>
	%cst_1 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_1 = mo.static.broadcast_to(%cst_1) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
	%add_1 = mo.add(%arg0, %bcast_1) : !mo.tensor<[2, 2], si32>
	%in_1 = mo.static.reshape(%add_1) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 2, 2], si32>
	%cst_2 = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_2 = mo.static.broadcast_to(%cst_2) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
	%add_2 = mo.add(%arg0, %bcast_2) : !mo.tensor<[2, 2], si32>
	%in_2 = mo.static.reshape(%add_2) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 2, 2], si32>
	%cst_3 = mo.constant <#M.dense_array<4> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_3 = mo.static.broadcast_to(%cst_3) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
	%add_3 = mo.add(%arg0, %bcast_3) : !mo.tensor<[2, 2], si32>
	%in_3 = mo.static.reshape(%add_3) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 2, 2], si32>
	%cst_4 = mo.constant <#M.dense_array<5> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_4 = mo.static.broadcast_to(%cst_4) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
	%add_4 = mo.add(%arg0, %bcast_4) : !mo.tensor<[2, 2], si32>
	%in_4 = mo.static.reshape(%add_4) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 2, 2], si32>
	%cst_5 = mo.constant <#M.dense_array<6> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_5 = mo.static.broadcast_to(%cst_5) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
	%add_5 = mo.add(%arg0, %bcast_5) : !mo.tensor<[2, 2], si32>
	%in_5 = mo.static.reshape(%add_5) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 2, 2], si32>
	%cst_6 = mo.constant <#M.dense_array<7> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_6 = mo.static.broadcast_to(%cst_6) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
	%add_6 = mo.add(%arg0, %bcast_6) : !mo.tensor<[2, 2], si32>
	%in_6 = mo.static.reshape(%add_6) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 2, 2], si32>
	%cst_7 = mo.constant <#M.dense_array<8> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_7 = mo.static.broadcast_to(%cst_7) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
	%add_7 = mo.add(%arg0, %bcast_7) : !mo.tensor<[2, 2], si32>
	%in_7 = mo.static.reshape(%add_7) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 2, 2], si32>
	%cst_8 = mo.constant <#M.dense_array<9> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_8 = mo.static.broadcast_to(%cst_8) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
	%add_8 = mo.add(%arg0, %bcast_8) : !mo.tensor<[2, 2], si32>
	%in_8 = mo.static.reshape(%add_8) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 2, 2], si32>
	%cst_9 = mo.constant <#M.dense_array<10> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_9 = mo.static.broadcast_to(%cst_9) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
	%add_9 = mo.add(%arg0, %bcast_9) : !mo.tensor<[2, 2], si32>
	%in_9 = mo.static.reshape(%add_9) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 2, 2], si32>
	%axis = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
	%concat = mo.concat[0] (%in_0,%in_1,%in_2,%in_3,%in_4,%in_5,%in_6,%in_7,%in_8,%in_9) : (!mo.tensor<[1, 2, 2], si32>, !mo.tensor<[1, 2, 2], si32>, !mo.tensor<[1, 2, 2], si32>, !mo.tensor<[1, 2, 2], si32>, !mo.tensor<[1, 2, 2], si32>, !mo.tensor<[1, 2, 2], si32>, !mo.tensor<[1, 2, 2], si32>, !mo.tensor<[1, 2, 2], si32>, !mo.tensor<[1, 2, 2], si32>, !mo.tensor<[1, 2, 2], si32>) -> !mo.tensor<[10, 2, 2], si32>
	%factor = mo.constant <#M.dense_array<4> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_factor = mo.static.broadcast_to(%factor) : !mo.tensor<[1], si32> -> !mo.tensor<[10, 2, 2], si32>
	%res = mo.mul(%concat, %bcast_factor) : !mo.tensor<[10, 2, 2], si32>
	mo.output %res : !mo.tensor<[10, 2, 2], si32>
}

// CHECK-LABEL: --- Running 'concat_0axis_special_reshape_2_2_num10_test'
// CHECK: 'concat_0axis_special_reshape_2_2_num10_test' returned tensor<10x2x2xsi32> [4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16, 20, 20, 20, 20, 24, 24, 24, 24, 28, 28, 28, 28, 32, 32, 32, 32, 36, 36, 36, 36, 40, 40, 40, 40]


mo.graph @concat_0axis_special_reshape_2_2_4_num10_test(%arg0 : !mo.tensor<[2, 2, 4], si32>) -> (!mo.tensor<[10, 2, 2, 4], si32>) {
	%cst_0 = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_0 = mo.static.broadcast_to(%cst_0) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2, 4], si32>
	%add_0 = mo.add(%arg0, %bcast_0) : !mo.tensor<[2, 2, 4], si32>
	%in_0 = mo.static.reshape(%add_0) : !mo.tensor<[2, 2, 4], si32> -> !mo.tensor<[1, 2, 2, 4], si32>
	%cst_1 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_1 = mo.static.broadcast_to(%cst_1) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2, 4], si32>
	%add_1 = mo.add(%arg0, %bcast_1) : !mo.tensor<[2, 2, 4], si32>
	%in_1 = mo.static.reshape(%add_1) : !mo.tensor<[2, 2, 4], si32> -> !mo.tensor<[1, 2, 2, 4], si32>
	%cst_2 = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_2 = mo.static.broadcast_to(%cst_2) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2, 4], si32>
	%add_2 = mo.add(%arg0, %bcast_2) : !mo.tensor<[2, 2, 4], si32>
	%in_2 = mo.static.reshape(%add_2) : !mo.tensor<[2, 2, 4], si32> -> !mo.tensor<[1, 2, 2, 4], si32>
	%cst_3 = mo.constant <#M.dense_array<4> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_3 = mo.static.broadcast_to(%cst_3) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2, 4], si32>
	%add_3 = mo.add(%arg0, %bcast_3) : !mo.tensor<[2, 2, 4], si32>
	%in_3 = mo.static.reshape(%add_3) : !mo.tensor<[2, 2, 4], si32> -> !mo.tensor<[1, 2, 2, 4], si32>
	%cst_4 = mo.constant <#M.dense_array<5> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_4 = mo.static.broadcast_to(%cst_4) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2, 4], si32>
	%add_4 = mo.add(%arg0, %bcast_4) : !mo.tensor<[2, 2, 4], si32>
	%in_4 = mo.static.reshape(%add_4) : !mo.tensor<[2, 2, 4], si32> -> !mo.tensor<[1, 2, 2, 4], si32>
	%cst_5 = mo.constant <#M.dense_array<6> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_5 = mo.static.broadcast_to(%cst_5) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2, 4], si32>
	%add_5 = mo.add(%arg0, %bcast_5) : !mo.tensor<[2, 2, 4], si32>
	%in_5 = mo.static.reshape(%add_5) : !mo.tensor<[2, 2, 4], si32> -> !mo.tensor<[1, 2, 2, 4], si32>
	%cst_6 = mo.constant <#M.dense_array<7> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_6 = mo.static.broadcast_to(%cst_6) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2, 4], si32>
	%add_6 = mo.add(%arg0, %bcast_6) : !mo.tensor<[2, 2, 4], si32>
	%in_6 = mo.static.reshape(%add_6) : !mo.tensor<[2, 2, 4], si32> -> !mo.tensor<[1, 2, 2, 4], si32>
	%cst_7 = mo.constant <#M.dense_array<8> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_7 = mo.static.broadcast_to(%cst_7) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2, 4], si32>
	%add_7 = mo.add(%arg0, %bcast_7) : !mo.tensor<[2, 2, 4], si32>
	%in_7 = mo.static.reshape(%add_7) : !mo.tensor<[2, 2, 4], si32> -> !mo.tensor<[1, 2, 2, 4], si32>
	%cst_8 = mo.constant <#M.dense_array<9> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_8 = mo.static.broadcast_to(%cst_8) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2, 4], si32>
	%add_8 = mo.add(%arg0, %bcast_8) : !mo.tensor<[2, 2, 4], si32>
	%in_8 = mo.static.reshape(%add_8) : !mo.tensor<[2, 2, 4], si32> -> !mo.tensor<[1, 2, 2, 4], si32>
	%cst_9 = mo.constant <#M.dense_array<10> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_9 = mo.static.broadcast_to(%cst_9) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2, 4], si32>
	%add_9 = mo.add(%arg0, %bcast_9) : !mo.tensor<[2, 2, 4], si32>
	%in_9 = mo.static.reshape(%add_9) : !mo.tensor<[2, 2, 4], si32> -> !mo.tensor<[1, 2, 2, 4], si32>
	%axis = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
	%concat = mo.concat[0] (%in_0,%in_1,%in_2,%in_3,%in_4,%in_5,%in_6,%in_7,%in_8,%in_9) : (!mo.tensor<[1, 2, 2, 4], si32>, !mo.tensor<[1, 2, 2, 4], si32>, !mo.tensor<[1, 2, 2, 4], si32>, !mo.tensor<[1, 2, 2, 4], si32>, !mo.tensor<[1, 2, 2, 4], si32>, !mo.tensor<[1, 2, 2, 4], si32>, !mo.tensor<[1, 2, 2, 4], si32>, !mo.tensor<[1, 2, 2, 4], si32>, !mo.tensor<[1, 2, 2, 4], si32>, !mo.tensor<[1, 2, 2, 4], si32>) -> !mo.tensor<[10, 2, 2, 4], si32>
	%factor = mo.constant <#M.dense_array<4> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_factor = mo.static.broadcast_to(%factor) : !mo.tensor<[1], si32> -> !mo.tensor<[10, 2, 2, 4], si32>
	%res = mo.mul(%concat, %bcast_factor) : !mo.tensor<[10, 2, 2, 4], si32>
	mo.output %res : !mo.tensor<[10, 2, 2, 4], si32>
}

// CHECK-LABEL: --- Running 'concat_0axis_special_reshape_2_2_4_num10_test'
// CHECK: 'concat_0axis_special_reshape_2_2_4_num10_test' returned tensor<10x2x2x4xsi32> [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]


mo.graph @concat_0axis_special_reshape_1_2_1_1_2_1_4_num10_test(%arg0 : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>) -> (!mo.tensor<[10, 1, 2, 1, 1, 2, 1, 4], si32>) {
	%cst_0 = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_0 = mo.static.broadcast_to(%cst_0) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%add_0 = mo.add(%arg0, %bcast_0) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in_0 = mo.static.reshape(%add_0) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32> -> !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>
	%cst_1 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_1 = mo.static.broadcast_to(%cst_1) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%add_1 = mo.add(%arg0, %bcast_1) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in_1 = mo.static.reshape(%add_1) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32> -> !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>
	%cst_2 = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_2 = mo.static.broadcast_to(%cst_2) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%add_2 = mo.add(%arg0, %bcast_2) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in_2 = mo.static.reshape(%add_2) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32> -> !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>
	%cst_3 = mo.constant <#M.dense_array<4> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_3 = mo.static.broadcast_to(%cst_3) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%add_3 = mo.add(%arg0, %bcast_3) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in_3 = mo.static.reshape(%add_3) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32> -> !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>
	%cst_4 = mo.constant <#M.dense_array<5> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_4 = mo.static.broadcast_to(%cst_4) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%add_4 = mo.add(%arg0, %bcast_4) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in_4 = mo.static.reshape(%add_4) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32> -> !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>
	%cst_5 = mo.constant <#M.dense_array<6> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_5 = mo.static.broadcast_to(%cst_5) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%add_5 = mo.add(%arg0, %bcast_5) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in_5 = mo.static.reshape(%add_5) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32> -> !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>
	%cst_6 = mo.constant <#M.dense_array<7> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_6 = mo.static.broadcast_to(%cst_6) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%add_6 = mo.add(%arg0, %bcast_6) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in_6 = mo.static.reshape(%add_6) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32> -> !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>
	%cst_7 = mo.constant <#M.dense_array<8> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_7 = mo.static.broadcast_to(%cst_7) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%add_7 = mo.add(%arg0, %bcast_7) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in_7 = mo.static.reshape(%add_7) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32> -> !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>
	%cst_8 = mo.constant <#M.dense_array<9> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_8 = mo.static.broadcast_to(%cst_8) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%add_8 = mo.add(%arg0, %bcast_8) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in_8 = mo.static.reshape(%add_8) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32> -> !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>
	%cst_9 = mo.constant <#M.dense_array<10> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_9 = mo.static.broadcast_to(%cst_9) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%add_9 = mo.add(%arg0, %bcast_9) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32>
	%in_9 = mo.static.reshape(%add_9) : !mo.tensor<[1, 2, 1, 1, 2, 1, 4], si32> -> !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>
	%axis = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
	%concat = mo.concat[0] (%in_0,%in_1,%in_2,%in_3,%in_4,%in_5,%in_6,%in_7,%in_8,%in_9) : (!mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>, !mo.tensor<[1, 1, 2, 1, 1, 2, 1, 4], si32>) -> !mo.tensor<[10, 1, 2, 1, 1, 2, 1, 4], si32>
	%factor = mo.constant <#M.dense_array<4> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_factor = mo.static.broadcast_to(%factor) : !mo.tensor<[1], si32> -> !mo.tensor<[10, 1, 2, 1, 1, 2, 1, 4], si32>
	%res = mo.mul(%concat, %bcast_factor) : !mo.tensor<[10, 1, 2, 1, 1, 2, 1, 4], si32>
	mo.output %res : !mo.tensor<[10, 1, 2, 1, 1, 2, 1, 4], si32>
}

// CHECK-LABEL: --- Running 'concat_0axis_special_reshape_1_2_1_1_2_1_4_num10_test'
// CHECK: 'concat_0axis_special_reshape_1_2_1_1_2_1_4_num10_test' returned tensor<10x1x2x1x1x2x1x4xsi32> [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
