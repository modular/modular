// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// Generated from numpy:

// (input_data[start:stop:step] + input_data[start:stop:step]) * 3 for all in:
//
// input = np.arange(...)
// For all axes and input / indices shapes in the range:
// input_data_shapes = [(3, 11, 5)]
// start_stop_step = [
//        [(0, 0, 0), (2, -4, 3), (1, 1, 1)],
//        [(2, -4, 3), (0, 0, 0), (-1, -1, -1)],
//        [(1, 10, 0), (3, 0, 3), (1, -3, 2)],
//        [(3, 0, 3), (1, 10, 0), (-1, 3, -2)],
//        [(1, 4, 2), (-1, -2, -1), (1, 1, 1)],
//        [(-1, -2, -1), (1, 4, 2), (-1, -1, -1)]
//    ]

mo.graph @slice_fused_inshape_3_11_5_start_0_0_0_stop_2_n4_3_step_1_1_1_test() -> (!mo.tensor<[2, 7, 3], si32>) {
	%data = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164> : tensor<3x11x5xsi32>> : !mo.tensor<[3, 11, 5], si32>
	%start = mo.constant <#M.dense_array<0,0,0> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%stop = mo.constant <#M.dense_array<2,-4,3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%step = mo.constant <#M.dense_array<1,1,1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%const = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%res = mo.slice(%data, %start, %stop, %step) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[2, 7, 3], si32>
	%add = mo.add(%res, %res) : !mo.tensor<[2, 7, 3], si32>
	%bcast = mo.static.broadcast_to(%const) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 7, 3], si32>
	%pow = mo.mul(%add, %bcast) : !mo.tensor<[2, 7, 3], si32>
	mo.output %pow : !mo.tensor<[2, 7, 3], si32>
}

// CHECK: 'slice_fused_inshape_3_11_5_start_0_0_0_stop_2_n4_3_step_1_1_1_test' returned tensor<2x7x3xsi32> [0, 6, 12, 30, 36, 42, 60, 66, 72, 90, 96, 102, 120, 126, 132, 150, 156, 162, 180, 186, 192, 330, 336, 342, 360, 366, 372, 390, 396, 402, 420, 426, 432, 450, 456, 462, 480, 486, 492, 510, 516, 522]


mo.graph @slice_fused_inshape_3_11_5_start_2_n4_3_stop_0_0_0_step_n1_n1_n1_test() -> (!mo.tensor<[2, 7, 3], si32>) {
	%data = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164> : tensor<3x11x5xsi32>> : !mo.tensor<[3, 11, 5], si32>
	%start = mo.constant <#M.dense_array<2,-4,3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%stop = mo.constant <#M.dense_array<0,0,0> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%step = mo.constant <#M.dense_array<-1,-1,-1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%const = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%res = mo.slice(%data, %start, %stop, %step) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[2, 7, 3], si32>
	%add = mo.add(%res, %res) : !mo.tensor<[2, 7, 3], si32>
	%bcast = mo.static.broadcast_to(%const) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 7, 3], si32>
	%pow = mo.mul(%add, %bcast) : !mo.tensor<[2, 7, 3], si32>
	mo.output %pow : !mo.tensor<[2, 7, 3], si32>
}

// CHECK: 'slice_fused_inshape_3_11_5_start_2_n4_3_stop_0_0_0_step_n1_n1_n1_test' returned tensor<2x7x3xsi32> [888, 882, 876, 858, 852, 846, 828, 822, 816, 798, 792, 786, 768, 762, 756, 738, 732, 726, 708, 702, 696, 558, 552, 546, 528, 522, 516, 498, 492, 486, 468, 462, 456, 438, 432, 426, 408, 402, 396, 378, 372, 366]


mo.graph @slice_fused_inshape_3_11_5_start_1_10_0_stop_3_0_3_step_1_n3_2_test() -> (!mo.tensor<[2, 4, 2], si32>) {
	%data = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164> : tensor<3x11x5xsi32>> : !mo.tensor<[3, 11, 5], si32>
	%start = mo.constant <#M.dense_array<1,10,0> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%stop = mo.constant <#M.dense_array<3,0,3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%step = mo.constant <#M.dense_array<1,-3,2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%const = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%res = mo.slice(%data, %start, %stop, %step) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[2, 4, 2], si32>
	%add = mo.add(%res, %res) : !mo.tensor<[2, 4, 2], si32>
	%bcast = mo.static.broadcast_to(%const) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 4, 2], si32>
	%pow = mo.mul(%add, %bcast) : !mo.tensor<[2, 4, 2], si32>
	mo.output %pow : !mo.tensor<[2, 4, 2], si32>
}

// CHECK: 'slice_fused_inshape_3_11_5_start_1_10_0_stop_3_0_3_step_1_n3_2_test' returned tensor<2x4x2xsi32> [630, 642, 540, 552, 450, 462, 360, 372, 960, 972, 870, 882, 780, 792, 690, 702]


mo.graph @slice_fused_inshape_3_11_5_start_3_0_3_stop_1_10_0_step_n1_3_n2_test() -> (!mo.tensor<[1, 4, 2], si32>) {
	%data = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164> : tensor<3x11x5xsi32>> : !mo.tensor<[3, 11, 5], si32>
	%start = mo.constant <#M.dense_array<3,0,3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%stop = mo.constant <#M.dense_array<1,10,0> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%step = mo.constant <#M.dense_array<-1,3,-2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%const = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%res = mo.slice(%data, %start, %stop, %step) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[1, 4, 2], si32>
	%add = mo.add(%res, %res) : !mo.tensor<[1, 4, 2], si32>
	%bcast = mo.static.broadcast_to(%const) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 4, 2], si32>
	%pow = mo.mul(%add, %bcast) : !mo.tensor<[1, 4, 2], si32>
	mo.output %pow : !mo.tensor<[1, 4, 2], si32>
}

// CHECK: 'slice_fused_inshape_3_11_5_start_3_0_3_stop_1_10_0_step_n1_3_n2_test' returned tensor<1x4x2xsi32> [678, 666, 768, 756, 858, 846, 948, 936]


mo.graph @slice_fused_inshape_3_11_5_start_1_4_2_stop_n1_n2_n1_step_1_1_1_test() -> (!mo.tensor<[1, 5, 2], si32>) {
	%data = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164> : tensor<3x11x5xsi32>> : !mo.tensor<[3, 11, 5], si32>
	%start = mo.constant <#M.dense_array<1,4,2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%stop = mo.constant <#M.dense_array<-1,-2,-1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%step = mo.constant <#M.dense_array<1,1,1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%const = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%res = mo.slice(%data, %start, %stop, %step) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[1, 5, 2], si32>
	%add = mo.add(%res, %res) : !mo.tensor<[1, 5, 2], si32>
	%bcast = mo.static.broadcast_to(%const) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 5, 2], si32>
	%pow = mo.mul(%add, %bcast) : !mo.tensor<[1, 5, 2], si32>
	mo.output %pow : !mo.tensor<[1, 5, 2], si32>
}

// CHECK: 'slice_fused_inshape_3_11_5_start_1_4_2_stop_n1_n2_n1_step_1_1_1_test' returned tensor<1x5x2xsi32> [462, 468, 492, 498, 522, 528, 552, 558, 582, 588]


mo.graph @slice_fused_inshape_3_11_5_start_n1_n2_n1_stop_1_4_2_step_n1_n1_n1_test() -> (!mo.tensor<[1, 5, 2], si32>) {
	%data = mo.constant <#M.dense_array<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164> : tensor<3x11x5xsi32>> : !mo.tensor<[3, 11, 5], si32>
	%start = mo.constant <#M.dense_array<-1,-2,-1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%stop = mo.constant <#M.dense_array<1,4,2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%step = mo.constant <#M.dense_array<-1,-1,-1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
	%const = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%res = mo.slice(%data, %start, %stop, %step) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[1, 5, 2], si32>
	%add = mo.add(%res, %res) : !mo.tensor<[1, 5, 2], si32>
	%bcast = mo.static.broadcast_to(%const) : !mo.tensor<[1], si32> -> !mo.tensor<[1, 5, 2], si32>
	%pow = mo.mul(%add, %bcast) : !mo.tensor<[1, 5, 2], si32>
	mo.output %pow : !mo.tensor<[1, 5, 2], si32>
}

// CHECK: 'slice_fused_inshape_3_11_5_start_n1_n2_n1_stop_1_4_2_step_n1_n1_n1_test' returned tensor<1x5x2xsi32> [954, 948, 924, 918, 894, 888, 864, 858, 834, 828]
