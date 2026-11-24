// RUN: mo-opt %s --gc-pipeline="use-patterns=false" -o %t_new_api.mlir
// RUN: mt --execute --model-inputs=zeros --result-output-style=full %t_new_api.mlir | FileCheck %s --check-prefix=CHECK-NEW

// A simple graph that computes: concat[(0 + 1) * 4, (0 + 2) * 4]

// CHECK-IR-LABEL: @concat_0axis_fusion_test
mo.graph @concat_0axis_fusion_test(%arg0 : !mo.tensor<[10], si32>) -> (!mo.tensor<[20], si32>) {
	%cst_0 = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_0 = mo.static.broadcast_to(%cst_0) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%in_0 = mo.add(%arg0, %bcast_0) : !mo.tensor<[10], si32>


	%cst_1 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_1 = mo.static.broadcast_to(%cst_1) : !mo.tensor<[1], si32> -> !mo.tensor<[10], si32>
	%in_1 = mo.add(%arg0, %bcast_1) : !mo.tensor<[10], si32>

	%axis = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK-IR-LABEL:  @input_1_fn_tuple
  // CHECK-IR-NEXT:     "simd_load"
  // CHECK-IR-NEXT:     "simd_load"
  // CHECK-IR-NEXT:     "mo.add"
  // CHECK-IR-NEXT:     mogg.output

  // CHECK-IR-LABEL:  @input_1_fn_tuple
  // CHECK-IR-NEXT:     "simd_load"
  // CHECK-IR-NEXT:     "simd_load"
  // CHECK-IR-NEXT:     "mo.add"
  // CHECK-IR-NEXT:     mogg.output
	%concat = mo.concat[0] (%in_0,%in_1) : (!mo.tensor<[10], si32>, !mo.tensor<[10], si32>) -> !mo.tensor<[20], si32>

	%factor = mo.constant <#M.dense_array<4> : tensor<1xsi32>> : !mo.tensor<[1], si32>
	%bcast_factor = mo.static.broadcast_to(%factor) : !mo.tensor<[1], si32> -> !mo.tensor<[20], si32>
	%res = mo.mul(%concat, %bcast_factor) : !mo.tensor<[20], si32>

	mo.output %res : !mo.tensor<[20], si32>
}

 // CHECK-LABEL:  --- Running 'concat_0axis_fusion_test':
 // CHECK-NEXT:   In override concat.
 // CHECK-NEXT:   'concat_0axis_fusion_test' returned tensor<20xsi32> [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]


// CHECK-NEW-LABEL:  --- Running 'concat_0axis_fusion_test':
// CHECK-NEW-NEXT:   'concat_0axis_fusion_test' returned tensor<20xsi32> [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
