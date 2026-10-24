// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// A fused `TensorTuple`: `trait_tensor_tuple_fused_concat` takes
// `inputs: TensorTuple[*Ins]` with `Ins: TypeList[Trait=Input & Fused &
// LoadAccess, ...]`, so each producer is fused into its tuple element and the
// element is its `_MoggInputLoadFusionTensor`. Like `mo.concat`, the kernel
// captures the tuple in one closure that loads element `k` by comptime index.
// The elements differ in dtype and length.

// CHECK: 'trait_tensor_tuple_fused_concat_mixed' returned tensor<5xf32> [2.000000e+00, 4.000000e+00, 2.250000e+00, 6.250000e+00, 1.225000e+01]
mo.graph @trait_tensor_tuple_fused_concat_mixed() -> !mo.tensor<[5], f32> {
  %a = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %b = mo.constant <#M.dense_array<1.5, 2.5, 3.5> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %a2 = mo.add(%a, %a) : !mo.tensor<[2], si32>
  %b2 = mo.mul(%b, %b) : !mo.tensor<[3], f32>

  %res = mo.custom [symbol = "trait_tensor_tuple_fused_concat", device = #M.device_ref<"cpu", 0>] (%a2, %b2) : (
    !mo.tensor<[2], si32>,
    !mo.tensor<[3], f32>
  ) -> !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}

// Three elements, one of which is a plain graph constant with no producer to
// fuse.
// CHECK: 'trait_tensor_tuple_fused_concat_three' returned tensor<6xsi32> [2, 4, 30, 1, 2, 3]
mo.graph @trait_tensor_tuple_fused_concat_three() -> !mo.tensor<[6], si32> {
  %a = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %b = mo.constant <#M.dense_array<10> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %c = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %a2 = mo.add(%a, %a) : !mo.tensor<[2], si32>
  %b3 = mo.mul(%b, %b) : !mo.tensor<[1], si32>
  %b4 = mo.sub(%b3, %b3) : !mo.tensor<[1], si32>
  %b5 = mo.add(%b4, %b) : !mo.tensor<[1], si32>
  %b6 = mo.add(%b5, %b5) : !mo.tensor<[1], si32>
  %b7 = mo.add(%b6, %b) : !mo.tensor<[1], si32>

  %res = mo.custom [symbol = "trait_tensor_tuple_fused_concat", device = #M.device_ref<"cpu", 0>] (%a2, %b7, %c) : (
    !mo.tensor<[2], si32>,
    !mo.tensor<[1], si32>,
    !mo.tensor<[3], si32>
  ) -> !mo.tensor<[6], si32>
  mo.output %res : !mo.tensor<[6], si32>
}
