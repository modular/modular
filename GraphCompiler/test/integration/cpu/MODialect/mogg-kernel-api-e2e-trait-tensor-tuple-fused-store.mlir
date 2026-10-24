// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// A fused-output `TensorTuple`: `trait_tensor_tuple_fused_store_split` takes
// `outputs: TensorTuple[*Outs]` with `Outs: TypeList[Trait=Output & Fused &
// StoreAccess, ...]`, so each output's consumer is fused into its tuple element
// as an epilogue, and the kernel's store runs it. The elements differ in dtype
// and length.

// `%a`'s epilogue reads an extra operand, and `%b`'s casts.
// CHECK: 'trait_tensor_tuple_fused_store_split_mixed' returned tensor<2xf32> [1.100000e+01, 2.200000e+01]
// CHECK-NEXT: , tensor<3xf64> [3.000000e+00, 4.000000e+00, 5.000000e+00]
mo.graph @trait_tensor_tuple_fused_store_split_mixed() -> (!mo.tensor<[2], f32>, !mo.tensor<[3], f64>) {
  %x = mo.constant <#M.dense_array<1.5, 2.5, 3.5, 4.5, 5.5> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %c = mo.constant <#M.dense_array<9.5, 19.5> : tensor<2xf32>> : !mo.tensor<[2], f32>

  %a, %b = mo.custom [symbol = "trait_tensor_tuple_fused_store_split", device = #M.device_ref<"cpu", 0>] (%x) : (
    !mo.tensor<[5], f32>
  ) -> (!mo.tensor<[2], f32>, !mo.tensor<[3], si32>)
  %a2 = mo.add(%a, %c) : !mo.tensor<[2], f32>
  %b2 = mo.cast(%b) : (!mo.tensor<[3], si32>) -> !mo.tensor<[3], f64>
  mo.output %a2, %b2 : !mo.tensor<[2], f32>, !mo.tensor<[3], f64>
}

// Three elements; the middle one has no consumer to fuse, so its store writes
// the output directly.
// CHECK: 'trait_tensor_tuple_fused_store_split_three' returned tensor<1xsi32> [2]
// CHECK-NEXT: , tensor<2xsi32> [2, 3]
// CHECK-NEXT: , tensor<3xsi32> [8, 10, 12]
mo.graph @trait_tensor_tuple_fused_store_split_three() -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>) {
  %x = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6> : tensor<6xsi32>> : !mo.tensor<[6], si32>

  %a, %b, %c = mo.custom [symbol = "trait_tensor_tuple_fused_store_split", device = #M.device_ref<"cpu", 0>] (%x) : (
    !mo.tensor<[6], si32>
  ) -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>)
  %a2 = mo.add(%a, %a) : !mo.tensor<[1], si32>
  %c2 = mo.add(%c, %c) : !mo.tensor<[3], si32>
  mo.output %a2, %b, %c2 : !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
}
