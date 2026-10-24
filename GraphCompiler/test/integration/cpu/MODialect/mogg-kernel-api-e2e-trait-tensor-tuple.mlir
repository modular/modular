// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// `TensorTuple`s `xs: TensorTuple[*Ts]` with `Ts: TypeList[Trait=..., ...]`, one
// per variadic. `trait_tensor_tuple_add` takes a variadic output tuple and a
// variadic input tuple and computes `outputs[i] = inputs[i] + bias[0]`. Each element is its own trait argument,
// so the elements may differ in length and dtype.

// CHECK: 'trait_tensor_tuple_add_mixed_dtypes' returned tensor<2xf32> [3.000000e+00, 4.000000e+00]
// CHECK-NEXT: , tensor<3xsi32> [5, 6, 12]
mo.graph @trait_tensor_tuple_add_mixed_dtypes() -> (!mo.tensor<[2], f32>, !mo.tensor<[3], si32>) {
  %in1 = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %in2 = mo.constant <#M.dense_array<3.0, 4.0, 10.0> : tensor<3xf64>> : !mo.tensor<[3], f64>
  %bias = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[1], f32>

  %out1, %out2 = mo.custom [symbol = "trait_tensor_tuple_add", device = #M.device_ref<"cpu", 0>] (%bias, %in1, %in2) : (
    !mo.tensor<[1], f32>,
    !mo.tensor<[2], si32>,
    !mo.tensor<[3], f64>
  ) -> (!mo.tensor<[2], f32>, !mo.tensor<[3], si32>)
  mo.output %out1, %out2 : !mo.tensor<[2], f32>, !mo.tensor<[3], si32>
}

// CHECK: 'trait_tensor_tuple_add_size3_si32' returned tensor<2xsi32> [1001, 1002]
// CHECK-NEXT: , tensor<3xsi32> [1010, 1020, 1030]
// CHECK-NEXT: , tensor<1xsi32> [1100]
mo.graph @trait_tensor_tuple_add_size3_si32() -> (!mo.tensor<[2], si32>, !mo.tensor<[3], si32>, !mo.tensor<[1], si32>) {
  %in1 = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %in2 = mo.constant <#M.dense_array<10, 20, 30> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %in3 = mo.constant <#M.dense_array<100> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %bias = mo.constant <#M.dense_array<1000> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  %out1, %out2, %out3 = mo.custom [symbol = "trait_tensor_tuple_add", device = #M.device_ref<"cpu", 0>] (%bias, %in1, %in2, %in3) : (
    !mo.tensor<[1], si32>,
    !mo.tensor<[2], si32>,
    !mo.tensor<[3], si32>,
    !mo.tensor<[1], si32>
  ) -> (!mo.tensor<[2], si32>, !mo.tensor<[3], si32>, !mo.tensor<[1], si32>)
  mo.output %out1, %out2, %out3 : !mo.tensor<[2], si32>, !mo.tensor<[3], si32>, !mo.tensor<[1], si32>
}

// One-element tuples bind exactly as single trait arguments would.
// CHECK: 'trait_tensor_tuple_add_size1_f32' returned tensor<2xf32> [8.000000e+00, 9.000000e+00]
mo.graph @trait_tensor_tuple_add_size1_f32() -> !mo.tensor<[2], f32> {
  %in1 = mo.constant <#M.dense_array<7.0, 8.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %bias = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[1], f32>

  %res = mo.custom [symbol = "trait_tensor_tuple_add", device = #M.device_ref<"cpu", 0>] (%bias, %in1) : (
    !mo.tensor<[1], f32>,
    !mo.tensor<[2], f32>
  ) -> !mo.tensor<[2], f32>
  mo.output %res : !mo.tensor<[2], f32>
}
