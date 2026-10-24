// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" %s -o %t.mlir
// RUN: mt --execute %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'test_opaques_and_chains_non_raising'
// CHECK-LITERAL: MyCustomScalarSI32.__init__ 100
// CHECK-LITERAL-NEXT: MyCustomScalarSI32.__init__ 42
// CHECK-LITERAL: 'test_opaques_and_chains_non_raising' returned tensor([[142]], dtype=si32, shape=[1])
mo.graph @test_opaques_and_chains_non_raising() -> (!mo.tensor<[1], si32>) {
  %const100 = mo.constant <#M.dense_array<100> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %const42 = mo.constant <#M.dense_array<42> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  %ch0 = mo.chain.create()

  %opaque1, %ch1 = mo.custom [symbol = "tensor_to_custom_scalar_si32", device = #M.device_ref<"cpu", 0>] (%const100, %ch0) :
    (!mo.tensor<[1], si32>, !mo.chain) -> (!mo.opaque<"MyCustomScalarSI32">, !mo.chain)

  %opaque2, %ch2 = mo.custom [symbol = "tensor_to_custom_scalar_si32", device = #M.device_ref<"cpu", 0>] (%const42, %ch1) :
    (!mo.tensor<[1], si32>, !mo.chain) -> (!mo.opaque<"MyCustomScalarSI32">, !mo.chain)

  %res, %ch3 = mo.custom [symbol = "opaque_add_to_tensor_si32", device = #M.device_ref<"cpu", 0>] (%opaque1, %opaque2, %ch2) :
    (!mo.opaque<"MyCustomScalarSI32">, !mo.opaque<"MyCustomScalarSI32">, !mo.chain) -> (!mo.tensor<[1], si32>, !mo.chain)

  mo.output %res: !mo.tensor<[1], si32>
}

// CHECK-LABEL: Running 'test_opaques_and_chains_raising'
// CHECK-LITERAL: MyCustomScalarSI32.__init__ 100
// CHECK-LITERAL-NEXT: MyCustomScalarSI32.__init__ 42
// CHECK-LITERAL: 'test_opaques_and_chains_raising' returned tensor([[142]], dtype=si32, shape=[1])
mo.graph @test_opaques_and_chains_raising() -> (!mo.tensor<[1], si32>) {
  %const100 = mo.constant <#M.dense_array<100> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %const42 = mo.constant <#M.dense_array<42> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  %ch0 = mo.chain.create()

  %opaque1, %ch1 = mo.custom [symbol = "tensor_to_custom_scalar_si32", device = #M.device_ref<"cpu", 0>] (%const100, %ch0) :
    (!mo.tensor<[1], si32>, !mo.chain) -> (!mo.opaque<"MyCustomScalarSI32">, !mo.chain)

  %opaque2, %ch2 = mo.custom [symbol = "tensor_to_custom_scalar_si32", device = #M.device_ref<"cpu", 0>] (%const42, %ch1) :
    (!mo.tensor<[1], si32>, !mo.chain) -> (!mo.opaque<"MyCustomScalarSI32">, !mo.chain)

  %res, %ch3 = mo.custom [symbol = "opaque_add_to_tensor_si32_raises", device = #M.device_ref<"cpu", 0>] (%opaque1, %opaque2, %ch2) :
    (!mo.opaque<"MyCustomScalarSI32">, !mo.opaque<"MyCustomScalarSI32">, !mo.chain) -> (!mo.tensor<[1], si32>, !mo.chain)

  mo.output %res: !mo.tensor<[1], si32>
}
