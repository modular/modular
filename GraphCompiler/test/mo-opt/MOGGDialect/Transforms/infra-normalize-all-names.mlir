// RUN: mo-opt %s -mogg-normalize-all-names -allow-unregistered-dialect -split-input-file | FileCheck %s

// COM: Test that symbol names are normalized to __S0, __S1, etc.
// COM: and that references to those symbols are updated.

// CHECK-LABEL: kgen.generator @__S0
kgen.generator @my_first_function(%arg0: si32) -> si32 {
  hlcf.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @__S1
kgen.generator @my_second_function(%arg0: si32) -> si32 {
  // CHECK: kgen.call @__S0
  %x = kgen.call @my_first_function(%arg0) : (si32) -> si32
  hlcf.return %x : si32
}

// -----

// COM: Test call chain with multiple symbol references.

// CHECK-LABEL: kgen.generator @__S0
kgen.generator @func_a(%arg0: si32) -> si32 {
  hlcf.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @__S1
kgen.generator @func_b(%arg0: si32) -> si32 {
  // CHECK: kgen.call @__S0
  %x = kgen.call @func_a(%arg0) : (si32) -> si32
  hlcf.return %x : si32
}

// CHECK-LABEL: kgen.generator @__S2
kgen.generator @func_c(%arg0: si32) -> si32 {
  // CHECK: kgen.call @__S1
  %x = kgen.call @func_b(%arg0) : (si32) -> si32
  // CHECK: kgen.call @__S0
  %y = kgen.call @func_a(%arg0) : (si32) -> si32
  hlcf.return %x : si32
}

// -----

// COM: Test that mo.graph operations normalize both symbol names and input parameters.
// COM: This test is based on the structure in example.mlir to ensure:
// COM: 1. Graph symbol is normalized to __S0
// COM: 2. Graph signature parameters (D1, D10) are normalized to __P0, __P1
// COM: 3. Operations that declare output parameters (like mo.random.normal) are normalized
// COM: 4. kgen.param.declare operations in the body get subsequent numbers
// COM: 5. References to parameters are updated correctly throughout

// CHECK-LABEL: mo.graph @__S0<__P0, __P1>
// CHECK-SAME: %arg1: !mo.tensor<[0, __P0, __P1], bool>
mo.graph @my_graph<D1, D10>(%arg0: !mo.tensor<[1], ui64>, %arg1: !mo.tensor<[0, D1, D10], bool>) -> (!mo.tensor<[D10], f32>) {
  %c0 = mo.constant <#M.dense_array<0.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %c1 = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  // CHECK: mosh.param.to_value: !mosh.ape = <[__P1]>
  %shape_val = mosh.param.to_value: !mosh.ape = <[D10]>
  %shape_tensor = mo.shape.to_tensor(%shape_val) -> !mo.tensor<[1], si64>

  // CHECK: mo.random.normal<() -> __P2>
  // CHECK-SAME: !mo.tensor<[__P2], f32>
  %random = mo.random.normal<() -> RMO_OUTPUT_DIM>(%shape_tensor, %c0, %c1, %arg0) : (!mo.tensor<[1], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[RMO_OUTPUT_DIM], f32>

  // CHECK: kgen.param.declare __P3: scalar<bool> = <identical(:scalar<si64> __P1, __P2)>
  kgen.param.declare RMO_SH_EQ: scalar<bool> = <identical(:scalar<si64> RMO_OUTPUT_DIM, D10)>
  // CHECK: kgen.param.declare __P4: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "shape check failed" : !kgen.string}, __P3)>
  kgen.param.declare RMO_SH_CH: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "shape check failed" : !kgen.string}, RMO_SH_EQ)>

  // CHECK: mo.rebind(%{{.*}}) : !mo.tensor<[__P2], f32> -> !mo.tensor<[__P1], f32>
  %rebind = mo.rebind(%random) : !mo.tensor<[RMO_OUTPUT_DIM], f32> -> !mo.tensor<[D10], f32>
  // CHECK: mo.output %{{.*}} : !mo.tensor<[__P1], f32>
  mo.output %rebind : !mo.tensor<[D10], f32>
}
