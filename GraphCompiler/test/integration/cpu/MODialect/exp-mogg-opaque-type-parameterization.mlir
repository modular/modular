// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s

// RUN: mo-opt %s --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s --check-prefix=CHECK-MT

// CHECK-LABEL: @parameterized_opaque_type
mo.graph @parameterized_opaque_type() -> !mo.tensor<[12], si32> {

  // COM: CHECK: mogg.routine () : ()  -> !mo.opaque<"SIMDPair", {S0 = 4 : i64, S1 = 8 : i64}>
  // COM: CHECK: "mogg.to_mogg"({{.*}}) {{.*}} : (!kgen.struct<(simd<4, si32>, simd<8, si32>)>) -> !mo.opaque<"SIMDPair", {S0 = 4 : i64, S1 = 8 : i64}>
  %pair = mo.custom [symbol = "make_simd_pair", device = #M.device_ref<"cpu", 0>, parameters = {}] () : () -> (!mo.opaque<"SIMDPair", {S0 = 4, S1 = 8}>)

  // COM: CHECK: mogg.routine ({{.*}}) : (%{{.*}}: !mo.opaque<"SIMDPair", {S0 = 4 : i64, S1 = 8 : i64}>)  -> !mo.tensor<[12], si32>
  // COM: CHECK: "mogg.to_kgen"({{.*}}) <{staticTensorSpecName = "{{.*}}", tensorIOType = "OutputTensor"}> : (!mo.tensor<[12], si32>) -> !kgen.struct<(pointer<none>, array<1, scalar<si64>>, array<1, scalar<si64>>)>
  // COM: CHECK: "mogg.to_kgen"({{.*}}) : (!mo.opaque<"SIMDPair", {S0 = 4 : i64, S1 = 8 : i64}>) -> !kgen.pointer<struct<(simd<4, si32>, simd<8, si32>)>>
  %out = mo.custom [symbol = "kernel_with_parameterized_opaque", device = #M.device_ref<"cpu", 0>] (%pair) : (!mo.opaque<"SIMDPair", {S0 = 4, S1 = 8}>) -> (!mo.tensor<[12], si32>)
  mo.output %out : !mo.tensor<[12], si32>
}

// CHECK-MT: --- Running 'parameterized_opaque_type':
// CHECK-MT-NEXT: 'parameterized_opaque_type' returned tensor<12xsi32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
