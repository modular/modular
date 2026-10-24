// RUN: mo-opt %s -o %t_mojo.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t_mojo.mlir | FileCheck %s

// We use the simdwidth of int8 for this elementwise kernel.
// CHECK-LABEL: Running 'cast_si8_to_f32':
// CHECK-NEXT: width == simd_width_of[DType.int8](): True
// CHECK-NEXT: width == simd_width_of[DType.float32](): False
mo.graph @cast_si8_to_f32(%arg0: !mo.tensor<[512], si8>) -> !mo.tensor<[512], f32> {
    %0 = mo.custom [symbol = "print_vector_size", device = #M.device_ref<"cpu", 0>] (%arg0) : (!mo.tensor<[512], si8>) -> !mo.tensor<[512], si8>
    %1 = mo.custom [symbol = "imposter_cast_elementwise", device = #M.device_ref<"cpu", 0>] (%0) : (!mo.tensor<[512], si8>) -> !mo.tensor<[512], f32>
    mo.output %1 : !mo.tensor<[512], f32>
}

// We use the simdwidth of int8 for this elementwise kernel.
// CHECK-LABEL: Running 'cast_f32_to_si8':
// CHECK-NEXT: width == simd_width_of[DType.int8](): True
// CHECK-NEXT: width == simd_width_of[DType.float32](): False
mo.graph @cast_f32_to_si8(%arg0: !mo.tensor<[512], f32>) -> !mo.tensor<[512], si8> {
    %0 = mo.custom [symbol = "print_vector_size", device = #M.device_ref<"cpu", 0>] (%arg0) : (!mo.tensor<[512], f32>) -> !mo.tensor<[512], f32>
    %1 = mo.custom [symbol = "imposter_cast_elementwise", device = #M.device_ref<"cpu", 0>] (%0) : (!mo.tensor<[512], f32>) -> !mo.tensor<[512], si8>
    mo.output %1 : !mo.tensor<[512], si8>
}

// We use the simdwidth of int8 for this elementwise kernel.
// CHECK-LABEL: Running 'elementwise_si8':
// CHECK-NEXT: width == simd_width_of[DType.int8](): True
// CHECK-NEXT: width == simd_width_of[DType.float32](): False
mo.graph @elementwise_si8(%arg0: !mo.tensor<[512], si8>) -> !mo.tensor<[512], si8> {
    %0 = mo.custom [symbol = "print_vector_size", device = #M.device_ref<"cpu", 0>] (%arg0) : (!mo.tensor<[512], si8>) -> !mo.tensor<[512], si8>
    mo.output %0 : !mo.tensor<[512], si8>
}

// We use the simdwidth of f32 for this elementwise kernel.
// CHECK-LABEL: Running 'elementwise_f32':
// CHECK-NEXT: width == simd_width_of[DType.int8](): False
// CHECK-NEXT: width == simd_width_of[DType.float32](): True
mo.graph @elementwise_f32(%arg0: !mo.tensor<[512], f32>) -> !mo.tensor<[512], f32> {
    %0 = mo.custom [symbol = "print_vector_size", device = #M.device_ref<"cpu", 0>] (%arg0) : (!mo.tensor<[512], f32>) -> !mo.tensor<[512], f32>
    mo.output %0 : !mo.tensor<[512], f32>
}
