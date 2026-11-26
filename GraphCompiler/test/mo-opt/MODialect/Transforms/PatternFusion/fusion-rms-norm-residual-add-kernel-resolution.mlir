// `--mo-pattern-fusion` forms `mo.composite.rms_norm_residual_add` only where
// the kernel registry selects a kernel for it on the norm's device.
// RUN: mo-opt %s --split-input-file --mo-pattern-fusion | FileCheck %s

// A kernel registered for CUDA resolves on a CUDA device, so the single norm
// fuses.
!pred = !kgen.generator<!lit.generator<<"label": string, "api": string, "arch": string, "model": string>scalar<bool>>>
#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>
module attributes {mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100">}} {
mogg.kernel.decl @cuda_rms_norm_residual_add registration = "mo.composite.rms_norm_residual_add" platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred execute = #kf struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @singleNormResolves
// CHECK-NOT:     mo.composite.reduce.rms_norm
// CHECK:         mo.composite.rms_norm_residual_add
// CHECK-NOT:     mo.composite.reduce.rms_norm
mo.graph @singleNormResolves(%x: !mo.tensor<[8, 128], f32, gpu:0>, %r: !mo.tensor<[8, 128], f32, gpu:0>, %w: !mo.tensor<[128], f32, gpu:0>) -> (!mo.tensor<[8, 128], f32, gpu:0>, !mo.tensor<[8, 128], f32, gpu:0>) {
  %sum = mo.add(%r, %x) : !mo.tensor<[8, 128], f32, gpu:0>
  %y = mo.composite.reduce.rms_norm(%sum, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[8, 128], f32, gpu:0>, !mo.tensor<[128], f32, gpu:0>) -> !mo.tensor<[8, 128], f32, gpu:0>
  mo.output %y, %sum : !mo.tensor<[8, 128], f32, gpu:0>, !mo.tensor<[8, 128], f32, gpu:0>
}
}

// -----

// The same kernel doesn't resolve on a HIP device, so the add and the norm
// stay.
!pred = !kgen.generator<!lit.generator<<"label": string, "api": string, "arch": string, "model": string>scalar<bool>>>
#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>
module attributes {mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "hip", "gfx942", "AMD Instinct MI300X">}} {
mogg.kernel.decl @cuda_rms_norm_residual_add registration = "mo.composite.rms_norm_residual_add" platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred execute = #kf struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @singleNormOtherDevice
// CHECK-NOT:     mo.composite.rms_norm_residual_add
// CHECK:         mo.add
// CHECK:         mo.composite.reduce.rms_norm
// CHECK-NOT:     mo.composite.rms_norm_residual_add
mo.graph @singleNormOtherDevice(%x: !mo.tensor<[8, 128], f32, gpu:0>, %r: !mo.tensor<[8, 128], f32, gpu:0>, %w: !mo.tensor<[128], f32, gpu:0>) -> (!mo.tensor<[8, 128], f32, gpu:0>, !mo.tensor<[8, 128], f32, gpu:0>) {
  %sum = mo.add(%r, %x) : !mo.tensor<[8, 128], f32, gpu:0>
  %y = mo.composite.reduce.rms_norm(%sum, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[8, 128], f32, gpu:0>, !mo.tensor<[128], f32, gpu:0>) -> !mo.tensor<[8, 128], f32, gpu:0>
  mo.output %y, %sum : !mo.tensor<[8, 128], f32, gpu:0>, !mo.tensor<[8, 128], f32, gpu:0>
}
}

// -----

// With no kernel registered for the composite, it doesn't form.
// CHECK-LABEL: mo.graph @singleNormUnregistered
// CHECK-NOT:     mo.composite.rms_norm_residual_add
// CHECK:         mo.composite.reduce.rms_norm
mo.graph @singleNormUnregistered(%x: !mo.tensor<[8, 128], f32>, %r: !mo.tensor<[8, 128], f32>, %w: !mo.tensor<[128], f32>) -> (!mo.tensor<[8, 128], f32>, !mo.tensor<[8, 128], f32>) {
  %sum = mo.add(%r, %x) : !mo.tensor<[8, 128], f32>
  %y = mo.composite.reduce.rms_norm(%sum, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[8, 128], f32>, !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], f32>
  mo.output %y, %sum : !mo.tensor<[8, 128], f32>, !mo.tensor<[8, 128], f32>
}
