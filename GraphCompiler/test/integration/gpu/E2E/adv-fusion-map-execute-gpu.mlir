// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib"
// RUN: mt --execute %t.mlir --result-output-style=full --model-inputs=ones | FileCheck %s

// End-to-end GPU execution of the MAP-ABI path. CPU coverage lives in
// `cpu/E2E/adv-fusion-map-execute*.mlir`; GPU is the edge case, so this file
// keeps a small graph set that COMPILES the emitted Mojo for GPU, RUNS it, and
// checks the numeric result. Each `mo.custom` runs on
// `device=#M.device_ref<"gpu", 0>` over `gpu:0` tensors, with `mo.transfer`
// staging the graph inputs cpu->gpu and the results gpu->cpu (mirrors
// `add_one.mlir` and `mogg-kernel-api-e2e-trait-gpu.mlir`). Inputs are all-ones
// (`--model-inputs=ones`); the kernels are chosen so the results are still
// discriminating (add3 -> 3; add_sub -> 2 and 0).

// 1 input / 1 output: `res = a + 1`.
// CHECK-LABEL: Running 'add_one_gpu':
// CHECK: 'add_one_gpu' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @add_one_gpu(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %ch0 = mo.chain.create()
  %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
  %r_gpu = mo.custom [symbol = "map_lower_add_one", device = #M.device_ref<"gpu", 0>](%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  %r, %ch2 = mo.transfer[%ch1] %r_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
  mo.output %r : !mo.tensor<[4], f32>
}

// 3 inputs / 1 output: `res = a + b + c` -> 3.
// CHECK-LABEL: Running 'add3_gpu':
// CHECK: 'add3_gpu' returned tensor<4xf32> [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]
mo.graph @add3_gpu(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>, %c: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[4], f32> to <"gpu", 0>
  %b_gpu, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[4], f32> to <"gpu", 0>
  %c_gpu, %ch3 = mo.transfer[%ch2] %c : !mo.tensor<[4], f32> to <"gpu", 0>
  %r_gpu = mo.custom [symbol = "map_lower_add3", device = #M.device_ref<"gpu", 0>](%a_gpu, %b_gpu, %c_gpu) : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  %r, %ch4 = mo.transfer[%ch3] %r_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
  mo.output %r : !mo.tensor<[4], f32>
}

// 2 inputs / 2 outputs: `res0 = a + b` -> 2, `res1 = a - b` -> 0. Distinct
// per-output values exercise the multi-output MAP path; per-output lines are
// matched loosely (substring) so they hold regardless of the separator printed
// between the two returned tensors.
// CHECK-LABEL: Running 'add_sub_gpu':
// CHECK: 'add_sub_gpu' returned
// CHECK: tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
// CHECK: tensor<4xf32> [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
mo.graph @add_sub_gpu(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[4], f32> to <"gpu", 0>
  %b_gpu, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[4], f32> to <"gpu", 0>
  %r0_gpu, %r1_gpu = mo.custom [symbol = "map_lower_add_sub", device = #M.device_ref<"gpu", 0>](%a_gpu, %b_gpu) : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>)
  %r0, %ch3 = mo.transfer[%ch2] %r0_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
  %r1, %ch4 = mo.transfer[%ch3] %r1_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
  mo.output %r0, %r1 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}
