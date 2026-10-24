// REQUIRES: AMD-GPU
// RUN: mo-opt --gc-pipeline="default-device-label=gpu save-temp-prefix=%t" %s -o %t.mlir

// -quit to stop on the first file found
// RUN: find $(dirname %t) -type f -name "*.amdgcn" -print -quit | xargs grep "amdgcn_target"

mo.graph @graph(%a : !mo.tensor<[4], f32, gpu:0>, %b : !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0> {
  %c = mo.add (%a, %b) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[4], f32, gpu:0>
  mo.output %c : !mo.tensor<[4], f32, gpu:0>
}
