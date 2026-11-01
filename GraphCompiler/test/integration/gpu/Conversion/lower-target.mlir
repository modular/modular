// RUN: mo-opt --mo-validate-devices="default-device-target-triple=nvptx64-nvidia-cuda default-device-target-arch=sm_89" --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg %s | FileCheck --check-prefix=CHECK-IR %s

// CHECK-IR-LABEL: mo.graph @graph
// CHECK-IR: #M.device_ref<"gpu", 0>

mo.graph @graph(%a : !mo.tensor<[4], f32, gpu:0>, %b : !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
{
  %c = mo.add (%a, %b) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[4], f32, gpu:0>
  mo.output %c : !mo.tensor<[4], f32, gpu:0>
}
