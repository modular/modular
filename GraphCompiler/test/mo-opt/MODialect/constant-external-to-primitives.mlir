// RUN: mo-opt %s --mogg-to-mgp | FileCheck %s

// Test host weights with device output.
// CHECK-LABEL: @external_weights_cpu_to_gpu
// CHECK-DAG: mgp.string.create "foo"
// CHECK-DAG: [[HOST_BUFFER1:%.*]] = mgp.buffer.constant.external{{.*}}
// CHECK-DAG: [[CUDA_BUFFER1:%.*]] = mgp.buffer.alloc{{.*}}gpu
// CHECK: mgp.buffer.host_to_device{{.*}}[[HOST_BUFFER1]], [[CUDA_BUFFER1]]
// CHECK-DAG: mgp.string.create "bar"
// CHECK-DAG: [[HOST_BUFFER2:%.*]] = mgp.buffer.constant.external{{.*}}has_alias
// CHECK-DAG: [[CUDA_BUFFER2:%.*]] = mgp.buffer.alloc{{.*}}gpu
// CHECK: mgp.buffer.host_to_device{{.*}}[[HOST_BUFFER2]], [[CUDA_BUFFER2]]
mo.graph @external_weights_cpu_to_gpu() -> (!mo.tensor<[128], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) attributes {
    argument_names = [], result_names = ["output0"], target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_86", index_bit_width = 0>
} {
  %0 = mo.constant.external [name = "foo", align = 4, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[128], f32, gpu:0>
  %1 = mo.constant.external [name = "bar", align = 4, device = #M.device_ref<"cpu", 0>, hasAlias = true] : !mo.tensor<[256], f32, gpu:0>
  mo.output %0, %1 : !mo.tensor<[128], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>
}

// Test external weights that are already on device.
// CHECK-LABEL: @external_weights_gpu_resident
// CHECK: mgp.string.create "foo"
// CHECK: mgp.buffer.constant.external{{.*}}gpu
// CHECK-NOT: mgp.buffer.host_to_device
// CHECK: mgp.string.create "bar"
// CHECK: mgp.buffer.constant.external{{.*}}has_alias{{.*}}gpu
// CHECK-NOT: mgp.buffer.host_to_device
mo.graph @external_weights_gpu_resident() -> (!mo.tensor<[128], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) attributes {
    argument_names = [], result_names = ["output0"], target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_86", index_bit_width = 0>
} {
  %0 = mo.constant.external [name = "foo", align = 4, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[128], f32, gpu:0>
  %1 = mo.constant.external [name = "bar", align = 4, device = #M.device_ref<"gpu", 0>, hasAlias = true] : !mo.tensor<[256], f32, gpu:0>
  mo.output %0, %1 : !mo.tensor<[128], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>
}
