// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

// CHECK-LABEL: @constant_external
mo.graph @constant_external() -> (!mo.tensor<[128, 128], bf16, gpu:0>, !mo.tensor<[256, 256], bf16, gpu:0>, !mo.tensor<[128, 128], bf16, gpu:0>) {
  // CHECK: %{{.*}} = mo.constant.external [name = "foo", {{.*}}]
  %weight = mo.constant.external [name = "foo", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[128, 128], bf16, gpu:0>

  // CHECK: %{{.*}} = mo.constant.external [name = "bar", {{.*}}hasAlias = true]
  %weight_mutable = mo.constant.external [name = "bar", align = 16, device = #M.device_ref<"gpu", 0>, hasAlias = true] : !mo.tensor<[256, 256], bf16, gpu:0>

  // CHECK: %{{.*}} = mo.constant.external [name = "placeholder", {{.*}}isPlaceholder = true]
  %weight_placeholder = mo.constant.external [name = "placeholder", align = 16, device = #M.device_ref<"gpu", 0>, isPlaceholder = true] : !mo.tensor<[128, 128], bf16, gpu:0>

  mo.output %weight, %weight_mutable, %weight_placeholder : !mo.tensor<[128, 128], bf16, gpu:0>, !mo.tensor<[256, 256], bf16, gpu:0>, !mo.tensor<[128, 128], bf16, gpu:0>
}
