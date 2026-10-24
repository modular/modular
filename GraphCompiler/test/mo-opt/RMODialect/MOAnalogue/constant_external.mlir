// RUN: mo-opt %s

mo.graph @constant_fold_external() -> !mo.tensor<[256, 1], si32> {
  %0 = mo.constant.external [name = "blob", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[256, 1], si32>
  %1 = mo.add(%0, %0) : !mo.tensor<[256, 1], si32>
  mo.output %1 : !mo.tensor<[256, 1], si32>
}
