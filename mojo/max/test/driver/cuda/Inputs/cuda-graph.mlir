mo.graph @add(%0: !mo.tensor<[5], f32, {device=#M.device_ref<"cuda", 0>}>) -> (!mo.tensor<[5], f32, {device=#M.device_ref<"cuda", 0>}>) {
  %1 = mo.add(%0, %0) : !mo.tensor<[5], f32, {device=#M.device_ref<"cuda", 0>}>
  mo.output %1 : !mo.tensor<[5], f32, {device=#M.device_ref<"cuda", 0>}>
}
