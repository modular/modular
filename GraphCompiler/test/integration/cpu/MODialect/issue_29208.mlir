// RUN: mt --execute %s | FileCheck %s

mo.graph @forward() -> !mo.tensor<[5], bool> attributes {argument_names = [], result_names = ["result0"]} {
  %0 = mo.constant <#M.dense_array<-1.5, -0.5, 0.0, 0.5, 1.5> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.cast(%0) : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], bool>
  mo.output %1 : !mo.tensor<[5], bool>
}
// CHECK{LITERAL}: 'forward' returned tensor([[True, True, False, True, True]], dtype=bool, shape=[5])
