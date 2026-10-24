// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// The indices go through mo.negative so the canonicalizer cannot normalize
// them, which leaves negative and unsorted indices for the GPU kernel.

mo.graph @squeeze_shape_si64_neg_unsorted() -> !mo.tensor<[2], si64> {
  %shape = mo.constant <#M.dense_array<2, 1, 3, 1, 1> : tensor<5xsi64>> : !mo.tensor<[5], si64, gpu:0>
  %neg_indices = mo.constant <#M.dense_array<1, -1, -3> : tensor<3xsi32>> : !mo.tensor<[3], si32, gpu:0>
  %indices = mo.negative(%neg_indices) : !mo.tensor<[3], si32, gpu:0>
  %res = mo.squeeze_shape(%shape, %indices) : (!mo.tensor<[5], si64, gpu:0>, !mo.tensor<[3], si32, gpu:0>) -> !mo.tensor<[2], si64, gpu:0>
  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %res : !mo.tensor<[2], si64, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[2], si64>
}
// CHECK: 'squeeze_shape_si64_neg_unsorted' returned tensor<2xsi64> [2, 3]

mo.graph @squeeze_shape_si32_si64_indices() -> !mo.tensor<[1], si32> {
  %shape = mo.constant <#M.dense_array<1, 7, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32, gpu:0>
  %neg_indices = mo.constant <#M.dense_array<-2, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64, gpu:0>
  %indices = mo.negative(%neg_indices) : !mo.tensor<[2], si64, gpu:0>
  %res = mo.squeeze_shape(%shape, %indices) : (!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[2], si64, gpu:0>) -> !mo.tensor<[1], si32, gpu:0>
  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %res : !mo.tensor<[1], si32, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[1], si32>
}
// CHECK: 'squeeze_shape_si32_si64_indices' returned tensor<1xsi32> [7]

mo.graph @squeeze_shape_constant_indices() -> !mo.tensor<[2], si32> {
  %shape = mo.constant <#M.dense_array<4, 5, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32, gpu:0>
  %indices = mo.constant <#M.dense_array<-1> : tensor<1xsi32>> : !mo.tensor<[1], si32, gpu:0>
  %res = mo.squeeze_shape(%shape, %indices) : (!mo.tensor<[3], si32, gpu:0>, !mo.tensor<[1], si32, gpu:0>) -> !mo.tensor<[2], si32, gpu:0>
  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %res : !mo.tensor<[2], si32, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[2], si32>
}
// CHECK: 'squeeze_shape_constant_indices' returned tensor<2xsi32> [4, 5]

// mo.reshape reads its shape operand on the host, so the squeezed shape comes
// back from the GPU first.
mo.graph @squeeze_shape_with_reshape<() -> D0, D1, D2>() -> !mo.tensor<[D0, D1, D2], si8> {
  %input_tensor = mo.constant <#M.dense_array<10, 11, 12, 13> : tensor<4xsi8>> : !mo.tensor<[2, 1, 2, 1], si8>
  %shape = mo.shape_of(%input_tensor) : (!mo.tensor<[2, 1, 2, 1], si8>) -> !mo.tensor<[4], si32>
  %ch0 = mo.chain.create()
  %shape_gpu, %ch1 = mo.transfer[%ch0] %shape : !mo.tensor<[4], si32> to <"gpu", 0>
  %neg_indices = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32, gpu:0>
  %indices = mo.negative(%neg_indices) : !mo.tensor<[1], si32, gpu:0>
  %new_shape_gpu = mo.squeeze_shape(%shape_gpu, %indices) : (!mo.tensor<[4], si32, gpu:0>, !mo.tensor<[1], si32, gpu:0>) -> !mo.tensor<[3], si32, gpu:0>
  %new_shape, %ch2 = mo.transfer[%ch1] %new_shape_gpu : !mo.tensor<[3], si32, gpu:0> to <"cpu", 0>
  %res = mo.reshape<() -> D3, D4, D5>(%input_tensor, %new_shape) : (!mo.tensor<[2, 1, 2, 1], si8>, !mo.tensor<[3], si32>) -> !mo.tensor<[D3, D4, D5], si8>
  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], si8>
}
// CHECK: 'squeeze_shape_with_reshape' returned tensor<2x1x2xsi8>
