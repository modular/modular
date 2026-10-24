// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %s | FileCheck --check-prefix=CHECK-MT %s

// CHECK-MT-LABEL: --- Running 'print':
// CHECK-MT: tensor = tensor({{\[}}[0.0000, 1.0000, 2.0000, 3.0000]], dtype=f32, shape=[4])

mo.graph @print() -> (!mo.chain)
{
  %ch0 = mo.chain.create()
  %a = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %b, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[4], f32> to <"gpu", 0>
  %ch2 = mo.debug.tensor.print(%ch0, %b) [label = "tensor"] : !mo.tensor<[4], f32, gpu:0>
  mo.output %ch2 : !mo.chain
}
