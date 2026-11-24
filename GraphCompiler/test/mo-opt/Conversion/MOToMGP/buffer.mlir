// RUN: mo-opt --gc-pipeline %s | FileCheck %s

// CHECK: mef.callable @buffer_output$exec({{.*}}[[BUFFER:%.*]]: !mgp.tensor<2x2xf32, mut>) -> (!mef.chain, !mgp.tensor<2x2xf32, mut>)
// CHECK: mef.output %{{.*}}, [[BUFFER]] : !mef.chain, !mgp.tensor<2x2xf32, mut>
mo.graph @buffer_output(%buffer: !mo.buffer<[2, 2], f32>) -> !mo.buffer<[2, 2], f32>
{
  mo.output %buffer : !mo.buffer<[2, 2], f32>
}

mo.graph @buffer_as_scalar(%seed_buffer: !mo.buffer<[1], ui64>) -> !mo.tensor<[1], f32> {
  %stdev = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %mean = mo.constant <#M.dense_array<0.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %shape = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[1], si64>
  %chain0 = mo.chain.create()
  %seed_tensor, %chain1 = mo.mutable.load(%seed_buffer, %chain0) : (!mo.buffer<[1], ui64>, !mo.chain) -> (!mo.tensor<[1], ui64>, !mo.chain)
  %random = mo.random.normal(%shape, %mean, %stdev, %seed_tensor) : (!mo.tensor<[1], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[1], f32>

  mo.output %random : !mo.tensor<[1], f32>
}
