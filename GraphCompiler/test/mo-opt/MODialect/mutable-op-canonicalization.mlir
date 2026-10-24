// RUN: mo-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: all_same_buffer
// CHECK-SAME: [[buf:%.*]]: !mo.buffer
// CHECK: [[ch0:%.*]] = mo.chain.create
// CHECK: [[tensor:%.*]], [[ch1:%.*]] = mo.mutable.load([[buf]], [[ch0]])
// CHECK-NOT: mo.mutable.load
// CHECK-NOT: mo.mutable.load
// CHECK: mo.negative([[tensor]])
// CHECK: mo.abs([[tensor]])
// CHECK: mo.relu([[tensor]])
mo.graph @all_same_buffer(%buf: !mo.buffer<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
{
  %0 = mo.chain.create()
  %outTensor, %outChain = mo.mutable.load(%buf, %0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %outTensor_1, %outChain_0 = mo.mutable.load(%buf, %outChain) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %outTensor_3, %outChain_2 = mo.mutable.load(%buf, %outChain) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %1 = mo.negative(%outTensor) : !mo.tensor<[2, 2], f32>
  %2 = mo.abs(%outTensor_1) : !mo.tensor<[2, 2], f32>
  %3 = mo.relu(%outTensor_3) : !mo.tensor<[2, 2], f32>
  mo.output %1, %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: different_buffer_in_between
// CHECK-SAME: [[buf0:%.*]]: !mo.buffer<{{.*}}>, [[buf1:%.*]]: !mo.buffer<{{.*}}>
// CHECK: [[ch0:%.*]] = mo.chain.create
// CHECK: [[tensor0:%.*]], [[ch1:%.*]] = mo.mutable.load([[buf0]], [[ch0]])
// CHECK: [[tensor1:%.*]], [[ch2:%.*]] = mo.mutable.load([[buf1]], [[ch1]])
// CHECK: [[tensor2:%.*]], [[ch3:%.*]] = mo.mutable.load([[buf0]], [[ch2]])
// CHECK: mo.negative([[tensor0]])
// CHECK: mo.abs([[tensor1]])
// CHECK: mo.relu([[tensor2]])
mo.graph @different_buffer_in_between(%buf1: !mo.buffer<[2, 2], f32>, %buf2: !mo.buffer<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
{
  %0 = mo.chain.create()
  %outTensor, %outChain = mo.mutable.load(%buf1, %0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %outTensor_1, %outChain_0 = mo.mutable.load(%buf2, %outChain) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %outTensor_3, %outChain_2 = mo.mutable.load(%buf1, %outChain_0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %1 = mo.negative(%outTensor) : !mo.tensor<[2, 2], f32>
  %2 = mo.abs(%outTensor_1) : !mo.tensor<[2, 2], f32>
  %3 = mo.relu(%outTensor_3) : !mo.tensor<[2, 2], f32>
  mo.output %1, %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
