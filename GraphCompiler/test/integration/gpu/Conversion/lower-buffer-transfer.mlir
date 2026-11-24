// RUN: mt --execute --default-device-label=gpu --model-inputs=ones %s

!dbt = !mo.buffer<[2, 2], f32, gpu:0>
!dtt = !mo.tensor<[2, 2], f32, gpu:0>

!hbt = !mo.buffer<[2, 2], f32>
!htt = !mo.tensor<[2, 2], f32>

!ct = !mo.chain

// CHECK-LABEL: --- Running 'buffer_transfer':
// CHECK-LITERAL: 'buffer_transfer' returned tensor([[2.0000, 2.0000],
// CHECK-LITERAL: [2.0000, 2.0000]], dtype=f32, shape=[2,2])
mo.graph @buffer_transfer(%arg0: !hbt, %arg1: !htt) -> !hbt {
  %ch0 = mo.chain.create()

  %darg0 = mo.buffer.create : !dbt
  %darg1, %ch1 = mo.transfer[%ch0] %arg1 : !mo.tensor<[2, 2], f32> to <"gpu", 0>
  %ch2 = mo.buffer.transfer(%arg0, %darg0, %ch1) : !hbt, !dbt

  %outTensor, %ch3 = mo.mutable.load(%darg0, %ch2) : (!dbt, !mo.chain) -> (!dtt, !mo.chain)
  %res = mo.add(%outTensor, %darg1) : !dtt
  %ch4 = mo.mutable.store(%darg0, %res, %ch3) : (!dbt, !dtt, !mo.chain) -> !mo.chain

  %ch5 = mo.buffer.transfer(%darg0, %arg0, %ch4) : !dbt, !hbt

  mo.output %arg0 : !hbt
}
