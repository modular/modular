// RUN: mt --execute --default-device-label=gpu --model-inputs=ones %s

!dbt = !mo.buffer<[2, 2], f32, gpu:0>
!dtt = !mo.tensor<[2, 2], f32, gpu:0>

!hbt = !mo.buffer<[2, 2], f32>
!htt = !mo.tensor<[2, 2], f32>

!ct = !mo.chain

// CHECK-LABEL: --- Running 'mixed_usage':
// CHECK-LITERAL: 'mixed_usage' returned tensor([[2.0000, 2.0000],
// CHECK-LITERAL: [2.0000, 2.0000]], dtype=f32, shape=[2,2])
mo.graph @mixed_usage(%arg0: !hbt, %arg1: !htt) -> (!hbt, !ct) {
  %ch0 = mo.chain.create()
  %darg0 = mo.buffer.create : !dbt

  %host_copied, %ch1 = mo.mutable.load(%arg0, %ch0) : (!hbt, !ct) -> (!htt, !ct)
  %2, %ch2 = mo.transfer[%ch1] %host_copied : !htt to <"gpu", 0>
  %ch3 = mo.mutable.store(%darg0, %2, %ch2) : (!dbt, !dtt, !ct) -> !ct

  %outTensor, %ch4 = mo.mutable.load(%darg0, %ch3) : (!dbt, !ct) -> (!dtt, !ct)
  // TODO: also try with a value coming in that wasn't loaded from a mo.buffer
  %res = mo.add(%outTensor, %outTensor) : !dtt
  %ch5 = mo.mutable.store(%darg0, %res, %ch4) : (!dbt, !dtt, !ct) -> !ct


  %finalDevBuf, %ch6 = mo.mutable.load(%darg0, %ch5) : (!dbt, !ct) -> (!dtt, !ct)
  %finalHostSideDevbBuf, %ch7 = mo.transfer[%ch6] %finalDevBuf : !dtt to <"cpu", 0>
  %ch8 = mo.mutable.store(%arg0, %finalHostSideDevbBuf, %ch7) : (!hbt, !htt, !ct) -> !ct

  mo.output %arg0, %ch8 : !hbt, !ct
}
