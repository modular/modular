!dbt = !mo.buffer<[2, 2], f32, gpu:0>
!dtt = !mo.tensor<[2, 2], f32, gpu:0>

!hbt = !mo.buffer<[2, 2], f32>
!htt = !mo.tensor<[2, 2], f32>

!ct = !mo.chain

mo.graph @mixed_usage(%arg0: !hbt, %arg1: !htt) -> (!dbt)
attributes {argument_names = ["input0", "input1"], result_names = ["output0"]}
{
  %ch0 = mo.chain.create()
  %darg0 = mo.buffer.create : !dbt

  %darg1, %ch1 = mo.transfer[%ch0] %arg1 : !htt to <"gpu", 0>

  %ch2 = mo.buffer.transfer(%arg0, %darg0, %ch1) : !hbt, !dbt

  %outTensor, %ch3 = mo.mutable.load(%darg0, %ch2) : (!dbt, !ct) -> (!dtt, !ct)
  %res = mo.add(%outTensor, %darg1) : !dtt
  %ch4 = mo.mutable.store(%darg0, %res, %ch3) : (!dbt, !dtt, !ct) -> !ct

  %ch6 = mo.buffer.transfer(%darg0, %arg0, %ch4) : !dbt, !hbt

  // Do some more operations on the host-side buffer so the result dirverges from the device buffer we ultimately return.
  %btens, %ch7 = mo.mutable.load(%arg0, %ch6)  : (!hbt, !ct) -> (!htt, !ct)
  %final_res = mo.add(%btens, %arg1)  : !htt
  %ch8 = mo.mutable.store(%arg0, %final_res, %ch7) : (!hbt, !htt, !ct) -> !ct

  mo.output %darg0 : !dbt
}
