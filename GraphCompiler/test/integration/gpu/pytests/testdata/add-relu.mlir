!tt = !mo.tensor<[2, 2], f32>
!tt_gpu = !mo.tensor<[2, 2], f32, gpu:0>
!bt_gpu = !mo.buffer<[2, 2], f32, gpu:0>
!ct = !mo.chain

mo.graph @add_relu(%buf: !bt_gpu, %rhs: !tt)
attributes {argument_names = ["buffer0", "tensor0"], result_names = []}
{
  %ch0 = mo.chain.create(%buf) : !bt_gpu
  %rhs_gpu, %chA = mo.transfer[%ch0] %rhs : !tt to <"gpu", 0>

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %chA)  : (!bt_gpu, !ct) -> (!tt_gpu, !ct)
  %res1 = rmo.mo.add(%vbuf, %rhs_gpu) : (!tt_gpu, !tt_gpu) -> !tt_gpu
  %ch2 = rmo.mo.mutable.store(%buf, %res1, %ch1)   : (!bt_gpu, !tt_gpu, !ct) -> (!ct)

  %vbuf2, %ch3 = rmo.mo.mutable.load(%buf, %ch2)  : (!bt_gpu, !ct) -> (!tt_gpu, !ct)
  %res2 = rmo.mo.relu(%vbuf2) : (!tt_gpu) -> !tt_gpu
  %ch4 = rmo.mo.mutable.store(%buf, %res2, %ch3) : (!bt_gpu, !tt_gpu, !ct) -> (!ct)

  mo.output
}
