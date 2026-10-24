!tt  = !mo.tensor<[d2, d3], f32, gpu:0>
!stt = !mo.tensor<[d0, d1], f32, gpu:0>
!bt  = !mo.buffer<[d2, d3], f32, gpu:0>
!st = !mo.tensor<[2], si64>
!c   = !mo.chain

mo.graph @graph<d0, d1, d2, d3>(%buf: !bt, %rhs: !stt, %start: !st, %stop: !st, %step: !st)
attributes {argument_names = ["buffer0", "rhs", "start", "stop", "step"], result_names = []}
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)

  %res = rmo.mo.slice(%vbuf, %start, %stop, %step) : (!tt, !st, !st, !st) -> !stt
  %res2 = rmo.mo.add(%res, %rhs) : (!stt, !stt) -> !stt
  %slice = rmo.mo.negative(%res2) : (!stt) -> !stt

  rmo.mo.mutable.store.slice(%buf, %slice, %start, %stop, %step, %ch1) : (!bt, !stt, !st, !st, !st, !c) -> !c

  mo.output
}
