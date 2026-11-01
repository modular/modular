// RUN: mo-opt %s --split-input-file --verify-diagnostics

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!c   = !mo.chain

mo.graph @mutable_load_ok<d0, d1, d2, d3>(%buf: !bt) -> !tt
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)

  mo.output %vbuf : !tt
}

// -----

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!c   = !mo.chain

mo.graph @mutable_store_ok<d0, d1, d2, d3>(%buf: !bt, %tensor: !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1 = rmo.mo.mutable.store(%buf, %tensor, %ch0) : (!bt, !tt, !c) -> (!c)

  mo.output
}

// -----

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f64>
!c   = !mo.chain

mo.graph @mutable_load_dtype_error<d0, d1, d2, d3, d4, d5>(%buf: !bt) -> !tt
{
  %ch0 = mo.chain.create(%buf) : !bt

  // expected-error @+1{{'rmo.mo.mutable.load' op failed to verify that all of {inBuffer, outTensor} have same dtype}}
  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)

  mo.output %vbuf : !tt
}

// -----

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f64>
!c   = !mo.chain

mo.graph @mutable_store_dtype_errorok<d0, d1, d2, d3, d4, d5>(%buf: !bt, %tensor: !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  // expected-error @+1{{'rmo.mo.mutable.store' op failed to verify that all of {inBuffer, inTensor} have same dtype}}
  %ch1 = rmo.mo.mutable.store(%buf, %tensor, %ch0) : (!bt, !tt, !c) -> (!c)

  mo.output
}
