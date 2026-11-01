// TODO(GEX-2037): Fix and re-enable test
// REQUIRES: disabled
// RUN: mo-opt --mo-infer-layouts --mogg-import-packages --mo-to-mogg --split-input-file --verify-diagnostics %s

!tt = !mo.tensor<[10, 10], f32>
!bt = !mo.buffer<[10, 10], f32>
!c = !mo.chain

mo.graph @add(%buf: !bt, %rhs: !tt) -> (!bt, !bt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf) : (!c, !bt) -> (!c, !tt)
  // expected-error@below{{Currently only elementwise compute ops are allowed to be in-place}}
  %res = mo.matmul(%vbuf, %rhs) : (!tt, !tt) -> !tt
  %ch2 = mo.mutable.store(%ch1, %buf, %res) : (!c, !bt, !tt) -> (!c)

  %ch3 = mo.chain.create(%ch2) : !c

  %gb = mo.guard[%ch3](%buf) : !bt

  mo.output %gb, %buf : !bt, !bt
}

// -----

mo.graph @add(%buf: !bt, %rhs: !tt) -> (!bt, !bt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1p, %vbufp = mo.mutable.load(%ch0, %buf) : (!c, !bt) -> (!c, !tt)
  %ch1, %vbuf = mo.mutable.load(%ch0, %buf) : (!c, !bt) -> (!c, !tt)
  // expected-error@below{{More than one input coming from a mo.mutable.store operation}}
  %res = mo.add(%vbuf, %vbufp) : !tt
  %ch2 = mo.mutable.store(%ch1, %buf, %res) : (!c, !bt, !tt) -> (!c)

  %ch3 = mo.chain.create(%ch2) : !c

  %gb = mo.guard[%ch3](%buf) : !bt

  mo.output %gb, %buf : !bt, !bt
}

// -----

mo.graph @add(%buf: !bt, %rhs: !tt) -> !tt
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf) : (!c, !bt) -> (!c, !tt)
  // expected-error@below{{Load/Store op flanking kernel op aren't operating on the same chain}}
  %res = mo.add(%vbuf, %rhs) : !tt
  %ch2 = mo.mutable.store(%ch0, %buf, %res) : (!c, !bt, !tt) -> (!c)

  %ch3, %final = mo.mutable.load(%ch2, %buf) : (!c, !bt) -> (!c, !tt)

  mo.output %final : !tt
}
