// RUN: mo-opt %s --canonicalize --split-input-file --verify-diagnostics

mo.graph @unranked_parameterized<Sh0: !mosh.ape>(
  // expected-error@+1 {{couldn't parse shape parameter: expected literal shape reference (i.e. `[...]`)}}
  %b0: !mo.buffer<Sh0, f32>
) {
  mo.output
}

// -----

mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f64>) {
  // expected-error@+1 {{'mo.mutable.load' op failed to verify that all of {inBuffer, outTensor} have same dtype}}
  %out:2 = mo.mutable.load(%buf, %ch) :
    (!mo.buffer<[10, 20], f64>, !mo.chain) ->
    (!mo.tensor<[10, 20], f32>, !mo.chain)
  mo.output
}

// -----

mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>) {
  // expected-error@+1 {{'mo.mutable.load' op failed to verify that all of {inBuffer, outTensor} have same shape}}
  %out:2 = mo.mutable.load(%buf, %ch) :
    (!mo.buffer<[10, 20], f32>, !mo.chain) ->
    (!mo.tensor<[30, 20], f32>, !mo.chain)
  mo.output
}

// -----

mo.graph @mutable_store(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %tensor: !mo.tensor<[10, 20], f64>) {
  // expected-error@+1 {{'mo.mutable.store' op failed to verify that all of {inBuffer, inTensor} have same dtype}}
  %outChain = mo.mutable.store(%buf, %tensor, %ch) :
    (!mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f64>, !mo.chain) -> (!mo.chain)
  mo.output
}

// -----

mo.graph @mutable_store(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %tensor: !mo.tensor<[10, 30], f32>) {
  // expected-error@+1 {{'mo.mutable.store' op failed to verify that all of {inBuffer, inTensor} have same shape}}
  %outChain = mo.mutable.store(%buf, %tensor, %ch) :
    (!mo.buffer<[10, 20], f32>, !mo.tensor<[10, 30], f32>, !mo.chain) -> (!mo.chain)
  mo.output
}
