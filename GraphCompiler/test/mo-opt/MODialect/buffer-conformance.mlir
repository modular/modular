// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @missing_chain_input(%arg0: !mo.buffer<[10, 10], f32>) {
  // expected-error @+1 {{buffer consuming op must have at least one chain input/output}}
  %res = mo.custom [symbol = "foo"] (%arg0) : (!mo.buffer<[10, 10], f32>) -> (!mo.tensor<[10, 10], f32>)
  mo.output
}

// -----

mo.graph @missing_chain_result(%arg0: !mo.buffer<[10, 10], f32>) {
  %ch0 = mo.chain.create()
  // expected-error @+1 {{number of input chains (1) does not match number of output chains (0)}}
  %res = mo.custom [symbol = "foo"] (%arg0, %ch0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>)
  mo.output
}

// -----

mo.graph @chain_not_first_operand(%arg0: !mo.buffer<[10, 10], f32>) {
  %ch0 = mo.chain.create()
  // expected-error @+1 {{A mo.chain must be the last output for ops with buffer inputs.}}
  %res:2 = mo.custom [symbol = "foo"] (%arg0, %ch0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  mo.output
}

// -----

mo.graph @chain_not_first_result(%arg0: !mo.buffer<[10, 10], f32>) {
  %ch0 = mo.chain.create()
  // expected-error @+1 {{A mo.chain must be the last input for ops with mo.buffer inputs.}}
  %res:2 = mo.custom [symbol = "foo"] (%ch0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  mo.output
}
