// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// End-to-end check that inlining a broadcast constant scalar produces correct
// results. The rank-0 constant 10.0 is broadcast and added to
// [1, 2, 3, 4], yielding [11, 12, 13, 14].
mo.graph @scalar_add_const() -> (!mo.tensor<[4], f32>) {
  %x = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %s = mo.constant <#M.dense_array<1.000000e+01> : tensor<f32>> : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %r = mo.add(%x, %b) : !mo.tensor<[4], f32>
  mo.output %r : !mo.tensor<[4], f32>
}
// CHECK{LITERAL}: 'scalar_add_const' returned tensor<4xf32> [1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01]

// A non-integral constant must emit a bare float literal (e.g. "0.5"), not a
// `FloatLiteral(...)` wrapper: `FloatLiteral` has no constructor taking another
// `FloatLiteral`, so wrapping a fractional value fails to compile. Adding 0.5
// to [1, 2, 3, 4] yields [1.5, 2.5, 3.5, 4.5].
mo.graph @scalar_add_fractional() -> (!mo.tensor<[4], f32>) {
  %x = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %s = mo.constant <#M.dense_array<5.000000e-01> : tensor<f32>> : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %r = mo.add(%x, %b) : !mo.tensor<[4], f32>
  mo.output %r : !mo.tensor<[4], f32>
}
// CHECK{LITERAL}: 'scalar_add_fractional' returned tensor<4xf32> [1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00]

// A -inf constant (no Mojo float literal) is inlined via the numerics helper.
// Adding -inf yields -inf for every element.
mo.graph @scalar_add_neg_inf() -> (!mo.tensor<[4], f32>) {
  %x = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %s = mo.constant <#M.dense_array<0xFF800000> : tensor<f32>> : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %r = mo.add(%x, %b) : !mo.tensor<[4], f32>
  mo.output %r : !mo.tensor<[4], f32>
}
// CHECK{LITERAL}: 'scalar_add_neg_inf' returned tensor<4xf32> [-INF, -INF, -INF, -INF]

// A broadcast constant inlined inside an input (prologue) lambda must compile
// and run. `mul(arg0, broadcast(0.5))` fuses as a prologue into softmax. With
// all-ones input, every element becomes 0.5, so softmax over each row of 3
// equal values yields 1/3.
mo.graph @softmax_mul_const_prologue(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  %s = mo.constant <#M.dense_array<5.000000e-01> : tensor<f32>> : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[2, 3], f32>
  %mul = mo.mul(%arg0, %b) : !mo.tensor<[2, 3], f32>
  %sm = mo.reduce.softmax(%mul) [axis = 1] : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %sm : !mo.tensor<[2, 3], f32>
}
// CHECK{LITERAL}: 'softmax_mul_const_prologue' returned tensor<2x3xf32> [3.333333e-01, 3.333333e-01, 3.333333e-01, 3.333333e-01, 3.333333e-01, 3.333333e-01]

// A broadcast constant inlined inside an output (epilogue) lambda must compile
// and run. `mul(matmul(A, B), broadcast(0.25))` fuses as an epilogue on the
// matmul. With all-ones inputs, each matmul element is 4, so scaling by 0.25
// yields 1 for every element.
mo.graph @matmul_mul_const_epilogue(%A: !mo.tensor<[5, 4], f32>, %B: !mo.tensor<[4, 4], f32>) -> (!mo.tensor<[5, 4], f32>) {
  %s = mo.constant <#M.dense_array<2.500000e-01> : tensor<f32>> : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[5, 4], f32>
  %mm = mo.matmul(%A, %B) : (!mo.tensor<[5, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 4], f32>
  %mul = mo.mul(%mm, %b) : !mo.tensor<[5, 4], f32>
  mo.output %mul : !mo.tensor<[5, 4], f32>
}
// CHECK{LITERAL}: 'matmul_mul_const_epilogue' returned tensor<5x4xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
