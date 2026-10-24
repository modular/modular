// RUN: mo-opt --mo-symbolic-optimize %s | FileCheck %s

// The forward `mo.relu -> max` canonicalization was removed, so a
// standalone `mo.relu` must survive symbolic optimization unchanged
// rather than being decomposed into a `mo.max`.
// CHECK-LABEL: @relu_not_decomposed
mo.graph @relu_not_decomposed(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  // CHECK: mo.relu(%{{.*}})
  // CHECK-NOT: mo.max
  %0 = mo.relu(%x) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// max(x, bcast(0)) -> mo.relu(x)
// CHECK-LABEL: @max_to_relu
mo.graph @max_to_relu(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  // CHECK: mo.relu(%{{.*}})
  %zero_s = mo.constant <#M.dense_array<0.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %zero = mo.static.broadcast_to(%zero_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %0 = mo.max(%x, %zero) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// div(1, add(1, exp(-x))) -> mo.sigmoid(x)
// CHECK-LABEL: @div_exp_to_sigmoid
mo.graph @div_exp_to_sigmoid(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  // CHECK: mo.sigmoid(%{{.*}})
  %one_s = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %one = mo.static.broadcast_to(%one_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %neg = mo.negative(%x) : !mo.tensor<[4], f32>
  %exp = mo.exp(%neg) : !mo.tensor<[4], f32>
  %den = mo.add(%one, %exp) : !mo.tensor<[4], f32>
  %0 = mo.div(%one, %den) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// mul(x, mo.sigmoid(x)) -> mo.silu(x)
// CHECK-LABEL: @sigmoid_mul_to_silu
mo.graph @sigmoid_mul_to_silu(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  // CHECK: mo.silu(%{{.*}})
  %s = mo.sigmoid(%x) : !mo.tensor<[4], f32>
  %0 = mo.mul(%x, %s) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// mul(x, mo.sigmoid(mul(1.702, x))) -> mo.gelu_quick(x)
// CHECK-LABEL: @sigmoid_mul_to_gelu_quick
mo.graph @sigmoid_mul_to_gelu_quick(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  // CHECK: mo.gelu_quick(%{{.*}})
  %c_s = mo.constant <#M.dense_array<1.702> : tensor<1xf32>> : !mo.tensor<[], f32>
  %c = mo.static.broadcast_to(%c_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %scaled = mo.mul(%c, %x) : !mo.tensor<[4], f32>
  %s = mo.sigmoid(%scaled) : !mo.tensor<[4], f32>
  %0 = mo.mul(%x, %s) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// mul(mul(0.5, x), add(1, erf(div(x, sqrt(2))))) -> mo.gelu(x)
// CHECK-LABEL: @erf_to_gelu
mo.graph @erf_to_gelu(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  // CHECK: mo.gelu(%{{.*}})
  %half_s = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[], f32>
  %half = mo.static.broadcast_to(%half_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %one_s = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %one = mo.static.broadcast_to(%one_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %sqrt2_s = mo.constant <#M.dense_array<1.4142135623730951> : tensor<1xf32>> : !mo.tensor<[], f32>
  %sqrt2 = mo.static.broadcast_to(%sqrt2_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %hx = mo.mul(%half, %x) : !mo.tensor<[4], f32>
  %xdiv = mo.div(%x, %sqrt2) : !mo.tensor<[4], f32>
  %erf = mo.erf(%xdiv) : !mo.tensor<[4], f32>
  %inner = mo.add(%one, %erf) : !mo.tensor<[4], f32>
  %0 = mo.mul(%hx, %inner) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// mul(mul(0.5, x),
//     add(1, tanh(mul(0.7978845608028654, add(x, mul(0.044715, x^3))))))
//   -> mo.gelu_tanh(x)
//
// The `x^3` is written as `mo.pow(x, 3)`; the forward canonicalization first
// decomposes it into `x * x * x`, which the backward pattern still recognizes.
// CHECK-LABEL: @tanh_to_gelu_tanh
mo.graph @tanh_to_gelu_tanh(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  // CHECK: mo.gelu_tanh(%{{.*}})
  %half_s = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[], f32>
  %half = mo.static.broadcast_to(%half_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %one_s = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %one = mo.static.broadcast_to(%one_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %c1_s = mo.constant <#M.dense_array<0.7978845608028654> : tensor<1xf32>> : !mo.tensor<[], f32>
  %c1 = mo.static.broadcast_to(%c1_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %c2_s = mo.constant <#M.dense_array<0.044715> : tensor<1xf32>> : !mo.tensor<[], f32>
  %c2 = mo.static.broadcast_to(%c2_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %three_s = mo.constant <#M.dense_array<3.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %three = mo.static.broadcast_to(%three_s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %hx = mo.mul(%x, %half) : !mo.tensor<[4], f32>
  %cube = mo.pow(%x : !mo.tensor<[4], f32>, %three : !mo.tensor<[4], f32>) : !mo.tensor<[4], f32>
  %c2cube = mo.mul(%c2, %cube) : !mo.tensor<[4], f32>
  %poly = mo.add(%x, %c2cube) : !mo.tensor<[4], f32>
  %scaled = mo.mul(%c1, %poly) : !mo.tensor<[4], f32>
  %tanh = mo.tanh(%scaled) : !mo.tensor<[4], f32>
  %inner = mo.add(%one, %tanh) : !mo.tensor<[4], f32>
  %0 = mo.mul(%hx, %inner) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}
