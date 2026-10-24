// RUN: mo-opt --mo-symbolic-optimize="symbolic-ffast-math=true symbolic-no-undefined-math=true" %s | FileCheck %s

// The first few tests have different data types and negative checks for constants.

// x + 0 -> x
mo.graph @add_with_cst_zero(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %zero = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 0.0> : tensor<6xf64>> : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.add(%input, %zero) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @add_with_cst_zero
// CHECK-NEXT: mo.output

// x + bcast(0) -> x
mo.graph @add_with_bcast_zero(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %zero = mo.constant <#M.dense_array<0.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %bcast_zero = mo.static.broadcast_to(%zero) : !mo.tensor<[], f32> -> !mo.tensor<[1, 2, 3], f32>
  %0 = mo.add(%input, %bcast_zero) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: @add_with_bcast_zero
// CHECK-NEXT: mo.output

// 0 + x -> x
mo.graph @add_with_cst_zero_reversed(%input: !mo.tensor<[1, 2, 3], si32>) -> !mo.tensor<[1, 2, 3], si32> {
  %zero = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi32>> : !mo.tensor<[1, 2, 3], si32>
  %0 = mo.add(%zero, %input) : !mo.tensor<[1, 2, 3], si32>
  mo.output %0 : !mo.tensor<[1, 2, 3], si32>
}
// CHECK-LABEL: @add_with_cst_zero_reversed
// CHECK-NEXT: mo.output

// bcast(0) + x -> x
mo.graph @add_with_bcast_zero_reversed(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %zero = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[], si64>
  %bcast_zero = mo.static.broadcast_to(%zero) : !mo.tensor<[], si64> -> !mo.tensor<[1, 2, 3], si64>
  %0 = mo.add(%bcast_zero, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @add_with_bcast_zero_reversed
// CHECK-NEXT: mo.output

mo.graph @add_with_non_zero(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %zero = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 1.0> : tensor<6xf64>> : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.add(%input, %zero) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @add_with_non_zero
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[ADD:.*]] = mo.add
// CHECK-NEXT: mo.output %[[ADD]]

// x - 0 -> x
mo.graph @sub_with_cst_zero(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %zero = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 0.0> : tensor<6xf64>> : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.sub(%input, %zero) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @sub_with_cst_zero
// CHECK-NEXT: mo.output

// x - x -> bcast(0)
mo.graph @sub_with_same(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.sub(%input, %input) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: @sub_with_same
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x * 1 -> x
mo.graph @mul_with_cst_one(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %one = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0, 1.0, 1.0> : tensor<6xf64>> : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.mul(%input, %one) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @mul_with_cst_one
// CHECK-NEXT: mo.output

// x * 0 -> bcast(0)
mo.graph @mul_with_cst_zero(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %zero = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 0.0> : tensor<6xf64>> : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.mul(%input, %zero) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @mul_with_cst_zero
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// 0 / x -> bcast(0)
mo.graph @zero_div(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %zero = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 0.0> : tensor<6xf64>> : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.div(%zero, %input) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @zero_div
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x / 1 -> x
mo.graph @div_by_one(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %one = mo.constant <#M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.div(%input, %one) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @div_by_one
// CHECK-NEXT: mo.output

// x ** 1 -> x
mo.graph @pow_by_one(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %one = mo.constant <#M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], si64>, %one : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @pow_by_one
// CHECK-NEXT: mo.output

// x ** 0 -> bcast(1)
mo.graph @pow_by_zero(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %zero = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], si64>, %zero : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @pow_by_zero
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x ** 0 -> bcast(1)
mo.graph @pow_of_zero(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %zero = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.pow(%zero : !mo.tensor<[1, 2, 3], si64>, %input : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @pow_of_zero
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x ** 0 -> bcast(1)
mo.graph @pow_of_one(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %one = mo.constant <#M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.pow(%one : !mo.tensor<[1, 2, 3], si64>, %input : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @pow_of_one
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// max(x, x) -> x
mo.graph @max_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %0 = mo.max(%input, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @max_same_operand
// CHECK-NEXT: mo.output

// min(x, x) -> x
mo.graph @min_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %0 = mo.min(%input, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @min_same_operand
// CHECK-NEXT: mo.output

// min(x, max(x, y)) -> x
mo.graph @min_of_max(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %cst = mo.constant <#M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.max(%input, %cst) : !mo.tensor<[1, 2, 3], si64>
  %1 = mo.min(%0, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %1 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @min_of_max
// CHECK-NEXT: mo.output

// max(x, min(x, y)) -> x
mo.graph @max_of_min(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %cst = mo.constant <#M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.min(%input, %cst) : !mo.tensor<[1, 2, 3], si64>
  %1 = mo.max(%0, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %1 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @max_of_min
// CHECK-NEXT: mo.output

// mod(x, x) -> x
mo.graph @mod_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %0 = mo.mod(%input, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @mod_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// mod(x, 1) -> bcast(0)
mo.graph @mod_by_1(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %one = mo.constant <#M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.mod(%input, %one) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @mod_by_1
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x >= x -> bcast(true)
mo.graph @ge_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.greater_equal(%input, %input) : (!mo.tensor<[1, 2, 3], si64>, !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @ge_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x > x -> bcast(false)
mo.graph @gt_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.greater(%input, %input) : (!mo.tensor<[1, 2, 3], si64>, !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @gt_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x != x -> bcast(false)
mo.graph @ne_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.not_equal(%input, %input) : (!mo.tensor<[1, 2, 3], si64>, !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @ne_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x == x -> bcast(false)
mo.graph @eq_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.equal(%input, %input) : (!mo.tensor<[1, 2, 3], si64>, !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @eq_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x && x -> bcast(true)
mo.graph @and_same_operand(%input: !mo.tensor<[1, 2, 3], bool>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.and(%input, %input) : !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @and_same_operand
// CHECK-NEXT: mo.output

// x || x -> bcast(true)
mo.graph @or_same_operand(%input: !mo.tensor<[1, 2, 3], bool>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.or(%input, %input) : !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @or_same_operand
// CHECK-NEXT: mo.output

// x ^ x -> bcast(true)
mo.graph @xor_same_operand(%input: !mo.tensor<[1, 2, 3], bool>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.xor(%input, %input) : !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @xor_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// ((x ^ y) ^ z) -> x ^ (y * z)
mo.graph @pow_chain(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %cst_4 = mo.constant <#M.dense_array<4,4,4,4,4,4> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %cst_3 = mo.constant <#M.dense_array<3,3,3,3,3,3> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], si64>, %cst_4 : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  %1 = mo.pow(%0 : !mo.tensor<[1, 2, 3], si64>, %cst_3 : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  mo.output %1 : !mo.tensor<[1, 2, 3], si64>
}


// CHECK-LABEL: @pow_chain
// CHECK-NEXT: %[[CST4:.*]] = mo.constant
// CHECK-NEXT: %[[CST3:.*]] = mo.constant
// CHECK-NEXT: %[[ADD:.*]] = mo.mul(%[[CST4]], %[[CST3]])
// CHECK-NEXT: %[[POW0:.*]] = mo.pow({{.*}}, %[[ADD]] : !mo.tensor<[1, 2, 3], si64>)
// CHECK-NEXT: mo.output %[[POW0]]

// Should not optimize because exponent of different types.
mo.graph @pow_chain_different_types(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %cst_4 = mo.constant <#M.dense_array<4,4,4,4,4,4> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %cst_3 = mo.constant <#M.dense_array<5,5,5,5,5,5> : tensor<6xsi32>> : !mo.tensor<[1, 2, 3], si32>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], si64>, %cst_4 : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  %1 = mo.pow(%0 : !mo.tensor<[1, 2, 3], si64>, %cst_3 : !mo.tensor<[1, 2, 3], si32>) : !mo.tensor<[1, 2, 3], si64>
  mo.output %1 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @pow_chain_different_types
// CHECK:        mo.pow
// CHECK-NEXT:   mo.pow

// rsqrt(x) -> 1 / sqrt(x) -> rsqrt(x)
mo.graph @rsqrt_roundtrip(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.rsqrt(%input) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: @rsqrt_roundtrip
// CHECK-NEXT:   %[[OUT:.*]] = mo.rsqrt({{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:   mo.output %[[OUT]] : !mo.tensor<[1, 2, 3], f32>

// relu(x) -> max(x, 0)
mo.graph @relu_roundtrip(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.relu(%input) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: @relu_roundtrip
// CHECK-NEXT:   %[[OUT:.*]] = mo.relu({{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:   mo.output %[[OUT]] : !mo.tensor<[1, 2, 3], f32>

// x * 2 -> x + x
mo.graph @mul_by_2_roundtrip(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %cst = mo.constant <#M.dense_array<2,2,2,2,2,2> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.mul(%input, %cst) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @mul_by_2_roundtrip
// CHECK-NEXT:   %[[CST:.*]] = mo.constant
// CHECK-NEXT:   %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT:   %[[OUT:.*]] = mo.mul({{.*}}, %[[BCAST]]) : !mo.tensor<[1, 2, 3], si64>
// CHECK-NEXT:   mo.output %[[OUT]] : !mo.tensor<[1, 2, 3], si64>

// x ** -{1,2} -> 1 / (x ** {1,2})
mo.graph @pow_neg_1_2_roundtrip(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %cst = mo.constant <#M.dense_array<-2.,-2.,-2.,-2.,-2.,-2.> : tensor<6xf32>> : !mo.tensor<[1, 2, 3], f32>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], f32>, %cst : !mo.tensor<[1, 2, 3], f32>) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: @pow_neg_1_2_roundtrip
// CHECK-NEXT:   %[[CST:.*]] = mo.constant
// CHECK-NEXT:   %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT:   %[[MUL:.*]] = mo.mul({{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:   %[[OUT:.*]] = mo.div(%[[BCAST]], %[[MUL]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:   mo.output %[[OUT]] : !mo.tensor<[1, 2, 3], f32>

// 0 - x -> -x
mo.graph @sub_0_to_neg(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %cst = mo.constant <#M.dense_array<-0.,-0.,-0.,-0.,-0.,-0.> : tensor<6xf32>> : !mo.tensor<[1, 2, 3], f32>
  %0 = mo.sub(%cst, %input) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: @sub_0_to_neg
// CHECK-NEXT:    %[[OUT:.*]] = mo.negative({{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:    mo.output %[[OUT]] : !mo.tensor<[1, 2, 3], f32>

// x + (-y) -> x - y
mo.graph @add_with_negative(%input1: !mo.tensor<[1, 2, 3], f32>, %input2: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.negative(%input2) : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.add(%input1, %0) : !mo.tensor<[1, 2, 3], f32>
  mo.output %1 : !mo.tensor<[1, 2, 3], f32>
}
// CHECk-LABEL:  @add_with_negative
// CHECk-NEXT:    %[[OUT:.*]] = mo.sub
// CHECk-NEXT:    mo.output %[[OUT]]

// x - (-y) -> x + y
mo.graph @sub_with_negative(%input1: !mo.tensor<[1, 2, 3], f32>, %input2: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.negative(%input2) : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.sub(%input1, %0) : !mo.tensor<[1, 2, 3], f32>
  mo.output %1 : !mo.tensor<[1, 2, 3], f32>
}
// CHECk-LABEL:  @sub_with_negative
// CHECk-NEXT:    %[[OUT:.*]] = mo.add
// CHECk-NEXT:    mo.output %[[OUT]]

// x / (y / z) -> x * z / y
mo.graph @div_by_div_to_mul(%input1: !mo.tensor<[1, 2, 3], f32>, %input2: !mo.tensor<[1, 2, 3], f32>, %input3: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.div(%input1, %input2) : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.div(%input3, %0) : !mo.tensor<[1, 2, 3], f32>
  mo.output %1 : !mo.tensor<[1, 2, 3], f32>
}

// CHECK-LABEL:  @div_by_div_to_mul
// CHECK-NEXT:    %[[MUL:.*]] = mo.mul({{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:    %[[DIV:.*]] = mo.div(%[[MUL]], {{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:    mo.output %[[DIV]] : !mo.tensor<[1, 2, 3], f32>

// (x / y) / z -> x / (y * z)
mo.graph @div_of_div_to_div_of_mul(%input1: !mo.tensor<[1, 2, 3], f32>, %input2: !mo.tensor<[1, 2, 3], f32>, %input3: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.div(%input1, %input2) : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.div(%0, %input3) : !mo.tensor<[1, 2, 3], f32>
  mo.output %1 : !mo.tensor<[1, 2, 3], f32>
}

// CHECK-LABEL:  @div_of_div_to_div_of_mul
// CHECK-NEXT:    %[[MUL:.*]] = mo.mul({{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:    %[[DIV:.*]] = mo.div({{.*}}, %[[MUL]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:    mo.output %[[DIV]] : !mo.tensor<[1, 2, 3], f32>

// Make sure the rewrite won't fire for non-fp inputs.
// x / cst -> x * (1 / cst)
mo.graph @div_by_cst_to_mul_of_reciprocal_int(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %cst = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.div(%input, %cst) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL:  @div_by_cst_to_mul_of_reciprocal_int
// CHECK-NEXT:    %[[BCAST:.*]] = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6> : tensor<6xsi64>> : !mo.tensor<[1, 2, 3], si64>
// CHECK-NEXT:    %[[DIV:.*]] = mo.div({{.*}}) : !mo.tensor<[1, 2, 3], si64>
// CHECK-NEXT:    mo.output %[[DIV]] : !mo.tensor<[1, 2, 3], si64>

// x / cst -> x * (1 / cst)
mo.graph @div_by_cst_to_mul_of_reciprocal(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %cst = mo.constant <#M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<6xf32>> : !mo.tensor<[1, 2, 3], f32>
  %0 = mo.div(%input, %cst) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL:  @div_by_cst_to_mul_of_reciprocal
// CHECK:         %[[BCAST:.*]] = mo.static.broadcast_to({{.*}}) : !mo.tensor<[], f32> -> !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:    %[[DIV:.*]] = mo.div({{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:    %[[MUL:.*]] = mo.mul({{.*}}, %[[DIV]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:    mo.output %[[MUL]] : !mo.tensor<[1, 2, 3], f32>

// x / bcast(y) -> x * bcast(1 / y) when each element of y feeds enough outputs.
// The fan-out counts only static broadcast dims, so a symbolic row count on
// a per-row scale still qualifies.
mo.graph @div_by_bcast_small_symbolic_rows<N>(%input: !mo.tensor<[N, 16], f32>, %scale: !mo.tensor<[N, 1], f32>) -> !mo.tensor<[N, 16], f32> {
  %bcast = mo.static.broadcast_to(%scale) : !mo.tensor<[N, 1], f32> -> !mo.tensor<[N, 16], f32>
  %0 = mo.div(%input, %bcast) : !mo.tensor<[N, 16], f32>
  mo.output %0 : !mo.tensor<[N, 16], f32>
}
// CHECK-LABEL:  @div_by_bcast_small_symbolic_rows
// CHECK-SAME:     <[[N:.*]]>(%[[INPUT:.*]]: !mo.tensor<[[[N]], 16], f32>, %[[SCALE:.*]]: !mo.tensor<[[[N]], 1], f32>
// CHECK:         %[[ONE:.*]] = mo.static.broadcast_to({{.*}}) : !mo.tensor<[], f32> -> !mo.tensor<[[[N]], 1], f32>
// CHECK-NEXT:    %[[RECIP:.*]] = mo.div(%[[ONE]], %[[SCALE]]) : !mo.tensor<[[[N]], 1], f32>
// CHECK-NEXT:    %[[BCAST:.*]] = mo.static.broadcast_to(%[[RECIP]]) : !mo.tensor<[[[N]], 1], f32> -> !mo.tensor<[[[N]], 16], f32>
// CHECK-NEXT:    %[[MUL:.*]] = mo.mul(%[[INPUT]], %[[BCAST]]) : !mo.tensor<[[[N]], 16], f32>
// CHECK-NEXT:    mo.output %[[MUL]] : !mo.tensor<[[[N]], 16], f32>

// A static fan-out below the threshold keeps the division.
mo.graph @div_by_bcast_small_fan_out<N>(%input: !mo.tensor<[N, 4], f32>, %scale: !mo.tensor<[N, 1], f32>) -> !mo.tensor<[N, 4], f32> {
  %bcast = mo.static.broadcast_to(%scale) : !mo.tensor<[N, 1], f32> -> !mo.tensor<[N, 4], f32>
  %0 = mo.div(%input, %bcast) : !mo.tensor<[N, 4], f32>
  mo.output %0 : !mo.tensor<[N, 4], f32>
}
// CHECK-LABEL:  @div_by_bcast_small_fan_out
// CHECK-NOT:     mo.mul
// CHECK:         mo.div
// CHECK-NOT:     mo.mul

// A fan-out along only a symbolic dim is unknown, so the division stays.
mo.graph @div_by_bcast_symbolic_fan_out<N>(%input: !mo.tensor<[N, 16], f32>, %scale: !mo.tensor<[1, 16], f32>) -> !mo.tensor<[N, 16], f32> {
  %bcast = mo.static.broadcast_to(%scale) : !mo.tensor<[1, 16], f32> -> !mo.tensor<[N, 16], f32>
  %0 = mo.div(%input, %bcast) : !mo.tensor<[N, 16], f32>
  mo.output %0 : !mo.tensor<[N, 16], f32>
}
// CHECK-LABEL:  @div_by_bcast_symbolic_fan_out
// CHECK-NOT:     mo.mul
// CHECK:         mo.div
// CHECK-NOT:     mo.mul

mo.graph @pow_float_to_pow_int(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], f32> {
  %cst = mo.constant <#M.dense_array<5.> : tensor<1xf32>> : !mo.tensor<[], f32>
  %cst_bcast = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[1, 2, 3], f32>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], si64>, %cst_bcast : !mo.tensor<[1, 2, 3], f32>) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL:  @pow_float_to_pow_int
// CHECK-NEXT:    %[[CST:.*]] = mo.constant <#M.dense_array<5> : !M.array<1xsi32>> : !mo.tensor<[], si32>
// CHECK-NEXT:    %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]]) : !mo.tensor<[], si32> -> !mo.tensor<[1, 2, 3], si32>
// CHECK-NEXT:    %2 = mo.pow({{.*}}, %[[BCAST]] : !mo.tensor<[1, 2, 3], si32>) : !mo.tensor<[1, 2, 3], f32>

mo.graph @pow_float_to_pow_float(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %cst = mo.constant <#M.dense_array<5.> : tensor<1xf32>> : !mo.tensor<[], f32>
  %cst_bcast = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[1, 2, 3], f32>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], f32>, %cst_bcast : !mo.tensor<[1, 2, 3], f32>) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL:  @pow_float_to_pow_float
// CHECK-NEXT:    %[[CST:.*]] = mo.constant <#M.dense_array<5> : !M.array<1xsi32>> : !mo.tensor<[], si32>
// CHECK-NEXT:    %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]]) : !mo.tensor<[], si32> -> !mo.tensor<[1, 2, 3], si32>
// CHECK-NEXT:    %2 = mo.pow({{.*}}, %[[BCAST]] : !mo.tensor<[1, 2, 3], si32>) : !mo.tensor<[1, 2, 3], f32>

mo.graph @pow_float16_to_pow_float16(%input: !mo.tensor<[1, 2, 3], f16>) -> !mo.tensor<[1, 2, 3], f16> {
  %cst = mo.constant <#M.dense_array<5.> : tensor<1xf16>> : !mo.tensor<[], f16>
  %cst_bcast = mo.static.broadcast_to(%cst) : !mo.tensor<[], f16> -> !mo.tensor<[1, 2, 3], f16>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], f16>, %cst_bcast : !mo.tensor<[1, 2, 3], f16>) : !mo.tensor<[1, 2, 3], f16>
  mo.output %0 : !mo.tensor<[1, 2, 3], f16>
}
// CHECK-LABEL:  @pow_float16_to_pow_float16
// CHECK-NEXT:    %[[CST:.*]] = mo.constant <#M.dense_array<5> : !M.array<1xsi32>> : !mo.tensor<[], si32>
// CHECK-NEXT:    %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]]) : !mo.tensor<[], si32> -> !mo.tensor<[1, 2, 3], si32>
// CHECK-NEXT:    %2 = mo.pow({{.*}}, %[[BCAST]] : !mo.tensor<[1, 2, 3], si32>) : !mo.tensor<[1, 2, 3], f16>

// x ** 2 -> x * x
mo.graph @pow_square_to_mul(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %cst = mo.constant <#M.dense_array<2.,2.,2.,2.,2.,2.> : tensor<6xf32>> : !mo.tensor<[1, 2, 3], f32>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], f32>, %cst : !mo.tensor<[1, 2, 3], f32>) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: @pow_square_to_mul
// CHECK-NEXT:   %[[SQUARE:.*]] = mo.mul({{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:   mo.output %[[SQUARE]] : !mo.tensor<[1, 2, 3], f32>

// x ** 3 -> x * x * x
mo.graph @pow_cube_to_mul(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %cst = mo.constant <#M.dense_array<3.,3.,3.,3.,3.,3.> : tensor<6xf32>> : !mo.tensor<[1, 2, 3], f32>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], f32>, %cst : !mo.tensor<[1, 2, 3], f32>) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: @pow_cube_to_mul
// CHECK-NEXT:   %[[SQUARE:.*]] = mo.mul({{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:   %[[CUBE:.*]] = mo.mul(%[[SQUARE]], {{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT:   mo.output %[[CUBE]] : !mo.tensor<[1, 2, 3], f32>