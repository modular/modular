// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @add() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.constant <#M.dense_array<3.0, 1.0, -6.0, 2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'add' returned tensor<5xf32> [2.000000e+00, 3.000000e+00, -9.000000e+00, -2.000000e+00, 1.000000e+01]


mo.graph @and() -> (!mo.tensor<[4], bool>) {
  %lhs = mo.constant <#M.dense_array<true, true, false, false> : tensor<4xi1>> : !mo.tensor<[4], bool>
  %rhs = mo.constant <#M.dense_array<true, false, true, false> : tensor<4xi1>> : !mo.tensor<[4], bool>
  %res = mo.and(%lhs, %rhs) : !mo.tensor<[4], bool>
  mo.output %res : !mo.tensor<[4], bool>
}
// CHECK{LITERAL}: 'and' returned tensor<4xbool> [True, False, False, False]


mo.graph @max() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.constant <#M.dense_array<3.0, 1.0, -6.0, 2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %2 = mo.max(%0, %1) : !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'max' returned tensor<5xf32> [3.000000e+00, 2.000000e+00, -3.000000e+00, 2.000000e+00, 5.000000e+00]


mo.graph @mul() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.constant <#M.dense_array<3.0, 1.0, -6.0, 2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %2 = mo.mul(%0, %1) : !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'mul' returned tensor<5xf32> [-3.000000e+00, 2.000000e+00, 1.800000e+01, -8.000000e+00, 2.500000e+01]


mo.graph @min() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.constant <#M.dense_array<3.0, 1.0, -6.0, 2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %2 = mo.min(%0, %1) : !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'min' returned tensor<5xf32> [-1.000000e+00, 1.000000e+00, -6.000000e+00, -4.000000e+00, 5.000000e+00]


mo.graph @mod_integer() -> (!mo.tensor<[5], si32>) {
  %0 = mo.constant <#M.dense_array<-1, 2, -3, -4, 5> : tensor<5xsi32>> : !mo.tensor<[5], si32>
  %1 = mo.constant <#M.dense_array<3, 1, -6, 2, 5> : tensor<5xsi32>> : !mo.tensor<[5], si32>
  %2 = mo.mod(%0, %1) : !mo.tensor<[5], si32>
  mo.output %2 : !mo.tensor<[5], si32>
}
// CHECK{LITERAL}: 'mod_integer' returned tensor<5xsi32> [2, 0, -3, 0, 0]


mo.graph @mod_float() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.1, 2.1, -3.1, -4.1, 5.1> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.constant <#M.dense_array<3.2, 1.2, -6.2, 2.2, 5.2> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %2 = mo.mod(%0, %1) : !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'mod_float' returned tensor<5xf32> [2.100000e+00, 8.999999e-01, -3.100000e+00, 3.000002e-01, 5.100000e+00]


mo.graph @not() -> (!mo.tensor<[4], bool>) {
  %in = mo.constant <#M.dense_array<true, true, false, false> : tensor<4xi1>> : !mo.tensor<[4], bool>
  %res = mo.not(%in) : !mo.tensor<[4], bool>
  mo.output %res : !mo.tensor<[4], bool>
}
// CHECK{LITERAL}: 'not' returned tensor<4xbool> [False, False, True, True]


mo.graph @xor() -> (!mo.tensor<[4], bool>) {
  %lhs = mo.constant <#M.dense_array<true, true, false, false> : tensor<4xi1>> : !mo.tensor<[4], bool>
  %rhs = mo.constant <#M.dense_array<true, false, true, false> : tensor<4xi1>> : !mo.tensor<[4], bool>
  %res = mo.xor(%lhs, %rhs) : !mo.tensor<[4], bool>
  mo.output %res : !mo.tensor<[4], bool>
}
// CHECK{LITERAL}: 'xor' returned tensor<4xbool> [False, True, True, False]


mo.graph @select() -> (!mo.tensor<[5], f32>) {
  %c = mo.constant <#M.dense_array<true, false, true, true, false> : tensor<5xui8>> : !mo.tensor<[5], bool>
  %0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.constant <#M.dense_array<3.0, 1.0, -6.0, 2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %2 = mo.select(%c, %0, %1) : (!mo.tensor<[5], bool>, !mo.tensor<[5], f32>, !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'select' returned tensor<5xf32> [-1.000000e+00, 1.000000e+00, -3.000000e+00, -4.000000e+00, 5.000000e+00]

mo.graph @select_bool() -> !mo.tensor<[], bool> {
  %1 = mo.constant <#M.dense_array<-1> : tensor<si64>> : !mo.tensor<[], si64>
  %3 = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[], si64>
  %7 = mo.constant <#M.dense_array<false> : tensor<1xui8>> : !mo.tensor<[], bool>
  %8 = mo.constant <#M.dense_array<true> : tensor<1xui8>> : !mo.tensor<[], bool>
  %9 = mo.greater_equal(%1, %3) : (!mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[], bool>
  %12 = mo.select(%9, %7, %8) : (!mo.tensor<[], bool>, !mo.tensor<[], bool>, !mo.tensor<[], bool>) -> !mo.tensor<[], bool>
  mo.output %12 : !mo.tensor<[], bool>
}

mo.graph @sub() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.constant <#M.dense_array<3.0, 1.0, -6.0, 2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %2 = mo.sub(%0, %1) : !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'sub' returned tensor<5xf32> [-4.000000e+00, 1.000000e+00, 3.000000e+00, -6.000000e+00, 0.000000e+00]


mo.graph @relu() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.relu(%0) : !mo.tensor<[5], f32>
  mo.output %1 : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'relu' returned tensor<5xf32> [0.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00, 5.000000e+00]


mo.graph @combined() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.constant <#M.dense_array<3.0, 1.0, -6.0, 2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %add = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %relu = mo.relu(%add) : !mo.tensor<[5], f32>
  mo.output %relu : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'combined' returned tensor<5xf32> [2.000000e+00, 3.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+01]


mo.graph @erf() -> (!mo.tensor<[6], f32>) {
  %0 = mo.constant <#M.dense_array<-10.0, -1.0, -0.5, 0.5, 1.0, 10.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %1 = mo.erf(%0) : !mo.tensor<[6], f32>
  mo.output %1 : !mo.tensor<[6], f32>
}
// CHECK{LITERAL}: 'erf' returned tensor<6xf32> [-1.000000e+00, -8.427008e-01, -5.204999e-01, 5.204999e-01, 8.427008e-01, 1.000000e+00]
