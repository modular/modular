// RUN: mo-opt %s --gc-pipeline -o %t.mlir
// RUN: mt --execute --input-shapes=2x2x4 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s --check-prefix=CHECK-ONES
// RUN: mo-opt %s --gc-pipeline -o %t.mlir
// RUN: mt --execute --input-shapes=2x2x4 -model-inputs=zeros --result-output-style=full %t.mlir | FileCheck %s --check-prefix=CHECK-ZEROS

mo.graph @equal<D0, D1, D2>(%arg0 : !mo.tensor<[D0, D1, D2], si64>) -> !mo.tensor<[D0, D1, D2], bool> {
  %const  = mo.constant <#M.dense_array<0, 1, 0, 1, 2, 3, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0> : tensor<16xsi64>> : !mo.tensor<[16], si64>
  %shape = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, D2], si64>) -> !mo.tensor<[3], si64>
  %const_vals = mo.reshape (%const, %shape) : (!mo.tensor<[16], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[D0, D1, D2], si64>
  %res   = mo.equal(%const_vals, %arg0) : (!mo.tensor<[D0, D1, D2], si64>, !mo.tensor<[D0, D1, D2], si64>) -> !mo.tensor<[D0, D1, D2], bool>
  mo.output %res : !mo.tensor<[D0, D1, D2], bool>
}

// CHECK-IR: mgp.generic.execute
// CHECK-ONES-LABEL:--- Running 'equal':
// CHECK-ONES-NEXT: 'equal' returned tensor<2x2x4xbool> [False, True, False, True, False, False, False, True, True, False, False, False, False, True, True, False]

// CHECK-ZEROS-LABEL:--- Running 'equal':
// CHECK-ZEROS-NEXT: 'equal' returned tensor<2x2x4xbool> [True, False, True, False, False, False, True, False, False, True, True, True, True, False, False, True]

mo.graph @equal_bools(%arg0 : !mo.tensor<[2, 2, 4], bool>) -> (!mo.tensor<[2, 2, 4], bool>) {
  %const  = mo.constant <#M.dense_array<0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0> : tensor<16xsi64>> : !mo.tensor<[2, 2, 4], si64>
  %cast = mo.cast(%const): (!mo.tensor<[2, 2, 4], si64>) -> !mo.tensor<[2, 2, 4], bool>
  %res  = mo.equal(%cast, %arg0) : (!mo.tensor<[2, 2, 4], bool>, !mo.tensor<[2, 2, 4], bool>) -> !mo.tensor<[2, 2, 4], bool>
  mo.output %res : !mo.tensor<[2, 2, 4], bool>
}

// CHECK-IR: mgp.generic.execute
// CHECK-ONES-LABEL:--- Running 'equal_bools':
// CHECK-ONES-NEXT: 'equal_bools' returned tensor<2x2x4xbool> [False, True, False, True, False, True, False, True, True, False, False, False, False, True, True, False]

// CHECK-ZEROS-LABEL:--- Running 'equal_bools':
// CHECK-ZEROS-NEXT: 'equal_bools' returned tensor<2x2x4xbool> [True, False, True, False, True, False, True, False, False, True, True, True, True, False, False, True]


mo.graph @not_equal<D0, D1, D2>(%arg0 : !mo.tensor<[D0, D1, D2], si64>) -> !mo.tensor<[D0, D1, D2], bool> {
  %const  = mo.constant <#M.dense_array<0, 1, 0, 1, 2, 3, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0> : tensor<16xsi64>> : !mo.tensor<[16], si64>
  %shape = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, D2], si64>) -> !mo.tensor<[3], si64>
  %const_vals = mo.reshape (%const, %shape) : (!mo.tensor<[16], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[D0, D1, D2], si64>
  %res   = mo.not_equal(%const_vals, %arg0) : (!mo.tensor<[D0, D1, D2], si64>, !mo.tensor<[D0, D1, D2], si64>) -> !mo.tensor<[D0, D1, D2], bool>
  mo.output %res : !mo.tensor<[D0, D1, D2], bool>
}

// CHECK-IR: mgp.generic.execute
// CHECK-ONES-LABEL:--- Running 'not_equal':
// CHECK-ONES-NEXT: 'not_equal' returned tensor<2x2x4xbool> [True, False, True, False, True, True, True, False, False, True, True, True, True, False, False, True]

// CHECK-ZEROS-LABEL:--- Running 'not_equal':
// CHECK-ZEROS-NEXT: 'not_equal' returned tensor<2x2x4xbool> [False, True, False, True, True, True, False, True, True, False, False, False, False, True, True, False]


mo.graph @not_equal_bools(%arg0 : !mo.tensor<[2, 2, 4], bool>) -> (!mo.tensor<[2, 2, 4], bool>) {
  %const  = mo.constant <#M.dense_array<0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0> : tensor<16xsi64>> : !mo.tensor<[2, 2, 4], si64>
  %cast = mo.cast(%const): (!mo.tensor<[2, 2, 4], si64>) -> !mo.tensor<[2, 2, 4], bool>
  %res  = mo.not_equal(%cast, %arg0) : (!mo.tensor<[2, 2, 4], bool>, !mo.tensor<[2, 2, 4], bool>) -> !mo.tensor<[2, 2, 4], bool>
  mo.output %res : !mo.tensor<[2, 2, 4], bool>
}

// CHECK-IR: mgp.generic.execute
// CHECK-ONES-LABEL:--- Running 'not_equal_bools':
// CHECK-ONES-NEXT: 'not_equal_bools' returned tensor<2x2x4xbool> [True, False, True, False, True, False, True, False, False, True, True, True, True, False, False, True]

// CHECK-ZEROS-LABEL:--- Running 'not_equal_bools':
// CHECK-ZEROS-NEXT: 'not_equal_bools' returned tensor<2x2x4xbool> [False, True, False, True, False, True, False, True, True, False, False, False, False, True, True, False]


mo.graph @greater(%arg0 : !mo.tensor<[2, 2, 4], si64>) -> (!mo.tensor<[2, 2, 4], bool>) {
  %const = mo.constant <#M.dense_array<0, 2, 0, 6, 10, 4, 14, 15, 15, 15, 4, 143, -3, 1432, -242, 24242> : tensor<16xsi64>> : !mo.tensor<[2, 2, 4], si64>
  %const1 = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> : tensor<16xsi64>> : !mo.tensor<[2, 2, 4], si64>
  %add = mo.add(%const1, %arg0) : !mo.tensor<[2, 2, 4], si64>
  %res = mo.greater(%const, %add) : (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[2, 2, 4], si64>) -> !mo.tensor<[2, 2, 4], bool>
  mo.output %res : !mo.tensor<[2, 2, 4], bool>
}

// CHECK-IR: mgp.generic.execute
// CHECK-ONES-LABEL:--- Running 'greater':
// CHECK-ONES-NEXT: 'greater' returned tensor<2x2x4xbool> [False, False, False, True, True, False, True, True, True, True, False, True, False, True, False, True]

mo.graph @greater_equal(%arg0 : !mo.tensor<[2, 2, 4], si64>) -> (!mo.tensor<[2, 2, 4], bool>) {
  %const = mo.constant <#M.dense_array<0, 2, 0, 6, 10, 6, 14, 15, 15, 15, 4, 143, -3, 1432, -242, 24242> : tensor<16xsi64>> : !mo.tensor<[2, 2, 4], si64>
  %const1 = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> : tensor<16xsi64>> : !mo.tensor<[2, 2, 4], si64>
  %add = mo.add(%const1, %arg0) : !mo.tensor<[2, 2, 4], si64>
  %res = mo.greater_equal(%const, %add) : (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[2, 2, 4], si64>) -> !mo.tensor<[2, 2, 4], bool>
  mo.output %res : !mo.tensor<[2, 2, 4], bool>
}

// CHECK-IR: mgp.generic.execute
// CHECK-ONES-LABEL:--- Running 'greater_equal':
// CHECK-ONES-NEXT: 'greater_equal' returned tensor<2x2x4xbool> [False, True, False, True, True, True, True, True, True, True, False, True, False, True, False, True]
