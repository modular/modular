// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=10x10 --result-output-style=full %t.mlir | FileCheck %s --check-prefix=CHECK-ONES
// RUN: mt --execute --model-inputs=zeros --input-shapes=10x10 --result-output-style=full %t.mlir | FileCheck %s --check-prefix=CHECK-ZEROS

// A test to ensure we fully lower dynamic cases of ops. shape function x MOGG

mo.graph @arg_nonzero<D0, D1 -> D2>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, 2], si64> {
  %1 = mo.arg_nonzero<() -> D3>(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D3, 2], si64>
  mo.output<D3> %1 : !mo.tensor<[D3, 2], si64>
}

// CHECK-ONES-LABEL:--- Running 'arg_nonzero':
// CHECK-ONES-NEXT: 'arg_nonzero' returned tensor<100x2xsi64>

// CHECK-ZEROS-LABEL:--- Running 'arg_nonzero':
// CHECK-ZEROS-NEXT: 'arg_nonzero' returned tensor<0x2xsi64>
