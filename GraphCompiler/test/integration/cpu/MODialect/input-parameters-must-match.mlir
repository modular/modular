// RUN: mo-opt --gc-pipeline="constant-fold=false" %s -o %t.mlir

// RUN: not mt --execute  --model-inputs=ones --input-shapes=7x10 --input-shapes=10x10 --result-output-style=full %t.mlir | FileCheck --enable-var-scope --check-prefix=CHECK-FIRST-DIM %s

// RUN: not mt --execute  --model-inputs=ones --input-shapes=10x7 --input-shapes=10x10 --result-output-style=full %t.mlir | FileCheck --enable-var-scope --check-prefix=CHECK-SECOND-DIM %s

// CHECK-FIRST-DIM-LABEL: --- Running 'dims_must_match':
// CHECK-FIRST-DIM: 'dims_must_match' returned <error:
// CHECK-FIRST-DIM: symbolic dimension 'D0' for input 1 does not match prior uses of that dimension>

// CHECK-SECOND-DIM-LABEL: --- Running 'dims_must_match':
// CHECK-SECOND-DIM: 'dims_must_match' returned <error:
// CHECK-SECOND-DIM: symbolic dimension 'D1' for input 1 does not match prior uses of that dimension>
mo.graph @dims_must_match<D0, D1>(%X: !mo.tensor<[D0, D1], f32>, %Y: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
{
  %Z = mo.add(%X, %Y) : !mo.tensor<[D0, D1], f32>
  mo.output %Z: !mo.tensor<[D0, D1], f32>
}
