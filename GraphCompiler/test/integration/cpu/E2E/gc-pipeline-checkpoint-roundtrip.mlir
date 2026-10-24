// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_MAX_TEMPS_DIR=%t mo-opt %s --gc-pipeline=constant-fold=false -o %t/out.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t/out.mlir | FileCheck %s

// Each checkpoint the gc-pipeline dumps re-parses and survives a text and
// bytecode roundtrip.
// RUN: mo-opt %t/matmul_add.mo.mlir --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.mo-pre-map.mlir --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.mo-to-map.mlir --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.map-pre-mogg.mlir --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.map-to-mogg.mlir --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.mogg-pre-mgp.mlir --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.mogg-to-mgp.mlir --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.mgp.mlir --verify-roundtrip -o /dev/null

// The pipeline resumes from each checkpoint: every stage runs on the previous
// stage's dump, and the model compiled from the last dump still executes.
// RUN: mo-opt %t/matmul_add.mo.mlir --mo-pipeline=constant-fold=false --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.mo-pre-map.mlir --mo-to-map --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.mo-to-map.mlir --map-pipeline --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.map-pre-mogg.mlir --map-to-mogg --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.map-to-mogg.mlir --mogg-pipeline --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.mogg-pre-mgp.mlir --mogg-to-mgp --verify-roundtrip -o /dev/null
// RUN: mo-opt %t/matmul_add.mogg-to-mgp.mlir --mgp-pipeline --jit --mogg-strip-kernel-decls --jit-pipeline --verify-roundtrip -o %t/resumed.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t/resumed.mlir | FileCheck %s

// CHECK-LABEL: Running 'matmul_add':
// CHECK: 'matmul_add' returned tensor<4x4xf32> [5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00]

mo.graph @matmul_add(%a: !mo.tensor<[4, 4], f32>, %b: !mo.tensor<[4, 4], f32>, %c: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32> {
  %0 = mo.matmul(%a, %b) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %1 = mo.add(%0, %c) : !mo.tensor<[4, 4], f32>
  mo.output %1 : !mo.tensor<[4, 4], f32>
}
